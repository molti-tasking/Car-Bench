#!/usr/bin/env python
"""Simulator-noise audit for CAR-bench: quantifies how much of a run's failure
signal comes from the LLM user-simulator misfiring rather than from the agent.

A *misfire* is an unambiguous benchmark artifact: on a Hallucination or
Disambiguation task the simulator ends the conversation on the first turn
(user message == "###STOP###", the *_ERROR keyword fires) before the agent has
produced any tool call or substantive turn — so the agent is scored 0 for a
behavior it was never given the chance to exhibit.

Outputs, from the run JSONs already on disk (no benchmark calls):
  1. Corpus misfire rate and its share of all failures.
  2. Per-simulator-model misfire rate (which backends are unsafe as user sims).
  3. Variance decomposition on repeated-trial runs: agent-inconsistency vs
     simulator-induced flips.
  4. A fragile-task list: tasks that pass in some trials but misfire in others
     under the *same* agent — proof the flip is the simulator, not the agent.

Usage:  uv run python tools/simulator_audit.py [--json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = REPO_ROOT / "experiments" / "runs.jsonl"
RAW_DIR = REPO_ROOT / "experiments" / "raw"

# Categories whose scoring depends on the agent getting multiple turns; a
# first-turn kill on these cannot reflect real agent behavior.
MULTI_TURN_CATEGORIES = {"hallucination", "disambiguation"}


def _non_system(trajectory):
    return [m for m in (trajectory or []) if m.get("role") != "system"]


def is_misfire(row: dict, split: str) -> bool:
    """True iff this failed trial is a first-turn simulator kill on a category
    that requires the agent to act before it can be judged."""
    if (row.get("reward") or 0) >= 1:
        return False
    if split not in MULTI_TURN_CATEGORIES:
        return False
    traj = _non_system(row.get("trajectory"))
    first_user = next((m.get("content", "") for m in traj if m.get("role") == "user"), "")
    if first_user.strip() != "###STOP###":
        return False
    # Agent never produced a tool call (never had the chance to act).
    agent_toolcalls = sum(1 for m in traj if m.get("role") == "assistant" and m.get("tool_calls"))
    return agent_toolcalls == 0


def load_run_payloads():
    if not RUNS_FILE.exists():
        return []
    out = []
    for line in RUNS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        raw = REPO_ROOT / rec["output_path"]
        if raw.exists():
            out.append((rec, json.loads(raw.read_text())))
    return out


def sim_model(payload: dict) -> str:
    cfg = (payload.get("metadata") or {}).get("config") or {}
    return (cfg.get("user_model") or "unknown").split("/")[-1]


def audit():
    payloads = load_run_payloads()

    total_fail = total_misfire = total_rows = 0
    by_sim = defaultdict(lambda: {"fail": 0, "misfire": 0, "rows": 0})
    fragile = []  # (run_id, task_id, split, n_pass, n_misfire, n_other_fail)
    # Pass^3 impact: among tasks in >=3-trial runs that FAILED Pass^3 (not all
    # trials passed), how many would have passed if misfires didn't count?
    p3_broken = p3_broken_misfire_only = 0

    for rec, payload in payloads:
        sim = sim_model(payload)
        final = payload.get("final_result") or {}
        # group this run's rows by task to see within-agent trial flips
        per_task = defaultdict(list)
        for split, rows in (final.get("detailed_results_by_split") or {}).items():
            for row in rows or []:
                total_rows += 1
                by_sim[sim]["rows"] += 1
                mf = is_misfire(row, split)
                if (row.get("reward") or 0) < 1:
                    total_fail += 1
                    by_sim[sim]["fail"] += 1
                    if mf:
                        total_misfire += 1
                        by_sim[sim]["misfire"] += 1
                per_task[(split, row.get("task_id"))].append(
                    ("pass" if (row.get("reward") or 0) >= 1 else ("misfire" if mf else "fail"))
                )
        # fragile = same task, same agent/run, both a pass and a misfire across trials
        for (split, tid), outcomes in per_task.items():
            n_pass = outcomes.count("pass")
            n_mf = outcomes.count("misfire")
            n_of = outcomes.count("fail")
            if n_pass and n_mf:
                fragile.append((rec["run_id"], tid, split, n_pass, n_mf, n_of))
            # Pass^3 impact only meaningful for full 3-trial task rows
            if (rec.get("num_trials") or 0) >= 3 and len(outcomes) >= 3:
                if n_pass < len(outcomes):  # Pass^k broken (not all passed)
                    p3_broken += 1
                    # would flip to all-pass if misfires were the only non-passes
                    if n_of == 0 and n_mf > 0:
                        p3_broken_misfire_only += 1

    return {
        "total_rows": total_rows,
        "total_fail": total_fail,
        "total_misfire": total_misfire,
        "by_sim": {k: dict(v) for k, v in by_sim.items()},
        "fragile": fragile,
        "p3_broken": p3_broken,
        "p3_broken_misfire_only": p3_broken_misfire_only,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()
    a = audit()

    if args.json:
        print(json.dumps(a, indent=2))
        return

    tr, tf, tm = a["total_rows"], a["total_fail"], a["total_misfire"]
    print("CAR-bench simulator-noise audit")
    print("=" * 60)
    print(f"Trial rows analysed:        {tr}")
    print(f"Failed rows:                {tf}")
    print(f"  of which simulator misfire: {tm}  ({tm/tf*100:.1f}% of failures, "
          f"{tm/tr*100:.1f}% of all rows)")
    print()
    print("Per user-simulator model (failed rows -> misfire share):")
    print(f"  {'simulator':<18} {'rows':>6} {'fails':>6} {'misfire':>8} {'misfire/fail':>13}")
    for sim, s in sorted(a["by_sim"].items(), key=lambda kv: -kv[1]["misfire"]):
        share = f"{s['misfire']/s['fail']*100:.0f}%" if s["fail"] else "—"
        print(f"  {sim:<18} {s['rows']:>6} {s['fail']:>6} {s['misfire']:>8} {share:>13}")
    print()
    pb, pbm = a["p3_broken"], a["p3_broken_misfire_only"]
    print("Pass^3 impact (3-trial task instances):")
    print(f"  task-instances that failed Pass^3:            {pb}")
    print(f"  … that would flip to Pass^3-pass w/o misfires: {pbm}"
          f"  ({pbm/pb*100:.0f}% of Pass^3 breaks)" if pb else "  (none)")
    print()
    print(f"Fragile tasks (same agent, pass in some trials but misfire in others): {len(a['fragile'])}")
    for run_id, tid, split, np_, nm, nof in a["fragile"][:20]:
        print(f"  {tid:<18} {split:<15} pass×{np_} misfire×{nm} fail×{nof}   [{run_id}]")
    if len(a["fragile"]) > 20:
        print(f"  … and {len(a['fragile'])-20} more")
    print()
    print("Interpretation: misfire rows are scored 0 for a behavior the agent was")
    print("never given the turns to exhibit. They inflate apparent failure and add")
    print("Pass^k variance that is attributable to the simulator, not the agent.")


if __name__ == "__main__":
    main()
