#!/usr/bin/env python
"""Experiment toolkit for iterating on the CAR-bench agent.

Usage (from repo root):
    uv run python tools/experiment.py run --variant english_basic --smoke
    uv run python tools/experiment.py run --variant german_basic --split train --tasks 10 --trials 3
    uv run python tools/experiment.py analyze --run latest
    uv run python tools/experiment.py leaderboard

The loop: run → analyze → edit src/my_agent/prompts.py → run again.
Screen variants on the train split / small subsets; confirm finalists on the
full public test set (--split test --full --trials 3).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # the toolkit reads USER_SIM_MODEL, JUDGE_MODEL, LITE_LLM_* itself

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_FILE = EXPERIMENTS_DIR / "runs.jsonl"
RAW_DIR = EXPERIMENTS_DIR / "raw"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
SCENARIOS_DIR = EXPERIMENTS_DIR / "scenarios"
EVENTS_DIR = EXPERIMENTS_DIR / "guard_events"

sys.path.insert(0, str(REPO_ROOT / "src" / "my_agent"))
from prompts import PROMPT_VARIANTS  # noqa: E402
from observability import normalize_litellm_proxy_env  # noqa: E402
import guard_events  # noqa: E402
sys.path.pop(0)

MAX_JUDGED_FAILURES = 20


# ---------------------------------------------------------------- helpers

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return [json.loads(line) for line in RUNS_FILE.read_text().splitlines() if line.strip()]


def _append_run(record: dict) -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


TASK_SPLITS = REPO_ROOT / "third_party" / "car-bench" / "docs" / "reference_data" / "tasks" / "task_splits.json"
CATEGORIES = ("base", "hallucination", "disambiguation")


def load_task_ids(split: str, per_category: int) -> dict:
    """{category: [task_id, ...]} for a split, truncated to per_category.

    Sharding needs explicit task ids because the evaluator selects a shard via
    tasks_<cat>_task_id_filter (honoured in car-bench run.py), not an offset.
    """
    data = json.loads(TASK_SPLITS.read_text())
    out = {}
    for cat in CATEGORIES:
        ids = data.get(f"{cat}_{split}", [])
        out[cat] = ids if per_category in (-1, None) else ids[:per_category]
    return out


def _merge_shards(raw_paths: list[Path]) -> dict:
    """Recompute Pass^k across shards.

    Pass^k cannot be averaged across shards — it is the fraction of *tasks*
    whose every trial passed, so the per-task trial lists must be pooled first
    and the metric recomputed on the union.
    """
    by_split: dict = {}
    totals = {"tokens": 0, "cost": 0.0, "rows": 0, "fails": 0}
    for p in raw_paths:
        if not p.exists():
            continue
        final = (json.loads(p.read_text()).get("final_result") or {})
        for split, rows in (final.get("detailed_results_by_split") or {}).items():
            by_split.setdefault(split, []).extend(rows or [])
            for r in rows or []:
                totals["rows"] += 1
                totals["tokens"] += int(r.get("agent_total_tokens") or 0)
                totals["cost"] += r.get("total_agent_cost") or 0.0
                if (r.get("reward") or 0) < 1:
                    totals["fails"] += 1

    def passes(rows):
        per_task: dict = {}
        for r in rows:
            per_task.setdefault(r.get("task_id"), []).append((r.get("reward") or 0) >= 1)
        if not per_task:
            return None, None, 0
        k = max(len(v) for v in per_task.values())
        p_hat = sum(1 for v in per_task.values() if v and all(v)) / len(per_task)
        p_at = sum(1 for v in per_task.values() if any(v)) / len(per_task)
        return p_hat, p_at, k

    all_rows = [r for rows in by_split.values() for r in rows]
    p_hat, p_at, k = passes(all_rows)
    by_split_scores = {}
    for split, rows in by_split.items():
        sh, sa, sk = passes(rows)
        if sh is not None:
            by_split_scores[split] = {f"Pass^{sk}": sh}
    score = sum(1 for rows in [all_rows] for r in rows if (r.get("reward") or 0) >= 1)
    return {
        "score": float(score),
        "max_score": float(totals["rows"]),
        "pass_rate": (score / totals["rows"] * 100) if totals["rows"] else 0.0,
        "pass_power_k_scores": {f"Pass^{k}": p_hat} if p_hat is not None else {},
        "pass_at_k_scores": {f"Pass@{k}": p_at} if p_at is not None else {},
        "pass_power_k_scores_by_split": by_split_scores,
        "detailed_results_by_split": by_split,
        "_totals": totals,
    }


def _port_busy(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _preflight_ports(ports: list[int]) -> None:
    """Fail fast and legibly on an occupied port.

    The orchestrator's symptom for this is 'Agent process exited before
    readiness', which reads like an agent bug and costs a debugging session.
    The usual cause is another run still in flight — so name the occupant
    rather than suggesting anything be killed.
    """
    busy = [p for p in ports if _port_busy(p)]
    if not busy:
        return
    import shutil
    import subprocess as sp
    detail = ""
    if shutil.which("lsof"):
        try:
            out = sp.run(["lsof", "-nP", "-sTCP:LISTEN",
                          *[f"-iTCP:{p}" for p in busy]],
                         capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                detail = "\n" + out
        except Exception:
            pass
    raise SystemExit(
        f"[experiment] port(s) already in use: {busy}{detail}\n"
        f"[experiment] Another run is probably still going — check before killing "
        f"anything.\n[experiment] To run alongside it: --base-port "
        f"{max(busy) + 1}"
    )


def _free_port_pair(start: int = 8080) -> tuple[int, int]:
    """Two free localhost ports.

    Every run used to hardcode 8080/8081, so launching a second run while one
    was in flight failed the agent-readiness check and died leaving only an
    orphan scenario file — three ablation runs were lost that way before it
    was noticed. Bind-testing makes concurrent runs safe.
    """
    import socket

    found: list[int] = []
    port = start
    while len(found) < 2 and port < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                found.append(port)
            except OSError:
                pass
        port += 1
    if len(found) < 2:
        raise SystemExit("[experiment] no free port pair found in 8080-8280")
    return found[0], found[1]


def _iter_detailed_rows(final_result: dict):
    for split, rows in (final_result.get("detailed_results_by_split") or {}).items():
        for row in rows or []:
            yield split, row


# ---------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> None:
    if args.variant not in PROMPT_VARIANTS:
        raise SystemExit(f"Unknown variant '{args.variant}'. Available: {', '.join(PROMPT_VARIANTS)}")

    if args.smoke:
        split, tasks, trials = "train", 1, 1
    else:
        split = args.split
        tasks = -1 if args.full else args.tasks
        trials = args.trials

    variant_label = (args.variant + ("+selfcheck" if args.self_check else "")
                     + ("+askgate" if args.ask_gate else "")
                     + (f"+vote{args.vote}" if args.vote else "")
                     + ("+guard" if args.schema_guard else "")
                     + ("+firewall" if args.firewall else "")
                     # An ablated firewall must not share a label with the full
                     # one, or the leaderboard silently compares different agents.
                     + (f"[{args.firewall_checks}]"
                        if args.firewall and args.firewall_checks else ""))
    run_id = f"{_now_utc().strftime('%Y%m%d-%H%M%S')}-{variant_label}-{split}"
    raw_path = RAW_DIR / f"{run_id}.json"
    scenario_path = SCENARIOS_DIR / f"{run_id}.toml"

    # Auto-skip occupied ports. --base-port only helps if you remember to pass
    # it; three ablation runs were already lost to a silent collision with an
    # in-flight run, so the default has to be safe rather than merely settable.
    base_port = int(getattr(args, "base_port", 8080) or 8080)
    agent_port, eval_port = _free_port_pair(base_port)
    if (agent_port, eval_port) != (base_port, base_port + 1):
        print(f"[experiment] ports {base_port}/{base_port+1} busy — using {agent_port}/{eval_port}")
    scenario = {
        "evaluator": {
            "endpoint": f"http://127.0.0.1:{eval_port}",
            "cmd": f"python src/evaluator/server.py --host 127.0.0.1 --port {eval_port}",
        },
        "agent_under_test": {
            "endpoint": f"http://127.0.0.1:{agent_port}",
            "cmd": f"python src/my_agent/server.py --host 127.0.0.1 --port {agent_port}",
        },
        "config": {
            "num_trials": trials,
            "task_split": split,
            "tasks_base_num_tasks": tasks,
            "tasks_hallucination_num_tasks": tasks,
            "tasks_disambiguation_num_tasks": tasks,
            "max_steps": 50,
        },
    }
    # Optionally route the evaluator's simulated user / policy judge through a
    # different model (e.g. a self-hosted LiteLLM proxy: "litellm_proxy/<name>").
    # Official evaluation uses the organizers' defaults, so treat these as a
    # dev convenience, not something to tune.
    user_sim_model = os.getenv("USER_SIM_MODEL", "")
    if user_sim_model:
        scenario["config"]["user_model"] = user_sim_model
        scenario["config"]["user_provider"] = user_sim_model.split("/")[0]
    # user_thinking=True makes the simulator send reasoning_effort, which
    # custom_openai (self-hosted) backends reject. Set USER_SIM_THINKING=false
    # when the user simulator runs on a self-hosted model.
    user_sim_thinking = os.getenv("USER_SIM_THINKING", "")
    if user_sim_thinking:
        scenario["config"]["user_thinking"] = user_sim_thinking.lower() == "true"
    policy_eval_model = os.getenv("POLICY_EVAL_MODEL", "")
    if policy_eval_model:
        scenario["config"]["policy_evaluator_model"] = policy_eval_model
        scenario["config"]["policy_evaluator_provider"] = policy_eval_model.split("/")[0]
    import tomli_w
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(scenario_path, "wb") as f:
        tomli_w.dump(scenario, f)

    normalize_litellm_proxy_env()
    env = os.environ.copy()
    env["AGENT_PROMPT_VARIANT"] = args.variant
    env["RUN_ID"] = run_id
    if args.model:
        env["AGENT_LLM"] = args.model
    if args.self_check:
        env["AGENT_SELF_CHECK"] = "true"
    if args.ask_gate:
        env["AGENT_ASK_GATE"] = "true"
    if args.vote:
        env["AGENT_VOTE_K"] = str(args.vote)
    if args.schema_guard:
        env["AGENT_SCHEMA_GUARD"] = "true"
    if args.firewall:
        env["AGENT_FIREWALL"] = "true"
    if args.firewall_checks:
        env["AGENT_FIREWALL_CHECKS"] = args.firewall_checks
    # Per-run guard-event log: which mechanism fired, how often, on what. A
    # flat score with zero firings and a flat score with hundreds of firings
    # are different findings; the run record has to be able to say which.
    events_path = EVENTS_DIR / f"{run_id}.jsonl"
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    env["GUARD_EVENTS_PATH"] = str(events_path)

    print(f"[experiment] run_id={run_id}")
    print(f"[experiment] variant={args.variant} split={split} tasks/category={tasks} trials={trials}"
          f" model={args.model or env.get('AGENT_LLM', 'gemini/gemini-2.5-flash (default)')}")
    started_at = _now_utc()
    # Capture provenance at START. Read after the run instead, a long run that
    # spans an edit records the tree it finished next to rather than the tree
    # it actually ran — which silently misattributes the result to code that
    # never executed.
    git_sha, git_dirty = _git("rev-parse", "--short", "HEAD"), bool(_git("status", "--porcelain"))

    shards = max(1, int(getattr(args, "shards", 1) or 1))
    # Check every port this run will bind before spawning anything, so a
    # collision surfaces as a collision rather than as an agent failure.
    _preflight_ports([base_port + n for n in range(2 * shards)])
    if shards > 1:
        # The evaluator runs tasks sequentially (max_concurrency=1) and the work
        # is network-bound on the LLM proxy, so wall-clock scales ~1/N by
        # splitting the task ids across independent evaluator+agent pairs on
        # separate ports and pooling the results afterwards.
        ids_by_cat = load_task_ids(split, tasks)
        shard_procs, shard_raws = [], []
        for i in range(shards):
            s_scenario = json.loads(json.dumps(scenario))  # deep copy
            a_port, e_port = base_port + 2 * i, base_port + 1 + 2 * i
            s_scenario["agent_under_test"]["endpoint"] = f"http://127.0.0.1:{a_port}"
            s_scenario["agent_under_test"]["cmd"] = (
                f"python src/my_agent/server.py --host 127.0.0.1 --port {a_port}")
            s_scenario["evaluator"]["endpoint"] = f"http://127.0.0.1:{e_port}"
            s_scenario["evaluator"]["cmd"] = (
                f"python src/evaluator/server.py --host 127.0.0.1 --port {e_port}")
            for cat in CATEGORIES:
                shard_ids = ids_by_cat.get(cat, [])[i::shards]
                if shard_ids:
                    s_scenario["config"][f"tasks_{cat}_task_id_filter"] = shard_ids
                    s_scenario["config"][f"tasks_{cat}_num_tasks"] = -1
                else:
                    # No filter + 0 tasks, else the evaluator falls back to
                    # num_tasks and this shard re-runs the whole category.
                    s_scenario["config"].pop(f"tasks_{cat}_task_id_filter", None)
                    s_scenario["config"][f"tasks_{cat}_num_tasks"] = 0
            s_path = SCENARIOS_DIR / f"{run_id}-shard{i}.toml"
            s_raw = RAW_DIR / f"{run_id}-shard{i}.json"
            with open(s_path, "wb") as f:
                tomli_w.dump(s_scenario, f)
            s_env = dict(env)
            s_env["RUN_ID"] = f"{run_id}-shard{i}"
            # One events file per shard: these are separate processes, so a
            # shared file would interleave records past the atomic-append size.
            s_env["GUARD_EVENTS_PATH"] = str(EVENTS_DIR / f"{run_id}-shard{i}.jsonl")
            shard_raws.append(s_raw)
            shard_procs.append(subprocess.Popen(
                [sys.executable, "-m", "agentbeats.run_scenario",
                 str(s_path.relative_to(REPO_ROOT)), "--output", str(s_raw.relative_to(REPO_ROOT))],
                cwd=REPO_ROOT, env=s_env,
            ))
            counts = {c: len(ids_by_cat.get(c, [])[i::shards]) for c in CATEGORIES}
            print(f"[experiment]   shard {i}: ports {a_port}/{e_port}, tasks {counts}")
        rcs = [p.wait() for p in shard_procs]
        finished_at = _now_utc()
        ok = [r for r, rc in zip(shard_raws, rcs) if rc == 0 and r.exists()]
        if not ok:
            raise SystemExit("[experiment] all shards failed; no registry entry written")
        if len(ok) != shards:
            print(f"[experiment] WARNING: {shards - len(ok)}/{shards} shards failed — "
                  "metrics below cover only the surviving shards")
        final = _merge_shards(ok)
        payload = {"final_result": final, "metadata": {"model": env.get("AGENT_LLM")}}
        raw_path.write_text(json.dumps(payload, indent=2))
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "agentbeats.run_scenario",
             str(scenario_path.relative_to(REPO_ROOT)), "--output", str(raw_path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            env=env,
        )
        finished_at = _now_utc()
        if proc.returncode != 0:
            raise SystemExit(f"[experiment] benchmark run failed (exit {proc.returncode}); no registry entry written")
        if not raw_path.exists():
            raise SystemExit(f"[experiment] run finished but {raw_path} was not written")
        payload = json.loads(raw_path.read_text())
        final = payload.get("final_result") or {}

    total_agent_cost = 0.0
    total_tokens = 0
    failed_tasks = 0
    total_rows = 0
    for _, row in _iter_detailed_rows(final):
        total_rows += 1
        total_agent_cost += row.get("total_agent_cost") or 0.0
        total_tokens += int(row.get("agent_total_tokens") or 0)
        if (row.get("reward") or 0) < 1:
            failed_tasks += 1

    record = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "variant": variant_label,
        "model": payload.get("metadata", {}).get("model") or env.get("AGENT_LLM", "gemini/gemini-2.5-flash"),
        "split": split,
        "tasks_per_category": tasks,
        "num_trials": trials,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "score": final.get("score"),
        "max_score": final.get("max_score"),
        "pass_rate": final.get("pass_rate"),
        "pass_power_k": final.get("pass_power_k_scores"),
        "pass_at_k": final.get("pass_at_k_scores"),
        "pass_power_k_by_split": final.get("pass_power_k_scores_by_split"),
        "total_agent_cost": round(total_agent_cost, 4),
        "total_tokens": total_tokens,
        "failed_trials": failed_tasks,
        "total_trial_rows": total_rows,
        "output_path": str(raw_path.relative_to(REPO_ROOT)),
    }
    # Unsharded writes one file; sharded writes one per shard.
    event_files = sorted(EVENTS_DIR.glob(f"{run_id}*.jsonl"))
    guard_summary = guard_events.aggregate(event_files)
    if guard_summary:
        record["guard_events"] = guard_summary
        record["guard_events_path"] = [str(p.relative_to(REPO_ROOT)) for p in event_files]
    _append_run(record)

    print("\n[experiment] ===== run summary =====")
    print(f"  run_id:     {run_id}")
    print(f"  score:      {record['score']}/{record['max_score']} (pass_rate {record['pass_rate']})")
    print(f"  Pass^k:     {record['pass_power_k']}")
    print(f"  Pass@k:     {record['pass_at_k']}")
    print(f"  cost:       ${record['total_agent_cost']} agent-side, {total_tokens} tokens")
    print(f"  failures:   {failed_tasks}/{total_rows} trial rows")
    if guard_summary:
        print(f"  guards:     {guard_summary['total']} firings across "
              f"{guard_summary['contexts_touched']} contexts "
              f"{guard_summary['by_mechanism']}")
        if guard_summary.get("firewall_by_kind"):
            print(f"  firewall:   {guard_summary['firewall_by_kind']}")
    elif args.firewall or args.schema_guard:
        # The distinction that makes a flat result readable.
        print("  guards:     enabled but never fired — a flat score here is "
              "'no effect to measure', not 'mechanism did not help'")
    print(f"  raw result: {record['output_path']}")
    if failed_tasks:
        print(f"  next:       uv run python tools/experiment.py analyze --run {run_id}")


# ---------------------------------------------------------------- analyze

def _resolve_run(run_arg: str) -> dict:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No runs recorded yet. Start with: experiment.py run --variant baseline --smoke")
    if run_arg == "latest":
        return runs[-1]
    for record in reversed(runs):
        if record["run_id"] == run_arg:
            return record
    raise SystemExit(f"run_id '{run_arg}' not found in {RUNS_FILE}")


def cmd_analyze(args: argparse.Namespace) -> None:
    record = _resolve_run(args.run)
    raw_path = REPO_ROOT / record["output_path"]
    payload = json.loads(raw_path.read_text())
    final = payload.get("final_result") or {}

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from judge_prompts import build_judge_messages, format_failure
    sys.path.pop(0)

    failures = [(split, row) for split, row in _iter_detailed_rows(final) if (row.get("reward") or 0) < 1]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{record['run_id']}.md"

    header = (
        f"# Failure analysis — {record['run_id']}\n\n"
        f"- variant: `{record['variant']}` | model: `{record['model']}` | split: {record['split']}"
        f" | tasks/category: {record['tasks_per_category']} | trials: {record['num_trials']}\n"
        f"- Pass^k: {record.get('pass_power_k')} | Pass@k: {record.get('pass_at_k')}\n"
        f"- failed trial rows: {len(failures)}\n\n"
    )

    if not failures:
        report_path.write_text(header + "All tasks passed — nothing to analyze. "
                               "Consider a larger task subset or more trials.\n")
        print(f"[experiment] no failures in {record['run_id']}; report at {report_path}")
        return

    judged = failures[:MAX_JUDGED_FAILURES]
    dropped = len(failures) - len(judged)
    if dropped:
        print(f"[experiment] judging first {len(judged)} of {len(failures)} failures ({dropped} dropped — rerun with a smaller subset for full coverage)")

    variant = PROMPT_VARIANTS.get(record["variant"], {"prefix": "", "suffix": ""})
    messages = build_judge_messages(
        variant=record["variant"],
        prefix=variant["prefix"],
        suffix=variant["suffix"],
        failure_blocks=[format_failure(row, split) for split, row in judged],
    )

    normalize_litellm_proxy_env()
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5")
    print(f"[experiment] asking judge model {judge_model} to analyze {len(judged)} failures...")
    from litellm import completion
    # Reasoning judges (e.g. Kimi) can spend the whole completion budget on
    # internal reasoning, returning empty content — give headroom and retry
    # once with a bigger budget before falling back to the reasoning text.
    analysis = None
    for max_tokens in (32000, 64000):
        response = completion(model=judge_model, messages=messages, max_tokens=max_tokens)
        msg = response.choices[0].message
        analysis = msg.content
        if analysis:
            break
        finish = response.choices[0].finish_reason
        print(f"[experiment] judge returned empty content (finish_reason={finish}); retrying with max_tokens={max_tokens*2}")
    if not analysis:
        reasoning = getattr(msg, "reasoning_content", None)
        analysis = f"(judge returned no final content; raw reasoning follows)\n\n{reasoning}" if reasoning else "(judge returned no content)"

    note = f"\n> Note: {dropped} additional failures were not judged (cap {MAX_JUDGED_FAILURES}).\n" if dropped else ""
    report_path.write_text(header + note + "\n" + analysis + "\n")

    print(f"\n[experiment] report written to {report_path}\n")
    for line in analysis.splitlines():
        if line.startswith("#") or line.lstrip().startswith(("1.", "2.", "3.", "4.", "- **")):
            print(f"  {line.strip()}")


# ---------------------------------------------------------------- leaderboard

def cmd_leaderboard(_args: argparse.Namespace) -> None:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No runs recorded yet.")

    groups: dict[tuple, list[dict]] = {}
    for record in runs:
        key = (record["variant"], record["model"], record["split"])
        groups.setdefault(key, []).append(record)

    def _metric(record: dict, table: str, k: int):
        value = (record.get(table) or {}).get(f"Pass^{k}" if table == "pass_power_k" else f"Pass@{k}")
        return value

    header = f"{'variant':<18} {'model':<32} {'split':<6} {'tasks':>5} {'trials':>6} {'runs':>4} {'pass_rate':>9} {'Pass^3':>7} {'Pass@3':>7} {'$/run':>7}"
    print(header)
    print("-" * len(header))
    task_counts = set()
    for (variant, model, split), records in sorted(groups.items()):
        latest = records[-1]
        task_counts.add((split, latest["tasks_per_category"]))
        pass_rates = [r["pass_rate"] for r in records if r.get("pass_rate") is not None]
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else None
        p3 = _metric(latest, "pass_power_k", 3)
        a3 = _metric(latest, "pass_at_k", 3)
        costs = [r.get("total_agent_cost") or 0 for r in records]
        avg_cost = sum(costs) / len(costs) if costs else 0
        print(f"{variant:<18} {model:<32} {split:<6} {latest['tasks_per_category']:>5} {latest['num_trials']:>6} {len(records):>4} "
              f"{avg_pass_rate if avg_pass_rate is not None else '—':>9} "
              f"{p3 if p3 is not None else '—':>7} {a3 if a3 is not None else '—':>7} {avg_cost:>7.3f}")

    if len({tc for tc in task_counts}) > 1:
        print("\n⚠ runs cover different splits/task counts — compare rows only within the same split and task count.")
    print("\nPass^3/Pass@3 shown for the latest run of each group (needs trials ≥ 3).")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the benchmark for one prompt variant and record the result")
    p_run.add_argument("--variant", default="baseline", help=f"Prompt variant ({', '.join(PROMPT_VARIANTS)})")
    p_run.add_argument("--split", default="train", choices=["train", "test"], help="Task split (default train)")
    p_run.add_argument("--tasks", type=int, default=5, help="Tasks per category (default 5)")
    p_run.add_argument("--full", action="store_true", help="Run all tasks of the split (overrides --tasks)")
    p_run.add_argument("--trials", type=int, default=1, help="Trials per task (3 needed for Pass^3)")
    p_run.add_argument("--model", default=None, help="Override AGENT_LLM for this run")
    p_run.add_argument("--self-check", action="store_true", help="Enable the agent's pre-send self-check pass")
    p_run.add_argument("--ask-gate", action="store_true", help="Enable the preference-lookup nudge before clarifying questions")
    p_run.add_argument("--vote", type=int, default=0, help="Self-consistency voting: K samples per turn (0 = off)")
    p_run.add_argument("--schema-guard", action="store_true", help="Deterministic tool-call schema validation + corrective regen")
    p_run.add_argument("--firewall", action="store_true", help="Deterministic action firewall (ledger + provenance + compiled policy constraints)")
    p_run.add_argument("--base-port", type=int, default=8080,
                       help="First port to bind (agent=N, evaluator=N+1). Raise it to run "
                            "alongside another in-flight run (default 8080)")
    p_run.add_argument("--shards", type=int, default=1,
                       help="Split the task set across N parallel evaluator+agent pairs (~N x faster; the work is proxy-bound, not CPU-bound)")
    p_run.add_argument("--firewall-checks", default=None,
                       help="Ablate the firewall: comma-separated subset of precondition,default,provenance (default all)")
    p_run.add_argument("--smoke", action="store_true", help="Shortcut: train split, 1 task/category, 1 trial")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="LLM-judge failure clustering for a recorded run")
    p_analyze.add_argument("--run", default="latest", help="run_id or 'latest' (default)")
    p_analyze.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL env")
    p_analyze.set_defaults(func=cmd_analyze)

    p_board = sub.add_parser("leaderboard", help="Compare recorded runs by variant/model/split")
    p_board.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
