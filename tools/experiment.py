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

sys.path.insert(0, str(REPO_ROOT / "src" / "my_agent"))
from prompts import PROMPT_VARIANTS  # noqa: E402
from observability import normalize_litellm_proxy_env  # noqa: E402
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

    variant_label = args.variant + ("+selfcheck" if args.self_check else "") + ("+askgate" if args.ask_gate else "")
    run_id = f"{_now_utc().strftime('%Y%m%d-%H%M%S')}-{variant_label}-{split}"
    raw_path = RAW_DIR / f"{run_id}.json"
    scenario_path = SCENARIOS_DIR / f"{run_id}.toml"

    scenario = {
        "evaluator": {
            "endpoint": "http://127.0.0.1:8081",
            "cmd": "python src/evaluator/server.py --host 127.0.0.1 --port 8081",
        },
        "agent_under_test": {
            "endpoint": "http://127.0.0.1:8080",
            "cmd": "python src/my_agent/server.py --host 127.0.0.1 --port 8080",
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

    print(f"[experiment] run_id={run_id}")
    print(f"[experiment] variant={args.variant} split={split} tasks/category={tasks} trials={trials}"
          f" model={args.model or env.get('AGENT_LLM', 'gemini/gemini-2.5-flash (default)')}")
    started_at = _now_utc()
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
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
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
    _append_run(record)

    print("\n[experiment] ===== run summary =====")
    print(f"  run_id:     {run_id}")
    print(f"  score:      {record['score']}/{record['max_score']} (pass_rate {record['pass_rate']})")
    print(f"  Pass^k:     {record['pass_power_k']}")
    print(f"  Pass@k:     {record['pass_at_k']}")
    print(f"  cost:       ${record['total_agent_cost']} agent-side, {total_tokens} tokens")
    print(f"  failures:   {failed_tasks}/{total_rows} trial rows")
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
