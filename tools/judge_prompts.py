"""Judge prompts for failure analysis of CAR-bench runs."""
import json

JUDGE_SYSTEM_PROMPT = """You are an expert analyst of tool-using LLM agents, \
reviewing failed tasks from CAR-bench, an in-car voice-assistant benchmark. \
The agent under test receives a policy-laden system prompt, converses with an \
LLM-simulated user, and calls tools that the benchmark evaluator executes. \
Tasks come in three categories: Base (correct tool use, state, policy \
compliance), Hallucination (acknowledging missing capabilities/data instead \
of fabricating), and Disambiguation (resolving ambiguity via preferences or \
clarification before acting). A task fails when any required metric fails; \
the main competition metric is Pass^3 (pass all 3 trials), so consistency \
matters more than occasional brilliance.

Your job: root-cause analysis, not scoring — the evaluator already scored \
deterministically."""

JUDGE_TASK_PROMPT = """Below are the failed tasks from one benchmark run, each \
with its failed metrics (`reward_info`) and a compacted conversation \
trajectory. The agent's system prompt was the evaluator's policy prompt \
wrapped with the prompt-variant text shown first.

## Active prompt variant: {variant}

Prefix (prepended before the policy text):
```
{prefix}
```

Suffix (appended after the policy text):
```
{suffix}
```

## Failed tasks

{failures}

## Your analysis

Produce a markdown report with exactly these sections:

1. **Failure clusters** — group the failures into named failure modes (e.g.
   "acts before clarifying", "fabricates unavailable capability", "wrong tool
   argument", "ends conversation prematurely"). For each cluster: the task_ids
   in it, a one-paragraph diagnosis citing concrete trajectory evidence, and
   which reward_info metrics it trips.
2. **Prompt-edit suggestions** — concrete, minimal edits to the variant's
   prefix/suffix that target each cluster. Quote the exact wording you would
   add or change. Only suggest edits that address a failure *class*; do not
   suggest task-specific rules.
3. **Non-prompt issues** — failures that prompt edits likely cannot fix
   (harness logic, tool-result handling, model capability limits), with a
   short recommendation each.
4. **Overfitting risk** — anything in your own suggestions or in the current
   variant text that risks overfitting to these specific public tasks rather
   than generalizing to a hidden test set."""


def compact_trajectory(trajectory: list[dict], max_content_chars: int = 500) -> list[dict]:
    """Reduce a trajectory to the fields the judge needs, truncating long content."""
    compact = []
    for msg in trajectory:
        role = msg.get("role", "?")
        if role == "system":
            continue  # policy prompt is huge and identical across tasks
        entry: dict = {"role": role}
        content = msg.get("content")
        if content:
            text = content if isinstance(content, str) else json.dumps(content)
            if len(text) > max_content_chars:
                text = text[:max_content_chars] + f"…[{len(text) - max_content_chars} chars truncated]"
            entry["content"] = text
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "name": tc.get("function", {}).get("name", tc.get("tool_name", "?")),
                    "arguments": str(tc.get("function", {}).get("arguments", tc.get("arguments", "")))[:300],
                }
                for tc in tool_calls
            ]
        compact.append(entry)
    return compact


def format_failure(row: dict, split: str) -> str:
    """Render one failed task row as a judge-readable block."""
    lines = [
        f"### Task `{row.get('task_id')}` (category: {split}, trial {row.get('trial')}, reward {row.get('reward')})",
    ]
    if row.get("error"):
        lines.append(f"Run error: {row['error']}")
    reward_info = row.get("reward_info") or {}
    lines.append("Failed metrics (reward_info):")
    lines.append("```json\n" + json.dumps(reward_info, indent=2, default=str)[:3000] + "\n```")
    trajectory = compact_trajectory(row.get("trajectory") or [])
    lines.append("Trajectory (compacted):")
    lines.append("```json\n" + json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n```")
    return "\n".join(lines)


def build_judge_messages(variant: str, prefix: str, suffix: str, failure_blocks: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_TASK_PROMPT.format(
                variant=variant,
                prefix=prefix or "(empty)",
                suffix=suffix or "(empty)",
                failures="\n\n".join(failure_blocks),
            ),
        },
    ]
