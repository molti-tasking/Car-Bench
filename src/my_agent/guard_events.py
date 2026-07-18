"""Append-only sink for guard/firewall firing events.

A run whose Pass^3 is flat is uninterpretable without this: "the check never
fired", "it fired and did not help", and "it fired and hurt" are three
different results with three different follow-ups, and the run record cannot
tell them apart from the score alone.

The agent runs in its own process, so events go to a JSONL file named by
GUARD_EVENTS_PATH (set per run by tools/experiment.py) and are aggregated back
into the run record when the run finishes. Unset path = disabled, which keeps
Docker and official runs untouched.

Every operation swallows its own errors: instrumentation must never be able to
fail an evaluation.
"""
import json
import os
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_ENV_PATH = "GUARD_EVENTS_PATH"


def enabled() -> bool:
    return bool(os.getenv(_ENV_PATH))


def emit(mechanism: str, context_id: str, **fields) -> None:
    """Record one firing. `mechanism` is the layer (schema_guard, firewall,
    ask_gate, self_check); `fields` carry whatever that layer can attribute —
    for the firewall, the per-check `kinds` that fired.
    """
    path = os.getenv(_ENV_PATH)
    if not path:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mechanism": mechanism,
        "context_id": context_id,
        **fields,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:  # the agent serves turns concurrently across tasks
            with open(path, "a") as f:
                f.write(line + "\n")
    except Exception:
        pass  # instrumentation is never worth failing a run over


def aggregate(paths) -> dict:
    """Summarize one or more guard-events files into counts for the run record.

    Accepts a single path or an iterable — a sharded run writes one file per
    shard (concurrent processes appending to a shared file can interleave once
    a record exceeds the atomic-append size).

    -> {"total": n, "by_mechanism": {...}, "firewall_by_kind": {...},
        "contexts_touched": n}
    """
    from collections import Counter
    from pathlib import Path

    if isinstance(paths, (str, Path)):
        paths = [paths]
    by_mechanism: Counter = Counter()
    by_kind: Counter = Counter()
    contexts: set = set()
    total = 0
    seen_any = False
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        seen_any = True
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # a torn line loses one event, never the aggregation
            total += 1
            by_mechanism[rec.get("mechanism", "?")] += 1
            if rec.get("context_id"):
                contexts.add(rec["context_id"])
            for kind in rec.get("kinds") or []:
                by_kind[kind] += 1
    if not seen_any:
        return {}
    return {
        "total": total,
        "by_mechanism": dict(by_mechanism),
        "firewall_by_kind": dict(by_kind),
        "contexts_touched": len(contexts),
    }
