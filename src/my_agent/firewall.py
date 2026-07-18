"""Deterministic action firewall for the CAR-bench agent.

Compile the episode's constraints once, then enforce them in code on every
action — so policy compliance stops depending on the model re-reading 19
policies at every turn.

Three runtime structures, built ONLY from evaluator-provided inputs (tool
schemas, policy text, tool results, user messages). No hidden state, no prior
knowledge of the benchmark — everything here is derived at runtime from what
the evaluator sent this episode.

    StateLedger       mirror of vehicle state assembled from observed results
    ProvenanceIndex   every value this episode legitimately supports
    Constraints       policy prose compiled once into checkable rules

Design rule: every check FAILS OPEN. On any doubt the action proceeds
unchanged. An over-strict firewall demonstrably costs more than it saves —
v4's minimalism rule lifted Disambiguation but dropped Base Pass^3 93→80 — so
these are advisory signals feeding one corrective regeneration, never a hard
block on the agent's output.
"""
import json
import re
from typing import Any

# Verb-ish tokens in tool names that are not entities ("open_close_sunshade").
_VERBS = {
    "get", "set", "open", "close", "start", "stop", "send", "add", "remove",
    "create", "delete", "update", "check", "find", "search", "list", "call",
    "toggle", "adjust", "increase", "decrease", "activate", "deactivate",
}
# The checks check_action() can run, individually selectable so each can be
# measured on its own (see AGENT_FIREWALL_CHECKS).
CHECK_KINDS = frozenset({"precondition", "default", "provenance"})

_TRIVIAL_NUMBERS = {0.0, 1.0}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Words that license a numeric value without stating the digits — "open it
# fully" legitimately means 100. Without this, natural phrasing trips the
# provenance check. English + German (the champion prompt is German).
_IMPLIED_NUMBERS = {
    100.0: ("fully", "completely", "all the way", "maximum", "max ", "full",
            "ganz", "vollständig", "komplett", "voll"),
    50.0: ("half", "halfway", "halb", "hälfte"),
    25.0: ("quarter", "viertel"),
    0.0: ("close", "closed", "shut", "off", "zu ", "schließ", "aus"),
}


def _scalars(obj: Any, out: set) -> None:
    """Collect scalar leaves of a nested structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            _scalars(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _scalars(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(float(obj))
    elif isinstance(obj, str):
        out.add(obj)


class StateLedger:
    """What the agent has actually *verified* — assembled from tool results.

    Distinct from what the model believes. Successful tool calls and their
    returned payloads are the only things that land here.
    """

    def __init__(self) -> None:
        self.successful_tools: list[str] = []
        self.failed_tools: list[str] = []
        self.state: dict = {}
        self.result_scalars: set = set()

    @classmethod
    def from_messages(cls, messages: list[dict]) -> "StateLedger":
        ledger = cls()
        # Resolve each tool result to its call by id. Positional matching breaks
        # whenever results arrive out of order or a batch comes back partially,
        # and it fails silently — every later name in the episode shifts by one.
        # The id map is authoritative; the positional queue is only a fallback
        # for backends that omit tool_call_id.
        by_id: dict[str, str] = {}
        pending: list[str] = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                pending = []
                for tc in m["tool_calls"]:
                    name = tc["function"]["name"]
                    if tc.get("id"):
                        by_id[tc["id"]] = name
                    pending.append(name)
            elif m.get("role") == "tool":
                call_id = m.get("tool_call_id")
                if call_id and call_id in by_id:
                    name = by_id[call_id]
                    if name in pending:
                        pending.remove(name)
                else:
                    name = pending.pop(0) if pending else "?"
                raw = m.get("content") or ""
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    payload = None
                ok = isinstance(payload, dict) and str(payload.get("status", "")).upper() == "SUCCESS"
                (ledger.successful_tools if ok else ledger.failed_tools).append(name)
                if ok:
                    result = payload.get("result")
                    if isinstance(result, dict):
                        ledger.state.update(result)
                    _scalars(result, ledger.result_scalars)
        return ledger

    def called(self, tool: str) -> bool:
        return tool in self.successful_tools


class ProvenanceIndex:
    """Every value this episode legitimately supports.

    A tool-call argument whose value appears nowhere — not in the user's words,
    not in a tool result, not in the policy — was invented by the model. That
    is exactly the failure where the agent opened the sunroof to 100% when the
    stored preference said 50%.
    """

    def __init__(self) -> None:
        self.numbers: set = set()
        self.strings: set = set()

    @classmethod
    def build(cls, messages: list[dict], ledger: StateLedger) -> "ProvenanceIndex":
        idx = cls()
        # Policy text (system prompt) + everything the user said + tool results
        for m in messages:
            if m.get("role") in ("system", "user", "tool"):
                text = m.get("content") or ""
                if isinstance(text, str):
                    for n in _NUM_RE.findall(text):
                        try:
                            idx.numbers.add(float(n))
                        except ValueError:
                            pass
                    for tok in re.findall(r"\b[A-Z][A-Z_]{2,}\b", text):
                        idx.strings.add(tok)
                    lowered = text.lower()
                    for number, words in _IMPLIED_NUMBERS.items():
                        if any(w in lowered for w in words):
                            idx.numbers.add(number)
        for s in ledger.result_scalars:
            (idx.numbers if isinstance(s, float) else idx.strings).add(s)
        return idx

    def supports_number(self, value: float) -> bool:
        if float(value) in _TRIVIAL_NUMBERS:
            return True  # 0/1 are structural, not claims
        return any(abs(float(value) - n) < 1e-9 for n in self.numbers)


def compile_constraints(policy_text: str, tools: list[dict], completion_fn,
                        completion_kwargs: dict, logger=None) -> dict | None:
    """One LLM call per episode: turn policy prose into checkable rules.

    Returns {"preconditions": [{tool, requires_tools[]}],
             "defaults": [{tool, argument, value}]} or None (fail open).
    """
    tool_names = [t.get("function", {}).get("name") for t in (tools or [])]
    tool_names = [n for n in tool_names if n]
    if not policy_text or not tool_names:
        return None
    prompt = (
        "Extract machine-checkable rules from the vehicle assistant policy below.\n"
        "Return JSON with exactly two keys:\n"
        '  "preconditions": [{"tool": <tool name>, "requires_tools": [<tool names that '
        'must succeed FIRST>]}]\n'
        '  "defaults": [{"tool": <tool name>, "argument": <arg name>, "value": <number>}]\n'
        "Only use tool names from this list: " + ", ".join(tool_names) + "\n"
        "Only include a rule if the policy states it explicitly. Empty lists are fine.\n\n"
        "POLICY:\n" + policy_text[:12000]
    )
    try:
        kwargs = {k: v for k, v in completion_kwargs.items() if k not in ("tools", "metadata")}
        kwargs["temperature"] = 0.0
        resp = completion_fn(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **kwargs,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(data, dict):
            return None
        pre = [p for p in data.get("preconditions", []) or []
               if isinstance(p, dict) and p.get("tool") in tool_names]
        dfl = [d for d in data.get("defaults", []) or []
               if isinstance(d, dict) and d.get("tool") in tool_names
               and isinstance(d.get("value"), (int, float))]
        if logger:
            logger.info("Constraints compiled", preconditions=len(pre), defaults=len(dfl))
        return {"preconditions": pre, "defaults": dfl}
    except Exception as e:  # fail open — a bad compile must not break the agent
        if logger:
            logger.warning(f"Constraint compilation failed (continuing unconstrained): {e}")
        return None


def check_action(tool_calls: list[dict], ledger: StateLedger, prov: ProvenanceIndex,
                 constraints: dict | None, enabled: set | None = None) -> list[dict]:
    """Advisory violations for an outgoing set of tool calls.

    Returns [{"kind": <check name>, "message": <text for the nudge>}]. The kind
    is what makes a run interpretable: without per-check attribution a flat
    result cannot distinguish "never fired" from "fired and did not help".

    `enabled` selects which checks run (default: all of CHECK_KINDS). The three
    have very different risk profiles and must be measurable in isolation —
    `default` contradicts the model on the strength of one unverified LLM
    extraction of policy prose, while `provenance` is episode-global and may
    turn out to be near-inert.
    """
    enabled = CHECK_KINDS if enabled is None else enabled
    violations: list[dict] = []
    pre_map, default_map = {}, {}
    for p in (constraints or {}).get("preconditions", []):
        pre_map[p["tool"]] = [t for t in (p.get("requires_tools") or []) if isinstance(t, str)]
    for d in (constraints or {}).get("defaults", []):
        default_map[(d["tool"], d.get("argument"))] = d.get("value")

    emitted = [tc["function"]["name"] for tc in tool_calls or []]
    for tc in tool_calls or []:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            continue
        if not isinstance(args, dict):
            continue

        # 1. Precondition: required tools must already have succeeded (or be in
        #    this same batch, which the evaluator executes in order).
        if "precondition" in enabled:
            for req in pre_map.get(name, []):
                if not ledger.called(req) and req not in emitted:
                    violations.append({
                        "kind": "precondition",
                        "message": f"Policy precondition: '{req}' must succeed before '{name}', "
                                   f"but it has not been called.",
                    })

        for arg, val in args.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue

            # 2. Default deviation: the policy defines a value for this
            #    argument and the model chose a different one. Provenance
            #    cannot catch this (it is episode-global, so "open it fully"
            #    licenses 100 for every component), so check the default
            #    directly — this is the 100%-instead-of-50% failure.
            default = default_map.get((name, arg))
            if default is not None:
                if "default" in enabled and abs(float(val) - float(default)) > 1e-9:
                    violations.append({
                        "kind": "default",
                        "message": f"Policy default: '{name}' argument '{arg}' defaults to {default}, "
                                   f"but {val} was used. Use the default unless the user explicitly "
                                   f"asked for a different value.",
                    })
                # A policy-defined default is a stronger, more specific signal
                # than episode-global provenance either way — do not also
                # complain that the value is unsourced.
                continue

            # 3. Provenance: a numeric argument with no source at all was invented.
            if "provenance" in enabled and not prov.supports_number(val):
                violations.append({
                    "kind": "provenance",
                    "message": f"Unsourced value: '{name}' argument '{arg}'={val} appears nowhere in "
                               f"the user's request, the tool results, or the policy.",
                })
    return violations


def entity_vocabulary(tools: list[dict]) -> dict:
    """entity noun -> tools that can act on it, derived from tool names."""
    vocab: dict = {}
    for t in tools or []:
        name = t.get("function", {}).get("name") or ""
        for token in name.split("_"):
            tok = token.lower()
            if len(tok) < 4 or tok in _VERBS:
                continue
            vocab.setdefault(tok, set()).add(name)
    return vocab


def suspect_entities(draft: str, ledger: StateLedger, tools: list[dict]) -> list[str]:
    """Entities the draft talks about that no successful tool call supports.

    A cheap deterministic pre-filter for the LLM self-check: it narrows an
    open-ended 'did you fabricate anything?' into 'you mentioned the sunshade
    but never successfully moved it'.
    """
    if not draft:
        return []
    vocab = entity_vocabulary(tools)
    lowered = draft.lower()
    suspects = []
    for entity, owning_tools in vocab.items():
        if entity not in lowered:
            continue
        if not any(ledger.called(t) for t in owning_tools):
            suspects.append(entity)
    return sorted(suspects)


def unresolved_required_args(tool_schema: dict, prov: ProvenanceIndex) -> list[str]:
    """Required numeric args of a tool with no value available this episode.

    Powers a principled act-vs-ask decision: ask only when information is
    genuinely missing, rather than on a '?' heuristic.
    """
    params = (tool_schema or {}).get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    unresolved = []
    for req in params.get("required", []) or []:
        spec = props.get(req) or {}
        if spec.get("type") in ("number", "integer") and not prov.numbers:
            unresolved.append(req)
    return unresolved
