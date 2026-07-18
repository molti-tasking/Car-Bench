"""CAR-bench agent under test — Track 1, with configurable prompt variants.

A cleaned-up version of the Track 1 starter:
1. Receives policy/user text, tool definitions, and tool results from the
   evaluator over A2A.
2. Augments the evaluator-provided system prompt with a configurable
   prefix/suffix (see prompts.py) and calls a LiteLLM-compatible model with
   native tool calling.
3. Returns a user-facing text Part and/or a data Part with tool calls. The
   evaluator executes all CAR-bench tools.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers.proto_helpers import new_message, new_text_part, new_data_part
from a2a.types import Role
from google.protobuf.json_format import MessageToDict
from litellm import completion

from observability import tracing_configured, trace_metadata
from firewall import (
    CHECK_KINDS, StateLedger, ProvenanceIndex, compile_constraints, check_action,
    suspect_entities,
)
import guard_events

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import (
    TURN_METRICS_KEY, PROMPT_TOKENS, COMPLETION_TOKENS, COST, MODEL,
    THINKING_TOKENS, NUM_LLM_CALLS, AVG_LLM_CALL_TIME_MS, NUM_PASSES,
)
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="-")


class SelfCheckVerdict(BaseModel):
    """Structured verdict of the pre-send self-check pass."""
    ok: bool
    revised_response: str | None = None


def _action_signature(message) -> str:
    """Canonical signature of a model decision, for majority voting.

    Tool-call turns vote on the multiset of (tool, canonical-args); pure
    text replies all share the 'respond' signature.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return "respond"
    parts = []
    for tc in tool_calls:
        try:
            args = json.dumps(json.loads(tc.function.arguments), sort_keys=True)
        except Exception:
            args = str(tc.function.arguments)
        parts.append(f"{tc.function.name}({args})")
    return "|".join(sorted(parts))


ASK_GATE_NUDGE = (
    "Interner Hinweis: Du wolltest gerade eine Rückfrage stellen. Prüfe zuerst"
    " mit get_user_preferences (mit passenden Kategorien) und den in den"
    " Richtlinien definierten Standardwerten, ob sich die Frage erübrigt."
    " Nur wenn beides die Mehrdeutigkeit nicht auflöst, stelle die Rückfrage."
)

# v2 trigger: only genuine clarification questions (asking the user to supply
# a missing value/choice) gate. Confirmation questions proposing a specific
# action ("Should I set it to 21°C?") carry no clarify interrogative and pass
# through — v1 firing on those derailed straightforward base flows.
_CLARIFY_QUESTION_RE = re.compile(
    r"\b(which|what|where|whose|how (?:many|much|warm|cool|hot|cold|high|low|far|long))\b",
    re.IGNORECASE,
)


def _is_clarification_question(text: str) -> bool:
    return "?" in text and bool(_CLARIFY_QUESTION_RE.search(text))


SELF_CHECK_PROMPT = """Du prüfst den Antwortentwurf eines Auto-Sprachassistenten \
vor dem Absenden. Melde NUR diese zwei Fehler:
1. Der Entwurf behauptet, eine Aktion oder Zustandsänderung sei erfolgt oder \
erfolge automatisch, obwohl kein passender erfolgreicher Werkzeugaufruf in der \
Aufruf-Liste steht (auch: eine andere Komponente habe sich "mitbewegt").
2. Der Nutzer hat eine Aktion bestätigt, aber kein Werkzeugaufruf hat sie \
ausgeführt, und der Entwurf tut so, als sei sie erledigt.

Wenn keiner der beiden Fehler vorliegt: ok=true, revised_response=null. \
Ändere NICHTS anderes (keine Stilfragen, keine zusätzlichen Rückfragen). \
Wenn ein Fehler vorliegt: ok=false und eine minimal korrigierte Antwort auf \
Englisch, die nur belegte Aussagen enthält."""


class MyAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench agent under test using native tool calling."""

    def __init__(
        self,
        model: str,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt_prefix: str = "",
        system_prompt_suffix: str = "",
        self_check: bool = False,
        self_check_model: str | None = None,
        ask_gate: bool = False,
        ask_gate_v2: bool = False,
        vote_k: int = 0,
        vote_temperature: float = 0.7,
        schema_guard: bool = False,
        firewall: bool = False,
        firewall_checks: set | None = None,
    ):
        self.firewall = firewall
        self.firewall_checks = CHECK_KINDS if firewall_checks is None else firewall_checks
        self.ctx_id_to_constraints: dict[str, dict | None] = {}
        self.self_check = self_check
        self.self_check_model = self_check_model
        self.ask_gate = ask_gate
        self.ask_gate_v2 = ask_gate_v2
        self.vote_k = vote_k
        self.vote_temperature = vote_temperature
        self.schema_guard = schema_guard
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key
        self.api_base = api_base
        self.system_prompt_prefix = system_prompt_prefix
        self.system_prompt_suffix = system_prompt_suffix
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        # Per-context turn metrics accumulation (reset when final response is sent)
        self.ctx_id_to_turn_metrics: dict[str, dict] = {}

    def _vote_completion(self, messages: list[dict], completion_kwargs: dict, turn_m: dict, ctx_logger):
        """Self-consistency voting: sample K decisions, majority-vote on the
        action signature. Ties/no-majority fall back to a temperature-0 call.

        Pass^3 rewards consistency; voting converts 'right most of the time'
        into 'right almost always'. Track 1 has no compute constraints.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _one(temp: float):
            kwargs = dict(completion_kwargs)
            kwargs["temperature"] = temp
            return completion(messages=messages, **kwargs)

        with ThreadPoolExecutor(max_workers=self.vote_k) as pool:
            futures = [pool.submit(_one, self.vote_temperature) for _ in range(self.vote_k)]
            responses = []
            for f in futures:
                try:
                    responses.append(f.result())
                except Exception as e:
                    ctx_logger.warning(f"Vote sample failed: {e}")

        for r in responses:
            usage = getattr(r, "usage", None)
            if usage:
                turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
            turn_m[NUM_LLM_CALLS] += 1

        votes: dict[str, list] = {}
        for r in responses:
            votes.setdefault(_action_signature(r.choices[0].message), []).append(r)
        if votes:
            best_sig, best = max(votes.items(), key=lambda kv: len(kv[1]))
            if len(best) > self.vote_k // 2 or len(votes) == 1:
                ctx_logger.info("Vote decided", signature=best_sig[:80], votes=f"{len(best)}/{len(responses)}")
                return best[0]
        # No majority: deterministic anchor call
        ctx_logger.info("Vote inconclusive; falling back to temperature-0 call")
        response = _one(0.0)
        usage = getattr(response, "usage", None)
        if usage:
            turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
            turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
        turn_m[NUM_LLM_CALLS] += 1
        return response

    @staticmethod
    def validate_tool_calls(tool_calls: list[dict], tools: list[dict]) -> list[str]:
        """Deterministic pre-flight of tool calls against the evaluator-provided
        schema. Returns human-readable violations (empty = valid). Uses only the
        schema the evaluator sent — no hidden state, benchmark-legal.

        With native tool calling the model rarely invents tool *names*, so the
        real value is argument validation: missing required args, invalid enum
        values (a hallucinated ambient color), malformed JSON.
        """
        schema = {}
        for t in tools or []:
            fn = t.get("function", {})
            if fn.get("name"):
                schema[fn["name"]] = fn.get("parameters", {}) or {}
        violations: list[str] = []
        for tc in tool_calls or []:
            name = tc.get("function", {}).get("name", "")
            if name not in schema:
                violations.append(f"Tool '{name}' is not among the available tools.")
                continue
            params = schema[name]
            props = params.get("properties", {}) or {}
            required = params.get("required", []) or []
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                violations.append(f"Tool '{name}' has malformed JSON arguments.")
                continue
            if not isinstance(args, dict):
                violations.append(f"Tool '{name}' arguments are not an object.")
                continue
            for req in required:
                if req not in args:
                    violations.append(f"Tool '{name}' is missing required argument '{req}'.")
            for arg, val in args.items():
                spec = props.get(arg)
                if not isinstance(spec, dict):
                    continue
                enum = spec.get("enum")
                if enum is not None and val not in enum:
                    violations.append(
                        f"Tool '{name}' argument '{arg}'={val!r} is not one of the allowed values {enum}."
                    )
        return violations

    def _run_self_check(self, messages: list[dict], draft: str, turn_m: dict,
                        suspects: list[str] | None = None) -> str | None:
        """One verification pass before a user-facing reply is sent.

        Returns a revised response iff the draft claims an unexecuted action;
        otherwise None. Any checker failure degrades to sending the draft.
        """
        tool_log = []
        pending_names: list[str] = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                pending_names = [tc["function"]["name"] for tc in m["tool_calls"]]
            elif m.get("role") == "tool":
                name = pending_names.pop(0) if pending_names else "?"
                content = str(m.get("content") or "")
                status = "SUCCESS" if '"status": "SUCCESS"' in content or "'status': 'SUCCESS'" in content else "OTHER"
                tool_log.append(f"{name} -> {status}")
        recent_user = [str(m.get("content") or "")[:300] for m in messages if m.get("role") == "user"][-3:]

        check_input = (
            f"Werkzeugaufrufe in diesem Gespräch (Reihenfolge, mit Status):\n"
            f"{json.dumps(tool_log, ensure_ascii=False)}\n\n"
            f"Letzte Nutzernachrichten:\n{json.dumps(recent_user, ensure_ascii=False)}\n\n"
            f"Antwortentwurf:\n{draft}"
        )
        if suspects:
            check_input += (
                "\n\nDeterministischer Vorbefund: Der Entwurf erwähnt "
                + ", ".join(suspects)
                + " — für diese Komponenten gibt es KEINEN erfolgreichen Werkzeugaufruf."
                " Prüfe besonders, ob der Entwurf hier etwas behauptet."
            )
        completion_kwargs = {
            "model": self.self_check_model or self.model,
            "temperature": 0.0,
            "timeout": float(os.getenv("AGENT_LLM_TIMEOUT", "300")),
            "response_format": SelfCheckVerdict,
        }
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        response = completion(
            messages=[
                {"role": "system", "content": SELF_CHECK_PROMPT},
                {"role": "user", "content": check_input},
            ],
            **completion_kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage:
            turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
            turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
        turn_m[NUM_LLM_CALLS] += 1
        verdict = SelfCheckVerdict.model_validate(json.loads(response.choices[0].message.content))
        if not verdict.ok and verdict.revised_response:
            return verdict.revised_response
        return None

    def _build_system_prompt(self, evaluator_system_prompt: str) -> str:
        """Wrap the evaluator-provided policy prompt with the configured variant.

        The evaluator prompt always stays intact — policy compliance is scored.
        """
        return f"{self.system_prompt_prefix}{evaluator_system_prompt}{self.system_prompt_suffix}"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}")

        messages = self.ctx_id_to_messages.setdefault(context.context_id, [])
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # Parse the incoming A2A Message Parts
        user_message_text = None
        incoming_tool_results = None

        try:
            for part in inbound_message.parts:
                content_type = part.WhichOneof("content")
                if content_type == "text":
                    text = part.text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message of a task: system prompt + first user turn
                        parts_split = text.split("\n\nUser:", 1)
                        evaluator_system_prompt = parts_split[0].replace("System:", "").strip()
                        user_message_text = parts_split[1].strip()
                        if not messages:  # Only add the system prompt once
                            messages.append({
                                "role": "system",
                                "content": self._build_system_prompt(evaluator_system_prompt),
                            })
                    else:
                        user_message_text = text

                elif content_type == "data":
                    data = MessageToDict(part.data)
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        incoming_tool_results = data["tool_results"]

            # Fallback if no text part and no structured tool results found
            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

            ctx_logger.info(
                "Received message",
                turn=len(messages) + 1,
                message_preview=(user_message_text[:100] if user_message_text else
                                 f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else "")
            )

        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # If the previous assistant message called tools, feed the results back
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                # Match each structured result to its tool_call_id by tool name
                tool_call_by_name: dict[str, list[dict]] = {}
                for tc in prev_tool_calls:
                    tool_call_by_name.setdefault(tc["function"]["name"], []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "") if isinstance(tr, dict) else tr.get("toolName", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        # Pop the first matching call to handle duplicate tool names
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        ctx_logger.warning("No matching tool_call_id for tool result", tool_name=tr_name)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", tr.get("toolCallId", f"unknown_{tr_name}")),
                            "content": tr.get("content", ""),
                        })
            else:
                # Fallback: no structured tool results, use the text message for all calls
                tool_results = [
                    {"role": "tool", "tool_call_id": tc["id"], "content": user_message_text or ""}
                    for tc in prev_tool_calls
                ]

            messages.extend(tool_results)
        else:
            messages.append({"role": "user", "content": user_message_text})

        # Call the LLM with native tool calling
        try:
            # Prompt caching (guard against empty lists)
            if tools:
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            if messages:
                messages[0]["cache_control"] = {"type": "ephemeral"}

            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None,
                # A hung provider connection must fail the turn, not the run.
                "timeout": float(os.getenv("AGENT_LLM_TIMEOUT", "300")),
            }
            if self.temperature is not None:
                completion_kwargs["temperature"] = self.temperature
            if self.reasoning_effort:
                completion_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base
            if tracing_configured():
                completion_kwargs["metadata"] = trace_metadata(context.context_id)

            call_start_time = time.perf_counter()
            if self.vote_k >= 2:
                # turn_m must exist before voting accumulates usage into it
                turn_m = self.ctx_id_to_turn_metrics.setdefault(context.context_id, {
                    PROMPT_TOKENS: 0, COMPLETION_TOKENS: 0, THINKING_TOKENS: 0,
                    COST: 0.0, NUM_LLM_CALLS: 0, "_total_llm_time_ms": 0.0,
                })
                response = self._vote_completion(messages, completion_kwargs, turn_m, ctx_logger)
            else:
                response = completion(messages=messages, **completion_kwargs)
            call_elapsed_ms = (time.perf_counter() - call_start_time) * 1000.0

            # Accumulate turn metrics for this LLM call
            turn_m = self.ctx_id_to_turn_metrics.setdefault(context.context_id, {
                PROMPT_TOKENS: 0,
                COMPLETION_TOKENS: 0,
                THINKING_TOKENS: 0,
                COST: 0.0,
                NUM_LLM_CALLS: 0,
                "_total_llm_time_ms": 0.0,
            })
            if self.vote_k < 2:  # voting already accumulated its samples' usage
                usage = getattr(response, "usage", None)
                if usage:
                    turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                    turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        turn_m[THINKING_TOKENS] += getattr(details, "reasoning_tokens", 0) or 0
                turn_m[COST] += getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
                turn_m[NUM_LLM_CALLS] += 1
            turn_m["_total_llm_time_ms"] += call_elapsed_ms

            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)
            tool_calls = assistant_content.get("tool_calls")

            # Optional schema-guard: deterministically reject invalid tool calls
            # (unknown tool, missing required arg, hallucinated enum value) and
            # regenerate once with a corrective nudge, so the model fixes the
            # call or tells the user it cannot do it — rather than emitting a
            # fabricated action.
            if self.schema_guard and tool_calls:
                violations = self.validate_tool_calls(tool_calls, tools)
                if violations:
                    try:
                        ctx_logger.info("Schema-guard caught invalid tool calls; regenerating",
                                        violations=violations)
                        guard_events.emit("schema_guard", context.context_id,
                                          count=len(violations), violations=violations)
                        nudge = (
                            "Interner Hinweis: Der zuletzt erzeugte Werkzeugaufruf ist ungültig:\n- "
                            + "\n- ".join(violations)
                            + "\nVerwende ausschließlich die verfügbaren Werkzeuge mit gültigen"
                            " Argumentwerten. Wenn die benötigte Funktion oder ein gültiger Wert"
                            " nicht verfügbar ist, sage dem Nutzer offen, dass du das nicht tun"
                            " kannst, statt einen ungültigen Aufruf zu erzwingen."
                        )
                        regen = messages + [{"role": "system", "content": nudge}]
                        response = completion(messages=regen, **completion_kwargs)
                        usage = getattr(response, "usage", None)
                        if usage:
                            turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                            turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
                        turn_m[NUM_LLM_CALLS] += 1
                        assistant_content = response.choices[0].message.model_dump(exclude_unset=True)
                        tool_calls = assistant_content.get("tool_calls")
                        # Fail-safe: drop any calls still invalid after regeneration
                        if tool_calls:
                            valid = [tc for tc in tool_calls if not self.validate_tool_calls([tc], tools)]
                            if len(valid) != len(tool_calls):
                                ctx_logger.warning("Schema-guard dropping still-invalid tool calls",
                                                   dropped=len(tool_calls) - len(valid))
                            assistant_content["tool_calls"] = valid
                            tool_calls = valid or None
                    except Exception as guard_err:
                        ctx_logger.warning(f"Schema-guard regeneration failed, keeping draft: {guard_err}")

            # Optional action firewall: deterministic policy-precondition and
            # value-provenance checks against a ledger of verified tool results.
            # Advisory only — violations trigger one corrective regeneration.
            if self.firewall and tool_calls:
                try:
                    if context.context_id not in self.ctx_id_to_constraints:
                        policy = next((m.get("content", "") for m in messages
                                       if m.get("role") == "system"), "")
                        self.ctx_id_to_constraints[context.context_id] = compile_constraints(
                            policy, tools, completion, completion_kwargs, ctx_logger)
                        turn_m[NUM_LLM_CALLS] += 1
                    constraints = self.ctx_id_to_constraints.get(context.context_id)
                    ledger = StateLedger.from_messages(messages)
                    prov = ProvenanceIndex.build(messages, ledger)
                    fw_violations = check_action(tool_calls, ledger, prov, constraints,
                                                 self.firewall_checks)
                    if fw_violations:
                        kinds = [v["kind"] for v in fw_violations]
                        messages_out = [v["message"] for v in fw_violations]
                        ctx_logger.info("Firewall flagged action", violations=messages_out)
                        guard_events.emit("firewall", context.context_id,
                                          kinds=kinds, violations=messages_out)
                        nudge = (
                            "Interner Hinweis — die geplante Aktion verletzt überprüfbare Regeln:\n- "
                            + "\n- ".join(messages_out)
                            + "\nKorrigiere die Werkzeugaufrufe entsprechend (Standardwerte und"
                            " Vorbedingungen aus den Richtlinien beachten), oder erkläre dem Nutzer,"
                            " warum du anders vorgehst."
                        )
                        resp2 = completion(messages=messages + [{"role": "system", "content": nudge}],
                                           **completion_kwargs)
                        usage = getattr(resp2, "usage", None)
                        if usage:
                            turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                            turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
                        turn_m[NUM_LLM_CALLS] += 1
                        assistant_content = resp2.choices[0].message.model_dump(exclude_unset=True)
                        tool_calls = assistant_content.get("tool_calls")
                except Exception as fw_err:
                    ctx_logger.warning(f"Firewall check failed, keeping action: {fw_err}")

            # Optional ask-gate: about to ask a clarifying question without
            # ever having consulted stored preferences? One internal nudge to
            # look up first, then the regenerated decision stands. v2 narrows
            # the trigger to genuine clarification questions so confirmation
            # questions ("Should I ...?") pass through untouched.
            gate_mode = "v2" if self.ask_gate_v2 else ("v1" if self.ask_gate else None)
            draft_text = assistant_content.get("content") or ""
            if (
                gate_mode
                and not tool_calls
                and "?" in draft_text
                and (gate_mode == "v1" or _is_clarification_question(draft_text))
                and any(t.get("function", {}).get("name") == "get_user_preferences" for t in (tools or []))
                and not any(
                    tc["function"]["name"] == "get_user_preferences"
                    for m in messages if m.get("role") == "assistant"
                    for tc in (m.get("tool_calls") or [])
                )
            ):
                try:
                    ctx_logger.info("Ask-gate (%s): regenerating with preference-lookup nudge", gate_mode)
                    guard_events.emit("ask_gate" if gate_mode == "v1" else "ask_gate_v2", context.context_id)
                    nudged = messages + [{"role": "system", "content": ASK_GATE_NUDGE}]
                    response = completion(messages=nudged, **completion_kwargs)
                    usage = getattr(response, "usage", None)
                    if usage:
                        turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                        turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
                    turn_m[NUM_LLM_CALLS] += 1
                    llm_message = response.choices[0].message
                    assistant_content = llm_message.model_dump(exclude_unset=True)
                    tool_calls = assistant_content.get("tool_calls")
                except Exception as gate_err:
                    ctx_logger.warning(f"Ask-gate failed, keeping draft: {gate_err}")

            # Optional pre-send self-check on user-facing replies (no tool
            # calls this turn): catches claimed-but-unexecuted actions.
            if self.self_check and not tool_calls and assistant_content.get("content"):
                try:
                    # Firewall cascade: a cheap deterministic pre-filter names the
                    # entities the draft talks about that no successful tool call
                    # supports, so the LLM checker gets a pointed question instead
                    # of an open-ended "did you fabricate anything?".
                    suspects = []
                    if self.firewall:
                        suspects = suspect_entities(
                            assistant_content["content"], StateLedger.from_messages(messages), tools)
                        if suspects:
                            ctx_logger.info("Firewall flagged unsupported entities", suspects=suspects)
                            guard_events.emit("suspect_entities", context.context_id,
                                              suspects=suspects)
                    revised = self._run_self_check(
                        messages, assistant_content["content"], turn_m, suspects)
                    if revised:
                        ctx_logger.info("Self-check revised the draft response")
                        guard_events.emit("self_check_revised", context.context_id,
                                          had_suspects=bool(suspects))
                        assistant_content["content"] = revised
                except Exception as check_err:
                    ctx_logger.warning(f"Self-check failed, sending draft: {check_err}")

            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                content_length=len(assistant_content.get("content") or ""),
            )

            # Build the outbound A2A Message Parts
            parts = []
            if assistant_content.get("content"):
                parts.append(new_text_part(assistant_content["content"]))
            if tool_calls:
                tool_calls_data = ToolCallsData(tool_calls=[
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in tool_calls
                ])
                parts.append(new_data_part(tool_calls_data.model_dump()))
            if assistant_content.get("reasoning_content"):
                parts.append(new_data_part({"reasoning_content": assistant_content["reasoning_content"]}))
            if not parts:
                parts.append(new_text_part(assistant_content.get("content", "")))

        except Exception as e:
            logger.error(f"LLM error: {e}")
            parts = [new_text_part(f"Error processing request: {str(e)}")]
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # Add to history, preserving tool calls and thinking blocks
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        messages.append(assistant_message_for_history)

        response_message = new_message(
            parts=parts,
            context_id=context.context_id,
            role=Role.ROLE_AGENT,
        )

        # Attach turn_metrics on final response (no tool calls = turn complete)
        if not assistant_content.get("tool_calls") and context.context_id in self.ctx_id_to_turn_metrics:
            turn_m = self.ctx_id_to_turn_metrics.pop(context.context_id)
            num_calls = turn_m[NUM_LLM_CALLS]
            avg_time = (turn_m["_total_llm_time_ms"] / num_calls) if num_calls > 0 else 0.0
            response_message.metadata.update({TURN_METRICS_KEY: {
                PROMPT_TOKENS: turn_m[PROMPT_TOKENS],
                COMPLETION_TOKENS: turn_m[COMPLETION_TOKENS],
                COST: turn_m[COST],
                MODEL: self.model,
                THINKING_TOKENS: turn_m[THINKING_TOKENS],
                NUM_LLM_CALLS: num_calls,
                AVG_LLM_CALL_TIME_MS: round(avg_time, 1),
                NUM_PASSES: 1,
            }})

        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution and drop per-context state."""
        logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context",
            context_id=context.context_id[:8],
        )
        self.ctx_id_to_messages.pop(context.context_id, None)
        self.ctx_id_to_tools.pop(context.context_id, None)
        self.ctx_id_to_turn_metrics.pop(context.context_id, None)
