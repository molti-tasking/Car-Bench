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
    ):
        self.self_check = self_check
        self.self_check_model = self_check_model
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

    def _run_self_check(self, messages: list[dict], draft: str, turn_m: dict) -> str | None:
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

            # Optional pre-send self-check on user-facing replies (no tool
            # calls this turn): catches claimed-but-unexecuted actions.
            if self.self_check and not tool_calls and assistant_content.get("content"):
                try:
                    revised = self._run_self_check(messages, assistant_content["content"], turn_m)
                    if revised:
                        ctx_logger.info("Self-check revised the draft response")
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
