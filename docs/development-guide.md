# CAR-bench Agent Under Test Development Guide

This guide explains how to build any **agent under test** for CAR-bench
evaluation. It applies to the baseline LiteLLM template, the Codex reference
agents, and participant-owned agents. Every agent communicates with the
CAR-bench evaluator through the same **A2A (Agent-to-Agent) protocol**.

> **Reference implementations:** The same wire contract is demonstrated by:
> - [`src/agent_under_test/`](../src/agent_under_test/) — minimal LiteLLM-compatible template
> - [`src/agent_under_test_codex/`](../src/agent_under_test_codex/) — Codex next-action JSON adapter
> - [`src/agent_under_test_codex_planner/`](../src/agent_under_test_codex_planner/) — private planner plus Spark executor
> - [`src/agent_under_test_codex_python/`](../src/agent_under_test_codex_python/) — Python-call DSL adapter
>
> For more sophisticated harnessing, see
> [`agent-under-test-harnessing.md`](agent-under-test-harnessing.md) and
> [`codex-harness-patterns.md`](codex-harness-patterns.md).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [A2A Message Protocol](#a2a-message-protocol)
3. [Inbound Messages — What Your Agent Receives](#inbound-messages--what-your-agent-receives)
4. [Outbound Messages — What Your Agent Should Return](#outbound-messages--what-your-agent-should-return)
5. [Conversation Lifecycle](#conversation-lifecycle)
6. [Agent Executor Contract](#agent-executor-contract)
7. [Response Metadata](#response-metadata)
8. [Server Setup](#server-setup)
9. [Testing Locally](#testing-locally)
10. [Key Considerations](#key-considerations)

---

## Architecture Overview

```
┌─────────────────────┐        A2A Messages        ┌─────────────────────┐
│ Evaluator           │ ◄────────────────────────► │ Agent Under Test    │
│ (CAR-bench)         │    TextPart + DataPart     │ (Your Agent)        │
└─────────────────────┘                             └─────────────────────┘
```

The evaluator wraps the CAR-bench environment. It sends system prompt, available tools, user messages, and tool execution results to your agent under test. Your agent decides what to do — call tools, respond with text, or both — and sends a response back.

---

## A2A Message Protocol

All messages are exchanged as a list of **Parts**. Each Part is one of:

| Part Type    | Purpose                              | Examples                                    |
|-------------|--------------------------------------|---------------------------------------------|
| **TextPart** | Natural language content             | System prompt, user message, text responses |
| **DataPart** | Structured/machine-readable data     | Tool definitions, tool calls, reasoning     |

A single message can contain **multiple Parts** of different types. For example, a response can have a `TextPart` (explanation) and a `DataPart` (tool calls) simultaneously.

---

## Inbound Messages — What Your Agent Receives

### First Message (Task Initialization)

The first message in a conversation contains **two Parts**:

| Part | Type | Content |
|------|------|---------|
| 1    | `TextPart` | Combined system prompt and user message, formatted as: `"System: <policies and instructions>\n\nUser: <initial task>"` |
| 2    | `DataPart` | Tool definitions in `{"tools": [...]}` format (OpenAI function calling schema) |

**What each part contains:**

- **TextPart** — The `System:` section includes all 19 CAR-bench policies the agent must follow (e.g., check weather before opening sunroof, validate addresses). The `User:` section is the initial user request (e.g., "Navigate to Munich city center").

- **DataPart** — A dictionary with a `"tools"` key containing a list of tool definitions. Each tool follows the OpenAI function calling format:
  ```json
  {
    "type": "function",
    "function": {
      "name": "get_current_location",
      "description": "Get the current GPS location...",
      "parameters": { "type": "object", "properties": {...} }
    }
  }
  ```

See how the baseline agent parses this in
[`src/agent_under_test/car_bench_agent.py`](../src/agent_under_test/car_bench_agent.py),
inside the `execute()` method. The Codex agents reuse the same parsing contract
before converting the transcript into their own internal prompt format.

### Subsequent Messages

After the first turn, each message usually contains one Part. The content depends on what happened in the previous turn:

#### Alternative A: Tool Execution Results

If your agent called tools in its previous response, the evaluator executes them against the CAR-bench environment and returns the results as a **`DataPart`** with structured tool results:

```json
{
  "tool_results": [
    {
      "tool_name": "get_current_location",
      "tool_call_id": "call_abc123",
      "content": "{\"latitude\": 48.1351, \"longitude\": 11.5820, \"city\": \"Munich\"}"
    },
    {
      "tool_name": "get_weather",
      "tool_call_id": "call_def456",
      "content": "{\"temperature\": 15, \"condition\": \"sunny\"}"
    }
  ]
}
```

Each entry in `tool_results` includes the `tool_name` and `content` (the execution result), allowing your agent to match each result to the corresponding tool call from its previous response. The baseline agent matches results by `tool_name` against the previous turn's tool calls.

#### Alternative B: User Follow-up

If your agent responded with text only (no tool calls), the evaluator advances the conversation and sends the next user utterance as plain text. For example:

```
Yes, please navigate there.
```

#### Edge Case: Empty Messages

Occasionally, the message may be empty or whitespace-only. The evaluator replaces these with `"none"` before sending. Your agent should handle this gracefully.

### Inbound Message Metadata

The evaluator also attaches a small `Message.metadata` object to messages sent
to the agent under test:

```json
{"source": "user"}
```

or:

```json
{"source": "environment"}
```

`source = "user"` means the parts contain an initial request or simulated user
follow-up. `source = "environment"` means the parts contain tool execution
results. Agents should still parse the `TextPart` and `DataPart` contents
directly; the metadata is an optional convenience tag for harnesses that want to
route user turns and tool-result turns differently. The shared constants live in
[`src/turn_metrics.py`](../src/turn_metrics.py) as `SOURCE_KEY`,
`SOURCE_USER`, and `SOURCE_ENVIRONMENT`.

---

## Outbound Messages — What Your Agent Should Return

Your agent sends its response as an A2A agent `Message` containing one or more
parts. The reference agents use `new_message(...)` plus `new_text_part(...)`
and `new_data_part(...)` from `a2a.helpers.proto_helpers`. There are several
valid response shapes:

### Option 1: Text Response Only

Return a single `TextPart` with your response text. Use this when your agent is responding directly to the user without needing to call any tools.

See the baseline agent's `execute()` method — when the LLM returns content but no tool calls, it creates a `TextPart` with the content text.

### Option 2: Tool Call(s) Only

Return a single `DataPart` containing the tool calls. The reference agents use
the `ToolCallsData` model in
[`src/agent_under_test/tool_call_types.py`](../src/agent_under_test/tool_call_types.py)
or
[`src/agent_under_test_codex/tool_call_types.py`](../src/agent_under_test_codex/tool_call_types.py)
to structure the data:

The DataPart's `data` field should be the `.model_dump()` of a `ToolCallsData` instance, which produces:
```json
{
  "tool_calls": [
    {"tool_name": "get_current_location", "arguments": {}},
    {"tool_name": "get_weather", "arguments": {}}
  ]
}
```

You can call **multiple tools** in a single response by adding multiple `ToolCall` entries to the list.

### Option 3: Text + Tool Call(s)

Return both a `TextPart` and a `DataPart`. The text serves as a natural language explanation of what the agent is doing, while the DataPart contains the actual tool calls.

This is the most common pattern in the baseline agent; see
[`src/agent_under_test/car_bench_agent.py`](../src/agent_under_test/car_bench_agent.py)
for the concrete response-building code. The Codex agents intentionally return
either a text response or tool-call data for each step, then let the evaluator
drive the next turn.

### Optional: Reasoning Content

If your LLM produces reasoning/thinking output (e.g., Claude extended thinking), you can include it as an additional `DataPart` with `{"reasoning_content": "..."}`. The evaluator will capture it for debugging but it doesn't affect evaluation.

### Message Parts vs Metadata

The evaluator scores behavior from the response **message parts**, not from
metadata. Put all benchmark-visible actions in parts:

- User-facing text goes in `TextPart`.
- Tool calls go in `DataPart({"tool_calls": [...]})`.
- Optional debug reasoning goes in `DataPart({"reasoning_content": "..."})`.

Do not put tool calls, hidden observations, private plans, or final answers in
`Message.metadata`. The evaluator ignores metadata for behavior and uses it only
for run accounting such as latency and token/cost metrics.

---

## Conversation Lifecycle

```
Turn 1:  Evaluator → Agent Under Test:  TextPart(System + User) + DataPart(tools)
         Agent Under Test → Evaluator:  TextPart(text) + DataPart(tool_calls)

Turn 2:  Evaluator → Agent Under Test:  DataPart(tool results)
         Agent Under Test → Evaluator:  TextPart(text) + DataPart(tool_calls)

Turn 3:  Evaluator → Agent Under Test:  DataPart(tool results)
         Agent Under Test → Evaluator:  TextPart(final answer)      ← no tool calls = done

Turn 4:  Evaluator → Agent Under Test:  TextPart(next user utterance)
         Agent Under Test → Evaluator:  ...
```

Key points:
- The conversation continues until the environment/task is complete (managed by the evaluator).
- Each `context_id` represents one independent conversation (one CAR-bench task).
- Your agent should maintain conversation state per `context_id` (see `ctx_id_to_messages` in the baseline).
- Clean up state when `cancel()` is called.

---

## Agent Executor Contract

Your agent must implement the `AgentExecutor` interface from `a2a.server.agent_execution`:

```python
class AgentExecutor:
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process an incoming message and enqueue a response."""
        ...

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation — clean up conversation state."""
        ...
```

**Key objects:**
- `context.message` — The inbound `Message` with `.parts` (list of `Part` objects)
- `context.context_id` — Unique conversation identifier
- `event_queue.enqueue_event(response)` — Send your response back
- `new_message(parts=..., context_id=..., role=Role.ROLE_AGENT)` — Helper to build the response message

See [`src/agent_under_test/car_bench_agent.py`](../src/agent_under_test/car_bench_agent.py)
and [`src/agent_under_test_codex/car_bench_agent.py`](../src/agent_under_test_codex/car_bench_agent.py)
for complete implementations of this executor contract.

---

## Response Metadata

Agents may attach a `turn_metrics` object to `Message.metadata`. The evaluator
uses this metadata to populate CAR-bench latency and cost accounting, but not to
decide task success.

Attach `turn_metrics` only when the response is a final user-facing response for
the current assistant step, meaning the response has **no** `tool_calls`
DataPart. If your agent calls a tool, accumulate metrics internally and attach
the aggregate metrics on the later response after the evaluator sends tool
results back.

The metadata shape is:

```json
{
  "turn_metrics": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost": 0.0,
    "model": "model-or-harness-name",
    "thinking_tokens": 0,
    "num_llm_calls": 1,
    "avg_llm_call_time_ms": 1234.5,
    "num_passes": 1
  }
}
```

Field meanings:

- `prompt_tokens`, `completion_tokens`, `thinking_tokens`: report provider
  usage when available; use `0` when unavailable. For Codex app-server, the
  reference clients read these from `thread/tokenUsage/updated`.
- `cost`: provider cost for the internal calls in this assistant step; use
  `0.0` for subscription-backed runtimes that do not expose reliable cost.
- `model`: the model or harness description, such as
  `gpt-5.3-codex-spark` or `gpt-5.5->gpt-5.3-codex-spark`.
- `num_llm_calls`: number of internal model calls made before returning this
  final response.
- `avg_llm_call_time_ms`: average duration of those internal model calls.
- `num_passes`: number of internal inference passes if the harness has a
  multi-pass planner, executor, ensemble, or validator. Use `1` for a normal
  single-pass agent.

The evaluator adds its own measured `turn_time_ms` after receiving the response.
Agents should not send `turn_time_ms` themselves.

The reference agents conform to this contract:

| Agent | Message Parts | Metadata |
|-------|---------------|----------|
| `src/agent_under_test/` | `TextPart`, `DataPart({"tool_calls": ...})`, optional `reasoning_content` | Aggregated LiteLLM usage on final no-tool-call responses |
| `src/agent_under_test_codex/` | `TextPart` for `respond`, `DataPart({"tool_calls": ...})` for actions | Codex latency/call count plus app-server token usage when emitted; cost remains zero |
| `src/agent_under_test_codex_planner/` | Same as Codex JSON agent | Planner plus executor call counts, combined model label, and aggregated app-server token usage |
| `src/agent_under_test_codex_python/` | Same as Codex JSON agent after parsing Python-call DSL | Same as Codex JSON agent |

For shared constants, see [`src/turn_metrics.py`](../src/turn_metrics.py).

---

## Server Setup

Your agent needs an HTTP server to expose it via A2A. The server setup involves:

1. **AgentCard** — Metadata describing your agent (name, skills, URL). See
   `prepare_agent_card()` in the server files listed below.
2. **RequestHandler** — Wraps your executor. Use `DefaultRequestHandler` from `a2a.server.request_handlers`.
3. **A2AStarletteApplication** — The ASGI app. Takes the agent card and request handler.
4. **uvicorn** — Runs the ASGI app.

The server also accepts CLI arguments and environment variables for LLM
configuration. The exact flags depend on the reference agent. For examples, see:

- [`src/agent_under_test/server.py`](../src/agent_under_test/server.py)
- [`src/agent_under_test_codex/server.py`](../src/agent_under_test_codex/server.py)
- [`src/agent_under_test_codex_planner/server.py`](../src/agent_under_test_codex_planner/server.py)
- [`src/agent_under_test_codex_python/server.py`](../src/agent_under_test_codex_python/server.py)

---

## Testing Locally

1. **Start your agent under test:**
   ```bash
   python src/agent_under_test/server.py --host localhost --port 8080 --agent-llm "gemini/gemini-2.5-flash"
   ```

2. **Configure the scenario** (`scenarios/agent_under_test/local.toml`) so the
   evaluator is started by the runner and points at your agent:
   ```toml
   [evaluator]
   endpoint = "http://localhost:8081"
   cmd = "python src/evaluator/server.py --host localhost --port 8081"

   [agent_under_test]
   endpoint = "http://localhost:8080"
   cmd = ""  # Already running in the first terminal.
   ```

3. **Run evaluation** (in another terminal):
   ```bash
   uv run car-bench-run scenarios/agent_under_test/local.toml --show-logs
   ```

4. **Check results** — The evaluator will report per-task pass/fail and overall metrics.

---

## Key Considerations

### Policy Compliance
The system prompt in the first message includes all 19 CAR-bench policies. Your agent must follow them to pass evaluation. Examples:
- Check weather before opening the sunroof
- Validate addresses before navigating
- Confirm actions with the user when required

You can perform prompt optimization on the system prompt, however the original policies are used for code-based and LLM-as-a-Judge evaluation (so changing the rules/logic will likely result in error).

### Tool Calling Format
- Tools are provided in **OpenAI function calling format** (see DataPart in first message)
- Return tool calls using the `ToolCallsData` shape from the reference
  `tool_call_types.py` modules.
- Arguments must match the tool's parameter schema

You can edit tool descriptions and parameter descriptions inside your own
internal prompt. Do not change the tool name, parameter names, parameter types,
or parameter structure returned to the evaluator. Hallucination and tool
execution metrics depend on the evaluator seeing the raw action your agent chose.

### Conversation State
- Maintain message history per `context_id`
- The baseline agent uses `ctx_id_to_messages` and `ctx_id_to_tools` dicts
- Clean up in `cancel()` to avoid memory leaks

### Error Handling
- Handle missing or malformed message parts gracefully
- Return error messages as `TextPart` if something fails
- The baseline agent has a fallback using `context.get_user_input()` if part parsing fails

### LLM Flexibility
You are **not** limited to the baseline approach. You can use:
- Any LLM provider (OpenAI, Anthropic, Google, local models) or finetuned LLM
- Any framework (LangChain, LlamaIndex, etc.)
- Rule-based logic, retrieval-augmented generation, or hybrid approaches
- The only requirement is conforming to the A2A message protocol described above

Advanced harnesses may also use internal planning, validation, reranking, memory,
or sub-agent-style components. These components must stay inside the benchmark
boundary: use only the prompt, transcript, tool definitions, and tool results
sent by the evaluator, then return one benchmark-compatible A2A response.
Do not execute CAR-bench tools directly, inspect hidden task/evaluator state, add
private vehicle tools, or give your runtime shell/file/network abilities that can
bypass the recorded A2A trajectory.
