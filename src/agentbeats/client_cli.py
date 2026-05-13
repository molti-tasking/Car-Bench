import sys
import json
import asyncio
from pathlib import Path

import httpx
import tomllib

from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    SendMessageRequest,
    TaskState,
)
from google.protobuf.json_format import MessageToDict

from agentbeats.client import create_message
from agentbeats.models import EvalRequest


class AgentFailedError(Exception):
    """Raised when an agent returns a non-successful terminal status."""
    pass


def parse_toml(d: dict[str, object]) -> tuple[EvalRequest, str]:
    if "green_agent" in d or "participants" in d:
        raise ValueError("Old scenario shape is unsupported; use [evaluator] and [agent_under_test].")

    evaluator = d.get("evaluator")
    if not isinstance(evaluator, dict) or "endpoint" not in evaluator:
        raise ValueError("evaluator.endpoint is required in TOML")
    evaluator_endpoint: str = evaluator["endpoint"]

    agent_under_test = d.get("agent_under_test")
    if not isinstance(agent_under_test, dict) or "endpoint" not in agent_under_test:
        raise ValueError("agent_under_test.endpoint is required in TOML")

    eval_req = EvalRequest(
        agent_under_test=agent_under_test["endpoint"],
        config=d.get("config", {}) or {}
    )
    return eval_req, evaluator_endpoint


def parse_parts(parts) -> tuple[list, list]:
    """Parse protobuf Parts into text and data lists."""
    text_parts = []
    data_parts = []

    for part in parts:
        content_type = part.WhichOneof("content")
        if content_type == "text":
            try:
                data_item = json.loads(part.text)
                data_parts.append(data_item)
            except Exception:
                text_parts.append(part.text.strip())
        elif content_type == "data":
            data_parts.append(MessageToDict(part.data))

    return text_parts, data_parts


def print_parts(parts, task_state: str | None = None):
    text_parts, data_parts = parse_parts(parts)

    output = []
    if task_state:
        output.append(f"[Status: {task_state}]")
    if text_parts:
        output.append("\n".join(text_parts))
    if data_parts:
        output.extend(json.dumps(item, indent=2) for item in data_parts)

    print("\n".join(output) + "\n")


_STATE_NAMES = {
    TaskState.TASK_STATE_SUBMITTED: "submitted",
    TaskState.TASK_STATE_WORKING: "working",
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "failed",
    TaskState.TASK_STATE_CANCELED: "canceled",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth-required",
}


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_cli.py <scenario.toml> [output.json]")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not scenario_path.exists():
        print(f"File not found: {scenario_path}")
        sys.exit(1)

    toml_data = scenario_path.read_text()
    data = tomllib.loads(toml_data)

    req, evaluator_url = parse_toml(data)

    # Collect artifacts from streaming events
    collected_artifacts = []

    # Send message via streaming
    async with httpx.AsyncClient(timeout=300) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=evaluator_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=True,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        outbound_msg = create_message(text=req.model_dump_json())
        request = SendMessageRequest(message=outbound_msg)

        try:
            async for event in client.send_message(request):
                payload_type = event.WhichOneof("payload")

                if payload_type == "message":
                    msg = event.message
                    print_parts(msg.parts)

                elif payload_type == "task":
                    task = event.task
                    state_name = _STATE_NAMES.get(task.status.state, "unknown")
                    parts = task.status.message.parts if task.status.message.parts else []
                    print_parts(parts, state_name)
                    if task.status.state == TaskState.TASK_STATE_COMPLETED:
                        collected_artifacts.extend(task.artifacts)
                    elif task.status.state not in (
                        TaskState.TASK_STATE_SUBMITTED,
                        TaskState.TASK_STATE_WORKING,
                    ):
                        raise AgentFailedError(f"Agent returned status {state_name}.")

                elif payload_type == "status_update":
                    update = event.status_update
                    state_name = _STATE_NAMES.get(update.status.state, "unknown")
                    parts = update.status.message.parts if update.status.message.parts else []
                    print_parts(parts, state_name)
                    if update.status.state == TaskState.TASK_STATE_COMPLETED:
                        pass  # Artifacts come via artifact_update events
                    elif update.status.state not in (
                        TaskState.TASK_STATE_SUBMITTED,
                        TaskState.TASK_STATE_WORKING,
                    ):
                        raise AgentFailedError(f"Agent returned status {state_name}.")

                elif payload_type == "artifact_update":
                    update = event.artifact_update
                    if update.artifact:
                        print_parts(update.artifact.parts, "Artifact update")
                        collected_artifacts.append(update.artifact)

        except AgentFailedError as e:
            print(str(e))
            sys.exit(1)

    if output_path:
        all_data_parts = []
        for artifact in collected_artifacts:
            _, data_parts = parse_parts(artifact.parts)
            all_data_parts.extend(data_parts)

        output_data = {
            "agent_under_test": str(req.agent_under_test),
            "results": all_data_parts
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
            print(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
