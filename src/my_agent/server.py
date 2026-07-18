"""Server entry point for the CAR-bench agent under test.

All model/prompt configuration is read from environment variables (a
submission requirement — organizers must be able to reconfigure the agent
without rebuilding the image):

    AGENT_LLM                    LiteLLM model string (required in practice)
    AGENT_API_KEY                optional explicit API key passed to LiteLLM;
                                 provider-specific vars (ANTHROPIC_API_KEY, ...)
                                 also work when this is unset
    AGENT_API_BASE               optional API base URL passed to LiteLLM
    AGENT_TEMPERATURE            optional float; unset = 0.0
    AGENT_REASONING_EFFORT       optional: none/low/medium/high (passed to LiteLLM)
    AGENT_PROMPT_VARIANT         named variant from prompts.py (default: baseline)
    AGENT_SYSTEM_PROMPT_PREFIX   optional free-text override of the variant prefix
    AGENT_SYSTEM_PROMPT_SUFFIX   optional free-text override of the variant suffix
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette

load_dotenv()

# Suppress Pydantic serialization warnings from litellm types
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.main",
)

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_jsonrpc_routes, create_agent_card_routes
from a2a.types import AgentCard

from agent import MyAgentExecutor
from observability import normalize_litellm_proxy_env, setup_tracing
from prompts import PROMPT_VARIANTS

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="server")


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the CAR-bench agent under test."""
    card = AgentCard(
        name="my_agent",
        description="In-car voice assistant agent for CAR-bench evaluation",
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant"
    skill.description = "Helps drivers with navigation, communication, charging, and other in-car tasks"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant"])

    return card


def resolve_config() -> dict:
    """Resolve all agent configuration from environment variables."""
    normalize_litellm_proxy_env()
    model = os.getenv("AGENT_LLM", "gemini/gemini-2.5-flash")

    temperature_raw = os.getenv("AGENT_TEMPERATURE", "")
    temperature = float(temperature_raw) if temperature_raw else 0.0

    reasoning_effort = os.getenv("AGENT_REASONING_EFFORT", "") or None

    variant_name = os.getenv("AGENT_PROMPT_VARIANT", "baseline")
    if variant_name not in PROMPT_VARIANTS:
        raise SystemExit(
            f"Unknown AGENT_PROMPT_VARIANT '{variant_name}'. "
            f"Available: {', '.join(PROMPT_VARIANTS)}"
        )
    variant = PROMPT_VARIANTS[variant_name]
    prefix = os.getenv("AGENT_SYSTEM_PROMPT_PREFIX") or variant["prefix"]
    suffix = os.getenv("AGENT_SYSTEM_PROMPT_SUFFIX") or variant["suffix"]

    return {
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "self_check": os.getenv("AGENT_SELF_CHECK", "false").lower() == "true",
        "self_check_model": os.getenv("AGENT_SELF_CHECK_MODEL") or None,
        "ask_gate": os.getenv("AGENT_ASK_GATE", "false").lower() == "true",
        "vote_k": int(os.getenv("AGENT_VOTE_K", "0") or 0),
        "vote_temperature": float(os.getenv("AGENT_VOTE_TEMPERATURE", "0.7")),
        "schema_guard": os.getenv("AGENT_SCHEMA_GUARD", "false").lower() == "true",
        "firewall": os.getenv("AGENT_FIREWALL", "false").lower() == "true",
        # For litellm_proxy/* models the proxy credentials come from the
        # normalized LITELLM_PROXY_* env vars; explicit AGENT_API_KEY/BASE
        # still override. Direct provider models (anthropic/..., gemini/...)
        # must not receive the proxy key.
        "api_key": os.getenv("AGENT_API_KEY") or None,
        "api_base": os.getenv("AGENT_API_BASE") or None,
        "prompt_variant": variant_name,
        "system_prompt_prefix": prefix,
        "system_prompt_suffix": suffix,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the CAR-bench agent under test.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    config = resolve_config()
    setup_tracing(logger)

    logger.info(
        "Starting CAR-bench agent",
        model=config["model"],
        temperature=config["temperature"],
        reasoning_effort=config["reasoning_effort"],
        prompt_variant=config["prompt_variant"],
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=MyAgentExecutor(
            model=config["model"],
            temperature=config["temperature"],
            reasoning_effort=config["reasoning_effort"],
            api_key=config["api_key"],
            api_base=config["api_base"],
            system_prompt_prefix=config["system_prompt_prefix"],
            system_prompt_suffix=config["system_prompt_suffix"],
            self_check=config["self_check"],
            self_check_model=config["self_check_model"],
            ask_gate=config["ask_gate"],
            vote_k=config["vote_k"],
            vote_temperature=config["vote_temperature"],
            schema_guard=config["schema_guard"],
            firewall=config["firewall"],
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)

    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
