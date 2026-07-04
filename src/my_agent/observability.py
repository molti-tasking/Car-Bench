"""Optional Langfuse tracing for LiteLLM calls, gated by environment variables.

Tracing activates only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
set AND the langfuse package is installed. The langfuse package lives in the
dev dependency group, which the Docker image excludes (--no-dev), so the
submitted agent image never traces and behaves identically minus telemetry.

The same setup function is reused by the local evaluator server for
development runs; the official evaluator image is unaffected.
"""
import os


def normalize_litellm_proxy_env() -> None:
    """Map the user's LITE_LLM_* env vars onto litellm's native names.

    litellm resolves `litellm_proxy/<model>` calls via LITELLM_PROXY_API_KEY /
    LITELLM_PROXY_API_BASE. We accept LITE_LLM_API_KEY / LITE_LLM_API_BASE as
    aliases so one pair of .env entries configures the agent, the evaluator's
    simulated user/policy judge, and the experiment toolkit's judge model.
    """
    if os.getenv("LITE_LLM_API_KEY") and not os.getenv("LITELLM_PROXY_API_KEY"):
        os.environ["LITELLM_PROXY_API_KEY"] = os.environ["LITE_LLM_API_KEY"]
    if os.getenv("LITE_LLM_API_BASE") and not os.getenv("LITELLM_PROXY_API_BASE"):
        os.environ["LITELLM_PROXY_API_BASE"] = os.environ["LITE_LLM_API_BASE"]


def tracing_configured() -> bool:
    """True when Langfuse credentials are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def setup_tracing(logger=None) -> bool:
    """Register the Langfuse success/failure callback on LiteLLM.

    Returns True if tracing is active. Never raises: missing credentials or a
    missing langfuse package degrade to a no-op so the agent runs unchanged.
    """
    if not tracing_configured():
        if logger:
            logger.info("Langfuse tracing disabled (no LANGFUSE_* credentials)")
        return False
    try:
        import langfuse  # noqa: F401 — availability check only
    except ImportError:
        if logger:
            logger.warning(
                "LANGFUSE_* credentials set but langfuse package not installed; "
                "tracing disabled (install the dev dependency group)"
            )
        return False

    import litellm
    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse")
    if logger:
        logger.info(
            "Langfuse tracing enabled",
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return True


def trace_metadata(context_id: str, side: str = "agent") -> dict:
    """Per-call LiteLLM metadata so traces group by conversation and run.

    session_id groups all calls of one task conversation; tags carry the run
    and prompt-variant labels injected by the experiment toolkit (RUN_ID) or
    the scenario env (AGENT_PROMPT_VARIANT).
    """
    tags = [f"side:{side}"]
    if os.getenv("RUN_ID"):
        tags.append(f"run:{os.getenv('RUN_ID')}")
    if os.getenv("AGENT_PROMPT_VARIANT"):
        tags.append(f"variant:{os.getenv('AGENT_PROMPT_VARIANT')}")
    return {
        "session_id": context_id,
        "trace_name": f"car-bench-{side}",
        "tags": tags,
    }
