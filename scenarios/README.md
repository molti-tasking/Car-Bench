# Scenario Map

Scenarios mirror the agent-under-test package names under `src/`.

| Agent Package | Scenario Directory | Files |
|---------------|--------------------|-------|
| `src/agent_under_test/` | `scenarios/agent_under_test/` | `local.toml`, `smoke.toml`, `docker-local.toml`, `ghcr.toml` |
| `src/agent_under_test_codex/` | `scenarios/agent_under_test_codex/` | `smoke.toml`, `docker-local.toml` |
| `src/agent_under_test_codex_planner/` | `scenarios/agent_under_test_codex_planner/` | `smoke.toml`, `docker-local.toml` |
| `src/agent_under_test_codex_python/` | `scenarios/agent_under_test_codex_python/` | `smoke.toml`, `docker-local.toml` |

Use `smoke.toml` for quick local checks, `local.toml` for the fuller baseline
local run, `docker-local.toml` for local Docker builds, and `ghcr.toml` for the
published baseline image smoke.
