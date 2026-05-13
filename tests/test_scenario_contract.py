import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from loguru import logger as loguru_logger

from agentbeats.client_cli import parse_toml as parse_client_toml
from agentbeats.run_scenario import parse_toml as parse_runner_toml
from generate_compose import (
    generate_a2a_scenario,
    generate_docker_compose,
    parse_scenario,
)

loguru_logger.disable("agentbeats.run_scenario")


class ScenarioContractTest(unittest.TestCase):
    def _scenario(self) -> dict:
        return {
            "evaluator": {
                "build": {
                    "context": ".",
                    "dockerfile": "src/evaluator/Dockerfile.evaluator",
                },
                "env": {"GEMINI_API_KEY": "${GEMINI_API_KEY}"},
            },
            "agent_under_test": {
                "build": {
                    "context": ".",
                    "dockerfile": "src/agent_under_test/Dockerfile.agent-under-test",
                },
                "env": {"AGENT_LLM": "gemini/gemini-2.5-flash"},
            },
            "config": {"num_trials": 1},
        }

    def test_compose_generation_uses_evaluator_and_agent_under_test(self) -> None:
        compose = generate_docker_compose(self._scenario())

        self.assertIn("  evaluator:", compose)
        self.assertIn("  agent-under-test:", compose)
        self.assertIn("  a2a-client:", compose)
        self.assertIn("agent-network:", compose)
        self.assertNotIn("green-agent", compose)

    def test_generated_a2a_scenario_uses_singular_aut_contract(self) -> None:
        scenario = generate_a2a_scenario(self._scenario())

        self.assertIn("[evaluator]", scenario)
        self.assertIn("[agent_under_test]", scenario)
        self.assertIn('endpoint = "http://agent-under-test:9009"', scenario)
        self.assertNotIn("[[participants]]", scenario)

    def test_client_cli_parses_new_shape(self) -> None:
        req, evaluator_url = parse_client_toml(
            {
                "evaluator": {"endpoint": "http://127.0.0.1:8081"},
                "agent_under_test": {"endpoint": "http://127.0.0.1:8080"},
                "config": {"num_trials": 1},
            }
        )

        self.assertEqual(evaluator_url, "http://127.0.0.1:8081")
        self.assertEqual(str(req.agent_under_test), "http://127.0.0.1:8080/")
        self.assertEqual(req.config["num_trials"], 1)

    def test_local_runner_parses_new_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[evaluator]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/evaluator/server.py --host 127.0.0.1 --port 8081"

[agent_under_test]
endpoint = "http://127.0.0.1:8080"
cmd = "python src/agent_under_test/server.py --host 127.0.0.1 --port 8080"

[config]
num_trials = 1
""".strip()
            )

            cfg = parse_runner_toml(str(path))

        self.assertEqual(cfg["evaluator"]["port"], 8081)
        self.assertEqual(cfg["agent_under_test"]["port"], 8080)
        self.assertEqual(cfg["config"]["num_trials"], 1)

    def test_old_scenario_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_client_toml(
                {
                    "green_agent": {"endpoint": "http://127.0.0.1:8081"},
                    "participants": [
                        {"role": "agent", "endpoint": "http://127.0.0.1:8080"}
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[green_agent]
endpoint = "http://127.0.0.1:8081"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:8080"
""".strip()
            )
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
                parse_scenario(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.toml"
            path.write_text(
                """
[evaluator]
endpoint = "http://127.0.0.1:8081"

[[participants]]
name = "agent"
endpoint = "http://127.0.0.1:8080"
""".strip()
            )
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
                parse_scenario(path)


if __name__ == "__main__":
    unittest.main()
