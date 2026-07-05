#!/bin/zsh
# Round 5 queue — run when the machine will stay awake ~3h (plugged in,
# lid open). Usage:  zsh experiments/resume_round5.sh
# Produces: wide baseline anchor, v4_german, and v4_german+selfcheck runs,
# then the leaderboard. Each run ~1h; all self-hosted via the LiteLLM proxy.
set -u
cd "$(dirname "$0")/.."

run() {
  echo "[round5] START $*"
  caffeinate -i .venv/bin/python tools/experiment.py run "$@" \
    && echo "[round5] DONE: $*" || echo "[round5] FAILED: $*"
}

run --variant baseline --tasks 15 --trials 3
run --variant v4_german --tasks 15 --trials 3
run --variant v4_german --self-check --tasks 15 --trials 3

echo "[round5] COMPLETE"
.venv/bin/python tools/experiment.py leaderboard
.venv/bin/python tools/dashboard.py
echo "[round5] dashboard regenerated — ask Claude to redeploy the artifact"
