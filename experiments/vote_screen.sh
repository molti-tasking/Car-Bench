#!/bin/zsh
# Waits for the in-flight baseline test run to finish (frees ports 8080/8081),
# then screens the self-consistency voting harness on the 45-task train set
# against the champion's 73.3% plateau. Sleep-guarded; plugged-in + lid-open.
set -u
cd "$(dirname "$0")/.."

echo "[vote] waiting for baseline test run to release ports @ $(date -u +%H:%M)"
while pgrep -f "split test" >/dev/null 2>&1; do sleep 120; done
echo "[vote] ports free @ $(date -u +%H:%M) — starting voting screen"

run() {
  echo "[vote] START $* @ $(date -u +%H:%M)"
  caffeinate -s .venv/bin/python tools/experiment.py run "$@" \
    && echo "[vote] DONE: $*" || echo "[vote] FAILED: $*"
}

# The decider: champion config + 3-sample voting, wide train, 3 trials.
run --variant v4_german --self-check --vote 3 --tasks 15 --trials 3

echo "[vote] COMPLETE @ $(date -u +%H:%M)"
.venv/bin/python tools/experiment.py leaderboard
