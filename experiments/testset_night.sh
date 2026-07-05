#!/bin/zsh
# Overnight full public test-set measurement: champion first, then baseline.
# 254 tasks × 3 trials each (~6h/run awake; laptop sleep pauses and resumes).
set -u
cd "$(dirname "$0")/.."

run() {
  echo "[testset] START $* @ $(date -u +%H:%M)"
  caffeinate -i .venv/bin/python tools/experiment.py run "$@" \
    && echo "[testset] DONE: $*" || echo "[testset] FAILED: $*"
}

run --variant v4_german --self-check --split test --full --trials 3
run --variant baseline --split test --full --trials 3

echo "[testset] ALL DONE @ $(date -u +%H:%M)"
.venv/bin/python tools/experiment.py leaderboard
