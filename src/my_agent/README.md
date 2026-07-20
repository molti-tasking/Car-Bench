# My Agent (Track 1)

A clean Track 1 agent under test, based on the official starter, built for
**system-prompt experiments**: the evaluator-provided policy prompt is wrapped
with a configurable prefix/suffix selected by env var, so prompt variants
(e.g. English vs. German instructions) can be compared without code changes.

## How prompting works here

On the first turn of each task, the evaluator sends `System: <policy context>
User: <request>`. This agent keeps the policy text intact (policy compliance
is scored) and wraps it:

```text
<AGENT_SYSTEM_PROMPT_PREFIX or variant prefix>
<evaluator policy prompt — never modified>
<AGENT_SYSTEM_PROMPT_SUFFIX or variant suffix>
```

Variants live in [`prompts.py`](prompts.py):

| Variant | Idea |
| --- | --- |
| `baseline` | Evaluator prompt unchanged |
| `english_basic` | Minimal English reliability instructions |
| `german_basic` | Same instructions in German, English user-facing output |
| `german_reasoning` | Explicit German internal reasoning, English output |

Add a new variant by adding an entry to `PROMPT_VARIANTS` — no other change
needed.

## Configuration (all via env vars)

| Env var | Default | Purpose |
| --- | --- | --- |
| `AGENT_LLM` | `gemini/gemini-2.5-flash` | LiteLLM model string |
| `AGENT_API_KEY` | – | Explicit API key (provider vars like `ANTHROPIC_API_KEY` also work) |
| `AGENT_API_BASE` | – | Custom API base URL |
| `AGENT_TEMPERATURE` | `0.0` | Sampling temperature |
| `AGENT_REASONING_EFFORT` | – | Optional `low`/`medium`/`high`, passed to LiteLLM |
| `AGENT_PROMPT_VARIANT` | `baseline` | Named variant from `prompts.py` |
| `AGENT_SYSTEM_PROMPT_PREFIX` | – | Free-text override of the variant prefix |
| `AGENT_SYSTEM_PROMPT_SUFFIX` | – | Free-text override of the variant suffix |
| `AGENT_SELF_CHECK` | `false` | Pre-send verification pass (**in the champion config**) |
| `AGENT_ASK_GATE` | `false` | Preference-lookup gate before clarifying questions: `true`/`v1` = any question, `v2` = only genuine clarification questions (confirmations pass through) |
| `AGENT_SELF_CHECK_MODEL` | – | Different model for the self-check pass (cross-model verification) |
| `AGENT_VOTE_K` | `0` | Self-consistency voting (confirmed dead end — 71.1% at 3× tokens) |
| `AGENT_SCHEMA_GUARD` | `false` | Deterministic tool-call schema validation + corrective regen |
| `AGENT_FIREWALL` | `false` | Action firewall: ledger + provenance + compiled policy |
| `AGENT_FIREWALL_CHECKS` | all | Ablate the firewall: subset of `precondition,default,provenance` |
| `GUARD_EVENTS_PATH` | – | JSONL sink for guard firings (set per run by `experiment.py`) |

The evaluator additionally needs `GEMINI_API_KEY`.

## Measuring a guard, not just scoring it

A guard that never fires and a guard that fires uselessly both show up as a
flat Pass^3. `GUARD_EVENTS_PATH` separates them: every firing is logged with
its mechanism and, for the firewall, the specific check that tripped. The
experiment toolkit sets the path automatically and folds the counts into the
run record as `guard_events`, so `runs.jsonl` can answer "did this mechanism
have anything to act on?" before anyone argues about whether it helped.

The firewall's three checks have very different risk profiles and should be
measured separately before the full combination is trusted:

- `precondition` — ordering rules from the compiled policy; low risk.
- `default` — **highest risk**: it contradicts the model on the strength of a
  single unverified LLM extraction of policy prose. One hallucinated default
  rule fires on every matching action for the whole episode.
- `provenance` — episode-global, so any number appearing anywhere in the
  policy text licenses it everywhere; expect this one to fire rarely.

```bash
# Isolate the risky check before running the full firewall wide
uv run python tools/experiment.py run --variant v4_german --self-check \
    --firewall --firewall-checks precondition,provenance --tasks 15 --trials 3
```

Ablated runs get their own variant label (`…+firewall[precondition,provenance]`)
so the leaderboard never compares two different agents under one name.

## Run

Local smoke (one task per task type):

```bash
uv run car-bench-run scenarios/my_agent/local_smoke.toml --show-logs
```

Compare two prompt variants on the public test set:

```bash
AGENT_PROMPT_VARIANT=english_basic uv run car-bench-run scenarios/my_agent/local_test_set.toml
AGENT_PROMPT_VARIANT=german_basic  uv run car-bench-run scenarios/my_agent/local_test_set.toml
```

Results land in `output/`; compare `Pass^3` and `Pass@3` between runs. For
cheap screening, copy `local_test_set.toml` and lower the three task-count
fields (e.g. 10 each) before running the full 254-task set on finalists.

Docker smoke:

```bash
uv run python generate_compose.py --scenario scenarios/my_agent/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/my_agent/docker-compose.yml up --abort-on-container-exit
```

## Iteration loop (experiment toolkit)

The loop for optimizing prompt variants is driven by `tools/experiment.py`:

```bash
# 1. Run a variant (screening: train split, small subset)
uv run python tools/experiment.py run --variant german_basic --tasks 10 --trials 3

# 2. Cluster the failures with an LLM judge (writes experiments/reports/<run_id>.md)
uv run python tools/experiment.py analyze --run latest

# 3. Edit PROMPT_VARIANTS in prompts.py based on the report, then go to 1.

# 4. Compare variants
uv run python tools/experiment.py leaderboard

# 5. Confirm only your finalists on the full public test set (254 tasks — expensive)
uv run python tools/experiment.py run --variant german_basic --split test --full --trials 3
```

Every run is recorded in `experiments/runs.jsonl` (variant, model, split, git
SHA, Pass^k/Pass@k, cost) with the raw result JSON in `experiments/raw/`.

Rules of thumb:

- **Screen on train, confirm on test.** Final ranking uses a *hidden* set, so
  fixes must target failure classes, not individual public tasks. The judge
  report flags overfitting risks explicitly.
- Pass^3 needs `--trials 3`; single-trial runs only show pass rate.

### Langfuse tracing (dev-only)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and optionally
`LANGFUSE_HOST`) in `.env` to trace every LiteLLM call from both the agent and
the locally-run evaluator (simulated user + policy judge) to Langfuse Cloud.
Traces are grouped per conversation (`session_id` = A2A context id) and tagged
with the run id and prompt variant. The `langfuse` package lives in the dev
dependency group, which the Docker build excludes (`--no-dev`) — the submitted
image never traces and needs no Langfuse configuration.

## Submission checklist

1. Build and push a public `linux/amd64` image:

   ```bash
   docker build --platform linux/amd64 \
     -f src/my_agent/Dockerfile.my-agent \
     -t ghcr.io/yourusername/car-bench-my-agent:latest .
   docker push ghcr.io/yourusername/car-bench-my-agent:latest
   ```

2. Set the GHCR package visibility to **Public**, then validate with
   `scenarios/my_agent/ghcr_smoke.toml`.
3. Pin the digest and finalize [`scenarios/my_agent/submission.toml`](../../scenarios/my_agent/submission.toml)
   (`task_split = "hidden"`, task counts `-1`, image `@sha256:...`).
4. Write the 4-page IJCAI technical report citing CAR-bench.
