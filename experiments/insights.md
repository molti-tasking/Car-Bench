# Campaign insights (confirmed findings only)

Accumulating knowledge base of the CAR-bench improvement campaign. Every entry
cites run evidence. The leaderboard says *what* won; this file says *why*.

## 2026-07-04 — Round 1 (variant screen, kimi user-sim)

- **Kimi-K2.5 cannot simulate hallucination-task users.** It emits
  `HALLUCINATION_ERROR` on the first turn, before the agent speaks (4/5
  hallucination tasks in `20260704-113008`, same in `-112815` smoke; MiniMax
  shows the same defect in `simtest-minimax`). GLM runs the conversations
  correctly (`simtest-glm`, 3/3 real conversations). → **User-sim = GLM** from
  Round 2 on; Kimi stays as policy/failure judge. Round-1 hallucination
  columns are artifacts; Base/Disambiguation columns are valid.
- **Naive reliability suffixes hurt GLM on Base/Disambiguation** (baseline 60%
  vs english_basic/german_basic 46.7%). Judge-confirmed mechanism (report
  `20260704-113008`): over-disambiguation — the agent keeps clarifying after
  the user already confirmed (disambiguation_0), and asks the user instead of
  checking stored preferences (disambiguation_4). Advisory "ask when unsure"
  prose amplifies exactly this. → v2 uses an ordered decision procedure with
  "resolve ambiguity yourself first" and "act on confirmation".
- **German processing (german_reasoning, 53.3%) ≈ baseline on valid columns**
  — no evidence yet that reasoning language matters for GLM; revisit with
  trials=3 once the protocol content is settled.
- **DISAMBIGUATION_ERROR fires when the agent asks the user for something the
  benchmark expects it to resolve internally** (user_end_conversation.py
  schema) — "clarify more" is the *wrong* fix for this category; "look it up
  first" is the right one.

## 2026-07-04 — Round 2 (GLM user-sim, baseline vs v2 protocol pair)

- **Simulator choice dominated round-1 numbers**: baseline 60% → 86.7% after
  switching user-sim kimi→GLM (`20260704-120859`). Any cross-round comparison
  must hold the simulator fixed.
- **XML vs prose markup: zero difference for GLM** — v2_protocol and
  v2_protocol_xml scored identically (80%) and failed the *same tasks the
  same way* (`-121407` vs `-121925`). Structure ablation answered; don't
  spend more runs on markup.
- **"Sensible defaults" instruction backfired twice**: agent fabricated a
  parallel sunshade action without a tool call (hallucination_0 → real
  HALLUCINATION_ERROR), and defaulted the sunroof to 100% where the policy
  default is 50% (disambiguation_0 expected `open_close_sunroof(50)`).
  Defaults must be routed to policy text, and claims to tool results.
- **"Check stored preferences" is too abstract** — the benchmark has a
  `get_user_preferences` tool; the instruction must name the mechanism
  (disambiguation_4 expects `get_user_preferences` → `set_ambient_lights`
  with the stored PURPLE).
- **GLM user-sim still misfires occasionally** (stochastic first-turn
  `###STOP###` in `-121407` hallucination_0 that the conversation recovered
  from, but the failure flag sticks). Small-sample category scores carry
  simulator noise; confirm anything important with trials=3.

## 2026-07-04 — Round 3 (v3 refinements + German protocol)

- **german_protocol: 100% (15/15), first clean sweep** — including both
  stubborn disambiguation tasks. Trajectory-verified: the agent proactively
  calls `get_user_preferences`, applies the stored 50% sunroof default and
  the PURPLE evening preference, and answers the user in English throughout.
- **The same protocol in English (v3_grounded) scored 80%** — it did NOT
  reliably trigger the preference lookup. The language of the *instructions*
  changed tool-use behavior with identical semantics. Single-trial evidence;
  Pass^3 confirmation running (Round 4). If it holds, this is the paper's
  headline: instruction-language sensitivity of tool-using behavior in GLM.
- **v3_minimal (two rules) ties baseline (86.7%)** — brevity preserves the
  baseline but the two rules alone didn't beat it on this subset.

## 2026-07-04 — Round 4 (Pass^3 confirmation, 3 trials × 15 tasks)

- **german_protocol confirmed champion**: Pass^3 86.7%, Pass@3 100%,
  trial-level pass rate 95.6% (`20260704-1837xx`). The round-3 clean sweep
  softened but held: per-trial rates 93–100% vs English variants at 80–87%.
  Per category: Base Pass^3 100%, Disambiguation 80% (one flake in one
  trial), Hallucination 80% (one task fails consistently — next target).
- **v3_minimal**: Pass^3 80%, Pass@3 93.3% — clearly behind.
- **Language effect replicated across 4 independent trials** (round 3 + 3×
  round 4): German-language instructions reliably induce the
  preference-lookup behavior that identical English instructions do not.
- **Ops**: one dead proxy connection wedged a 3-trial run for ~3h —
  litellm timeouts now set on both agent (AGENT_LLM_TIMEOUT) and local
  evaluator (EVALUATOR_LLM_TIMEOUT), commit dd1befe. Baseline's 3-trial run
  was interrupted (user stop); its single-trial 86.7% stands as reference.

## 2026-07-05 — Round 5 (wide 45-task runs: anchors, v4, self-check harness)

- **New champion: v4_german + self-check — Pass^3 73.3%, Pass@3 91.1%, pass
  rate 82.2%** (`20260705-134537`). First configuration to beat baseline at
  full width. Token cost +32% (~6.2M/run vs ~4.7M).
- **Wide baseline anchor: Pass^3 71.1%** — which means *no prompt variant
  alone beats no-prompt at 45 tasks* (v4_german ties 71.1%, german_protocol
  66.7%). The 15-task German-language advantage did not generalize —
  subset effects can invert at scale; always anchor at equal width.
- **v4_german traded categories, not totals**: minimalism/defaults rules
  lifted Disambiguation 40→60% and Hallucination 67→73% Pass^3 but dropped
  Base 93→80% (over-minimalism suppresses required auxiliary actions).
  Self-check on top recovered Base to 87% while keeping the gains
  (87/73/60) — verification composes where prompting trades off.
- **Ops lesson**: harness-managed background tasks die on laptop
  sleep/wake; detached `nohup` processes survive. Long runs must be
  launched detached with a disposable registry watcher.

## 2026-07-05 — Round 6 (v5 prompt refinement)

- **v5_german+selfcheck: Pass^3 73.3% — ties the champion but loses the
  tiebreak** (Pass@3 86.7 vs 91.1, pass rate 80.0 vs 82.2,
  `20260705-152847`). Category see-saw again: state-verification rules fixed
  Base (87→93) but stricter no-invented-constraints dropped Disambiguation
  (60→53). **Prompt-text iteration has plateaued at 73.3%** across two
  refinement rounds; every rule buys one category with another.
- Champion remains **v4_german+selfcheck**. Round 7 shifts fully to harness
  logic: ask-gate (one internal regeneration nudge when the agent is about
  to ask a clarifying question without having called get_user_preferences)
  targeting the Disambiguation Pass^3 ceiling.

## 2026-07-05 — Round 7 (ask-gate) and plateau declaration

- **Ask-gate: Pass^3 73.3% — third consecutive configuration at exactly this
  number** (`20260705-171624`). Target category improved (Disambiguation
  60→67) but Base paid (87→80): the "?"-heuristic occasionally derails
  straightforward flows with unneeded lookups.
- **Plateau is real, not noise**: only 1–3 failing rows per run (4–12%) are
  simulator misfire artifacts → true ceiling ≈ 73–75% Pass^3 for
  GLM + prompt + single-pass verification on this benchmark.
- **Final champion: v4_german + self-check** (Pass^3 73.3%, Pass@3 91.1%,
  pass rate 82.2%) — wins the Pass@3 tiebreak over v5 (86.7) and ask-gate
  (86.7). Ask-gate stays in the codebase (env-gated off) as an ablation.
- Campaign arc for the report: baseline 71.1% → prompt engineering
  plateaued at ≤71.1% (trades categories) → verification harness broke to
  73.3% (composes) → further gating shuffles but doesn't lift. The
  remaining Pass@3−Pass^3 gap (~18pp) is trial-to-trial inconsistency of a
  capable model — the benchmark's core thesis, reproduced.

## 2026-07-06 — Full public test-set measurement

- **Champion generalizes: v4_german+selfcheck on the full test split (125
  tasks × 3 trials) — Pass^3 71.3%, Pass@3 89.3%, pass rate 83.7%**
  (`20260705-235030`). Only ~2pp below its train-wide number → the
  train-fitted rules transfer. Category shift on unseen tasks: Base 78
  (weaker), Hallucination 76 (stronger), Disambiguation 60 (same).
- 31.4M agent tokens for the run (~84k/task-trial). Baseline test run in
  progress for the comparison gap.
- The ~18pp Pass@3−Pass^3 consistency gap persists on test — the voting
  harness (built, unmeasured) targets exactly this.

## Open questions

- Does XML markup of the same protocol content change GLM's adherence?
  (v2_protocol vs v2_protocol_xml, Round 2)
- Same-model user-sim (GLM sim for GLM agent): watch for correlated blind
  spots; revalidate finalists with the official Gemini fixture before any
  submission decision.
