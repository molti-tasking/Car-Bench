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

## Open questions

- Does XML markup of the same protocol content change GLM's adherence?
  (v2_protocol vs v2_protocol_xml, Round 2)
- Same-model user-sim (GLM sim for GLM agent): watch for correlated blind
  spots; revalidate finalists with the official Gemini fixture before any
  submission decision.
