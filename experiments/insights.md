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

## Open questions

- Does XML markup of the same protocol content change GLM's adherence?
  (v2_protocol vs v2_protocol_xml, Round 2)
- Same-model user-sim (GLM sim for GLM agent): watch for correlated blind
  spots; revalidate finalists with the official Gemini fixture before any
  submission decision.
