"""System-prompt variants for the CAR-bench agent under test.

Each variant AUGMENTS the evaluator-provided system prompt (the policy
context) — it never replaces it. Policy compliance is scored directly, so the
evaluator's policy text must always remain in the system prompt. `prefix` is
prepended above the policy text, `suffix` appended below it.

Select a variant at runtime with the AGENT_PROMPT_VARIANT env var. Free-text
overrides via AGENT_SYSTEM_PROMPT_PREFIX / AGENT_SYSTEM_PROMPT_SUFFIX take
precedence over the selected variant (useful for quick experiments without a
code change or image rebuild).

Note: the simulated user, tasks, tools, and scoring are all English. Variants
in other languages should steer *internal* processing only and must instruct
the model to keep user-facing responses in English.
"""

PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    # Evaluator system prompt used unchanged.
    "baseline": {
        "prefix": "",
        "suffix": "",
    },
    # Minimal English reliability instructions.
    "english_basic": {
        "prefix": "",
        "suffix": (
            "\n\nAdditional instructions:\n"
            "- Follow the policies above exactly.\n"
            "- Never invent capabilities, tools, or data. If something is"
            " unavailable, say so plainly.\n"
            "- If the user's request is ambiguous, ask a clarifying question"
            " before acting."
        ),
    },
    # The same instructions in German; user-facing output stays English.
    "german_basic": {
        "prefix": "",
        "suffix": (
            "\n\nZusätzliche Anweisungen:\n"
            "- Befolge die oben genannten Richtlinien exakt.\n"
            "- Erfinde niemals Funktionen, Werkzeuge oder Daten. Wenn etwas"
            " nicht verfügbar ist, sage das offen.\n"
            "- Wenn die Anfrage des Nutzers mehrdeutig ist, stelle zuerst eine"
            " Rückfrage, bevor du handelst.\n"
            "- Antworte dem Nutzer immer auf Englisch."
        ),
    },
    # Explicitly ask for German internal reasoning with English output.
    "german_reasoning": {
        "prefix": "",
        "suffix": (
            "\n\nZusätzliche Anweisungen:\n"
            "- Denke intern auf Deutsch: Analysiere die Anfrage, die"
            " Richtlinien und die verfügbaren Werkzeuge Schritt für Schritt"
            " auf Deutsch, bevor du antwortest.\n"
            "- Antworte dem Nutzer und formuliere alle Werkzeugaufrufe"
            " ausschließlich auf Englisch.\n"
            "- Erfinde niemals Funktionen oder Daten; sage offen, wenn etwas"
            " nicht verfügbar ist.\n"
            "- Kläre mehrdeutige Anfragen durch eine Rückfrage, bevor du"
            " handelst."
        ),
    },
}
