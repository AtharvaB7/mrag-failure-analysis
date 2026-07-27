"""
Optional GPT-based answer extraction, used only as a fallback when
`evaluation.metrics.extract_choice_letter` (rule-based) returns None and
config sets `evaluation.answer_extraction: "gpt_fallback"`.

Needs OPENAI_API_KEY set in the environment. Kept isolated in its own module
so the core pipeline has zero hard dependency on the OpenAI API -- most runs
should need this for only a small tail of malformed generations.
"""
from __future__ import annotations

import os


def gpt_extract_choice_letter(raw_output: str, choices: list[str]) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    letters = "ABCDEFGH"
    choices_block = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    prompt = (
        "A model was asked a multiple-choice question and gave the following raw response. "
        "Extract which choice letter (A-H) it selected. If it's genuinely ambiguous, respond 'NONE'.\n\n"
        f"Choices:\n{choices_block}\n\nModel response:\n{raw_output}\n\nLetter:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().upper()
    return answer if answer in letters else None
