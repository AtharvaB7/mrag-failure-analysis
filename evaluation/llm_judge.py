"""
LLM-assisted failure-mode labeling for the subset of instances that
rule_based_prefilter.py couldn't resolve (see that module's docstring for
which cases those are).

Supports either an Anthropic or OpenAI backend -- pick whichever you have
API access to; the classification prompt and output parsing are identical
either way. Per standard LLM-as-judge practice (and consistent with the
judge-validation approach used elsewhere in this project), you should
validate this judge against a small hand-labeled sample (e.g. Cohen's kappa
between the LLM judge and a human label on ~50-100 instances) before trusting
its labels across the full stratified sample -- this module doesn't do that
validation itself, it just produces the labels to validate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from evaluation.failure_taxonomy import FAILURE_MODES, LABEL_DESCRIPTIONS

JUDGE_PROMPT_TEMPLATE = """You are labeling why a vision-language model got a multiple-choice question wrong.

Question: {question}
Choices:
{choices_block}
Ground-truth answer: {ground_truth}
Model's raw output: {prediction}
Retrieval setting: {retrieval_type}
Retrieval recall for this question (fraction of ground-truth evidence images actually retrieved): {recall}

Choose exactly ONE label from this fixed list that best explains the failure:
{label_options_block}

Respond with strict JSON only, no other text:
{{"label": "<one of the labels above, exact spelling>", "justification": "<one sentence>"}}
"""


@dataclass
class JudgeLabel:
    qid: str
    label: str
    justification: str
    raw_response: str


def _build_prompt(instance: dict) -> str:
    letters = "ABCDEFGH"
    choices_block = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(instance["choices"]))
    label_options_block = "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in FAILURE_MODES)
    return JUDGE_PROMPT_TEMPLATE.format(
        question=instance["question"],
        choices_block=choices_block,
        ground_truth=instance["ground_truth"],
        prediction=instance["prediction"],
        retrieval_type=instance["retrieval_type"],
        recall=instance.get("recall", "N/A"),
        label_options_block=label_options_block,
    )


def _parse_response(raw_text: str) -> tuple[str, str]:
    """Parse the judge's JSON response, tolerating markdown code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(cleaned)
    label = parsed["label"].strip()
    if label not in FAILURE_MODES:
        raise ValueError(f"Judge returned an unrecognized label: {label!r}")
    return label, parsed.get("justification", "")


def label_with_anthropic(instance: dict, model: str = "claude-sonnet-5") -> JudgeLabel:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = _build_prompt(instance)
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = resp.content[0].text
    label, justification = _parse_response(raw_text)
    return JudgeLabel(qid=instance["id"], label=label, justification=justification, raw_response=raw_text)


def label_with_openai(instance: dict, model: str = "gpt-4o-mini") -> JudgeLabel:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(instance)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    raw_text = resp.choices[0].message.content
    label, justification = _parse_response(raw_text)
    return JudgeLabel(qid=instance["id"], label=label, justification=justification, raw_response=raw_text)


def label_batch(instances: list[dict], backend: str = "anthropic") -> list[JudgeLabel]:
    """instances: each dict needs id, question, choices, ground_truth,
    prediction, retrieval_type, and optionally recall."""
    labeler = label_with_anthropic if backend == "anthropic" else label_with_openai
    labels = []
    for instance in instances:
        try:
            labels.append(labeler(instance))
        except Exception as e:  # noqa: BLE001 -- log and continue; don't let one bad instance kill the batch
            print(f"WARNING: failed to label {instance['id']}: {e}")
    return labels
