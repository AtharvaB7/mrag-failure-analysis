"""
Core metrics: answer extraction, correctness, retrieval recall, and the
per-scenario contamination check flagged as important by Yifei's feedback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def extract_choice_letter(raw_output: str) -> str | None:
    """Pull a single A-H letter out of the model's raw generation.

    Rule-based first (handles the overwhelming majority of cases when the
    prompt explicitly asks for "only the letter"); falls back to
    evaluation/gpt_fallback.py's GPT-based extractor only if this returns
    None and `evaluation.answer_extraction == "gpt_fallback"` in config.
    """
    raw_output = raw_output.strip()
    # Common patterns: "A", "A.", "(A)", "Answer: A", "The answer is A." (case-insensitive,
    # since models don't reliably respect requested casing).
    match = re.search(r"\b([A-Ha-h])\b", raw_output)
    if match:
        return match.group(1).upper()
    return None


def is_correct(raw_output: str, gt_answer: str) -> bool:
    pred = extract_choice_letter(raw_output)
    return pred is not None and pred.upper() == gt_answer.strip().upper()


def retrieval_recall_at_k(retrieved_ids: list[str], gt_image_ids: list[str]) -> float:
    """Fraction of ground-truth images that appear anywhere in the retrieved set.
    Returns 1.0 if there are no GT images to find (vacuously true; exclude
    these from aggregate recall if you want a stricter denominator)."""
    if not gt_image_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for gt in gt_image_ids if gt in retrieved_set)
    return hits / len(gt_image_ids)


@dataclass
class ContaminationFlag:
    scenario: str
    no_retrieval_acc: float
    hybrid_acc: float
    gap: float
    flagged: bool


def check_contamination(
    no_retrieval_acc_by_scenario: dict[str, float],
    hybrid_acc_by_scenario: dict[str, float],
    threshold: float = 0.05,
) -> list[ContaminationFlag]:
    """Per Yifei's feedback: a model's no-retrieval accuracy being suspiciously
    close to its hybrid-retrieval accuracy on a given scenario suggests that
    scenario's answers may already be memorized from pretraining (i.e., the
    dataset has leaked into the model's parametric knowledge), rather than
    retrieval genuinely being unhelpful there. Flag, don't silently exclude --
    the decision to drop a scenario from headline results should be a
    deliberate, documented step, not automatic.
    """
    flags = []
    for scenario, no_ret_acc in no_retrieval_acc_by_scenario.items():
        hybrid_acc = hybrid_acc_by_scenario.get(scenario)
        if hybrid_acc is None:
            continue
        gap = abs(no_ret_acc - hybrid_acc)
        flags.append(
            ContaminationFlag(
                scenario=scenario,
                no_retrieval_acc=no_ret_acc,
                hybrid_acc=hybrid_acc,
                gap=gap,
                flagged=gap < threshold,
            )
        )
    return flags
