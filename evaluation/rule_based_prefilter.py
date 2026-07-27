"""
Rule-based prefilter for failure-mode labeling.

Two cases are cheap and reliable to resolve without any LLM call:
  1. Correct predictions -> label "correct", trivially.
  2. Incorrect predictions where the retriever had zero recall on the
     ground-truth evidence -> label "retrieval_missed_evidence", since if
     none of the correct evidence was even retrieved, the model had no
     chance of using it, and no amount of examining the model's reasoning
     changes that diagnosis.

Everything else (incorrect + some/all evidence was retrieved, or incorrect
under the no-retrieval setting where there's no retrieval signal to check)
requires actually looking at the model's answer/reasoning to disambiguate
missing-knowledge vs. visual-grounding vs. OCR vs. hallucination vs.
multi-hop failure -- that's the LLM-assisted pass in llm_judge.py, applied
only to a stratified sample per stratified_sampling.py, not to every instance.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluation.failure_taxonomy import CORRECT
from evaluation.metrics import is_correct, retrieval_recall_at_k


@dataclass
class PrefilterResult:
    qid: str
    label: str | None  # None means "needs LLM-assisted labeling"
    resolved_by_rule: bool


def prefilter_instance(
    qid: str,
    prediction: str,
    ground_truth: str,
    retrieval_type: str,
    retrieved_ids: list[str],
    gt_image_ids: list[str],
) -> PrefilterResult:
    if is_correct(prediction, ground_truth):
        return PrefilterResult(qid=qid, label=CORRECT, resolved_by_rule=True)

    if retrieval_type == "none":
        # No retrieval happened at all -- nothing rule-based to check.
        return PrefilterResult(qid=qid, label=None, resolved_by_rule=False)

    recall = retrieval_recall_at_k(retrieved_ids, gt_image_ids)
    if recall == 0.0 and gt_image_ids:
        return PrefilterResult(qid=qid, label="retrieval_missed_evidence", resolved_by_rule=True)

    return PrefilterResult(qid=qid, label=None, resolved_by_rule=False)


def prefilter_run(predictions: list[dict], retrieval_by_qid: dict[str, dict]) -> list[PrefilterResult]:
    """predictions: list of dicts as saved in predictions.json.
    retrieval_by_qid: qid -> {"retrieved_ids": [...]} as saved in retrieval.json (keyed)."""
    results = []
    for pred in predictions:
        qid = pred["id"]
        r = retrieval_by_qid.get(qid, {})
        results.append(
            prefilter_instance(
                qid=qid,
                prediction=pred["prediction"],
                ground_truth=pred["ground_truth"],
                retrieval_type=pred["retrieval_type"],
                retrieved_ids=r.get("retrieved_ids", []),
                gt_image_ids=pred.get("gt_image_ids", []),
            )
        )
    return results
