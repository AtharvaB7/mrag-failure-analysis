"""
The fixed failure-mode taxonomy from the project proposal. Kept as a single
source of truth so the rule-based prefilter, the LLM-judge prompt, and the
transition-matrix code all reference the same category strings -- a mismatch
between any two of these (e.g. a typo in the LLM prompt) would silently
corrupt the transition matrix, so this module is the only place these
strings are allowed to be spelled out.
"""
from __future__ import annotations

CORRECT = "correct"

# Every non-"correct" label a failed instance can carry. Order matters only
# for display purposes (e.g. matrix row/column ordering).
FAILURE_MODES = [
    "missing_factual_knowledge",
    "missing_cultural_contextual_knowledge",
    "retrieval_missed_evidence",
    "evidence_retrieved_but_not_used",
    "visual_grounding_error",
    "ocr_error",
    "hallucination_despite_correct_evidence",
    "multi_hop_reasoning_failure",
]

ALL_LABELS = [CORRECT] + FAILURE_MODES

LABEL_DESCRIPTIONS = {
    "missing_factual_knowledge": (
        "The model lacks the factual/world knowledge needed to answer, and "
        "retrieval either wasn't available or didn't supply it."
    ),
    "missing_cultural_contextual_knowledge": (
        "The model lacks culturally or contextually specific knowledge "
        "(e.g. region-specific objects, conventions, or practices) needed to answer."
    ),
    "retrieval_missed_evidence": (
        "The retriever failed to surface any of the ground-truth evidence "
        "images for this question (retrieval recall == 0 for this instance)."
    ),
    "evidence_retrieved_but_not_used": (
        "The ground-truth evidence was retrieved (recall > 0) but the model's "
        "answer doesn't reflect having used it -- it answered as if unretrieved."
    ),
    "visual_grounding_error": (
        "The model attended to or described the wrong region/object/attribute "
        "in an image it did have access to."
    ),
    "ocr_error": (
        "The model misread text/numbers/labels rendered within an image."
    ),
    "hallucination_despite_correct_evidence": (
        "The correct evidence was both retrieved and, based on the model's "
        "reasoning, apparently attended to -- yet the final answer contradicts it."
    ),
    "multi_hop_reasoning_failure": (
        "The model needed to combine multiple pieces of evidence (retrieved "
        "and/or in-image) across steps, and the failure is in that combination "
        "step rather than in any single piece of evidence."
    ),
}


def validate_label(label: str) -> None:
    if label not in ALL_LABELS:
        raise ValueError(f"'{label}' is not a recognized label. Valid labels: {ALL_LABELS}")
