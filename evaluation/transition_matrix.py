"""
Build the failure-mode transition matrix: for a fixed model, how does an
instance's failure label change as the retrieval setting improves
(no_retrieval -> sparse -> dense -> hybrid)? This is the centerpiece result
described in the project proposal.

Two things this module is careful about:
  1. Coverage: an instance can only appear in a transition if it has a
     resolved label (via rule or LLM judge) at BOTH settings being compared.
     Instances outside the stratified LLM sample and not resolved by rule are
     excluded from the matrix, not silently imputed -- `unlabeled_count` in
     the returned result tells you how much coverage you actually have, so
     you can judge whether the matrix is trustworthy or needs a bigger sample.
  2. Per Yifei's confound concern: this module builds ONE matrix per model.
     It deliberately does not pool across models -- call it once per model
     and compare the two matrices side by side, rather than averaging them
     into a single matrix that would obscure model-dependent differences.
"""
from __future__ import annotations

from collections import defaultdict

from evaluation.failure_taxonomy import ALL_LABELS


def merge_labels(
    prefilter_results: list,  # list[PrefilterResult]
    judge_labels: list,  # list[JudgeLabel]
) -> dict[str, str]:
    """Combine rule-resolved labels and LLM-judge labels into one qid -> label
    dict. Rule-resolved labels always win if somehow both exist for the same
    qid (they shouldn't, since prefilter_run only leaves rule-UNresolved
    instances as candidates for judging, but this keeps the merge order
    explicit rather than relying on that invariant silently)."""
    merged: dict[str, str] = {}
    for judge_label in judge_labels:
        merged[judge_label.qid] = judge_label.label
    for pf in prefilter_results:
        if pf.label is not None:
            merged[pf.qid] = pf.label  # rule-based takes precedence
    return merged


def build_transition_matrix(
    labels_by_setting: dict[str, dict[str, str]],
    from_setting: str,
    to_setting: str,
) -> tuple[list[list[int]], int]:
    """labels_by_setting: {"no_retrieval": {qid: label, ...}, "sparse": {...}, ...}
    Returns (matrix, unlabeled_count) where matrix[i][j] = number of instances
    whose label was ALL_LABELS[i] at from_setting and ALL_LABELS[j] at
    to_setting. unlabeled_count = instances present in from_setting but
    missing a label at to_setting (or vice versa) -- these are excluded from
    the matrix and should be reported alongside it, not swept under the rug.
    """
    from_labels = labels_by_setting[from_setting]
    to_labels = labels_by_setting[to_setting]

    label_index = {label: i for i, label in enumerate(ALL_LABELS)}
    n = len(ALL_LABELS)
    matrix = [[0] * n for _ in range(n)]

    unlabeled_count = 0
    all_qids = set(from_labels) | set(to_labels)
    for qid in all_qids:
        from_label = from_labels.get(qid)
        to_label = to_labels.get(qid)
        if from_label is None or to_label is None:
            unlabeled_count += 1
            continue
        matrix[label_index[from_label]][label_index[to_label]] += 1

    return matrix, unlabeled_count


def matrix_to_readable_rows(matrix: list[list[int]]) -> list[dict]:
    """Turn the raw matrix into a list of {from, to, count} rows -- easier to
    dump to CSV / inspect than a raw nested list."""
    rows = []
    for i, from_label in enumerate(ALL_LABELS):
        for j, to_label in enumerate(ALL_LABELS):
            if matrix[i][j] > 0:
                rows.append({"from": from_label, "to": to_label, "count": matrix[i][j]})
    return rows


def build_full_chain_matrices(
    labels_by_setting: dict[str, dict[str, str]],
    setting_order: list[str] = ("no_retrieval", "sparse", "dense", "hybrid"),
) -> dict[str, tuple[list[list[int]], int]]:
    """Build a transition matrix for every consecutive pair in setting_order
    (no_retrieval->sparse, sparse->dense, dense->hybrid), plus one endpoint
    matrix (no_retrieval->hybrid) for the headline result."""
    setting_order = list(setting_order)
    results = {}
    for a, b in zip(setting_order[:-1], setting_order[1:]):
        results[f"{a}_to_{b}"] = build_transition_matrix(labels_by_setting, a, b)
    if len(setting_order) > 2:
        results[f"{setting_order[0]}_to_{setting_order[-1]}"] = build_transition_matrix(
            labels_by_setting, setting_order[0], setting_order[-1]
        )
    return results
