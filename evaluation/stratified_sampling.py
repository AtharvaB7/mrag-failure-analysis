"""
Stratified sampling for the LLM-assisted annotation pass.

Full manual/LLM labeling of every rule-unresolved instance (recall this is
already a subset -- 'correct' and zero-recall instances were resolved for
free by rule_based_prefilter.py) is unnecessary and, at MRAG-Bench's scale
(1,353 questions x 4 settings x 2 models), needlessly expensive. Instead we
sample a fixed budget per (scenario, retrieval_type) stratum, so every
scenario and every retrieval setting gets adequate representation in the
labeled set rather than the sample being dominated by whichever stratum
happens to have the most unresolved instances.
"""
from __future__ import annotations

import random
from collections import defaultdict


def stratified_sample(
    unresolved_qids: list[tuple[str, str, str]],
    per_stratum_budget: int,
    seed: int = 42,
) -> list[str]:
    """unresolved_qids: list of (qid, scenario, retrieval_type) tuples for
    instances rule_based_prefilter.py couldn't resolve.
    Returns: the sampled subset of qids, up to per_stratum_budget per
    (scenario, retrieval_type) stratum (fewer if the stratum is smaller)."""
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, scenario, retrieval_type in unresolved_qids:
        by_stratum[(scenario, retrieval_type)].append(qid)

    sampled = []
    for stratum_qids in by_stratum.values():
        stratum_qids_sorted = sorted(stratum_qids)  # sort first so shuffle is reproducible regardless of input order
        rng.shuffle(stratum_qids_sorted)
        sampled.extend(stratum_qids_sorted[:per_stratum_budget])
    return sampled


def coverage_report(
    unresolved_qids: list[tuple[str, str, str]],
    sampled_qids: list[str],
) -> dict[tuple[str, str], dict[str, int]]:
    """Sanity-check helper: how many unresolved instances existed per stratum
    vs. how many were actually sampled, so you can see at a glance whether any
    stratum was starved."""
    sampled_set = set(sampled_qids)
    by_stratum: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"unresolved": 0, "sampled": 0})
    for qid, scenario, retrieval_type in unresolved_qids:
        key = (scenario, retrieval_type)
        by_stratum[key]["unresolved"] += 1
        if qid in sampled_set:
            by_stratum[key]["sampled"] += 1
    return dict(by_stratum)
