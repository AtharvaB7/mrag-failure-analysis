import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.failure_taxonomy import ALL_LABELS, CORRECT, FAILURE_MODES, validate_label
from evaluation.llm_judge import JudgeLabel, _parse_response
from evaluation.rule_based_prefilter import prefilter_instance, prefilter_run
from evaluation.stratified_sampling import coverage_report, stratified_sample
from evaluation.transition_matrix import build_full_chain_matrices, build_transition_matrix, merge_labels


# ---------- taxonomy ----------

def test_all_labels_includes_correct_plus_failure_modes():
    assert ALL_LABELS[0] == CORRECT
    assert set(ALL_LABELS[1:]) == set(FAILURE_MODES)


def test_validate_label_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        validate_label("not_a_real_label")


# ---------- rule-based prefilter ----------

def test_prefilter_correct_prediction():
    r = prefilter_instance("q1", "Answer: A", "A", "hybrid", ["img1"], ["img1"])
    assert r.label == CORRECT
    assert r.resolved_by_rule


def test_prefilter_no_retrieval_setting_is_never_rule_resolved_when_wrong():
    r = prefilter_instance("q2", "Answer: B", "A", "none", [], [])
    assert r.label is None
    assert not r.resolved_by_rule


def test_prefilter_zero_recall_resolves_to_retrieval_missed_evidence():
    r = prefilter_instance("q3", "Answer: B", "A", "hybrid", ["img5", "img6"], ["img1"])
    assert r.label == "retrieval_missed_evidence"
    assert r.resolved_by_rule


def test_prefilter_nonzero_recall_wrong_answer_needs_llm():
    r = prefilter_instance("q4", "Answer: B", "A", "hybrid", ["img1", "img2"], ["img1"])
    assert r.label is None
    assert not r.resolved_by_rule


def test_prefilter_run_batches_correctly():
    predictions = [
        {"id": "q1", "prediction": "A", "ground_truth": "A", "retrieval_type": "hybrid", "gt_image_ids": ["img1"]},
        {"id": "q2", "prediction": "B", "ground_truth": "A", "retrieval_type": "hybrid", "gt_image_ids": ["img1"]},
    ]
    retrieval_by_qid = {
        "q1": {"retrieved_ids": ["img1"]},
        "q2": {"retrieved_ids": ["imgX"]},  # missed the GT evidence entirely
    }
    results = prefilter_run(predictions, retrieval_by_qid)
    labels = {r.qid: r.label for r in results}
    assert labels["q1"] == CORRECT
    assert labels["q2"] == "retrieval_missed_evidence"


# ---------- stratified sampling ----------

def test_stratified_sample_respects_per_stratum_budget():
    unresolved = [(f"q{i}", "scenario_a", "hybrid") for i in range(10)] + [
        (f"q{i}", "scenario_b", "hybrid") for i in range(10, 13)
    ]
    sampled = stratified_sample(unresolved, per_stratum_budget=2, seed=0)
    assert len(sampled) == 4  # 2 from scenario_a, 2 from scenario_b


def test_stratified_sample_is_deterministic():
    unresolved = [(f"q{i}", "scenario_a", "hybrid") for i in range(20)]
    s1 = stratified_sample(unresolved, per_stratum_budget=5, seed=7)
    s2 = stratified_sample(unresolved, per_stratum_budget=5, seed=7)
    assert s1 == s2


def test_stratified_sample_smaller_stratum_not_padded():
    unresolved = [("q1", "rare_scenario", "sparse")]  # only 1 instance in this stratum
    sampled = stratified_sample(unresolved, per_stratum_budget=5, seed=0)
    assert sampled == ["q1"]


def test_coverage_report_flags_stratum_sizes():
    unresolved = [(f"q{i}", "scenario_a", "hybrid") for i in range(5)]
    sampled = stratified_sample(unresolved, per_stratum_budget=2, seed=0)
    report = coverage_report(unresolved, sampled)
    assert report[("scenario_a", "hybrid")]["unresolved"] == 5
    assert report[("scenario_a", "hybrid")]["sampled"] == 2


# ---------- LLM judge response parsing ----------

def test_parse_response_handles_plain_json():
    label, justification = _parse_response('{"label": "ocr_error", "justification": "misread the sign"}')
    assert label == "ocr_error"
    assert justification == "misread the sign"


def test_parse_response_strips_markdown_fences():
    raw = '```json\n{"label": "visual_grounding_error", "justification": "wrong object"}\n```'
    label, _ = _parse_response(raw)
    assert label == "visual_grounding_error"


def test_parse_response_rejects_unrecognized_label():
    import pytest

    with pytest.raises(ValueError):
        _parse_response('{"label": "made_up_label", "justification": "x"}')


# ---------- transition matrix ----------

def test_merge_labels_prefers_rule_based_over_judge():
    from evaluation.rule_based_prefilter import PrefilterResult

    prefilter_results = [PrefilterResult(qid="q1", label="retrieval_missed_evidence", resolved_by_rule=True)]
    judge_labels = [JudgeLabel(qid="q1", label="ocr_error", justification="x", raw_response="{}")]
    merged = merge_labels(prefilter_results, judge_labels)
    assert merged["q1"] == "retrieval_missed_evidence"


def test_build_transition_matrix_counts_correctly():
    labels_by_setting = {
        "no_retrieval": {"q1": "missing_factual_knowledge", "q2": CORRECT, "q3": "visual_grounding_error"},
        "hybrid": {"q1": CORRECT, "q2": CORRECT, "q3": "visual_grounding_error"},
    }
    matrix, unlabeled = build_transition_matrix(labels_by_setting, "no_retrieval", "hybrid")
    from evaluation.failure_taxonomy import ALL_LABELS

    i_missing = ALL_LABELS.index("missing_factual_knowledge")
    i_correct = ALL_LABELS.index(CORRECT)
    i_visual = ALL_LABELS.index("visual_grounding_error")

    assert matrix[i_missing][i_correct] == 1   # q1: missing_knowledge -> correct
    assert matrix[i_correct][i_correct] == 1   # q2: correct -> correct
    assert matrix[i_visual][i_visual] == 1     # q3: unchanged, still visual grounding error
    assert unlabeled == 0


def test_build_transition_matrix_reports_unlabeled_instances():
    labels_by_setting = {
        "no_retrieval": {"q1": "missing_factual_knowledge"},
        "hybrid": {},  # q1 has no label at hybrid (e.g. not in the LLM sample)
    }
    matrix, unlabeled = build_transition_matrix(labels_by_setting, "no_retrieval", "hybrid")
    assert unlabeled == 1
    assert sum(sum(row) for row in matrix) == 0


def test_build_full_chain_matrices_covers_all_consecutive_pairs_plus_endpoints():
    labels_by_setting = {
        "no_retrieval": {"q1": CORRECT},
        "sparse": {"q1": CORRECT},
        "dense": {"q1": CORRECT},
        "hybrid": {"q1": CORRECT},
    }
    chains = build_full_chain_matrices(labels_by_setting)
    assert "no_retrieval_to_sparse" in chains
    assert "sparse_to_dense" in chains
    assert "dense_to_hybrid" in chains
    assert "no_retrieval_to_hybrid" in chains  # headline endpoint matrix
