import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import check_contamination, extract_choice_letter, is_correct, retrieval_recall_at_k


def test_extract_choice_letter_handles_common_formats():
    assert extract_choice_letter("A") == "A"
    assert extract_choice_letter("A.") == "A"
    assert extract_choice_letter("(B)") == "B"
    assert extract_choice_letter("The answer is C.") == "C"
    assert extract_choice_letter("I'm not sure") is None


def test_is_correct_case_insensitive():
    assert is_correct("Answer: b", "B") is True
    assert is_correct("Answer: c", "B") is False


def test_retrieval_recall_partial_and_full():
    assert retrieval_recall_at_k(["img1", "img2", "img3"], ["img2"]) == 1.0
    assert retrieval_recall_at_k(["img1"], ["img1", "img2"]) == 0.5
    assert retrieval_recall_at_k([], []) == 1.0  # vacuous case


def test_contamination_flag_triggers_on_small_gap():
    no_ret = {"scenario_a": 0.80, "scenario_b": 0.20}
    hybrid = {"scenario_a": 0.83, "scenario_b": 0.75}
    flags = check_contamination(no_ret, hybrid, threshold=0.05)
    flag_map = {f.scenario: f.flagged for f in flags}
    assert flag_map["scenario_a"] is True   # 0.03 gap < 0.05 threshold -> flagged
    assert flag_map["scenario_b"] is False  # 0.55 gap -> not flagged
