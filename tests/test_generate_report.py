import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_report import discover_model_names, generate_report


def _write_run(results_root: Path, model: str, setting: str, accuracy: float, recall: float | None):
    run_dir = results_root / f"{model}_{setting}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = [
        {
            "id": f"q{i}",
            "prediction": f"Answer: {'A' if i < int(accuracy * 10) else 'B'}",
            "ground_truth": "A",
            "scenario": "scenario_a" if i % 2 == 0 else "scenario_b",
            "gt_image_ids": ["img1"],
            "retrieval_type": "none" if setting == "no_retrieval" else "hybrid",
            "latency_seconds": 0.5,
        }
        for i in range(10)
    ]
    retrieval = (
        []
        if setting == "no_retrieval"
        else [{"id": f"q{i}", "retrieved_ids": ["img1"] if recall and i < int(recall * 10) else ["distractor"]}
              for i in range(10)]
    )
    with open(run_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)
    with open(run_dir / "retrieval.json", "w") as f:
        json.dump(retrieval, f)

    metrics = {
        "overall_accuracy": accuracy,
        "accuracy_by_scenario": {"scenario_a": accuracy, "scenario_b": accuracy},
        "retrieval_recall_mean": recall,
        "latency_seconds_mean": 0.5,
        "latency_seconds_p95": 0.6,
        "n_examples": 10,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


def test_discover_model_names_finds_both_models():
    with tempfile.TemporaryDirectory() as tmp:
        results_root = Path(tmp)
        _write_run(results_root, "modelA", "no_retrieval", 0.5, None)
        _write_run(results_root, "modelB", "no_retrieval", 0.6, None)
        names = discover_model_names(results_root)
        assert set(names) == {"modelA", "modelB"}


def test_generate_report_end_to_end_with_synthetic_data():
    with tempfile.TemporaryDirectory() as tmp:
        results_root = Path(tmp)

        # modelA: accuracy improves with retrieval (normal pattern)
        _write_run(results_root, "modelA", "no_retrieval", 0.3, None)
        _write_run(results_root, "modelA", "sparse_bm25", 0.4, 0.1)
        _write_run(results_root, "modelA", "dense_siglip", 0.5, 0.4)
        _write_run(results_root, "modelA", "hybrid_rrf", 0.6, 0.5)

        # modelB: no-retrieval suspiciously close to hybrid (contamination-flag case)
        _write_run(results_root, "modelB", "no_retrieval", 0.55, None)
        _write_run(results_root, "modelB", "sparse_bm25", 0.4, 0.1)
        _write_run(results_root, "modelB", "dense_siglip", 0.5, 0.4)
        _write_run(results_root, "modelB", "hybrid_rrf", 0.57, 0.5)  # gap to no_retrieval = 0.02, should flag

        out_path = results_root / "_summary" / "report.md"
        generate_report(results_root, out_path)

        assert out_path.exists()
        content = out_path.read_text()

        # Sanity: both models show up, key sections present
        assert "modelA" in content and "modelB" in content
        assert "## 1. Overall Accuracy" in content
        assert "## 4. Contamination Check" in content
        assert "## 5. Failure-Mode Transition Matrix" in content
        # modelB's tiny no_retrieval/hybrid gap should get flagged
        assert "flagged" in content.lower()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
