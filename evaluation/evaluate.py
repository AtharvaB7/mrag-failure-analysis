"""
Aggregate a completed run's predictions.json + retrieval.json into metrics.json:
overall accuracy, per-scenario accuracy, retrieval recall, latency stats, and
the contamination flags.

NOTE: This module handles the *baseline* metrics only (this stage's scope,
per the request). Failure-mode labeling and the failure-mode transition
matrix are a separate downstream stage that consumes these same
predictions.json/retrieval.json files -- kept out of this stage so the
retrieval-evaluation pipeline can be fully validated (accuracy, recall,
latency all sane) before spending time on the more labor-intensive
annotation pass.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from evaluation.metrics import check_contamination, is_correct, retrieval_recall_at_k


def evaluate_run(results_dir: str) -> dict:
    results_dir = Path(results_dir)
    with open(results_dir / "predictions.json") as f:
        predictions = json.load(f)
    with open(results_dir / "retrieval.json") as f:
        retrieval = json.load(f)  # keyed by question id -> list of retrieved doc ids
    retrieval_by_qid = {r["id"]: r for r in retrieval}

    correct_flags, scenario_correct, scenario_total = [], {}, {}
    recalls = []
    latencies = []

    for pred in predictions:
        qid = pred["id"]
        correct = is_correct(pred["prediction"], pred["ground_truth"])
        correct_flags.append(correct)

        scenario = pred["scenario"]
        scenario_total[scenario] = scenario_total.get(scenario, 0) + 1
        scenario_correct[scenario] = scenario_correct.get(scenario, 0) + int(correct)

        latencies.append(pred["latency_seconds"])

        r = retrieval_by_qid.get(qid)
        if r is not None:
            recalls.append(retrieval_recall_at_k(r["retrieved_ids"], pred.get("gt_image_ids", [])))

    metrics = {
        "overall_accuracy": sum(correct_flags) / len(correct_flags) if correct_flags else 0.0,
        "accuracy_by_scenario": {
            s: scenario_correct[s] / scenario_total[s] for s in scenario_total
        },
        "retrieval_recall_mean": statistics.mean(recalls) if recalls else None,
        "latency_seconds_mean": statistics.mean(latencies) if latencies else None,
        "latency_seconds_p95": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=None)
        ),
        "n_examples": len(predictions),
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_contamination_check(no_retrieval_results_dir: str, hybrid_results_dir: str, threshold: float = 0.05) -> list:
    """Compare a model's no-retrieval run against its hybrid-retrieval run,
    per scenario, and flag suspiciously-small gaps (see evaluation/metrics.py
    docstring for why this matters)."""
    no_ret_metrics = evaluate_run(no_retrieval_results_dir)
    hybrid_metrics = evaluate_run(hybrid_results_dir)
    flags = check_contamination(
        no_ret_metrics["accuracy_by_scenario"], hybrid_metrics["accuracy_by_scenario"], threshold=threshold
    )
    flagged = [f for f in flags if f.flagged]
    if flagged:
        print(f"WARNING: {len(flagged)} scenario(s) flagged as possibly contaminated:")
        for f in flagged:
            print(f"  - {f.scenario}: no_retrieval_acc={f.no_retrieval_acc:.3f}, "
                  f"hybrid_acc={f.hybrid_acc:.3f}, gap={f.gap:.3f}")
    return flags


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.results_dir), indent=2))
