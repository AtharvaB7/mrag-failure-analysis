"""
Failure-mode annotation pipeline (Stage 2) -- consumes the predictions.json/
retrieval.json produced by scripts/run_experiment.py (Stage 1) for one
model's four runs (no_retrieval, sparse, dense, hybrid), and produces:

  1. Rule-based prefilter labels (free, no LLM calls)
  2. A stratified sample of the rule-unresolved instances
  3. LLM-judge labels for that sample
  4. Merged final labels per setting
  5. The failure-mode transition matrices (consecutive pairs + headline endpoint)

Usage:
    python scripts/run_failure_annotation.py \\
        --model_name qwen2vl-7b \\
        --results_root results \\
        --per_stratum_budget 5 \\
        --judge_backend anthropic

Requires an ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment
depending on --judge_backend. Rule-based prefiltering and stratified
sampling run with no external calls at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.llm_judge import label_batch
from evaluation.metrics import retrieval_recall_at_k
from evaluation.rule_based_prefilter import prefilter_run
from evaluation.stratified_sampling import coverage_report, stratified_sample
from evaluation.transition_matrix import build_full_chain_matrices, matrix_to_readable_rows, merge_labels

SETTINGS = ["no_retrieval", "sparse", "dense", "hybrid"]
RETRIEVAL_TYPE_TO_RUN_SUFFIX = {
    "no_retrieval": "no_retrieval",
    "sparse": "sparse_bm25",
    "dense": "dense_siglip",
    "hybrid": "hybrid_rrf",
}


def load_run(results_root: Path, model_name: str, setting: str) -> tuple[list[dict], dict[str, dict]]:
    run_dir = results_root / f"{model_name}_{RETRIEVAL_TYPE_TO_RUN_SUFFIX[setting]}"
    with open(run_dir / "predictions.json") as f:
        predictions = json.load(f)
    with open(run_dir / "retrieval.json") as f:
        retrieval = json.load(f)
    retrieval_by_qid = {r["id"]: r for r in retrieval}
    return predictions, retrieval_by_qid


def process_one_setting(
    predictions: list[dict],
    retrieval_by_qid: dict[str, dict],
    per_stratum_budget: int,
    judge_backend: str,
    seed: int,
) -> tuple[dict[str, str], dict]:
    prefilter_results = prefilter_run(predictions, retrieval_by_qid)
    predictions_by_qid = {p["id"]: p for p in predictions}

    unresolved = [
        (r.qid, predictions_by_qid[r.qid]["scenario"], predictions_by_qid[r.qid]["retrieval_type"])
        for r in prefilter_results
        if r.label is None
    ]
    sampled_qids = stratified_sample(unresolved, per_stratum_budget=per_stratum_budget, seed=seed)
    coverage = coverage_report(unresolved, sampled_qids)

    judge_instances = []
    for qid in sampled_qids:
        pred = predictions_by_qid[qid]
        recall_info = retrieval_by_qid.get(qid, {})
        recall = retrieval_recall_at_k(recall_info.get("retrieved_ids", []), pred.get("gt_image_ids", []))
        judge_instances.append(
            {
                "id": qid,
                "question": pred["question"],
                "choices": pred.get("choices", []),
                "ground_truth": pred["ground_truth"],
                "prediction": pred["prediction"],
                "retrieval_type": pred["retrieval_type"],
                "recall": recall,
            }
        )
    judge_labels = label_batch(judge_instances, backend=judge_backend) if judge_instances else []

    merged = merge_labels(prefilter_results, judge_labels)

    stats = {
        "n_total": len(predictions),
        "n_resolved_by_rule": sum(1 for r in prefilter_results if r.resolved_by_rule),
        "n_unresolved": len(unresolved),
        "n_sampled_for_llm": len(sampled_qids),
        "n_final_labeled": len(merged),
        "coverage_by_stratum": {f"{k[0]}|{k[1]}": v for k, v in coverage.items()},
    }
    return merged, stats


def discover_model_names(results_root: Path) -> list[str]:
    """Scan results/ for '<model_name>_no_retrieval' folders and return the
    model_name prefixes found -- lets you run this script without having to
    remember/retype the exact model.name string from your config."""
    names = []
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and d.name.endswith("_no_retrieval"):
            names.append(d.name[: -len("_no_retrieval")])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, help="e.g. qwen2vl-7b, llava-next-7b. If omitted, auto-discovers all models found in --results_root.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--per_stratum_budget", type=int, default=5)
    parser.add_argument("--judge_backend", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None, help="defaults to results/_failure_analysis/<model_name>/")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    model_names = [args.model_name] if args.model_name else discover_model_names(results_root)
    if not model_names:
        raise SystemExit(f"No '<model>_no_retrieval' folders found under {results_root}. Check --results_root.")
    print(f"Processing model(s): {model_names}")

    for model_name in model_names:
        run_for_model(model_name, results_root, args)


def run_for_model(model_name: str, results_root: Path, args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "_failure_analysis" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_setting: dict[str, dict[str, str]] = {}
    all_stats = {}

    for setting in SETTINGS:
        print(f"=== Processing {model_name} / {setting} ===")
        predictions, retrieval_by_qid = load_run(results_root, model_name, setting)
        merged, stats = process_one_setting(
            predictions, retrieval_by_qid, args.per_stratum_budget, args.judge_backend, args.seed
        )
        labels_by_setting[setting] = merged
        all_stats[setting] = stats
        print(json.dumps(stats, indent=2))

    with open(out_dir / "labels_by_setting.json", "w") as f:
        json.dump(labels_by_setting, f, indent=2)
    with open(out_dir / "labeling_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    chains = build_full_chain_matrices(labels_by_setting, setting_order=SETTINGS)
    matrix_output = {}
    for chain_name, (matrix, unlabeled) in chains.items():
        matrix_output[chain_name] = {
            "rows": matrix_to_readable_rows(matrix),
            "unlabeled_count": unlabeled,
        }
        print(f"\n--- {chain_name} (unlabeled/excluded: {unlabeled}) ---")
        for row in matrix_output[chain_name]["rows"]:
            print(f"  {row['from']:45s} -> {row['to']:45s} : {row['count']}")

    with open(out_dir / "transition_matrices.json", "w") as f:
        json.dump(matrix_output, f, indent=2)

    print(f"\nSaved labels_by_setting.json, labeling_stats.json, transition_matrices.json -> {out_dir}")


if __name__ == "__main__":
    main()
