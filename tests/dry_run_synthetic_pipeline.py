"""
Not a pytest test (needs no assertions to be meaningful) -- a smoke test that
fabricates a tiny synthetic results/ tree matching Stage 1's exact output
format, then runs the Stage 2 pipeline's non-LLM parts (prefilter, sampling,
merge, transition matrix) over it end-to-end, to catch any wiring bugs
between run_experiment.py's output format and run_failure_annotation.py's
expected input format. The LLM judge call itself is stubbed out (no network).

Run with: python tests/dry_run_synthetic_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.rule_based_prefilter import prefilter_run
from evaluation.stratified_sampling import coverage_report, stratified_sample
from evaluation.transition_matrix import build_full_chain_matrices, matrix_to_readable_rows, merge_labels
from evaluation.llm_judge import JudgeLabel

SCENARIOS = ["angle", "occlusion", "deformation"]
SETTINGS = ["none", "sparse", "dense", "hybrid"]


def fake_predictions(setting: str, n_per_scenario: int = 6):
    preds, retrieval = [], []
    for scenario in SCENARIOS:
        for i in range(n_per_scenario):
            qid = f"{scenario}_{i}"
            # fabricate a plausible pattern: accuracy improves as retrieval improves
            correct_prob = {"none": 0.3, "sparse": 0.45, "dense": 0.55, "hybrid": 0.65}[setting]
            is_correct_instance = (hash((qid, setting)) % 100) / 100 < correct_prob
            gt = "A"
            pred_letter = "A" if is_correct_instance else "B"
            gt_image_ids = [f"{qid}_gt_img"]
            retrieved_ids = (
                [] if setting == "none"
                else [f"{qid}_gt_img"] if (hash((qid, setting, "recall")) % 2 == 0)
                else [f"{qid}_distractor_img"]
            )
            preds.append(
                {
                    "id": qid,
                    "question": f"Fake question for {qid}",
                    "choices": ["choice A", "choice B", "choice C"],
                    "prediction": f"Answer: {pred_letter}",
                    "ground_truth": gt,
                    "scenario": scenario,
                    "gt_image_ids": gt_image_ids,
                    "retrieval_type": setting,
                    "model_name": "fake-model",
                    "latency_seconds": 1.2,
                    "timestamp": 0,
                }
            )
            if setting != "none":
                retrieval.append({"id": qid, "retrieved_ids": retrieved_ids, "scores": [0.9]})
    return preds, retrieval


def stub_label_batch(instances, backend="anthropic"):
    """Stand-in for evaluation.llm_judge.label_batch that needs no network:
    deterministically assigns a plausible failure label based on qid hash."""
    from evaluation.failure_taxonomy import FAILURE_MODES

    labels = []
    for inst in instances:
        label = FAILURE_MODES[hash(inst["id"]) % len(FAILURE_MODES)]
        labels.append(JudgeLabel(qid=inst["id"], label=label, justification="stubbed", raw_response="{}"))
    return labels


def main():
    setting_name_map = {"none": "no_retrieval", "sparse": "sparse", "dense": "dense", "hybrid": "hybrid"}
    labels_by_setting = {}
    all_stats = {}

    for setting in SETTINGS:
        preds, retrieval = fake_predictions(setting)
        retrieval_by_qid = {r["id"]: r for r in retrieval}

        prefilter_results = prefilter_run(preds, retrieval_by_qid)
        preds_by_qid = {p["id"]: p for p in preds}
        unresolved = [
            (r.qid, preds_by_qid[r.qid]["scenario"], preds_by_qid[r.qid]["retrieval_type"])
            for r in prefilter_results
            if r.label is None
        ]
        sampled = stratified_sample(unresolved, per_stratum_budget=2, seed=42)
        cov = coverage_report(unresolved, sampled)

        judge_instances = [preds_by_qid[qid] | {"recall": None} for qid in sampled]
        judge_labels = stub_label_batch(judge_instances)

        merged = merge_labels(prefilter_results, judge_labels)
        labels_by_setting[setting_name_map[setting]] = merged

        all_stats[setting] = {
            "n_total": len(preds),
            "n_resolved_by_rule": sum(1 for r in prefilter_results if r.resolved_by_rule),
            "n_unresolved": len(unresolved),
            "n_sampled": len(sampled),
            "n_final_labeled": len(merged),
            "coverage": {f"{k[0]}|{k[1]}": v for k, v in cov.items()},
        }

    print("=== Per-setting labeling stats ===")
    print(json.dumps(all_stats, indent=2))

    chains = build_full_chain_matrices(
        labels_by_setting, setting_order=["no_retrieval", "sparse", "dense", "hybrid"]
    )
    print("\n=== Transition matrices (non-zero cells only) ===")
    for chain_name, (matrix, unlabeled) in chains.items():
        print(f"\n-- {chain_name} (unlabeled/excluded: {unlabeled}) --")
        for row in matrix_to_readable_rows(matrix):
            print(f"  {row['from']:45s} -> {row['to']:45s} : {row['count']}")

    print("\nDRY RUN COMPLETE -- no exceptions, pipeline wiring is consistent end to end.")


if __name__ == "__main__":
    main()
