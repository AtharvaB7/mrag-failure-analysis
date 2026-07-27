"""
Consolidates everything into one markdown report, ready to pull findings
from for the actual paper: accuracy/recall/latency across all (model x
retrieval setting) runs, the contamination check per model, and the
failure-mode transition matrix highlights per model.

Run this LAST, after:
  1. All 8 (2 model x 4 retrieval) runs from scripts/run_experiment.py
  2. scripts/run_failure_annotation.py for each model

Usage:
    python scripts/generate_report.py --results_root results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SETTINGS = ["no_retrieval", "sparse_bm25", "dense_siglip", "hybrid_rrf"]
SETTING_DISPLAY = {
    "no_retrieval": "No Retrieval",
    "sparse_bm25": "Sparse (BM25)",
    "dense_siglip": "Dense (SigLIP)",
    "hybrid_rrf": "Hybrid (RRF)",
}


def discover_model_names(results_root: Path) -> list[str]:
    names = []
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and d.name.endswith("_no_retrieval"):
            names.append(d.name[: -len("_no_retrieval")])
    return names


def load_metrics(results_root: Path, model_name: str, setting: str) -> dict | None:
    path = results_root / f"{model_name}_{setting}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_accuracy_table(results_root: Path, model_names: list[str]) -> str:
    lines = ["| Retrieval Setting | " + " | ".join(model_names) + " |",
             "|---" * (len(model_names) + 1) + "|"]
    for setting in SETTINGS:
        row = [SETTING_DISPLAY[setting]]
        for model in model_names:
            m = load_metrics(results_root, model, setting)
            row.append(f"{m['overall_accuracy']:.3f}" if m else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_recall_latency_table(results_root: Path, model_names: list[str]) -> str:
    lines = [
        "| Model | Retrieval Setting | Retrieval Recall | Mean Latency (s) | P95 Latency (s) |",
        "|---|---|---|---|---|",
    ]
    for model in model_names:
        for setting in SETTINGS:
            m = load_metrics(results_root, model, setting)
            if m is None:
                continue
            recall = f"{m['retrieval_recall_mean']:.3f}" if m["retrieval_recall_mean"] is not None else "n/a"
            lines.append(
                f"| {model} | {SETTING_DISPLAY[setting]} | {recall} "
                f"| {m['latency_seconds_mean']:.3f} | {m['latency_seconds_p95']:.3f} |"
            )
    return "\n".join(lines)


def build_scenario_breakdown(results_root: Path, model_names: list[str], setting: str = "no_retrieval") -> str:
    """Per-scenario accuracy for a given setting, one table per model --
    useful for spotting which scenarios are hardest/easiest and cross-
    referencing against contamination flags."""
    sections = []
    for model in model_names:
        m = load_metrics(results_root, model, setting)
        if m is None:
            continue
        lines = [f"**{model} ({SETTING_DISPLAY[setting]})**", "", "| Scenario | Accuracy |", "|---|---|"]
        for scenario, acc in sorted(m["accuracy_by_scenario"].items(), key=lambda kv: kv[1]):
            lines.append(f"| {scenario} | {acc:.3f} |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_contamination_section(results_root: Path, model_names: list[str]) -> str:
    from evaluation.evaluate import run_contamination_check

    sections = []
    for model in model_names:
        no_ret_dir = results_root / f"{model}_no_retrieval"
        hybrid_dir = results_root / f"{model}_hybrid_rrf"
        if not (no_ret_dir.exists() and hybrid_dir.exists()):
            continue
        flags = run_contamination_check(str(no_ret_dir), str(hybrid_dir))
        flagged = [f for f in flags if f.flagged]
        lines = [f"**{model}**", ""]
        if flagged:
            lines.append("| Scenario | No-Retrieval Acc | Hybrid Acc | Gap | Flagged |")
            lines.append("|---|---|---|---|---|")
            for f in flags:
                marker = "⚠️ YES" if f.flagged else "no"
                lines.append(f"| {f.scenario} | {f.no_retrieval_acc:.3f} | {f.hybrid_acc:.3f} | {f.gap:.3f} | {marker} |")
            lines.append("")
            lines.append(
                f"**{len(flagged)} scenario(s) flagged** -- no-retrieval accuracy is suspiciously close to "
                f"hybrid-retrieval accuracy here, which could indicate the model already has this "
                f"scenario's answers memorized from pretraining rather than retrieval being genuinely unhelpful. "
                f"Treat conclusions about these scenarios with caution in the write-up."
            )
        else:
            lines.append("No scenarios flagged -- no-retrieval and hybrid-retrieval accuracy differ enough "
                          "per scenario that memorization/contamination looks unlikely to be driving the results.")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_transition_matrix_section(results_root: Path, model_names: list[str], top_n: int = 15) -> str:
    sections = []
    for model in model_names:
        matrix_path = results_root / "_failure_analysis" / model / "transition_matrices.json"
        if not matrix_path.exists():
            sections.append(f"**{model}**: no transition_matrices.json found -- "
                             f"run `scripts/run_failure_annotation.py --model_name {model}` first.")
            continue
        with open(matrix_path) as f:
            chains = json.load(f)

        lines = [f"**{model}**", ""]
        headline_key = "no_retrieval_to_hybrid"
        if headline_key in chains:
            headline = chains[headline_key]
            lines.append(f"*Headline: no_retrieval → hybrid (unlabeled/excluded: {headline['unlabeled_count']})*")
            lines.append("")
            lines.append("| From | To | Count |")
            lines.append("|---|---|---|")
            # Sort by count desc, show top N, skip trivial correct->correct rows
            # from cluttering the interesting part of the table (still counted,
            # just deprioritized in display -- shown last if room permits).
            interesting = [r for r in headline["rows"] if not (r["from"] == "correct" and r["to"] == "correct")]
            trivial = [r for r in headline["rows"] if r["from"] == "correct" and r["to"] == "correct"]
            for row in sorted(interesting, key=lambda r: -r["count"])[:top_n]:
                lines.append(f"| {row['from']} | {row['to']} | {row['count']} |")
            if trivial:
                lines.append(f"| correct | correct | {trivial[0]['count']} (unchanged, shown separately) |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def generate_report(results_root: Path, out_path: Path) -> None:
    model_names = discover_model_names(results_root)
    if not model_names:
        raise SystemExit(f"No '<model>_no_retrieval' folders found under {results_root}.")

    report = f"""# Multimodal RAG Failure-Mode Analysis -- Consolidated Report

Models found: {', '.join(model_names)}

## 1. Overall Accuracy

{build_accuracy_table(results_root, model_names)}

## 2. Retrieval Recall & Latency

{build_recall_latency_table(results_root, model_names)}

## 3. Per-Scenario Accuracy (No-Retrieval Baseline)

{build_scenario_breakdown(results_root, model_names, setting="no_retrieval")}

## 4. Contamination Check (No-Retrieval vs. Hybrid, Per Scenario)

{build_contamination_section(results_root, model_names)}

## 5. Failure-Mode Transition Matrix (No-Retrieval → Hybrid)

Per-model, not pooled -- see `evaluation/transition_matrix.py` for why pooling
across models would obscure model-dependent differences. `unlabeled_count`
indicates how many instances were excluded from the matrix due to incomplete
coverage (outside the LLM-annotation sample and not rule-resolved) -- treat
matrices with a high unlabeled_count relative to n_examples cautiously.

{build_transition_matrix_section(results_root, model_names)}

---
*Generated by scripts/generate_report.py*
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_path", default="results/_summary/report.md")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    generate_report(Path(args.results_root), Path(args.out_path))
