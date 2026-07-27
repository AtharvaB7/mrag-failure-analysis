"""
Generate comparison plots + CSV summaries across all runs in results/.

Usage:
    python evaluation/plots.py --results_root results --out_dir results/_summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_all_metrics(results_root: str) -> pd.DataFrame:
    rows = []
    for run_dir in Path(results_root).iterdir():
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        # run_name convention: "{model}_{retrieval}"
        model, retrieval = run_dir.name.rsplit("_", 1) if "_" in run_dir.name else (run_dir.name, "unknown")
        rows.append(
            {
                "run_name": run_dir.name,
                "model": model,
                "retrieval": retrieval,
                "overall_accuracy": m["overall_accuracy"],
                "retrieval_recall_mean": m["retrieval_recall_mean"],
                "latency_seconds_mean": m["latency_seconds_mean"],
                "n_examples": m["n_examples"],
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("retrieval")
        ax.plot(sub["retrieval"], sub["overall_accuracy"], marker="o", label=model)
    ax.set_xlabel("Retrieval setting")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Accuracy across retrieval settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=200)
    plt.close(fig)


def plot_recall_comparison(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[df["retrieval"] != "no_retrieval"]
    for model in sub["model"].unique():
        s = sub[sub["model"] == model].sort_values("retrieval")
        ax.bar(
            [f"{r}\n({model})" for r in s["retrieval"]], s["retrieval_recall_mean"], label=model
        )
    ax.set_ylabel("Mean retrieval recall")
    ax.set_title("Retrieval recall by setting/model")
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_recall_comparison.png", dpi=200)
    plt.close(fig)


def plot_latency_comparison(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("retrieval")
        ax.plot(sub["retrieval"], sub["latency_seconds_mean"], marker="o", label=model)
    ax.set_xlabel("Retrieval setting")
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Latency across retrieval settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_dir", default="results/_summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_metrics(args.results_root)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Loaded {len(df)} runs -> {out_dir / 'summary.csv'}")

    if not df.empty:
        plot_accuracy_comparison(df, out_dir)
        plot_recall_comparison(df, out_dir)
        plot_latency_comparison(df, out_dir)
        print(f"Figures saved to {out_dir}")
