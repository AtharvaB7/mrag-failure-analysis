#!/usr/bin/env bash
# Run every (model x retrieval) combination for the full experiment grid.
# Run from the project root: bash scripts/run_all.sh
set -euo pipefail

MODELS=(qwen2vl internvl)          # add llava_next here if InternVL doesn't reproduce cleanly
RETRIEVALS=(none sparse dense hybrid)

for model in "${MODELS[@]}"; do
  for retrieval in "${RETRIEVALS[@]}"; do
    echo "=== Running model=${model} retrieval=${retrieval} ==="
    python scripts/run_experiment.py model="${model}" retrieval="${retrieval}"
  done
done

echo "=== Generating comparison plots + summary CSV ==="
python evaluation/plots.py --results_root results --out_dir results/_summary

echo "=== Running contamination check (no_retrieval vs hybrid, per model) ==="
python - <<'PYEOF'
from evaluation.evaluate import run_contamination_check

for model in ["qwen2vl-7b", "internvl2_5-8b"]:
    print(f"\n--- {model} ---")
    run_contamination_check(
        no_retrieval_results_dir=f"results/{model}_no_retrieval",
        hybrid_results_dir=f"results/{model}_hybrid_rrf",
    )
PYEOF
