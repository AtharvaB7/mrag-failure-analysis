# Failure-Mode Transitions in Multimodal RAG — Baseline Pipeline

Baseline evaluation pipeline for the project: compare four retrieval settings
(no retrieval, sparse/BM25, dense/SigLIP, hybrid/RRF) on MRAG-Bench using
Qwen2-VL-7B and InternVL2.5-8B, and save everything needed for the later
failure-mode annotation stage (out of scope for this stage — see
`evaluation/evaluate.py` docstring).

**Important environment note:** this repo was built in a sandbox with no
network access to `huggingface.co` and no GPU, so the code below is written
and unit-tested for correctness (see `tests/`) but the end-to-end
model/dataset pipeline has **not** been executed here. Run it on a machine
with GPU + HF Hub access (your own box, a university cluster, Colab Pro,
Lambda/RunPod, etc.) — see "GPU requirements" below for what tier you need.

## Repo layout

```
project/
├── data/            # dataset download + loading
├── models/          # VLM wrappers (Qwen2-VL, InternVL2.5, LLaVA-NeXT fallback)
├── retrieval/        # BM25 / SigLIP-dense / hybrid-RRF retrievers
├── evaluation/       # metrics, contamination check, plotting
├── scripts/          # entry points (setup, download, run experiments)
├── configs/           # Hydra configs (model/, retrieval/, top-level config.yaml)
├── results/          # per-run outputs (gitignored except .gitkeep)
├── logs/              # per-run logs (gitignored except .gitkeep)
└── tests/            # unit tests for logic that needs no GPU/network
```

## 1. Install

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This installs `requirements.txt` plus `qwen-vl-utils` (needed for Qwen2-VL's
image preprocessing). If you'd rather use conda: `conda env create -f environment.yml`.

## 2. Download data

```bash
bash scripts/download_data.sh
```

This pulls the MRAG-Bench QA split (`uclanlp/MRAG-Bench` on the HF Hub,
1,353 MCQs across 9 scenarios) into `data/mrag_bench/qa/`. **The 16,130-image
retrieval corpus is a separate archive released by the MRAG-Bench authors**
(not bundled in the HF dataset object) — check
[github.com/mragbench/MRAG-Bench](https://github.com/mragbench/MRAG-Bench)
for the current corpus download link (their README's "image corpus release"
section), then run:

```bash
python data/download_mrag_bench.py --out_dir data/mrag_bench --image_archive path/to/corpus.zip
```

Expected layout after this step:
```
data/mrag_bench/
├── qa/test.json          # flat JSON: question, choices, answer, scenario, gt_image_ids
├── qa/test/               # HF datasets on-disk cache (same data, HF format)
└── images/*.jpg           # the 16,130-image retrieval corpus
```

## 3. Run an experiment

Each run is one (model × retrieval setting) combination:

```bash
python scripts/run_experiment.py model=qwen2vl retrieval=none
python scripts/run_experiment.py model=qwen2vl retrieval=sparse
python scripts/run_experiment.py model=qwen2vl retrieval=dense
python scripts/run_experiment.py model=qwen2vl retrieval=hybrid
python scripts/run_experiment.py model=internvl retrieval=hybrid
```

Or run the full 2-model × 4-retrieval grid (8 runs) plus generate all
comparison plots and the contamination check in one go:

```bash
bash scripts/run_all.sh
```

Each run writes to `results/<model>_<retrieval>/`:
```
predictions.json   # per-question: prediction, ground truth, scenario, latency, timestamp, ...
retrieval.json     # per-question: retrieved doc ids + scores (empty for no_retrieval)
metrics.json       # overall accuracy, per-scenario accuracy, retrieval recall, latency stats
config.yaml        # exact Hydra config used for this run (full reproducibility)
log.txt            # run log
```

## Changing models / retrievers

Everything is Hydra config groups — swap via CLI overrides, no code changes needed:

```bash
# Use LLaVA-NeXT instead of InternVL if InternVL's trust_remote_code causes issues
python scripts/run_experiment.py model=llava_next retrieval=hybrid

# Override a specific config value
python scripts/run_experiment.py model=qwen2vl retrieval=sparse retrieval.top_k=10
```

To add a new retriever entirely: drop a new YAML in `configs/retrieval/`,
implement it under `retrieval/`, and add a branch in `retrieval/factory.py`.
Same pattern for a new model under `models/` + `models/factory.py`.

## 4. Failure-mode annotation (Stage 2)

Once you have all four runs for a model (`results/<model>_no_retrieval/`,
`..._sparse_bm25/`, `..._dense_siglip/`, `..._hybrid_rrf/`), label failure
modes and build the transition matrices:

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY if using --judge_backend openai
python scripts/run_failure_annotation.py \
    --model_name qwen2vl-7b \
    --per_stratum_budget 5 \
    --judge_backend anthropic
```

This runs in three stages, none of which relabel an instance twice:
1. **Rule-based prefilter** (free): correct predictions get labeled `correct`;
   incorrect predictions with zero retrieval recall on the ground-truth
   evidence get labeled `retrieval_missed_evidence`. No LLM call needed for
   either case (see `evaluation/rule_based_prefilter.py`).
2. **Stratified sampling**: everything the rule couldn't resolve is sampled
   per (scenario, retrieval setting) stratum, up to `--per_stratum_budget`
   per stratum, so small scenarios aren't starved and large ones don't
   dominate the LLM budget.
3. **LLM-assisted judging**: the sampled instances get a single fixed-taxonomy
   label from an LLM judge (`evaluation/llm_judge.py`). **Before trusting
   this at scale, validate the judge against ~50-100 hand-labeled instances**
   (e.g. Cohen's kappa) — this script produces labels, it doesn't validate them.

Output, in `results/_failure_analysis/<model_name>/`:
- `labels_by_setting.json` — final merged label per question, per retrieval setting
- `labeling_stats.json` — how many instances were rule-resolved vs. sampled vs. still unlabeled, per setting
- `transition_matrices.json` — the failure-mode transition matrices for every
  consecutive setting pair (no_retrieval→sparse→dense→hybrid) plus the
  headline no_retrieval→hybrid endpoint matrix, each with an explicit
  `unlabeled_count` so you can see how much of the matrix rests on actual
  labels vs. coverage gaps

**Per Yifei's feedback, this is run once per model and the two matrices are
compared side by side** (`--model_name qwen2vl-7b` and then `--model_name
internvl2_5-8b`) rather than pooled into one matrix — see
`evaluation/transition_matrix.py`'s docstring for why pooling would obscure
model-dependent differences.

A synthetic end-to-end smoke test for this whole stage (no GPU/API calls, just
verifies the wiring) lives at `tests/dry_run_synthetic_pipeline.py` — run it
directly to see the full label→matrix flow on fabricated data.

## 5. Reproducing figures

```bash
python evaluation/plots.py --results_root results --out_dir results/_summary
```

Produces `results/_summary/summary.csv` plus three figures: accuracy
comparison, retrieval recall comparison, and latency comparison, all faceted
by model and retrieval setting.

## Contamination check (per Yifei's feedback)

Before drawing any conclusion from a low "no retrieval" accuracy, we check
whether that low accuracy might instead reflect that the model has *not*
memorized that scenario's answers (i.e., retrieval is doing genuine work),
versus flagging scenarios where no-retrieval accuracy is suspiciously close
to hybrid-retrieval accuracy (possible leakage/memorization):

```bash
python -c "
from evaluation.evaluate import run_contamination_check
run_contamination_check('results/qwen2vl-7b_no_retrieval', 'results/qwen2vl-7b_hybrid_rrf')
"
```

Flagged scenarios (gap < `dataset.contamination_flag_threshold` in
`configs/config.yaml`, default 0.05) should be called out explicitly in any
write-up rather than folded silently into headline numbers — see the
Methodology section of the project proposal for why this matters.

## GPU requirements

| Model | Min VRAM (bf16 inference) | Recommended GPU |
|---|---|---|
| Qwen2-VL-7B-Instruct | ~18 GB | single RTX 4090 (24GB), L40S, or A100 |
| InternVL2.5-8B | ~20 GB | L40S or A100 (24GB 4090 is tight with tiling at `num_image_tiles=12`; lower it if you hit OOM) |
| LLaVA-NeXT-7B (fallback) | ~16 GB | RTX 4090 or better |
| SigLIP-SO400M (dense retriever) | ~4 GB | any of the above; can run on CPU but much slower for index-building |

Colab Pro's A100 (40GB) tier can run either VLM comfortably one at a time.
Running the full 8-run grid sequentially on a single 4090-class GPU: budget
roughly 1,353 questions × ~2-4 sec/question × 8 runs ≈ 6-12 hours end to end
(varies a lot with `max_new_tokens` and whether you batch — this pipeline
runs one example at a time for simplicity; batching is a reasonable
follow-up optimization if runtime becomes a bottleneck).

## Reproducibility

- Seeds are set for `random`, `numpy`, and `torch` (see `scripts/run_experiment.py:set_seed`).
- `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled by default (`configs/config.yaml: determinism.torch_deterministic`).
- Every run's exact resolved config is saved to `results/<run_name>/config.yaml`.
- Package versions: run `pip freeze > results/<run_name>/pip_freeze.txt` after `setup_env.sh` if you want a pinned snapshot per machine (not automated here since it's environment-specific, not experiment-specific).

## Common errors / troubleshooting

- **`trust_remote_code` errors on InternVL2.5-8B**: version mismatch between the InternVL repo's custom code and your installed `transformers`. Try pinning `transformers==4.46.0` first; if it persists, switch to `model=llava_next` (fully native `transformers` support, no custom code).
- **CUDA OOM on InternVL**: lower `model.num_image_tiles` in `configs/model/internvl.yaml` (e.g. 12 → 6).
- **`FileNotFoundError` on `qa/test.json`**: you haven't run `scripts/download_data.sh` yet, or ran it without HF Hub network access.
- **Dense/hybrid retrieval "requires images" error**: same root cause — the image corpus archive step (see step 2) hasn't been completed; the QA-split download alone isn't enough to build the retrieval index.
- **Everything's on CPU / very slow**: check `torch.cuda.is_available()` — the model wrappers use `device_map="auto"`, which silently falls back to CPU if no CUDA device is visible; that's a correctness-preserving fallback but will be extremely slow for 7-8B models.

## Tests

The retrieval-fusion logic and answer-extraction/metrics logic (i.e., the
parts with no GPU/network dependency) have real unit tests:

```bash
pip install pytest
python -m pytest tests/ -v
```

All 9 currently pass (verified in this repo's build environment).

## Suggested git commit order

1. `configs/`, `.gitignore`, `requirements.txt`, `environment.yml`, `README.md`
2. `retrieval/` + `tests/test_retrieval.py`
3. `evaluation/metrics.py` + `tests/test_metrics.py`
4. `data/`
5. `models/`
6. `evaluation/evaluate.py`, `evaluation/plots.py`, `evaluation/gpt_fallback.py`
7. `scripts/`
