"""
Run one full (model x retrieval-setting) experiment over MRAG-Bench and save
predictions/retrieval/metrics/config/log to results/<run_name>/.

Usage (from project root):
    python scripts/run_experiment.py model=qwen2vl retrieval=hybrid
    python scripts/run_experiment.py model=internvl retrieval=none

Requires GPU + huggingface.co access for model/dataset downloads -- this
sandbox's network is restricted, so this script is meant to be run on your
own machine/cluster/Colab, not executed here.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import MRAGBenchDataset
from models.factory import build_model
from retrieval.factory import build_retriever


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_corpus(data_dir: Path) -> tuple[list[str], list[str], list[Image.Image]]:
    """Returns (doc_ids, doc_texts, images) for building sparse+dense indices
    over the shared retrieval corpus built by data/download_mrag_bench.py
    (data_dir/images/*.jpg + data_dir/image_metadata.json)."""
    image_dir = data_dir / "images"
    meta_path = data_dir / "image_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Couldn't find {meta_path}. Run data/download_mrag_bench.py first."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    doc_ids, doc_texts, images = [], [], []
    for img_path in sorted(image_dir.glob("*.jpg")):
        doc_id = img_path.stem
        doc_ids.append(doc_id)
        doc_texts.append(metadata.get(doc_id, {}).get("caption", doc_id.replace("_", " ")))
        images.append(Image.open(img_path).convert("RGB"))
    return doc_ids, doc_texts, images


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    if cfg.determinism.torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = cfg.determinism.cudnn_benchmark

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=results_dir / "log.txt",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting run: %s", cfg.run_name)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    with open(results_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ---- Load data ----
    dataset = MRAGBenchDataset(cfg.paths.data_dir)
    logger.info("Loaded %d examples across %d scenarios", len(dataset), len(dataset.scenarios()))

    # ---- Build retriever (skip entirely for no_retrieval) ----
    retriever = None
    if cfg.retrieval.type != "none":
        doc_ids, doc_texts, images = load_image_corpus(Path(cfg.paths.data_dir))
        retriever = build_retriever(cfg.retrieval, doc_ids, doc_texts, images)
        logger.info("Built %s retriever over %d images", cfg.retrieval.type, len(doc_ids))

    # ---- Build model ----
    model = build_model(cfg.model)
    logger.info("Loaded model %s", cfg.model.name)

    predictions, retrieval_records = [], []
    image_dir = Path(cfg.paths.data_dir) / "images"

    for ex in tqdm(dataset.examples, desc=f"{cfg.run_name}"):
        query_image = Image.open(ex.query_image_full_path(Path(cfg.paths.data_dir))).convert("RGB")

        retrieved_images, retrieved_ids = [], []
        if retriever is not None:
            retrieved = retriever.retrieve(ex.question, top_k=cfg.retrieval.top_k)
            retrieved_ids = [r.doc_id for r in retrieved]
            retrieved_images = [Image.open(image_dir / f"{doc_id}.jpg").convert("RGB") for doc_id in retrieved_ids]
            retrieval_records.append(
                {
                    "id": ex.id,
                    "retrieved_ids": retrieved_ids,
                    "scores": [r.score for r in retrieved],
                }
            )

        # The query image is ALWAYS shown -- it's the actual subject of the
        # question (MRAG-Bench's whole design: an ambiguous/transformed view
        # that the retrieved evidence images are meant to help disambiguate).
        # Retrieved images are additional context, not a replacement for it.
        response = model.generate(
            question=ex.question,
            choices=ex.choices,
            images=[query_image] + retrieved_images,
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )

        predictions.append(
            {
                "id": ex.id,
                "question": ex.question,
                "choices": ex.choices,
                "prediction": response.text,
                "ground_truth": ex.answer,
                "scenario": ex.scenario,
                "gt_image_ids": ex.gt_image_ids,
                "retrieval_type": cfg.retrieval.type,
                "model_name": cfg.model.name,
                "latency_seconds": response.latency_seconds,
                "timestamp": time.time(),
            }
        )

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    with open(results_dir / "retrieval.json", "w") as f:
        json.dump(retrieval_records, f, indent=2)
    logger.info("Saved %d predictions to %s", len(predictions), results_dir)

    # ---- Baseline metrics for this run ----
    from evaluation.evaluate import evaluate_run

    metrics = evaluate_run(str(results_dir))
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
