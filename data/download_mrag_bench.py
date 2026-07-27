"""
Download and cache MRAG-Bench locally.

MRAG-Bench (ICLR 2025, Hu et al.) is hosted on the Hugging Face Hub as
`uclanlp/MRAG-Bench`. Confirmed real schema (via `load_dataset(...).column_names`):

    ['id', 'aspect', 'scenario', 'image', 'gt_images', 'question',
     'A', 'B', 'C', 'D', 'answer_choice', 'answer', 'image_type',
     'source', 'retrieved_images']

Key facts this script depends on:
  - 'image' is the QUERY image (the ambiguous/transformed photo the question
    is actually about) -- this must be shown to the model in EVERY setting,
    including no-retrieval, since it's the subject of the question itself.
  - 'gt_images' is a list of PIL images: the ground-truth evidence that
    supports the answer. These are NOT pre-assigned stable IDs by the
    dataset -- they're embedded PIL objects per-row. To build a single
    shared retrieval corpus (needed so our own BM25/dense/hybrid retrievers
    have something consistent to index and search over across ALL
    questions), this script extracts every unique gt_image across the
    whole dataset (deduplicated by content hash, since the same evidence
    image can support multiple questions), assigns each a stable
    "corpus_XXXXXX" id, and saves it to disk once.
  - 'retrieved_images' is MRAG-Bench's OWN precomputed retrieval results
    (from the benchmark authors' default retriever). We do NOT use this --
    our whole project is comparing OUR OWN retrieval settings (none/sparse/
    dense/hybrid) against each other, so using their precomputed retrieval
    would defeat the point. It's ignored here.
  - 'answer_choice' is the ground-truth letter (A/B/C/D) -- use this as
    `answer`, not the free-text 'answer' column, since the eval pipeline
    grades on the extracted letter.
  - Per-image captions for BM25 (sparse retrieval) use the text of the
    answer choice the image was evidence for (e.g. gt_images for a
    "silky_terrier" question get captioned "silky_terrier") -- these images
    are frequently sourced from labeled datasets like ImageNet
    ('source': 'Imagenet'), so the row's own answer text is a genuine,
    reasonably accurate caption, not a leaked shortcut specific to our setup.

Run this on a machine with internet access to huggingface.co (this repo's
dev sandbox network is restricted and cannot reach the HF Hub -- run this
step in Colab / on your GPU box, not in a network-restricted environment).

Usage:
    python data/download_mrag_bench.py --out_dir data/mrag_bench
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def _image_hash(img: Image.Image) -> str:
    """Content hash for dedup -- two PIL Image objects loaded independently
    from the same underlying JPEG bytes won't be `==` to each other, so we
    hash the encoded bytes rather than relying on object identity/equality."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return hashlib.sha1(buf.getvalue()).hexdigest()[:16]


def download(out_dir: str) -> None:
    from datasets import load_dataset

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    query_images_dir = out_dir / "query_images"
    qa_dir = out_dir / "qa"
    for d in (images_dir, query_images_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/3] Downloading MRAG-Bench (uclanlp/MRAG-Bench, split='test') ...")
    ds = load_dataset("uclanlp/MRAG-Bench", split="test")
    print(f"    Loaded {len(ds)} rows. Columns: {ds.column_names}")

    print("[2/3] Building shared retrieval corpus from gt_images (deduplicated) ...")
    hash_to_id: dict[str, str] = {}
    captions: dict[str, str] = {}
    next_id = 0

    records = []
    for i, row in enumerate(ds):
        # Save this row's query image (always shown to the model, every setting).
        query_image_path = query_images_dir / f"{row['id']}.jpg"
        if not query_image_path.exists():
            row["image"].convert("RGB").save(query_image_path, format="JPEG", quality=95)

        # Register each gt_image in the shared corpus, deduplicating by content hash.
        gt_image_ids = []
        for gt_img in row["gt_images"]:
            h = _image_hash(gt_img)
            if h not in hash_to_id:
                corpus_id = f"corpus_{next_id:06d}"
                hash_to_id[h] = corpus_id
                captions[corpus_id] = row["answer"]  # e.g. "silky_terrier"
                gt_img.convert("RGB").save(images_dir / f"{corpus_id}.jpg", format="JPEG", quality=95)
                next_id += 1
            gt_image_ids.append(hash_to_id[h])

        records.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "choices": [row["A"], row["B"], row["C"], row["D"]],
                "answer": row["answer_choice"],  # letter (A/B/C/D), matches eval's extract_choice_letter
                "scenario": row["scenario"],
                "aspect": row["aspect"],
                "image_type": row["image_type"],
                "source": row["source"],
                "gt_image_ids": gt_image_ids,
                "query_image_path": str(query_image_path.relative_to(out_dir)),
            }
        )
        if (i + 1) % 200 == 0:
            print(f"    processed {i + 1}/{len(ds)} rows, corpus size so far: {next_id}")

    print(f"    Final corpus size: {next_id} unique images (from {len(ds)} questions)")

    with open(images_dir.parent / "image_metadata.json", "w") as f:
        json.dump({cid: {"caption": cap} for cid, cap in captions.items()}, f, indent=2)

    # Atomic write: build in a temp file, only replace the real file on full success.
    tmp_path = qa_dir / "test.json.tmp"
    final_path = qa_dir / "test.json"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    tmp_path.replace(final_path)

    print(f"[3/3] Saved {len(records)} questions -> {final_path}")
    print(f"       Saved {next_id} corpus images -> {images_dir}")
    print(f"       Saved {len(records)} query images -> {query_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/mrag_bench")
    args = parser.parse_args()
    download(args.out_dir)
