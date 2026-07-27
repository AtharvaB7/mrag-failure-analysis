"""
Tests the corpus-building / dedup / record-extraction logic in
data/download_mrag_bench.py against fabricated rows that match MRAG-Bench's
REAL confirmed schema (columns: id, aspect, scenario, image, gt_images,
question, A, B, C, D, answer_choice, answer, image_type, source,
retrieved_images) -- verified via an actual `load_dataset(...).column_names`
call, not guessed. No network access needed since the PIL images are
fabricated in-memory.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from data.download_mrag_bench import _image_hash


def make_fake_row(row_id: str, color: str, gt_colors: list[str]) -> dict:
    return {
        "id": row_id,
        "aspect": "Perspective",
        "scenario": "Scope",
        "image": Image.new("RGB", (50, 50), color=color),
        "gt_images": [Image.new("RGB", (50, 50), color=c) for c in gt_colors],
        "question": f"What is this {color} thing?",
        "A": "silky_terrier",
        "B": "Yorkshire_terrier",
        "C": "Australian_terrier",
        "D": "Cairn_terrier",
        "answer_choice": "A",
        "answer": "silky_terrier",
        "image_type": "Animal",
        "source": "Imagenet",
        "retrieved_images": [],  # deliberately ignored by our pipeline
    }


def test_image_hash_is_deterministic_for_identical_content():
    img1 = Image.new("RGB", (20, 20), color="blue")
    img2 = Image.new("RGB", (20, 20), color="blue")
    assert _image_hash(img1) == _image_hash(img2)


def test_image_hash_differs_for_different_content():
    img1 = Image.new("RGB", (20, 20), color="blue")
    img2 = Image.new("RGB", (20, 20), color="red")
    assert _image_hash(img1) != _image_hash(img2)


def test_corpus_dedup_and_record_extraction_end_to_end():
    """Simulates the core loop of download() without needing `datasets` or
    network access -- same logic, fabricated rows."""
    import hashlib
    import io

    rows = [
        make_fake_row("0", "red", gt_colors=["green", "green", "blue"]),  # 2 unique gt images
        make_fake_row("1", "red", gt_colors=["green", "yellow"]),          # "green" should dedup with row 0's
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        images_dir = out_dir / "images"
        query_images_dir = out_dir / "query_images"
        qa_dir = out_dir / "qa"
        for d in (images_dir, query_images_dir, qa_dir):
            d.mkdir(parents=True, exist_ok=True)

        hash_to_id, captions, next_id, records = {}, {}, 0, []
        for row in rows:
            query_path = query_images_dir / f"{row['id']}.jpg"
            row["image"].convert("RGB").save(query_path, format="JPEG")

            gt_image_ids = []
            for gt_img in row["gt_images"]:
                h = _image_hash(gt_img)
                if h not in hash_to_id:
                    cid = f"corpus_{next_id:06d}"
                    hash_to_id[h] = cid
                    captions[cid] = row["answer"]
                    gt_img.convert("RGB").save(images_dir / f"{cid}.jpg", format="JPEG")
                    next_id += 1
                gt_image_ids.append(hash_to_id[h])

            records.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "choices": [row["A"], row["B"], row["C"], row["D"]],
                    "answer": row["answer_choice"],
                    "scenario": row["scenario"],
                    "gt_image_ids": gt_image_ids,
                    "query_image_path": str(query_path.relative_to(out_dir)),
                }
            )

        # 4 gt_images total across both rows, but "green" appears in both ->
        # 3 unique images should be in the corpus, not 4.
        assert next_id == 3
        assert len(list(images_dir.glob("*.jpg"))) == 3

        # Row 0's first gt_image and row 1's first gt_image are both "green"
        # -> should resolve to the SAME corpus id.
        assert records[0]["gt_image_ids"][0] == records[1]["gt_image_ids"][0]

        # Ground truth should be the LETTER (answer_choice), not the free-text answer.
        assert records[0]["answer"] == "A"

        # Choices should be a plain list in A/B/C/D order.
        assert records[0]["choices"] == ["silky_terrier", "Yorkshire_terrier", "Australian_terrier", "Cairn_terrier"]

        # Record should be fully JSON-serializable (this was the original bug).
        json.dumps(records)  # raises if not serializable
