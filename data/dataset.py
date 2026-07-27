"""
Thin, framework-agnostic wrapper around the locally-cached MRAG-Bench data
(post data/download_mrag_bench.py). Kept deliberately dumb: just loads the
flat JSON + resolves image paths, so retrievers/models/evaluation code don't
need to know about `datasets` or HF internals at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MRAGExample:
    id: str
    question: str
    choices: list[str]
    answer: str  # ground-truth choice letter, e.g. "A"
    scenario: str
    aspect: str
    image_type: str
    source: str
    query_image_path: str  # relative to data_dir; the image the question is actually about
    gt_image_ids: list[str] = field(default_factory=list)  # ids into the shared corpus (images/)

    def query_image_full_path(self, data_dir: Path) -> Path:
        return data_dir / self.query_image_path

    def gt_image_paths(self, image_dir: Path) -> list[Path]:
        return [image_dir / f"{img_id}.jpg" for img_id in self.gt_image_ids]


class MRAGBenchDataset:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.query_image_dir = self.data_dir / "query_images"
        qa_path = self.data_dir / "qa" / "test.json"
        if not qa_path.exists():
            raise FileNotFoundError(
                f"Couldn't find {qa_path}. Run data/download_mrag_bench.py first "
                "(on a machine with huggingface.co access)."
            )
        with open(qa_path) as f:
            raw: list[dict[str, Any]] = json.load(f)
        self.examples = [
            MRAGExample(
                id=str(r["id"]),
                question=r["question"],
                choices=r["choices"],
                answer=r["answer"],
                scenario=r["scenario"],
                aspect=r["aspect"],
                image_type=r["image_type"],
                source=r["source"],
                query_image_path=r["query_image_path"],
                gt_image_ids=r.get("gt_image_ids", []),
            )
            for r in raw
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MRAGExample:
        return self.examples[idx]

    def scenarios(self) -> list[str]:
        return sorted({ex.scenario for ex in self.examples})

    def by_scenario(self, scenario: str) -> list[MRAGExample]:
        return [ex for ex in self.examples if ex.scenario == scenario]
