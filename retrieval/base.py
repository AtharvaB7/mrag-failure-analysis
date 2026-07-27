"""Common interface every retriever (none/sparse/dense/hybrid) implements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    rank: int  # 1-indexed


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Return up to top_k documents/images, ranked best-first."""
        ...
