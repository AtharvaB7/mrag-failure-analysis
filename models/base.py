"""Common interface every VLM wrapper implements, so scripts/run_experiment.py
never needs to know which model it's talking to."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class VLMResponse:
    text: str
    latency_seconds: float


class VLMWrapper(Protocol):
    def generate(self, question: str, choices: list[str], images: list, max_new_tokens: int) -> VLMResponse:
        """images: list of PIL.Image (retrieved context images; empty list for no-retrieval)."""
        ...
