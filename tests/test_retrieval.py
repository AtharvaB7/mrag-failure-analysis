"""
Sanity tests for retrieval logic that requires no GPU, no model downloads,
and no network access -- safe to run anywhere, including CI.

Run with: pytest tests/test_retrieval.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.base import RetrievedDoc
from retrieval.hybrid_rrf import reciprocal_rank_fusion
from retrieval.sparse_bm25 import BM25Retriever, simple_lowercase_tokenize


def test_tokenizer_strips_punctuation_and_lowercases():
    assert simple_lowercase_tokenize("A cat, sitting on a MAT!") == [
        "a", "cat", "sitting", "on", "a", "mat"
    ]


def test_bm25_retrieves_lexically_relevant_doc_first():
    doc_ids = ["d1", "d2", "d3"]
    doc_texts = [
        "a red sports car parked on the street",
        "a golden retriever puppy playing fetch",
        "a bowl of ramen noodles with pork belly",
    ]
    retriever = BM25Retriever(doc_ids, doc_texts)
    results = retriever.retrieve("golden retriever dog", top_k=3)
    assert results[0].doc_id == "d2"
    # ranks should be 1-indexed and monotonically increasing
    assert [r.rank for r in results] == [1, 2, 3]


def test_bm25_top_k_respected():
    doc_ids = [f"d{i}" for i in range(10)]
    doc_texts = [f"document number {i} about topic {i % 3}" for i in range(10)]
    retriever = BM25Retriever(doc_ids, doc_texts)
    results = retriever.retrieve("topic 1", top_k=3)
    assert len(results) == 3


def test_rrf_favors_doc_ranked_high_by_both_arms():
    sparse = [
        RetrievedDoc(doc_id="a", score=9.1, rank=1),
        RetrievedDoc(doc_id="b", score=5.0, rank=2),
        RetrievedDoc(doc_id="c", score=1.0, rank=3),
    ]
    dense = [
        RetrievedDoc(doc_id="b", score=0.91, rank=1),
        RetrievedDoc(doc_id="a", score=0.80, rank=2),
        RetrievedDoc(doc_id="d", score=0.10, rank=3),
    ]
    fused = reciprocal_rank_fusion([sparse, dense], k=60, top_k=4)
    fused_ids = [d.doc_id for d in fused]
    # "a" and "b" each appear in top-2 of both lists -> should beat "c"/"d",
    # which only appear in one list each, near the bottom.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert fused_ids[-1] in {"c", "d"}


def test_rrf_is_robust_to_non_comparable_score_scales():
    """The whole point of RRF: a doc with a huge raw score in one arm but a
    bad rank shouldn't dominate a doc with consistently good ranks."""
    sparse = [
        RetrievedDoc(doc_id="huge_score_bad_rank", score=1000.0, rank=5),
        RetrievedDoc(doc_id="consistent", score=2.0, rank=1),
    ]
    dense = [
        RetrievedDoc(doc_id="consistent", score=0.5, rank=1),
        RetrievedDoc(doc_id="huge_score_bad_rank", score=0.01, rank=5),
    ]
    fused = reciprocal_rank_fusion([sparse, dense], k=60, top_k=2)
    assert fused[0].doc_id == "consistent"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
