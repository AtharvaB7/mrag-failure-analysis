"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why RRF specifically: BM25 scores and cosine-similarity scores live on
different, non-comparable scales (BM25 is unbounded and corpus-dependent;
cosine similarity is bounded in [-1, 1] and dominated by embedding geometry).
Naively averaging or summing the two would let whichever arm happens to have
larger-magnitude scores dominate the fusion for reasons that have nothing to
do with retrieval quality. RRF sidesteps this entirely by fusing on *rank
position* rather than raw score, using the standard formula from
Cormack, Clarke & Buettcher (2009):

    RRF_score(doc) = sum over retrievers r of  1 / (k + rank_r(doc))

where k (typically 60) dampens the influence of top-ranked-but-possibly-noisy
results and is not particularly sensitive to tuning.
"""
from __future__ import annotations

from retrieval.base import RetrievedDoc


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]],
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedDoc]:
    """Fuse N ranked lists (e.g. [sparse_results, dense_results]) into one."""
    fused_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for doc in ranked_list:
            fused_scores[doc.doc_id] = fused_scores.get(doc.doc_id, 0.0) + 1.0 / (k + doc.rank)

    ranked_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)[:top_k]
    return [
        RetrievedDoc(doc_id=doc_id, score=fused_scores[doc_id], rank=rank + 1)
        for rank, doc_id in enumerate(ranked_ids)
    ]


class HybridRRFRetriever:
    def __init__(self, sparse_retriever, dense_retriever, rrf_k: int = 60):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int, arm_top_k: int = 10) -> list[RetrievedDoc]:
        sparse_results = self.sparse.retrieve(query, top_k=arm_top_k)
        dense_results = self.dense.retrieve(query, top_k=arm_top_k)
        return reciprocal_rank_fusion([sparse_results, dense_results], k=self.rrf_k, top_k=top_k)
