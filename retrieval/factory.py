"""Build the right retriever object from a Hydra/OmegaConf `retrieval` config node."""
from __future__ import annotations

from typing import Any


def build_retriever(retrieval_cfg: Any, doc_ids: list[str], doc_texts: list[str], images: list | None = None):
    rtype = retrieval_cfg.type

    if rtype == "none":
        return None

    if rtype == "sparse":
        from retrieval.sparse_bm25 import BM25Retriever

        return BM25Retriever(doc_ids, doc_texts, k1=retrieval_cfg.k1, b=retrieval_cfg.b)

    if rtype == "dense":
        from retrieval.dense_embedding import SiglipDenseRetriever

        retriever = SiglipDenseRetriever(
            hf_id=retrieval_cfg.hf_id,
            embed_batch_size=retrieval_cfg.get("embed_batch_size", 32),
            dtype=retrieval_cfg.get("dtype", "float16"),
        )
        if images is None:
            raise ValueError("Dense retrieval requires `images` (list of PIL.Image) to build the index.")
        retriever.build_index(doc_ids, images)
        return retriever

    if rtype == "hybrid":
        from retrieval.dense_embedding import SiglipDenseRetriever
        from retrieval.hybrid_rrf import HybridRRFRetriever
        from retrieval.sparse_bm25 import BM25Retriever

        sparse = BM25Retriever(doc_ids, doc_texts)
        dense = SiglipDenseRetriever(
            hf_id=retrieval_cfg.dense.get("hf_id", "google/siglip-so400m-patch14-384"),
            embed_batch_size=retrieval_cfg.dense.get("embed_batch_size", 32),
            dtype=retrieval_cfg.dense.get("dtype", "float16"),
        )
        if images is None:
            raise ValueError("Hybrid retrieval requires `images` to build the dense index.")
        dense.build_index(doc_ids, images)
        return HybridRRFRetriever(sparse, dense, rrf_k=retrieval_cfg.rrf_k)

    raise ValueError(f"Unknown retrieval type: {rtype}")
