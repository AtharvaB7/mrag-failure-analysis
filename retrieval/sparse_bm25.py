"""
Sparse (BM25) retrieval over the text side of the corpus.

For MRAG-Bench specifically: images don't have natural free text attached,
so "sparse retrieval over documents" means BM25 over per-image captions/alt-
text/scenario metadata (whatever textual description each image has) -- this
mirrors how sparse retrieval is used in practice for image corpora that lack
rich text (caption-based indexing), and gives a genuinely different retrieval
signal from the dense (embedding) arm rather than just a weaker copy of it.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from retrieval.base import RetrievedDoc


def simple_lowercase_tokenize(text: str) -> list[str]:
    """Deliberately simple tokenizer: lowercase + strip punctuation + split on
    whitespace. BM25 is robust to tokenizer choice for short captions; a more
    elaborate tokenizer (stemming, stopword removal) is a reasonable ablation
    but not needed for the core experiment."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1: float = 1.5, b: float = 0.75):
        assert len(doc_ids) == len(doc_texts), "doc_ids and doc_texts must be aligned 1:1"
        self.doc_ids = doc_ids
        tokenized_corpus = [simple_lowercase_tokenize(t) for t in doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        tokenized_query = simple_lowercase_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedDoc(doc_id=self.doc_ids[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(ranked_idx)
        ]
