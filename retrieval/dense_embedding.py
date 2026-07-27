"""
Dense retrieval via SigLIP embeddings + a flat FAISS index (cosine sim via
inner product on L2-normalized vectors).

Why SigLIP over vanilla CLIP: SigLIP's sigmoid loss (vs. CLIP's softmax
contrastive loss) tends to produce better zero-shot image-text alignment at
comparable model size, and its embeddings are a drop-in replacement for CLIP's
in a retrieval pipeline (same "embed query text, embed corpus images,
nearest-neighbor search" recipe). ColPali/BGE-VL are reasonable alternatives
if you want a stronger-but-heavier ablation later; the retriever interface
here doesn't care which embedding model produced the vectors.

Requires network access to huggingface.co to pull model weights -- run on
your GPU machine, not in a network-restricted sandbox.
"""
from __future__ import annotations

import numpy as np
import torch

from retrieval.base import RetrievedDoc


def _extract_features(output):
    """SigLIP's get_image_features/get_text_features return a raw tensor in
    older `transformers` versions, but a BaseModelOutputWithPooling object in
    newer ones -- handle both so this doesn't silently break on a version bump."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


class SiglipDenseRetriever:
    def __init__(
        self,
        hf_id: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        embed_batch_size: int = 32,
        dtype: str = "float16",
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = device if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(hf_id, dtype=torch_dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.embed_batch_size = embed_batch_size
        self.index: np.ndarray | None = None  # [N, D], L2-normalized
        self.doc_ids: list[str] = []

    @torch.no_grad()
    def build_index(self, doc_ids: list[str], images: list) -> None:
        """images: list of PIL.Image, aligned 1:1 with doc_ids.

        Encodes in batches of `embed_batch_size` rather than all at once --
        a single forward pass over thousands of full-resolution images can
        need many GB of activation memory on top of whatever else (e.g. the
        VLM) is already resident on the same GPU, which is exactly what
        caused the CUDA OOM this replaces."""
        self.doc_ids = doc_ids
        all_feats = []
        for start in range(0, len(images), self.embed_batch_size):
            batch = images[start : start + self.embed_batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device, dtype=self.model.dtype)
            feats = _extract_features(self.model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.float().cpu().numpy())
            del inputs, feats
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.index = np.concatenate(all_feats, axis=0).astype("float32")

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve().")
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        text_feat = _extract_features(self.model.get_text_features(**inputs))
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1).float().cpu().numpy().astype("float32")

        sims = self.index @ text_feat[0]  # [N], cosine similarity since both sides normalized
        ranked_idx = np.argsort(-sims)[:top_k]
        return [
            RetrievedDoc(doc_id=self.doc_ids[i], score=float(sims[i]), rank=rank + 1)
            for rank, i in enumerate(ranked_idx)
        ]
