"""
Dense search: a shared-weight bi-encoder (two-tower) and exact nearest neighbours.

Each chunk's breadcrumb text is embedded once (mean pooling over tokens, padding
masked, L2-normalized), so a dot product is a cosine. At query time one matrix
multiply scores all ~19,000 chunks; at this size exact search costs milliseconds and
no approximate index is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from search import config


class Encoder:
    def __init__(self, model: str = config.ENCODER, max_length: int = config.ENCODER_MAX_LENGTH):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return t.nn.functional.normalize(pooled, p=2, dim=1).numpy()

    def encode(self, texts: list[str], batch: int = config.ENCODER_BATCH) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.vstack([self._encode(texts[s:s + batch]) for s in range(0, len(texts), batch)]).astype(np.float32)


class DenseIndex:
    """Row i of `vectors` belongs to corpus.ids[i]."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self.vectors = vectors

    @classmethod
    def load(cls, ids: list[str], path: Path) -> "DenseIndex":
        return cls(ids, np.load(path))

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.vectors @ qvec

    def search(self, qvec: np.ndarray, k: int = config.LIST_DEPTH) -> list[str]:
        s = self.scores(qvec)
        return [self.ids[i] for i in np.argsort(s)[::-1][:k]]

    def best_cosine(self, qvec: np.ndarray, rows: list[int]) -> float:
        """A section's similarity = its best-matching chunk."""
        return float(np.max(self.vectors[rows] @ qvec))
