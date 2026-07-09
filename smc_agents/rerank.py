"""
Cross-encoder reranker for a second, more expensive relevance pass over a
small candidate pool. Mirrors the retriever's injectable-encoder design: tests
pass a deterministic stub scorer, production lazy-loads the HF model on first
use so importing this module stays cheap and offline.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

Scorer = Callable[[Sequence[Tuple[str, str]]], List[float]]

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Scores (query, passage) pairs with a cross-encoder. Higher is more relevant.

    The heavy transformer is only materialized when `score` is first called and
    no scorer was injected, so constructing the reranker never touches the
    network.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        scorer: Optional[Scorer] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._scorer = scorer

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        if self._scorer is None:
            self._load_model()
        return list(self._scorer(pairs))

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device)
        model.eval()
        # This MiniLM checkpoint's fp32 forward returns NaN logits on CPU with
        # torch 2.2.x; float64 is numerically stable and the cost is trivial
        # over the small rerank pool.
        if device.type == "cpu":
            model = model.double()

        def _hf_score(pairs: Sequence[Tuple[str, str]]) -> List[float]:
            scores: List[float] = []
            with torch.no_grad():
                for start in range(0, len(pairs), self.batch_size):
                    batch = list(pairs[start : start + self.batch_size])
                    encoded = tokenizer(
                        [q for q, _ in batch],
                        [p for _, p in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    logits = model(**encoded).logits.squeeze(-1)
                    scores.extend(logits.detach().cpu().reshape(-1).tolist())
            return scores

        self._scorer = _hf_score
