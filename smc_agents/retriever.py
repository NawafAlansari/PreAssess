"""
Semantic retriever that pairs the SMC chunk metadata + embeddings.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    full_citation: Optional[str]
    section_heading: Optional[str]
    chapter_title: Optional[str]
    title_number: Optional[int]
    title_label: Optional[str]
    metadata: Dict[str, object]


class GroundedRetriever:
    """
    Loads the precomputed embeddings and metadata, then supplies
    semantic search with optional metadata/FTS filtering.
    """

    def __init__(
        self,
        *,
        embeddings_path: Path,
        chunks_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sqlite_path: Optional[Path] = None,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.sqlite_path = sqlite_path

        self._load_embeddings()
        self._load_metadata()
        self._load_encoder()

    def _load_embeddings(self) -> None:
        data = np.load(self.embeddings_path)
        self.chunk_ids: np.ndarray = data["chunk_ids"]
        self.embeddings: np.ndarray = data["embeddings"]

    def _load_metadata(self) -> None:
        metadata: Dict[str, Dict[str, object]] = {}
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunk_id = payload.get("chunk_id")
                if chunk_id is None:
                    continue
                metadata[chunk_id] = payload
        self.metadata = metadata

    def _load_encoder(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        with torch.no_grad():
            encoded = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

    @staticmethod
    def _mean_pool(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        summed = (model_output * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return summed / torch.clamp(counts, min=1e-9)

    def _candidate_mask(
        self,
        *,
        title_number: Optional[int],
        section_prefix: Optional[str],
        chunk_types: Optional[Sequence[str]],
        fts_query: Optional[str],
    ) -> np.ndarray:
        mask = np.ones(len(self.chunk_ids), dtype=bool)

        if title_number is not None:
            mask &= np.array(
                [
                    self.metadata[cid].get("title_number") == title_number
                    for cid in self.chunk_ids
                ],
                dtype=bool,
            )

        if section_prefix:
            mask &= np.array(
                [
                    str(self.metadata[cid].get("section_citation") or "").startswith(section_prefix)
                    for cid in self.chunk_ids
                ],
                dtype=bool,
            )

        if chunk_types:
            allowed = set(chunk_types)
            mask &= np.array(
                [self.metadata[cid].get("chunk_type") in allowed for cid in self.chunk_ids],
                dtype=bool,
            )

        if fts_query and self.sqlite_path:
            candidate_ids = self._fts_lookup(fts_query, title_number=title_number, section_prefix=section_prefix)
            allowed = set(candidate_ids)
            mask &= np.array([cid in allowed for cid in self.chunk_ids], dtype=bool)

        return mask

    def _fts_lookup(
        self,
        query: str,
        *,
        title_number: Optional[int],
        section_prefix: Optional[str],
        limit: int = 200,
    ) -> List[str]:
        sql = [
            "SELECT chunks.chunk_id",
            "FROM chunks_fts",
            "JOIN chunks ON chunks.rowid = chunks_fts.rowid",
            "WHERE chunks_fts MATCH ?",
        ]
        params: List[object] = [query]
        if title_number is not None:
            sql.append("AND chunks.title_number = ?")
            params.append(title_number)
        if section_prefix:
            sql.append("AND chunks.section_citation LIKE ?")
            params.append(f"{section_prefix}%")
        sql.append("LIMIT ?")
        params.append(limit)

        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [row[0] for row in rows]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        title_number: Optional[int] = None,
        section_prefix: Optional[str] = None,
        chunk_types: Optional[Sequence[str]] = None,
        fts_query: Optional[str] = None,
    ) -> List[RetrievalResult]:
        candidate_mask = self._candidate_mask(
            title_number=title_number,
            section_prefix=section_prefix,
            chunk_types=chunk_types,
            fts_query=fts_query,
        )

        vocab_indices = np.where(candidate_mask)[0]
        if vocab_indices.size == 0:
            return []

        query_embedding = self._encode([query])[0]
        candidate_embeddings = self.embeddings[vocab_indices]
        scores = candidate_embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[RetrievalResult] = []
        for idx in top_indices:
            corpus_idx = vocab_indices[idx]
            chunk_id = str(self.chunk_ids[corpus_idx])
            meta = self.metadata[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(scores[idx]),
                    text=meta.get("text", ""),
                    full_citation=meta.get("full_citation"),
                    section_heading=meta.get("section_heading"),
                    chapter_title=meta.get("chapter_title"),
                    title_number=meta.get("title_number"),
                    title_label=meta.get("title_label"),
                    metadata=meta,
                )
            )
        return results
