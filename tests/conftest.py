"""
Shared fixtures: a tiny hand-built corpus (chunks + embeddings + FTS db) and a
deterministic stub encoder, so the whole suite runs offline with no model
downloads and no API keys.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from smc_agents.retriever import GroundedRetriever

DIM = 8

# Deterministic "embeddings": each chunk gets a fixed unit vector; the stub
# encoder maps query keywords onto the same vectors so similarity is exact.
KEYWORD_AXES = {
    "setback": 0,
    "dwelling": 1,
    "permit": 2,
    "tree": 3,
    "parking": 4,
    "height": 5,
}

CHUNKS = [
    {
        "chunk_id": "c-2344-010",
        "text": "Minimum setbacks for accessory dwelling units in single-family zones.",
        "chunk_type": "section",
        "title_number": 23,
        "title_label": "Title 23",
        "chapter_citation": "23.44",
        "chapter_title": "Residential, Single-Family",
        "section_citation": "23.44.010",
        "section_heading": "Setback requirements",
        "full_citation": "SMC 23.44.010",
        "keywords": ["setback", "dwelling"],
    },
    {
        "chunk_id": "c-2344-020",
        "text": "Height limits for principal structures in single-family zones.",
        "chunk_type": "section",
        "title_number": 23,
        "title_label": "Title 23",
        "chapter_citation": "23.44",
        "chapter_title": "Residential, Single-Family",
        "section_citation": "23.44.020",
        "section_heading": "Height limits",
        "full_citation": "SMC 23.44.020",
        "keywords": ["height"],
    },
    {
        "chunk_id": "c-22801-050",
        "text": "Building permit application requirements and submittal documents.",
        "chunk_type": "section",
        "title_number": 22,
        "title_label": "Title 22",
        "chapter_citation": "22.801",
        "chapter_title": "Permits",
        "section_citation": "22.801.050",
        "section_heading": "Permit applications",
        "full_citation": "SMC 22.801.050",
        "keywords": ["permit"],
    },
    {
        "chunk_id": "c-2345-005",
        "text": "Parking space requirements for multifamily development.",
        "chunk_type": "definition",
        "title_number": 23,
        "title_label": "Title 23",
        "chapter_citation": "23.45",
        "chapter_title": "Multifamily",
        "section_citation": "23.45.005",
        "section_heading": "Parking",
        "full_citation": "SMC 23.45.005",
        "keywords": ["parking"],
    },
]


def _vector(keywords: Sequence[str]) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for kw in keywords:
        axis = KEYWORD_AXES.get(kw)
        if axis is not None:
            v[axis] = 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    out = []
    for text in texts:
        kws = [kw for kw in KEYWORD_AXES if kw in text.lower()]
        out.append(_vector(kws))
    return np.stack(out)


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")

    chunks_path = root / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in CHUNKS:
            payload = {k: v for k, v in chunk.items() if k != "keywords"}
            fh.write(json.dumps(payload) + "\n")

    ids = np.array([c["chunk_id"] for c in CHUNKS])
    embs = np.stack([_vector(c["keywords"]) for c in CHUNKS])
    np.savez(root / "embeddings.npz", chunk_ids=ids, embeddings=embs)

    db_path = root / "ground_truth.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE chunks (chunk_id TEXT, title_number INTEGER,"
            " section_citation TEXT, text TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks,"
            " content_rowid=rowid)"
        )
        for c in CHUNKS:
            cur = conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                (c["chunk_id"], c["title_number"], c["section_citation"], c["text"]),
            )
            conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                (cur.lastrowid, c["text"]),
            )
    return root


@pytest.fixture()
def retriever(corpus_dir: Path) -> GroundedRetriever:
    return GroundedRetriever(
        embeddings_path=corpus_dir / "embeddings.npz",
        chunks_path=corpus_dir / "chunks.jsonl",
        sqlite_path=corpus_dir / "ground_truth.db",
        encoder=stub_encoder,
    )
