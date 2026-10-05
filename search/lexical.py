"""
Keyword search: an SQLite FTS5 inverted index with three fields, scored by BM25.

FTS5 builds the inverted index when rows are inserted. At query time `MATCH` uses it
to find the chunks that contain any query word, and `bm25()` scores only those, with
per-column weights (path ×2, heading ×5, body ×1). FTS5's bm25() returns negative
numbers (lower is better), so ascending order is best-first.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from search import config
from search.corpus import Corpus


def or_query(text: str) -> str:
    """Lowercase, keep alphabetic words of 3+ letters, dedupe, join with OR.

    No stemming and no stop-word list: BM25's IDF already makes common words nearly
    worthless. OR, because requiring every word of a long complaint matches nothing.
    """
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return " OR ".join(dict.fromkeys(words))


def build(corpus: Corpus, path: Path) -> Path:
    """Fielded index: (chunk id, cleaned chapter title, cleaned section heading, chunk text)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("CREATE VIRTUAL TABLE f USING fts5(cid UNINDEXED, path, heading, body)")
    rows = []
    for cid in corpus.ids:
        c = corpus.chunk(cid)
        rows.append((cid, corpus.clean_chapter_title(c.get("chapter_title")), corpus.clean_heading(c.get("section_heading")), str(c.get("text", ""))))
    con.executemany("INSERT INTO f VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


class LexicalIndex:
    def __init__(self, path: Path):
        self.path = Path(path)

    def search(self, text: str, k: int = config.LIST_DEPTH, weights=config.BM25_FIELD_WEIGHTS) -> list[str]:
        match = or_query(text)
        if not match:
            return []
        sql = f"SELECT cid FROM f WHERE f MATCH ? ORDER BY bm25(f, 0, {weights[0]}, {weights[1]}, {weights[2]}) LIMIT ?"
        with sqlite3.connect(self.path) as con:
            try:
                return [r[0] for r in con.execute(sql, (match, k))]
            except sqlite3.OperationalError:   # malformed MATCH -> no lexical evidence
                return []
