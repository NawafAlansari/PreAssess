#!/usr/bin/env python3
"""
Populate a SQLite full-text index from the processed SMC chunk file.

Usage:
    python build_sqlite_index.py data/processed/smc_chunks.jsonl \
        --database data/processed/smc_ground_truth.db

The resulting database contains a `chunks` table with full metadata and an
FTS5 virtual table (`chunks_fts`) for fast text search. This can serve as the
retrieval layer feeding an LLM with grounded snippets plus citations.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence


DEFAULT_DB = Path("seattle-checker/data/processed/smc_ground_truth.db")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chunks_file",
        type=Path,
        help="Path to the JSONL file created by build_ground_truth.py.",
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=DEFAULT_DB,
        help=f"Destination SQLite database file (default: {DEFAULT_DB}).",
    )
    return parser.parse_args(argv)


def read_chunks(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def setup_database(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            chunk_type TEXT NOT NULL,
            text TEXT NOT NULL,
            source_path TEXT NOT NULL,
            title_number INTEGER,
            title_label TEXT,
            subtitle_number TEXT,
            subtitle_title TEXT,
            division_number TEXT,
            division_title TEXT,
            chapter_citation TEXT,
            chapter_title TEXT,
            section_citation TEXT,
            section_heading TEXT,
            full_citation TEXT,
            start_page INTEGER,
            end_page INTEGER,
            char_count INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            chunk_id,
            text,
            content='chunks',
            content_rowid='rowid'
        );
        """
    )
    conn.execute("DELETE FROM chunks;")
    conn.execute("DELETE FROM chunks_fts;")


def insert_chunk(conn: sqlite3.Connection, chunk: Dict[str, object]) -> None:
    fields = [
        "chunk_id",
        "chunk_type",
        "text",
        "source_path",
        "title_number",
        "title_label",
        "subtitle_number",
        "subtitle_title",
        "division_number",
        "division_title",
        "chapter_citation",
        "chapter_title",
        "section_citation",
        "section_heading",
        "full_citation",
        "start_page",
        "end_page",
        "char_count",
    ]
    values = [chunk.get(field) for field in fields]
    conn.execute(
        f"""
        INSERT INTO chunks ({", ".join(fields)})
        VALUES ({", ".join(["?"] * len(fields))});
        """,
        values,
    )
    conn.execute(
        "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (last_insert_rowid(), ?, ?);",
        (chunk["chunk_id"], chunk["text"]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(args.database) as conn:
        setup_database(conn)
        seen_ids: set[str] = set()
        total = 0
        skipped = 0
        for chunk in read_chunks(args.chunks_file):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id in seen_ids:
                skipped += 1
                continue
            insert_chunk(conn, chunk)
            seen_ids.add(chunk_id)
            total += 1
        conn.commit()

    print(
        f"Indexed {total} unique chunks into {args.database} "
        f"(skipped {skipped} duplicates). "
        "Use sqlite FTS queries to retrieve grounded passages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
