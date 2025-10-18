#!/usr/bin/env python3
"""
Simple CLI helper to search the SMC SQLite ground-truth database.

Example:
    python query_ground_truth.py "tree protection exceptional"
    python query_ground_truth.py "rear yard setback" --title 23 --limit 3
    python query_ground_truth.py "hazardous tree removal" --citation 25.11
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data/processed/smc_ground_truth.db"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search expression (FTS5 syntax works).")
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--title",
        type=int,
        help="Filter results to a specific SMC title number (e.g., 23).",
    )
    parser.add_argument(
        "--citation",
        help="Filter to sections whose citation starts with this text (e.g., '23.44').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of matches to return (default: %(default)s).",
    )
    return parser.parse_args(argv)


def build_filters(args: argparse.Namespace) -> Tuple[str, Iterable[object]]:
    filters = []
    params: list[object] = []

    if args.title is not None:
        filters.append("chunks.title_number = ?")
        params.append(args.title)

    if args.citation:
        filters.append("chunks.section_citation LIKE ?")
        params.append(f"{args.citation}%")

    where_clause = " AND ".join(filters)
    if where_clause:
        where_clause = "WHERE " + where_clause

    return where_clause, params


def query_database(
    database: Path,
    query: str,
    *,
    where_clause: str,
    params: Iterable[object],
    limit: int,
) -> Iterator[sqlite3.Row]:
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        sql = f"""
            SELECT
                chunks.chunk_id,
                chunks.full_citation,
                chunks.section_heading,
                chunks.text,
                chunks.title_number,
                chunks.title_label,
                rank
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            {where_clause}
            AND chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        for row in conn.execute(sql, (*params, query, limit)):
            yield row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    where_clause, params = build_filters(args)
    if "WHERE" not in where_clause:
        # ensure the base query has a WHERE for the MATCH clause
        where_clause = "WHERE 1=1"

    rows = list(
        query_database(
            args.database,
            args.query,
            where_clause=where_clause,
            params=params,
            limit=args.limit,
        )
    )
    if not rows:
        print("No matches found.")
        return 0

    for row in rows:
        print(f"{row['full_citation'] or row['chunk_id']}: {row['section_heading']}")
        print(f"Title {row['title_number']} – {row['title_label']}")
        print(row["text"])
        print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
