"""
Retrieval evaluation: recall@k and MRR for three configurations over the
hand-labeled query set in eval/queries.jsonl.

- dense:  embedding similarity over the whole corpus
- fts:    SQLite FTS5 BM25 ranking alone
- hybrid: FTS5 prefilter (OR-joined tokens) + dense rerank (the app's config)

A query counts as hit if any returned chunk's section citation matches an
expected prefix (section-level labels) or falls inside the expected chapter
(chapter-level labels). Run: python -m eval.retrieval_eval
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"
QUERIES = REPO_ROOT / "eval/queries.jsonl"
RESULTS = REPO_ROOT / "eval/results.md"

K_VALUES = (1, 3, 5)
TOP_K = max(K_VALUES)


def load_queries():
    with QUERIES.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def or_tokens(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z]{3,}", query.lower())
    return " OR ".join(dict.fromkeys(tokens))


def matches(section: str, expected: list[str]) -> bool:
    section = section or ""
    return any(
        section == exp or section.startswith(exp + ".") for exp in expected
    )


def rank_of_hit(sections: list[str], expected: list[str]) -> int | None:
    for i, section in enumerate(sections, start=1):
        if matches(section, expected):
            return i
    return None


def fts_search(db_path: Path, query: str, k: int) -> list[str]:
    sql = (
        "SELECT chunks.section_citation FROM chunks_fts "
        "JOIN chunks ON chunks.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
    )
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(sql, (or_tokens(query), k)).fetchall()
        except sqlite3.OperationalError:
            return []
    return [row[0] or "" for row in rows]


def evaluate(retriever, queries):
    per_config = {name: [] for name in ("dense", "fts", "prefilter", "rrf-fused")}

    for item in queries:
        q, expected = item["query"], item["expected"]

        dense = [
            str(h.metadata.get("section_citation") or "")
            for h in retriever.search(q, top_k=TOP_K)
        ]
        fts = fts_search(DATA_DIR / "smc_ground_truth.db", q, TOP_K)
        prefilter = [
            str(h.metadata.get("section_citation") or "")
            for h in retriever.search(q, top_k=TOP_K, fts_query=or_tokens(q))
        ]
        fused = [
            str(h.metadata.get("section_citation") or "")
            for h in retriever.search_fused(q, top_k=TOP_K)
        ]

        for name, sections in (
            ("dense", dense),
            ("fts", fts),
            ("prefilter", prefilter),
            ("rrf-fused", fused),
        ):
            per_config[name].append(rank_of_hit(sections, expected))

    return per_config


def summarize(per_config, n):
    lines = [
        "| config | recall@1 | recall@3 | recall@5 | MRR@5 |",
        "|--------|----------|----------|----------|-------|",
    ]
    for name, ranks in per_config.items():
        recalls = {
            k: sum(1 for r in ranks if r is not None and r <= k) / n for k in K_VALUES
        }
        mrr = sum(1.0 / r for r in ranks if r is not None) / n
        lines.append(
            f"| {name} | {recalls[1]:.2f} | {recalls[3]:.2f} | {recalls[5]:.2f} | {mrr:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    from smc_agents.retriever import GroundedRetriever

    queries = load_queries()
    retriever = GroundedRetriever(
        embeddings_path=DATA_DIR / "smc_embeddings.npz",
        chunks_path=DATA_DIR / "smc_chunks.jsonl",
        sqlite_path=DATA_DIR / "smc_ground_truth.db",
    )
    per_config = evaluate(retriever, queries)
    table = summarize(per_config, len(queries))

    misses = [
        q["query"]
        for q, r in zip(queries, per_config["rrf-fused"])
        if r is None
    ]
    report = (
        f"# Retrieval evaluation\n\n{len(queries)} hand-labeled queries "
        f"(eval/queries.jsonl), top-{TOP_K} retrieval.\n\n{table}\n\n"
        "The app uses rrf-fused (reciprocal-rank fusion of dense + BM25).\n"
        "`prefilter` is the previous design (FTS candidate filter + dense\n"
        "rerank), kept for comparison.\n\n"
        f"rrf-fused misses ({len(misses)}):\n"
        + "".join(f"- {m}\n" for m in misses)
    )
    RESULTS.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
