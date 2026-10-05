"""
First stage: five retrieval lists fused with weighted reciprocal rank fusion.

BM25 scores and cosine similarities are on different scales (and BM25 scores aren't
comparable across queries), so the lists are fused by rank:

    score(chunk) = Σ_lists  weight / (60 + rank of the chunk in that list)

k = 60 flattens the top, so a chunk that several lists agree on beats a chunk that
is first in only one. The top 100 fused chunks collapse to sections (each section
takes its best chunk), and the top 30 sections become the ranker's candidates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from search import config
from search.corpus import Corpus
from search.dense import DenseIndex
from search.lexical import LexicalIndex
from search.query import QueryVersions


def weighted_rrf(lists_weights, k: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in lists_weights:
        for r, cid in enumerate(ranking, start=1):
            scores[cid] += w / (k + r)
    return dict(scores)


@dataclass
class FirstStage:
    lists: dict[str, list[str]]        # list name -> ranked chunk ids
    fused: dict[str, float]            # chunk id -> fused score
    pool: list[str]                    # top fused chunk ids
    candidates: list[str]              # top sections, best first
    representative: dict[str, str]     # section -> its best chunk in the pool
    qvecs: dict[str, np.ndarray]       # query version -> query vector


def run(versions: QueryVersions, corpus: Corpus, lexical: LexicalIndex, dense: DenseIndex, encoder) -> FirstStage:
    texts = {"words": versions.words, "rewrite": versions.rewrite, "hyde": versions.hyde}
    vecs = dict(zip(texts, encoder.encode(list(texts.values()))))
    lists = {
        "dense_words": dense.search(vecs["words"]),
        "dense_rewrite": dense.search(vecs["rewrite"]),
        "dense_hyde": dense.search(vecs["hyde"]),
        "bm25_words": lexical.search(versions.words),
        "bm25_rewrite": lexical.search(versions.rewrite),
    }
    fused = weighted_rrf([(lists[name], w) for name, w in config.LIST_WEIGHTS])
    pool = sorted(fused, key=fused.get, reverse=True)[: config.POOL_CHUNKS]
    representative: dict[str, str] = {}
    for cid in pool:
        representative.setdefault(corpus.section(cid), cid)
    return FirstStage(lists, fused, pool, corpus.sections(pool)[: config.CANDIDATE_SECTIONS], representative, vecs)
