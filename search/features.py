"""
The 41 ranker features for each (complaint, candidate section).

  base (18)   fused rank and score; the section's rank in each of the 5 lists (101 if
              absent); its best-chunk cosine to the words, rewrite and hypothetical
              passage; agreement (how many lists have it in their top 10); title and
              chapter number; chunk count and length; complaint length; two complaint flags
  priors (3)  section popularity (smoothed relevance rate in training data, and log
              count) and complaint-category × chapter prior
  overlap (4) share of the complaint's / rewrite's content words in the section
              heading / text
  facets (16) per facet query (rule, definition, penalty, enforcement, exceptions):
              the section's dense rank, BM25 rank and best cosine; plus which facet fits best

Priors use training labels. A prior must be computed exactly as it could be at
serving time: with `leave_one_out=True`, a training complaint's own labels are
excluded from its features. The evaluated model used in-sample priors (each
complaint's own label was counted in its own popularity feature), which taught it
that never-seen sections are never relevant and hurt rare issues; leave-one-out
improved rare issues in cross-validation (rank 1 0.18 -> 0.33) at a small cost on
common ones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from search import config
from search.corpus import Corpus, chapter_of, content_words
from search.dense import DenseIndex
from search.first_stage import FirstStage
from search.lexical import LexicalIndex
from search.query import QueryVersions

BASE_NAMES = [
    "fused_rank", "fused_score", "rank_dense_words", "rank_dense_rewrite", "rank_dense_hyde", "rank_bm25_words",
    "rank_bm25_rewrite", "cos_words", "cos_rewrite", "cos_hyde", "agree_top10", "title", "chapter", "n_chunks",
    "sec_chars", "q_chars", "q_ai_drafted", "q_statute_words",
]
PRIOR_NAMES = ["pop_rate", "pop_log_count", "category_chapter_rate"]
OVERLAP_NAMES = ["words_in_heading", "words_in_text", "rewrite_in_heading", "rewrite_in_text"]
FACET_NAMES = (
    [f"facet_{f}_dense_rank" for f in config.FACETS]
    + [f"facet_{f}_bm25_rank" for f in config.FACETS]
    + [f"facet_{f}_cos" for f in config.FACETS]
    + ["best_facet"]
)
NAMES = BASE_NAMES + PRIOR_NAMES + OVERLAP_NAMES + FACET_NAMES      # 41


def _num(part: str) -> float:
    return float(part) if part.isdigit() else -1.0


def base(stage: FirstStage, versions: QueryVersions, corpus: Corpus, dense: DenseIndex, flags=(0.0, 0.0)) -> np.ndarray:
    names = [n for n, _ in config.LIST_WEIGHTS]
    section_lists = [corpus.sections(stage.lists[n]) for n in names]
    rows = []
    for r, s in enumerate(stage.candidates, start=1):
        idx = corpus.section_rows[s]
        ranks = [(lst.index(s) + 1 if s in lst else config.ABSENT_RANK) for lst in section_lists]
        cos = [dense.best_cosine(stage.qvecs[v], idx) for v in ("words", "rewrite", "hyde")]
        p = s.split(".")
        rows.append([
            r, stage.fused[stage.representative[s]], *ranks, *cos, sum(k <= 10 for k in ranks),
            _num(p[0]), _num(p[1]) if len(p) > 1 else -1.0, len(idx), corpus.section_chars(s),
            len(versions.words), *flags,
        ])
    return np.array(rows, dtype=np.float32)


def overlap(candidates: list[str], words: str, rewrite: str, corpus: Corpus) -> np.ndarray:
    heads, bodies = corpus.section_words
    tw, tr = content_words(words), content_words(rewrite)
    rows = []
    for s in candidates:
        h, b = heads.get(s, set()), bodies.get(s, set())
        rows.append([len(tw & h) / max(1, len(tw)), len(tw & b) / max(1, len(tw)), len(tr & h) / max(1, len(tr)), len(tr & b) / max(1, len(tr))])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def facets(stage: FirstStage, versions: QueryVersions, corpus: Corpus, lexical: LexicalIndex, dense: DenseIndex, encoder) -> np.ndarray:
    if not versions.has_facets:
        return np.full((len(stage.candidates), len(FACET_NAMES)), np.nan, dtype=np.float32)
    texts = [versions.facets[f] for f in config.FACETS]
    vecs = encoder.encode(texts)
    dl = [corpus.sections(dense.search(v)) for v in vecs]
    bl = [corpus.sections(lexical.search(t)) for t in texts]
    rows = []
    for s in stage.candidates:
        idx = corpus.section_rows[s]
        cos = [dense.best_cosine(v, idx) for v in vecs]
        rows.append([(x.index(s) + 1 if s in x else config.ABSENT_RANK) for x in dl]
                    + [(x.index(s) + 1 if s in x else config.ABSENT_RANK) for x in bl]
                    + cos + [float(np.argmax(cos))])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------- priors
@dataclass
class Priors:
    """Smoothed relevance rates learned from graded training complaints.

    labels:     complaint id -> {section: grade 0-3}
    categories: complaint id -> the complaint's City record type (e.g. "Noise")
    """
    rel: dict
    n: dict
    cat_rel: dict
    cat_n: dict
    ch_rel: dict
    ch_n: dict
    p0: float
    alpha: float = config.PRIOR_ALPHA

    @classmethod
    def fit(cls, labels: dict[str, dict[str, int]], categories: dict[str, str], alpha: float = config.PRIOR_ALPHA) -> "Priors":
        rel, n, crel, cn, chr_, chn = (defaultdict(float) for _ in range(6))
        for q, grades in labels.items():
            c = categories.get(q, "none")
            for s, g in grades.items():
                r = float(g >= config.RELEVANT_GRADE)
                ch = chapter_of(s)
                rel[s] += r; n[s] += 1; crel[(c, ch)] += r; cn[(c, ch)] += 1; chr_[ch] += r; chn[ch] += 1
        p0 = sum(rel.values()) / max(1, sum(n.values()))
        return cls(rel, n, crel, cn, chr_, chn, p0, alpha)

    def features(self, candidates: list[str], category: str, exclude: dict[str, int] | None = None) -> np.ndarray:
        """`exclude`: this complaint's own labels, removed for leave-one-out."""
        ex = exclude or {}
        a = self.alpha
        rows = []
        for s in candidates:
            ch = chapter_of(s)
            own = float(ex.get(s, -1) >= config.RELEVANT_GRADE) if s in ex else 0.0
            own_n = 1.0 if s in ex else 0.0
            own_ch = [float(g >= config.RELEVANT_GRADE) for t, g in ex.items() if chapter_of(t) == ch]
            rel_s, n_s = self.rel[s] - own, self.n[s] - own_n
            ch_rel, ch_n = self.ch_rel[ch] - sum(own_ch), self.ch_n[ch] - len(own_ch)
            c_rel, c_n = self.cat_rel[(category, ch)] - sum(own_ch), self.cat_n[(category, ch)] - len(own_ch)
            p_ch = (ch_rel + a * self.p0) / (ch_n + a)
            rows.append([(rel_s + a * self.p0) / (n_s + a), np.log1p(rel_s), (c_rel + a * p_ch) / (c_n + a)])
        return np.array(rows, dtype=np.float32).reshape(-1, 3)


def matrix(base_x: np.ndarray, prior_x: np.ndarray, overlap_x: np.ndarray, facet_x: np.ndarray) -> np.ndarray:
    return np.hstack([base_x, prior_x, overlap_x, facet_x])
