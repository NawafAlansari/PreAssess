"""
Ranking metrics over graded labels (0-3; relevant = 2 or 3), and paired bootstrap CIs.

rank 1     is the top result relevant? (P@1)
P@k        share of the top k that is relevant
hit@k      is at least one relevant section in the top k?
nDCG@k     order quality with graded gains (linear gain, 1/log2(position + 1)),
           divided by the best possible order of the judged sections
Unjudged sections count as not relevant, so scores are lower bounds.
"""

from __future__ import annotations

import math

import numpy as np

from search.config import RELEVANT_GRADE


def rank1(ranking: list[str], grades: dict[str, int]) -> float:
    return float(bool(ranking) and grades.get(ranking[0], 0) >= RELEVANT_GRADE)


def precision_at(ranking: list[str], grades: dict[str, int], k: int) -> float:
    return sum(grades.get(s, 0) >= RELEVANT_GRADE for s in ranking[:k]) / k


def hit_at(ranking: list[str], grades: dict[str, int], k: int) -> float:
    return float(any(grades.get(s, 0) >= RELEVANT_GRADE for s in ranking[:k]))


def ndcg_at(ranking: list[str], grades: dict[str, int], k: int) -> float:
    dcg = sum(grades.get(s, 0) / math.log2(i + 2) for i, s in enumerate(ranking[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0


def recall_at(ranking: list[str], grades: dict[str, int], k: int) -> float:
    relevant = {s for s, g in grades.items() if g >= RELEVANT_GRADE}
    return len(relevant & set(ranking[:k])) / len(relevant) if relevant else 0.0


def paired_bootstrap(a: list[float], b: list[float], resamples: int = 3000, seed: int = 0) -> dict:
    """Mean of b - a over the same queries, with a 95% interval from resampling queries."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
