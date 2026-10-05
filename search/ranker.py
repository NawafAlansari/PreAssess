"""
LambdaMART: gradient-boosted trees trained with LightGBM's lambdarank objective.

Each complaint is one query group; the model only compares sections within the same
complaint. For every pair in the wrong order the gradient is weighted by how much
swapping them would change nDCG, so mistakes at the top cost the most. Grades 0-3
become gains 0, 1, 3, 7 (LightGBM's default 2^g - 1).

Only judged candidates are used for training: with ~3 relevant sections per
complaint, an unjudged candidate is often relevant, and calling it 0 would teach
false negatives.

macOS note: PyTorch and LightGBM ship different OpenMP runtimes; using both in one
process can crash. Compute features and rank in separate processes if that happens
(`python -m search` does).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from search import config


@dataclass
class Group:
    candidates: list[str]          # sections, first-stage order
    X: np.ndarray                  # one feature row per candidate
    labels: dict[str, int] | None = None


def train(groups: list[Group], params: dict | None = None):
    import lightgbm as lgb

    Xs, ys, sizes = [], [], []
    for g in groups:
        keep = [j for j, s in enumerate(g.candidates) if g.labels and s in g.labels]
        if len(keep) < 2:
            continue
        Xs.append(g.X[keep])
        ys.append([g.labels[g.candidates[j]] for j in keep])
        sizes.append(len(keep))
    model = lgb.LGBMRanker(**(params or config.LGBM_PARAMS))
    model.fit(np.vstack(Xs), np.concatenate(ys), group=sizes)
    return model


def rank(model, group: Group) -> list[str]:
    scores = model.predict(group.X)
    return [group.candidates[j] for j in np.argsort(-scores, kind="stable")]


def save(model, path: Path) -> None:
    model.booster_.save_model(str(path))


def load(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))
