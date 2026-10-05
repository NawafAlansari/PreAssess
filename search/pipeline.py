"""
End to end: complaint -> query versions -> first stage -> features -> ranker -> (LLM reorder).

    s = Searcher.open()                         # indexes built by `python -m search build`
    out = s.search("neighbour runs a car repair shop in his driveway", llm=my_llm)
    out.candidates     # 30 sections, first-stage order
    out.X              # 41 features per candidate (for a trained ranker)

Ranking with a trained LambdaMART model and the LLM reorder are separate calls
(`search.ranker.rank`, `search.cascade.reorder`), so each tier can be served and
evaluated on its own: no LLM -> first stage only or the ranker on words-only
features; LLM rewrite -> ranker; + LLM reorder of the top 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from search import config, features, first_stage
from search.corpus import Corpus
from search.dense import DenseIndex, Encoder
from search.lexical import LexicalIndex
from search.query import LLM, QueryVersions, understand


def index_paths(index_dir: Path = config.INDEX_DIR, encoder: str = config.ENCODER) -> tuple[Path, Path]:
    tag = encoder.rstrip("/").split("/")[-1]
    return index_dir / "fts_fields.db", index_dir / f"vectors_{tag}.npy"


@dataclass
class Result:
    versions: QueryVersions
    stage: first_stage.FirstStage
    X: np.ndarray

    @property
    def candidates(self) -> list[str]:
        return self.stage.candidates


class Searcher:
    def __init__(self, corpus: Corpus, lexical: LexicalIndex, dense: DenseIndex, encoder, priors: features.Priors | None = None):
        self.corpus, self.lexical, self.dense, self.encoder, self.priors = corpus, lexical, dense, encoder, priors

    @classmethod
    def open(cls, index_dir: Path = config.INDEX_DIR, encoder: str = config.ENCODER, priors: features.Priors | None = None) -> "Searcher":
        corpus = Corpus()
        fts, vec = index_paths(index_dir, encoder)
        return cls(corpus, LexicalIndex(fts), DenseIndex.load(corpus.ids, vec), Encoder(encoder), priors)

    def search(self, complaint: str, llm: LLM | None = None, versions: QueryVersions | None = None,
               category: str = "none", flags=(0.0, 0.0)) -> Result:
        v = versions or understand(complaint, llm)
        stage = first_stage.run(v, self.corpus, self.lexical, self.dense, self.encoder)
        base = features.base(stage, v, self.corpus, self.dense, flags)
        prior = (self.priors.features(stage.candidates, category) if self.priors
                 else np.full((len(stage.candidates), len(features.PRIOR_NAMES)), np.nan, dtype=np.float32))
        X = features.matrix(base, prior, features.overlap(stage.candidates, v.words, v.rewrite, self.corpus),
                            features.facets(stage, v, self.corpus, self.lexical, self.dense, self.encoder))
        return Result(v, stage, X)
