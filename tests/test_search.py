"""
Offline tests for the search package: a tiny hand-built corpus and a stub encoder,
no model downloads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from search import features, first_stage, lexical, metrics
from search.corpus import Corpus, section_of
from search.dense import DenseIndex
from search.lexical import LexicalIndex, or_query
from search.query import QueryVersions

CHUNKS = [
    ("c1", "23.54.015", "Required parking", "LAND USE CODE", "Off-street parking spaces are required for each dwelling unit."),
    ("c2", "23.54.015", "Required parking", "LAND USE CODE", "Parking shall be located at least five feet from the lot line."),
    ("c3", "25.08.500", "Noise disturbance", "NOISE CONTROL", "It is unlawful to make a noise disturbance at night."),
    ("c4", "23.44.014", "Yards", "LAND USE CODE", "Front yard depth shall not be less than the average of adjacent yards."),
]
AXES = {"parking": 0, "noise": 1, "yard": 2}


class StubEncoder:
    """Maps keywords to fixed axes, so similarity is exact and deterministic."""

    def encode(self, texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, a in AXES.items():
                if w in t.lower():
                    out[i, a] = 1.0
            out[i, 3] = 0.1
            out[i] /= np.linalg.norm(out[i])
        return out


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "chunks.jsonl"
    with path.open("w") as fh:
        for cid, sec, head, chap, text in CHUNKS:
            fh.write(json.dumps({"chunk_id": cid, "section_citation": sec, "section_heading": head, "chapter_title": chap,
                                 "chapter_citation": ".".join(sec.split(".")[:2]), "title_number": int(sec.split(".")[0]), "text": text}) + "\n")
    np.savez(tmp_path / "ids.npz", chunk_ids=np.array([c[0] for c in CHUNKS]))
    return Corpus(path, tmp_path / "ids.npz")


@pytest.fixture()
def indexes(corpus, tmp_path):
    enc = StubEncoder()
    dense = DenseIndex(corpus.ids, enc.encode([corpus.context_text(c) for c in corpus.ids]))
    return LexicalIndex(lexical.build(corpus, tmp_path / "fts.db")), dense, enc


def test_or_query_keeps_words_of_three_letters():
    assert or_query("My dog, IS loud at 3am!!") == "dog OR loud"


def test_section_rolls_up_subsections():
    assert section_of("23.47A.004.D.3") == "23.47A.004"


def test_context_text_has_breadcrumb(corpus):
    assert corpus.context_text("c1").startswith("Title 23 > Chapter 23.54 LAND USE CODE > Section 23.54.015 Required parking\n")


def test_weighted_rrf_rewards_agreement():
    s = first_stage.weighted_rrf([(["a", "b"], 1.0), (["b", "c"], 1.0)])
    assert max(s, key=s.get) == "b"


def test_keyword_search_finds_matches_only(indexes):
    lex, _, _ = indexes
    assert set(lex.search("parking")) == {"c1", "c2"}
    assert lex.search("zzz") == []


def test_first_stage_collapses_to_sections(corpus, indexes):
    lex, dense, enc = indexes
    st = first_stage.run(QueryVersions("where can I park, parking"), corpus, lex, dense, enc)
    assert st.candidates[0] == "23.54.015"
    assert len(st.candidates) == len(set(st.candidates))


def test_base_features_shape_and_absent_rank(corpus, indexes):
    lex, dense, enc = indexes
    v = QueryVersions("parking")
    st = first_stage.run(v, corpus, lex, dense, enc)
    X = features.base(st, v, corpus, dense)
    assert X.shape == (len(st.candidates), len(features.BASE_NAMES))
    noise = st.candidates.index("25.08.500")
    assert X[noise, features.BASE_NAMES.index("rank_bm25_words")] == 101


def test_feature_names_total_41():
    assert len(features.NAMES) == 41


def test_leave_one_out_removes_own_label():
    labels = {"q1": {"23.54.015": 3}, "q2": {"23.54.015": 0}}
    pri = features.Priors.fit(labels, {"q1": "Land Use", "q2": "Land Use"})
    in_sample = pri.features(["23.54.015"], "Land Use")[0, 0]
    loo = pri.features(["23.54.015"], "Land Use", exclude=labels["q1"])[0, 0]
    assert loo < in_sample            # q1's own "relevant" no longer inflates the prior


def test_metrics():
    g = {"a": 3, "b": 1, "c": 2}
    assert metrics.rank1(["a", "b"], g) == 1.0
    assert metrics.precision_at(["a", "b", "c"], g, 3) == pytest.approx(2 / 3)
    assert metrics.hit_at(["b", "x", "y"], g, 3) == 0.0
    assert metrics.ndcg_at(["a", "c", "b"], g, 3) == pytest.approx(1.0)
