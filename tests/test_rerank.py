"""
Offline tests for the cross-encoder reranker and its retriever integration.
All scoring is done by injected stubs — no model download, no network.
"""

from __future__ import annotations

import pytest

from smc_agents import retriever as retriever_module
from smc_agents.rerank import CrossEncoderReranker


def _keyword_scorer(keyword: str):
    """A stub CE scorer: score = count of `keyword` occurrences in the passage."""

    def score(pairs):
        return [float(passage.lower().count(keyword)) for _, passage in pairs]

    return score


@pytest.fixture()
def install_reranker():
    """Install a stub reranker module-level and restore afterwards."""
    installed = []

    def _install(scorer):
        reranker = CrossEncoderReranker(scorer=scorer)
        retriever_module.set_reranker(reranker)
        installed.append(reranker)
        return reranker

    yield _install
    retriever_module.set_reranker(None)


def test_stub_scorer_empty_pairs_short_circuits():
    reranker = CrossEncoderReranker(scorer=lambda pairs: [1.0] * len(pairs))
    assert reranker.score([]) == []


def test_reranker_reorders_fused_results(retriever, install_reranker):
    # Base fused order puts the setback chunk first for this query.
    base = retriever.search_fused("setback dwelling permit", top_k=4, rerank=False)
    assert base[0].chunk_id == "c-2344-010"

    # A CE that rewards "permit" mentions must promote the permit chunk.
    install_reranker(_keyword_scorer("permit"))
    reranked = retriever.search_fused("setback dwelling permit", top_k=4, rerank=True)
    assert reranked[0].chunk_id == "c-22801-050"
    assert [h.chunk_id for h in reranked] != [h.chunk_id for h in base]


def test_rerank_arg_wins_over_env(retriever, install_reranker, monkeypatch):
    install_reranker(_keyword_scorer("permit"))
    monkeypatch.setenv("PREASSESS_RERANK", "1")
    # Explicit False overrides the env flag → no reranking.
    hits = retriever.search_fused("setback dwelling permit", top_k=4, rerank=False)
    assert hits[0].chunk_id == "c-2344-010"


def test_rerank_env_flag_enables(retriever, install_reranker, monkeypatch):
    install_reranker(_keyword_scorer("permit"))
    monkeypatch.setenv("PREASSESS_RERANK", "1")
    hits = retriever.search_fused("setback dwelling permit", top_k=4)
    assert hits[0].chunk_id == "c-22801-050"


def test_rerank_default_off(retriever, install_reranker, monkeypatch):
    monkeypatch.delenv("PREASSESS_RERANK", raising=False)
    install_reranker(_keyword_scorer("permit"))
    # Default (no arg, no env) leaves fused order untouched.
    hits = retriever.search_fused("setback dwelling permit", top_k=4)
    assert hits[0].chunk_id == "c-2344-010"


def test_rerank_respects_pool_size(retriever, install_reranker, monkeypatch):
    # The reranker must only see the fused top-RERANK_POOL candidates.
    seen = {}

    def spy(pairs):
        seen["n"] = len(pairs)
        return [0.0] * len(pairs)

    monkeypatch.setattr(retriever_module, "RERANK_POOL", 2)
    install_reranker(spy)
    retriever.search_fused("setback dwelling permit height parking", top_k=1, rerank=True)
    assert seen["n"] == 2


def test_retriever_unchanged_when_off(retriever, install_reranker):
    # Installing a reranker but leaving it off must not alter results at all.
    install_reranker(_keyword_scorer("permit"))
    off = retriever.search_fused("setback rules for a dwelling", top_k=3, rerank=False)
    assert off[0].chunk_id == "c-2344-010"
