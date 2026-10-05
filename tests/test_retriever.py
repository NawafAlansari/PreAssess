import logging

import numpy as np
import pytest


def test_dense_search_ranks_matching_chunk_first(retriever):
    hits = retriever.search("setback rules for a dwelling", top_k=2)
    assert hits[0].chunk_id == "c-2344-010"
    assert hits[0].score > hits[1].score


def test_title_filter_excludes_other_titles(retriever):
    hits = retriever.search("permit", top_k=4, title_number=23)
    assert hits
    assert all(h.title_number == 23 for h in hits)


def test_section_prefix_filter(retriever):
    hits = retriever.search("height", top_k=4, section_prefix="23.44")
    assert hits
    assert all(str(h.metadata["section_citation"]).startswith("23.44") for h in hits)


def test_chunk_type_filter(retriever):
    hits = retriever.search("parking", top_k=4, chunk_types=["definition"])
    assert [h.chunk_id for h in hits] == ["c-2345-005"]


def test_fts_prefilter_limits_candidates(retriever):
    hits = retriever.search("permit", top_k=4, fts_query="permit")
    assert [h.chunk_id for h in hits] == ["c-22801-050"]


def test_impossible_title_returns_empty_and_warns(retriever, caplog):
    with caplog.at_level(logging.WARNING):
        hits = retriever.search("tree removal", top_k=3, title_number=25)
    assert hits == []
    assert any("title_number=25" in r.message for r in caplog.records)
    assert any("[22, 23]" in r.message for r in caplog.records)


def test_top_k_respected(retriever):
    hits = retriever.search("setback dwelling permit parking height", top_k=2)
    assert len(hits) == 2


def test_results_carry_citation_metadata(retriever):
    hit = retriever.search("setback", top_k=1)[0]
    assert hit.full_citation == "SMC 23.44.010"
    assert hit.section_heading == "Setback requirements"
    assert hit.metadata["chapter_citation"] == "23.44"


def test_ingested_titles_and_citation_sets(retriever):
    assert retriever.ingested_titles == [22, 23]
    assert "23.44.010" in retriever.section_citations
    assert "22.801" in retriever.chapter_citations


@pytest.mark.parametrize(
    "citation,expected",
    [
        ("23.44.010", True),          # exact section
        ("23.44", True),              # chapter
        ("23.44.010.C", True),        # subsection resolves to parent section
        ("23.44.999", False),         # nonexistent section
        ("99.99.999", False),         # fabricated
        ("22.801", True),             # chapter-level
    ],
)
def test_has_citation(retriever, citation, expected):
    assert retriever.has_citation(citation) is expected


def test_encoder_injection_no_torch_needed(retriever):
    q = retriever._encode(["setback"])
    assert isinstance(q, np.ndarray)
    assert q.shape == (1, 8)


def test_fts_or_query_sanitizes_free_text():
    from smc_agents.retriever import GroundedRetriever

    assert (
        GroundedRetriever.fts_or_query("What setbacks apply? (ADU!)")
        == "what OR setbacks OR apply OR adu"
    )


def test_search_fused_returns_relevant_chunk_first(retriever):
    hits = retriever.search_fused("setback rules for a dwelling", top_k=3)
    assert hits[0].chunk_id == "c-2344-010"


def test_search_fused_promotes_lexical_only_match(retriever):
    # "submittal" appears in the permit chunk text but maps to no embedding
    # axis in the stub encoder — only the BM25 branch can find it.
    hits = retriever.search_fused("submittal documents", top_k=2)
    assert "c-22801-050" in [h.chunk_id for h in hits]


def test_search_fused_respects_filters(retriever):
    hits = retriever.search_fused("permit setback", top_k=4, title_number=23)
    assert hits
    assert all(h.title_number == 23 for h in hits)

    typed = retriever.search_fused("parking", top_k=4, chunk_types=["definition"])
    assert [h.chunk_id for h in typed] == ["c-2345-005"]


def test_search_fused_impossible_filter_returns_empty(retriever):
    assert retriever.search_fused("tree", top_k=3, title_number=25) == []


def test_search_fused_malformed_punctuation_still_works(retriever):
    hits = retriever.search_fused('"setback?!" -- (dwelling)', top_k=2)
    assert hits[0].chunk_id == "c-2344-010"
