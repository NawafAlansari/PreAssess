import pytest

from smc_agents.citation_check import (
    GROUNDED,
    IN_CORPUS_NOT_RETRIEVED,
    UNKNOWN,
    audit_report,
    parse_citations,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Per SMC 23.44.010 setbacks apply.", ["23.44.010"]),
        ("See Section 22.801 and Chapter 23.45.", ["22.801", "23.45"]),
        ("Inline [23.44.010] and [SMC 22.801.050].", ["23.44.010", "22.801.050"]),
        ("SMC Chapter 23.44 governs.", ["23.44"]),
        ("Subsection SMC 23.44.014.C.17.a is cited.", ["23.44.014.C.17.a"]),
        ("smc 23.44.010 lowercase works.", ["23.44.010"]),
        ("Trailing period SMC 23.44.010. Next sentence.", ["23.44.010"]),
        ("No citations here, just 42 and 3.14 chatter.", []),
    ],
)
def test_parse_citations(text, expected):
    assert parse_citations(text) == expected


def test_parse_deduplicates_preserving_order():
    text = "SMC 23.44.010 first, [23.44.010] again, then SMC 22.801."
    assert parse_citations(text) == ["23.44.010", "22.801"]


def test_audit_grounded_when_citation_in_evidence(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report("Setbacks per SMC 23.44.010.", hits)
    assert audit.verdicts[0].status == GROUNDED
    assert audit.verdicts[0].matched_chunk_ids == ["c-2344-010"]


def test_audit_chapter_citation_grounds_on_section_chunk(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report("See Chapter 23.44 generally.", hits)
    assert audit.verdicts[0].status == GROUNDED


def test_audit_subsection_grounds_on_parent_section(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report("Per SMC 23.44.010.C.2 the side setback is 5 feet.", hits)
    assert audit.verdicts[0].status == GROUNDED


def test_audit_in_corpus_not_retrieved(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report(
        "Permits per SMC 22.801.050.", hits, corpus_lookup=retriever.has_citation
    )
    assert audit.verdicts[0].status == IN_CORPUS_NOT_RETRIEVED
    assert audit.verdicts[0].matched_chunk_ids == []


def test_audit_unknown_for_fabricated_citation(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report(
        "As stated in SMC 99.99.999.", hits, corpus_lookup=retriever.has_citation
    )
    assert audit.verdicts[0].status == UNKNOWN


def test_audit_without_corpus_lookup_falls_back_to_unknown(retriever):
    hits = retriever.search("setback", top_k=1)
    audit = audit_report("Permits per SMC 22.801.050.", hits)
    assert audit.verdicts[0].status == UNKNOWN


def test_grounded_ratio_mixed(retriever):
    hits = retriever.search("setback", top_k=1)
    text = "SMC 23.44.010 applies. Also SMC 22.801.050 and SMC 99.99.999."
    audit = audit_report(text, hits, corpus_lookup=retriever.has_citation)
    statuses = [v.status for v in audit.verdicts]
    assert statuses == [GROUNDED, IN_CORPUS_NOT_RETRIEVED, UNKNOWN]
    assert audit.grounded_ratio == pytest.approx(1 / 3)


def test_grounded_ratio_empty_report():
    audit = audit_report("", [])
    assert audit.verdicts == []
    assert audit.grounded_ratio == 0.0


def test_audit_to_dict_shape(retriever):
    hits = retriever.search("setback", top_k=1)
    payload = audit_report("SMC 23.44.010.", hits).to_dict()
    assert set(payload) == {"verdicts", "grounded_ratio"}
    assert set(payload["verdicts"][0]) == {"citation", "status", "matched_chunk_ids"}
