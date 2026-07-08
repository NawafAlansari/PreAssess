"""
Verify SMC citations produced by the report agent against the evidence that
was actually retrieved (and, optionally, the full chunk corpus).

The report agent is *asked* to cite the Seattle Municipal Code; this module
turns that request into a check. For every citation found in the generated
text we assign one of three statuses:

- ``grounded``: the citation matches a chunk in the retrieved evidence set,
  so the model quoted something it was actually shown.
- ``in_corpus_not_retrieved``: the citation exists in the corpus but was not
  part of the evidence bundle (plausible, but not directly supported here).
- ``unknown``: the citation appears nowhere in the corpus (likely fabricated
  or from a title that was never ingested).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .retriever import RetrievalResult


CorpusLookup = Callable[[str], bool]

GROUNDED = "grounded"
IN_CORPUS_NOT_RETRIEVED = "in_corpus_not_retrieved"
UNKNOWN = "unknown"

# A citation number: 2-digit title, then dotted numeric/alpha segments.
# Examples: 22.801, 23.44.010, 23.60A.190, 23.44.014.C.17.a
_NUM = r"\d{2}\.\d+[A-Za-z]?(?:\.[0-9A-Za-z]+)*"

# One pass, two alternatives, so citations are returned in document order:
# bracketed form ("[23.44.010]", "[SMC 23.44.010]") or keyword-led form
# ("SMC 23.44.010", "Section 22.801", "SMC Chapter 23.44").
_CITATION_RE = re.compile(
    r"\[\s*(?:SMC\s+)?(" + _NUM + r")\s*\]"
    r"|(?:SMC|Section|Chapter)\s+(?:Chapter\s+)?(" + _NUM + r")",
    re.IGNORECASE,
)


@dataclass
class CitationVerdict:
    citation: str
    status: str
    matched_chunk_ids: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "citation": self.citation,
            "status": self.status,
            "matched_chunk_ids": self.matched_chunk_ids,
        }


@dataclass
class CitationAudit:
    verdicts: List[CitationVerdict]

    @property
    def grounded_ratio(self) -> float:
        if not self.verdicts:
            return 0.0
        grounded = sum(1 for v in self.verdicts if v.status == GROUNDED)
        return grounded / len(self.verdicts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "grounded_ratio": self.grounded_ratio,
        }


def parse_citations(text: str) -> List[str]:
    """Extract unique SMC citations from report text, in document order."""
    found: List[str] = []
    seen: set = set()
    for match in _CITATION_RE.finditer(text or ""):
        _add(match.group(1) or match.group(2), found, seen)
    return found


def _add(raw: str, found: List[str], seen: set) -> None:
    citation = raw.strip().rstrip(".")
    if citation and citation not in seen:
        seen.add(citation)
        found.append(citation)


def _matches_chunk(citation: str, meta: Dict[str, object]) -> bool:
    section = str(meta.get("section_citation") or "")
    chapter = str(meta.get("chapter_citation") or "")
    if section == citation or section.startswith(citation + "."):
        return True
    # Citation deeper than the chunk's section (e.g. 23.44.014.C.17.a cited,
    # chunk is section 23.44.014): the subsection is grounded by its parent.
    if section and citation.startswith(section + "."):
        return True
    if chapter == citation:
        return True
    return False


def matched_chunk_ids(citation: str, hits: Sequence[RetrievalResult]) -> List[str]:
    return [hit.chunk_id for hit in hits if _matches_chunk(citation, hit.metadata)]


def audit_report(
    text: str,
    retrieved_hits: Sequence[RetrievalResult],
    *,
    corpus_lookup: Optional[CorpusLookup] = None,
) -> CitationAudit:
    """
    Parse citations from ``text`` and classify each against the retrieved
    evidence, falling back to ``corpus_lookup`` (e.g. the retriever's
    ``has_citation``) to distinguish in-corpus misses from fabrications.
    """
    verdicts: List[CitationVerdict] = []
    for citation in parse_citations(text):
        matches = matched_chunk_ids(citation, retrieved_hits)
        if matches:
            status = GROUNDED
        elif corpus_lookup is not None and corpus_lookup(citation):
            status = IN_CORPUS_NOT_RETRIEVED
        else:
            status = UNKNOWN
        verdicts.append(CitationVerdict(citation, status, matches))
    return CitationAudit(verdicts)
