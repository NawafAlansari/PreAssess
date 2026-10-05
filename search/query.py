"""
Query understanding: turn a resident's complaint into versions the code's vocabulary can match.

Residents write "my neighbour's dog barks all night"; the code says "noise
disturbance". An LLM produces, blind (it sees only the complaint):
  - a rewrite: a 6-14 word query in the code's vocabulary,
  - a hypothetical passage (HyDE): one sentence written the way the governing
    section might read, which embeds closer to real sections than a question does,
  - facet queries: one code-style query each for the rule, the definitions, the
    penalty, enforcement, and exceptions.

Without an LLM, every version falls back to the complaint's own words.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from search import config

# Any function prompt -> completion text, e.g. a thin wrapper around an LLM API.
LLM = Callable[[str], str]

SCOPE = (
    "a search engine over the Seattle Municipal Code (the city's laws: zoning/land use, building and housing codes, "
    "tenant/landlord rules, noise, weeds and vegetation, junk storage, parking, utilities, animals, etc.)"
)

REWRITE_PROMPT = f"""You are the "query understanding" step of {SCOPE}. Residents write complaints in everyday \
language; the code is written in formal legal language. Rewrite the complaint so that it matches how the code \
itself would phrase the topic. Work only from the complaint and your general knowledge.

Return JSON with:
- "rewrite": a compact search query of 6 to 14 words in the vocabulary a municipal code would use (legal or \
administrative terms for the same concepts, e.g. "notice of entry by owner" for "landlord came in without telling \
me", "inoperable vehicle storage" for "old car that doesn't run"). No question words, no citations or section \
numbers, no invented facts.
- "hyde": one sentence (20 to 40 words) written the way the relevant code section itself would plausibly read, in \
formal regulatory language, with no specific numbers you cannot infer from the complaint.

Complaint: {{complaint}}
JSON:"""

FACET_PROMPT = f"""You are the "query understanding" step of {SCOPE}. A resident's complaint usually needs several \
kinds of code sections, not one. Write one short search query (6 to 14 words, in the vocabulary a municipal code \
would use) for EACH facet:
- "rule": the substantive requirement or prohibition the complaint is about
- "definition": the defined term(s) the rule depends on
- "penalty": penalties, fines, violations, civil enforcement amounts for this kind of issue
- "enforcement": how the City enforces it: notice of violation, inspection, complaint procedure, abatement, appeals
- "exceptions": exemptions, allowances or conditions under which it is permitted
No citations or section numbers, no invented facts. Return JSON with those five keys.

Complaint: {{complaint}}
JSON:"""


@dataclass
class QueryVersions:
    words: str                     # the resident's own text
    rewrite: str = ""
    hyde: str = ""
    facets: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # No-LLM fallback: every version is the complaint's words.
        self.rewrite = self.rewrite or self.words
        self.hyde = self.hyde or self.words

    @property
    def has_facets(self) -> bool:
        return all(self.facets.get(f) for f in config.FACETS)


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0)) if m else {}


def understand(complaint: str, llm: LLM | None = None, facets: bool = True) -> QueryVersions:
    if llm is None:
        return QueryVersions(complaint)
    rw = _json(llm(REWRITE_PROMPT.format(complaint=complaint)))
    fc = _json(llm(FACET_PROMPT.format(complaint=complaint))) if facets else {}
    return QueryVersions(complaint, rw.get("rewrite", ""), rw.get("hyde", ""), {f: str(fc.get(f, "")) for f in config.FACETS if fc.get(f)})
