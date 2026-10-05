"""
Optional last step: an LLM reads the complaint and the ranker's top 10 sections and
reorders them. It catches what features can't (a section that matches in words but
governs a different situation). It only touches the top 10 because LLM calls are slow
and cost money: cheap models on many candidates, the expensive one on few.

In the evaluation this was Claude Sonnet with the prompt below (adapted to one
complaint per call). Sections it omits keep their ranker order after the ones it ranks.
"""

from __future__ import annotations

import json
import re

from search import config
from search.corpus import Corpus
from search.query import LLM

PROMPT = """You are the final ranking step of a search engine over the Seattle Municipal Code. Below is a resident's \
complaint and {n} candidate code sections (citation, chapter, heading, opening text), in the order a ranking model \
put them. Put the sections in the order a city code-compliance expert would want to read them: first the section(s) \
that actually govern the resident's issue (the rule that is violated or that answers the question), then sections \
with important supporting parts (definitions, penalties, procedures, exceptions that apply), last the ones that are \
off-topic. Judge only from the complaint and the candidate texts.

Complaint: {complaint}

Candidates:
{candidates}

Return a JSON list of ALL {n} citations, best first, no duplicates.
JSON:"""


def reorder(complaint: str, ranking: list[str], corpus: Corpus, llm: LLM, top: int = config.CASCADE_TOP) -> list[str]:
    head = ranking[:top]
    cands = "\n".join(json.dumps(corpus.section_text(s)) for s in head)
    reply = llm(PROMPT.format(n=len(head), complaint=complaint, candidates=cands))
    m = re.search(r"\[.*\]", reply, flags=re.S)
    order = [s for s in (json.loads(m.group(0)) if m else []) if s in head]
    order = list(dict.fromkeys(order))
    return order + [s for s in ranking if s not in order]
