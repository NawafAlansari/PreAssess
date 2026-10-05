"""
High-level helper that pairs the semantic retriever with Groq LLM calls.

This module demonstrates a simple RAG workflow: fetch grounded municipal code
snippets, compose a prompt that mixes parcel details + user inputs + evidence,
and ask the LLM for a resident-friendly report with inline citations.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

from .citation_check import _matches_chunk, audit_report, parse_citations
from .retriever import GroundedRetriever, RetrievalResult


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"

# Any OpenAI-compatible endpoint works: Groq (default), OpenAI, a local
# Ollama, or a self-hosted gateway. Configure with LLM_BASE_URL + LLM_API_KEY
# + LLM_MODEL; the GROQ_* names are kept as fallbacks for compatibility.
# (llama-3.1-70b-versatile, the original pin, was decommissioned by Groq on
# 2025-01-24; llama-3.3-70b-versatile is its live successor.)
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = (
    os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
)


def resolve_api_key() -> Optional[str]:
    return os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")


def format_evidence(evidence: Dict[str, List["RetrievalResult"]]) -> str:
    """Evidence blocks for prompts. Never exposes internal chunk ids as
    citation labels: the model copies whatever label it sees into its answer."""
    blocks = []
    for label, hits in evidence.items():
        if not hits:
            continue
        snippets = []
        for hit in hits:
            citation = (
                hit.full_citation
                or hit.metadata.get("section_citation")
                or hit.metadata.get("chapter_citation")
                or "uncited"
            )
            heading = hit.section_heading or ""
            snippets.append(
                f"[{citation}] {heading}\n{textwrap.shorten(hit.text, width=1200, placeholder=' …')}"
            )
        blocks.append(f"{label.upper()}:\n" + "\n\n".join(snippets))
    return "\n\n".join(blocks) if blocks else "No evidence found."


FOLLOWUP_SYSTEM = """
You are a civic compliance research assistant answering a follow-up question in
an ongoing conversation about a Seattle property project. Use ONLY the
municipal code evidence provided in the latest message.
- Cite inline using [SMC chapter.section]; never invent citations.
- If the evidence does not answer the question, say so plainly.
- Information voice ("the code requires/allows ..."), not personal directives.
- Plain text only, no markdown syntax. Be concise: a short paragraph or two.
""".strip()


@dataclass
class EvidenceRequest:
    label: str
    query: str
    title_number: Optional[int] = None
    section_prefix: Optional[str] = None
    chunk_types: Optional[List[str]] = None
    fts_query: Optional[str] = None
    top_k: int = 4


class SeattleReportAgent:
    def __init__(
        self,
        retriever: GroundedRetriever,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.model = model or DEFAULT_MODEL
        api_key = api_key or resolve_api_key()
        if not api_key:
            raise RuntimeError("LLM_API_KEY (or GROQ_API_KEY) not set")
        self.client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)

    def gather_evidence(self, requests: Iterable[EvidenceRequest]) -> Dict[str, List[RetrievalResult]]:
        # Fused (dense + BM25, RRF) retrieval: best MRR and recall@1 of the
        # configurations measured in eval/results.md.
        evidence: Dict[str, List[RetrievalResult]] = {}
        for req in requests:
            hits = self.retriever.search_fused(
                req.query,
                top_k=req.top_k,
                title_number=req.title_number,
                section_prefix=req.section_prefix,
                chunk_types=req.chunk_types,
            )
            evidence[req.label] = hits
        return evidence

    def build_prompt(
        self,
        *,
        address_profile: Dict[str, object],
        user_inputs: Dict[str, object],
        evidence: Dict[str, List[RetrievalResult]],
    ) -> str:
        facts_block = json.dumps(
            {"address": address_profile, "user_inputs": user_inputs},
            indent=2,
            ensure_ascii=False,
        )

        evidence_text = format_evidence(evidence)

        instructions = """
You are a civic compliance assistant. Use only the evidence provided.
- Summarize requirements in plain language for the resident.
- Cite each requirement inline using [SMC chapter.section].
- If evidence is missing for a checklist item, state that it needs confirmation.
- Keep the tone practical and friendly; no legal disclaimers.
- Describe what the code provides, in information voice: "the code requires /
  allows / limits X [SMC ...]". Do not give personal directives or advice
  ("you must", "you should hire"); the reader decides what to do.
- Write plain text only: no markdown syntax (no asterisks, hashes, or backticks).
  Structure with short paragraphs and simple numbered lists like "1." on their
  own lines.
""".strip()

        prompt = f"""
{instructions}

Property context:
{facts_block}

Municipal code evidence:
{evidence_text}
""".strip()
        return prompt

    def _history_anchor_hits(
        self,
        history: Optional[List[Dict[str, str]]],
        *,
        max_citations: int = 4,
        per_citation: int = 2,
    ) -> List[RetrievalResult]:
        """Chunks for sections the conversation has already cited. A follow-up
        usually drills into what the report just said, and fresh retrieval on
        the follow-up's wording alone can miss the very section under
        discussion (e.g. "DADU" phrasing vs the code's "accessory dwelling
        unit"). Most recent citations win the cap."""
        citations: List[str] = []
        for turn in history or []:
            if turn.get("role") == "assistant":
                citations.extend(parse_citations(str(turn.get("content", ""))))
        unique = list(dict.fromkeys(reversed(citations)))[:max_citations]
        hits: List[RetrievalResult] = []
        for citation in unique:
            found = 0
            for chunk_id, meta in self.retriever.metadata.items():
                if found >= per_citation:
                    break
                if _matches_chunk(citation, meta):
                    hits.append(self.retriever._result(chunk_id, 0.0))
                    found += 1
        return hits

    def answer_followup(
        self,
        *,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        address_profile: Optional[Dict[str, object]] = None,
        top_k: int = 4,
    ) -> Dict[str, object]:
        """One audited conversational turn: retrieve fresh evidence for the
        question, anchor it with sections the conversation already cited,
        answer grounded in both, audit the citations."""
        fresh = self.retriever.search_fused(question, top_k=top_k)
        fresh_ids = {hit.chunk_id for hit in fresh}
        anchors = [
            hit
            for hit in self._history_anchor_hits(history)
            if hit.chunk_id not in fresh_ids
        ]
        hits = fresh + anchors
        evidence: Dict[str, List[RetrievalResult]] = {"followup": fresh}
        if anchors:
            evidence["conversation"] = anchors

        messages: List[Dict[str, str]] = [{"role": "system", "content": FOLLOWUP_SYSTEM}]
        for turn in (history or [])[-8:]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            content = str(turn.get("content", ""))[:2000]
            if content:
                messages.append({"role": role, "content": content})

        facts = ""
        if address_profile:
            facts = "Property context:\n" + json.dumps(address_profile, ensure_ascii=False)[:2000] + "\n\n"
        messages.append(
            {
                "role": "user",
                "content": f"{facts}Municipal code evidence:\n{format_evidence(evidence)}\n\nQuestion: {question}",
            }
        )

        completion = self.client.chat.completions.create(
            model=self.model, temperature=0.2, messages=messages
        )
        text = completion.choices[0].message.content or ""
        audit = audit_report(text, hits, corpus_lookup=self.retriever.has_citation)
        return {
            "answer": text,
            "citation_audit": [v.to_dict() for v in audit.verdicts],
            "grounded_ratio": audit.grounded_ratio,
            "evidence": {
                label: [hit.metadata for hit in label_hits]
                for label, label_hits in evidence.items()
            },
        }

    def generate_report(
        self,
        *,
        address_profile: Dict[str, object],
        user_inputs: Dict[str, object],
        evidence_requests: Iterable[EvidenceRequest],
    ) -> Dict[str, object]:
        evidence = self.gather_evidence(evidence_requests)
        prompt = self.build_prompt(
            address_profile=address_profile,
            user_inputs=user_inputs,
            evidence=evidence,
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content

        retrieved_hits = [hit for hits in evidence.values() for hit in hits]
        audit = audit_report(
            text,
            retrieved_hits,
            corpus_lookup=self.retriever.has_citation,
        )

        return {
            "report": text,
            "prompt": prompt,
            "evidence": {
                label: [result.metadata for result in hits]
                for label, hits in evidence.items()
            },
            "citation_audit": [verdict.to_dict() for verdict in audit.verdicts],
            "grounded_ratio": audit.grounded_ratio,
        }


def demo() -> None:
    """
    Example invocation for manual testing (requires GROQ_API_KEY set).
    """
    retriever = GroundedRetriever(
        embeddings_path=DATA_DIR / "smc_embeddings.npz",
        chunks_path=DATA_DIR / "smc_chunks.jsonl",
        sqlite_path=DATA_DIR / "smc_ground_truth.db",
    )
    agent = SeattleReportAgent(retriever=retriever)
    address_profile = {
        "address": "1234 Example Ave N, Seattle, WA",
        "zoning": "LR1",
        "lot_size_sqft": 4200,
    }
    user_inputs = {
        "project": "Add an accessory dwelling unit and plant new street trees.",
        "questions": [
            "What setbacks and lot coverage apply?",
            "What landscaping standards apply?",
            "What building permit do I need?",
        ],
    }
    # Every request targets an ingested title (22 and 23 only). Title 25
    # (trees) is not in the corpus, so we do not ask for it here.
    evidence_requests = [
        EvidenceRequest(
            label="zoning",
            query="accessory dwelling unit development standards setbacks lot coverage",
            title_number=23,
            section_prefix="23.44",
            chunk_types=["section"],
        ),
        EvidenceRequest(
            label="landscaping",
            query="tree planting and landscaping standards",
            title_number=23,
            section_prefix="23.45",
            chunk_types=["section"],
        ),
        EvidenceRequest(
            label="permits",
            query="building permit application requirements",
            title_number=22,
            section_prefix="22.801",
            chunk_types=["section"],
        ),
    ]
    bundle = agent.generate_report(
        address_profile=address_profile,
        user_inputs=user_inputs,
        evidence_requests=evidence_requests,
    )
    print(bundle["report"])
    print("\n--- citation audit ---")
    print(f"grounded_ratio: {bundle['grounded_ratio']:.2f}")
    for verdict in bundle["citation_audit"]:
        print(f"  {verdict['citation']}: {verdict['status']}")


if __name__ == "__main__":
    demo()
