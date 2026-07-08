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

from groq import Groq

from .citation_check import audit_report
from .retriever import GroundedRetriever, RetrievalResult


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"

# The original pin (llama-3.1-70b-versatile) was decommissioned by Groq on
# 2025-01-24; llama-3.3-70b-versatile is its live successor. Verified against
# the Groq models API. Override via GROQ_MODEL when the roster rotates.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

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

        evidence_blocks = []
        for label, hits in evidence.items():
            if not hits:
                continue
            snippets = []
            for hit in hits:
                # Never expose internal chunk ids as citation labels: the model
                # copies whatever label it sees into the report.
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
            evidence_blocks.append(f"{label.upper()}:\n" + "\n\n".join(snippets))

        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence found."

        instructions = """
You are a civic compliance assistant. Use only the evidence provided.
- Summarize requirements in plain language for the resident.
- Cite each requirement inline using [SMC chapter.section].
- If evidence is missing for a checklist item, state that it needs confirmation.
- Keep the tone practical and friendly; no legal disclaimers.
""".strip()

        prompt = f"""
{instructions}

Property context:
{facts_block}

Municipal code evidence:
{evidence_text}
""".strip()
        return prompt

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
