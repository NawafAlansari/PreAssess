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

from .retriever import GroundedRetriever, RetrievalResult


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
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

    def gather_evidence(self, requests: Iterable[EvidenceRequest]) -> Dict[str, List[RetrievalResult]]:
        evidence: Dict[str, List[RetrievalResult]] = {}
        for req in requests:
            hits = self.retriever.search(
                req.query,
                top_k=req.top_k,
                title_number=req.title_number,
                section_prefix=req.section_prefix,
                chunk_types=req.chunk_types,
                fts_query=req.fts_query,
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
                citation = hit.full_citation or hit.chunk_id
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
        return {
            "report": text,
            "prompt": prompt,
            "evidence": {
                label: [result.metadata for result in hits]
                for label, hits in evidence.items()
            },
        }


def demo() -> None:
    """
    Example invocation for manual testing (requires GROQ_API_KEY set).
    """
    retriever = GroundedRetriever(
        embeddings_path=Path("seattle-checker/data/processed/smc_embeddings.npz"),
        chunks_path=Path("seattle-checker/data/processed/smc_chunks.jsonl"),
        sqlite_path=Path("seattle-checker/data/processed/smc_ground_truth.db"),
    )
    agent = SeattleReportAgent(retriever=retriever)
    address_profile = {
        "address": "1234 Example Ave N, Seattle, WA",
        "zoning": "LR1",
        "lot_size_sqft": 4200,
        "tree_inventory": {"exceptional_trees": 1, "significant_trees": 3},
    }
    user_inputs = {
        "project": "Add an accessory dwelling unit and remove one hazardous tree.",
        "questions": ["Do I need a tree removal permit?", "What setbacks apply?"],
    }
    evidence_requests = [
        EvidenceRequest(
            label="zoning",
            query="accessory dwelling unit setbacks",
            title_number=23,
            section_prefix="23.44",
            chunk_types=["section"],
        ),
        EvidenceRequest(
            label="trees",
            query="tree removal permit hazardous",
            title_number=25,
            section_prefix="25.11",
            chunk_types=["section"],
        ),
        EvidenceRequest(
            label="permits",
            query="building permit accessory dwelling",
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


if __name__ == "__main__":
    demo()
