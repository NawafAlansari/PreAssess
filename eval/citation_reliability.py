"""
Citation reliability: the same model answering the same resident questions,
with and without PreAssess's retrieval + audit pipeline.

For each question we produce two answers:
  bare     - the LLM is asked to answer with SMC citations, no evidence given
             (what a resident gets from a generic chatbot)
  preassess- the full pipeline: fused retrieval -> evidence-grounded prompt

Both answers are audited with the same citation checker against the full
ingested corpus (19,419 chunks, 21 titles). We report, per system:
  citations/answer, % of cited sections that exist in the code,
  % fabricated (cited but nowhere in the code), and for the pipeline,
  % verified against the evidence actually retrieved.

Run: python -m eval.citation_reliability   (needs LLM_API_KEY / GROQ_API_KEY)
"""

from __future__ import annotations

import json
from pathlib import Path

from smc_agents.citation_check import GROUNDED, UNKNOWN, audit_report, parse_citations
from smc_agents.report_agent import EvidenceRequest, SeattleReportAgent
from smc_agents.retriever import GroundedRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"
RESULTS = REPO_ROOT / "eval/citation_reliability.md"

QUESTIONS = [
    "Do I need a permit to remove a large tree in my backyard?",
    "What setbacks apply if I build a detached ADU in a single family zone?",
    "How much bicycle parking does a new apartment building need?",
    "Can I run a small food business out of my home kitchen?",
    "What are the noise rules for construction near my house, and what hours apply?",
    "Do I need a license for my dog in Seattle?",
    "What happens if my rental unit has no working heat in winter?",
    "Are short term rentals like Airbnb regulated in Seattle?",
    "What are the rules for putting a fence between my yard and my neighbor's?",
    "Do I need a permit to put a dumpster on the street during a remodel?",
    "What are the sign rules for a small storefront?",
    "How steep can a driveway be for a new house?",
    "What are the requirements to convert a garage into living space?",
    "Is graffiti on my building my responsibility to remove?",
    "What drainage requirements apply when I pave part of my yard?",
]

BARE_SYSTEM = (
    "You are a Seattle civic compliance assistant. Answer the resident's "
    "question about Seattle Municipal Code requirements in plain text, citing "
    "the specific SMC sections that apply inline, like [SMC 23.44.010]."
)


def bare_answer(agent: SeattleReportAgent, question: str) -> str:
    completion = agent.client.chat.completions.create(
        model=agent.model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": BARE_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content or ""


def main() -> None:
    retriever = GroundedRetriever(
        embeddings_path=DATA_DIR / "smc_embeddings.npz",
        chunks_path=DATA_DIR / "smc_chunks.jsonl",
        sqlite_path=DATA_DIR / "smc_ground_truth.db",
    )
    agent = SeattleReportAgent(retriever=retriever)

    stats = {
        "bare": {"citations": 0, "exists": 0, "fabricated": 0, "answers": 0},
        "preassess": {
            "citations": 0,
            "exists": 0,
            "fabricated": 0,
            "grounded": 0,
            "answers": 0,
        },
    }
    fabricated_examples = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")

        text = bare_answer(agent, question)
        cited = parse_citations(text)
        stats["bare"]["answers"] += 1
        stats["bare"]["citations"] += len(cited)
        for citation in cited:
            if retriever.has_citation(citation):
                stats["bare"]["exists"] += 1
            else:
                stats["bare"]["fabricated"] += 1
                if len(fabricated_examples) < 8:
                    fabricated_examples.append((question, citation))

        bundle = agent.generate_report(
            address_profile={},
            user_inputs={"project": question},
            evidence_requests=[EvidenceRequest(label="q", query=question, top_k=5)],
        )
        audit = audit_report(
            bundle["report"],
            [],
            corpus_lookup=retriever.has_citation,
        )
        verdicts = bundle["citation_audit"]
        stats["preassess"]["answers"] += 1
        stats["preassess"]["citations"] += len(verdicts)
        for verdict in verdicts:
            if verdict["status"] == GROUNDED:
                stats["preassess"]["grounded"] += 1
                stats["preassess"]["exists"] += 1
            elif verdict["status"] == UNKNOWN:
                stats["preassess"]["fabricated"] += 1
            else:
                stats["preassess"]["exists"] += 1

    def pct(part, whole):
        return f"{100 * part / whole:.0f}%" if whole else "n/a"

    b, p = stats["bare"], stats["preassess"]
    lines = [
        "# Citation reliability: bare LLM vs PreAssess pipeline",
        "",
        f"Model: {agent.model}. {len(QUESTIONS)} resident questions. Audited against",
        "the full ingested corpus (21 SMC titles). 'Fabricated' = cited section",
        "appears nowhere in the code.",
        "",
        "| system | citations/answer | cited section exists | fabricated | verified against retrieved evidence |",
        "|--------|-----------------|----------------------|------------|--------------------------------------|",
        f"| bare LLM (no retrieval) | {b['citations']/b['answers']:.1f} | {pct(b['exists'], b['citations'])} | {pct(b['fabricated'], b['citations'])} | — (no evidence to verify against) |",
        f"| PreAssess pipeline | {p['citations']/p['answers']:.1f} | {pct(p['exists'], p['citations'])} | {pct(p['fabricated'], p['citations'])} | {pct(p['grounded'], p['citations'])} |",
        "",
        "Fabricated-citation examples from the bare LLM:",
    ] + [f"- \"{q}\" -> cited SMC {c} (does not exist)" for q, c in fabricated_examples]

    report = "\n".join(lines) + "\n"
    RESULTS.write_text(report)
    print()
    print(report)


if __name__ == "__main__":
    main()
