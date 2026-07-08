"""
PreAssess API: the retrieval + report pipeline behind an HTTP boundary.

The browser never talks to Groq and never sees an API key; it sends property
facts and a project description here, and gets back a report with a citation
audit and the evidence that grounded it.

Run: uvicorn api.main:app --reload  (from the repo root)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.gis import point_context
from smc_agents.report_agent import (
    DEFAULT_MODEL,
    EvidenceRequest,
    SeattleReportAgent,
    resolve_api_key,
)
from smc_agents.retriever import GroundedRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"
DIST_DIR = REPO_ROOT / "dist"

REPORT_RATE_LIMIT = int(os.getenv("REPORT_RATE_LIMIT", "10"))  # per minute per IP

app = FastAPI(title="PreAssess API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons, overridable in tests.
_retriever: Optional[GroundedRetriever] = None
_agent: Optional[SeattleReportAgent] = None
_report_hits: Dict[str, Deque[float]] = defaultdict(deque)


def get_retriever() -> GroundedRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GroundedRetriever(
            embeddings_path=DATA_DIR / "smc_embeddings.npz",
            chunks_path=DATA_DIR / "smc_chunks.jsonl",
            sqlite_path=DATA_DIR / "smc_ground_truth.db",
        )
    return _retriever


def get_agent() -> SeattleReportAgent:
    global _agent
    if _agent is None:
        if not resolve_api_key():
            raise HTTPException(
                status_code=503,
                detail="Report generation is not configured (LLM_API_KEY / GROQ_API_KEY "
                "missing on the server). Retrieval endpoints still work.",
            )
        _agent = SeattleReportAgent(retriever=get_retriever())
    return _agent


def _rate_limit(client: str) -> None:
    now = time.monotonic()
    hits = _report_hits[client]
    while hits and now - hits[0] > 60.0:
        hits.popleft()
    if len(hits) >= REPORT_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry in a minute.")
    hits.append(now)


class ReportRequest(BaseModel):
    address_profile: Dict[str, object] = Field(default_factory=dict)
    project_description: str = ""
    questions: List[str] = Field(default_factory=list, max_length=6)
    # Output of /api/context — lets the report retrieve the chapters that the
    # city's own GIS says govern this parcel (overlays, ECA, trees).
    context: Optional[Dict[str, object]] = None


def _evidence_requests(payload: ReportRequest) -> List[EvidenceRequest]:
    requests: List[EvidenceRequest] = []
    if payload.project_description.strip():
        requests.append(
            EvidenceRequest(label="project", query=payload.project_description, top_k=5)
        )
    for i, question in enumerate(payload.questions, start=1):
        if question.strip():
            requests.append(EvidenceRequest(label=f"question_{i}", query=question, top_k=4))

    ingested = set(get_retriever().ingested_titles)

    if 25 in ingested and "tree" in payload.project_description.lower():
        requests.append(
            EvidenceRequest(
                label="tree_protection",
                query="tree removal protection replacement requirements",
                section_prefix="25.11",
                top_k=4,
            )
        )

    if payload.context:
        eca_flags = payload.context.get("eca") or []
        if eca_flags and 25 in ingested:
            requests.append(
                EvidenceRequest(
                    label="eca",
                    query="development standards for environmentally critical areas "
                    + " ".join(str(f).replace("_", " ") for f in eca_flags),
                    section_prefix="25.09",
                    top_k=3,
                )
            )
        for overlay in payload.context.get("overlays", []) or []:
            prefix = overlay.get("chapter_prefix") if isinstance(overlay, dict) else None
            name = overlay.get("name") if isinstance(overlay, dict) else None
            if not prefix:
                continue
            try:
                title = int(prefix.split(".")[0])
            except ValueError:
                continue
            # Only request chapters we have ingested; others would silently
            # retrieve nothing (the retriever also warns).
            if title not in ingested:
                continue
            requests.append(
                EvidenceRequest(
                    label=f"overlay:{name or prefix}",
                    query=f"{name or ''} overlay district requirements and standards",
                    section_prefix=prefix,
                    top_k=3,
                )
            )

    if not requests:
        raise HTTPException(
            status_code=422,
            detail="Provide a project_description or at least one question.",
        )
    return requests


def _trim_evidence(evidence: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    trimmed: Dict[str, List[dict]] = {}
    for label, hits in evidence.items():
        trimmed[label] = [
            {
                "chunk_id": meta.get("chunk_id"),
                "citation": meta.get("full_citation") or meta.get("section_citation"),
                "section_citation": meta.get("section_citation"),
                "heading": meta.get("section_heading"),
                "chapter_title": meta.get("chapter_title"),
                "text": str(meta.get("text", ""))[:600],
            }
            for meta in hits
        ]
    return trimmed


@app.get("/api/health")
def health() -> dict:
    retriever = get_retriever()
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "report_enabled": bool(resolve_api_key()),
        "corpus": {
            "chunks": len(retriever.metadata),
            "sections": len(retriever.section_citations),
            "chapters": len(retriever.chapter_citations),
            "titles": retriever.ingested_titles,
        },
    }


@app.get("/api/stats")
def stats() -> dict:
    retriever = get_retriever()
    by_title: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    for meta in retriever.metadata.values():
        by_title[str(meta.get("title_number"))] += 1
        by_type[str(meta.get("chunk_type"))] += 1
    return {"chunks_by_title": dict(by_title), "chunks_by_type": dict(by_type)}


@app.get("/api/search")
def search(q: str, k: int = 5, title: Optional[int] = None) -> dict:
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be empty")
    k = max(1, min(k, 20))
    hits = get_retriever().search_fused(q, top_k=k, title_number=title)
    return {
        "query": q,
        "results": [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "citation": h.full_citation or h.metadata.get("section_citation"),
                "section_citation": h.metadata.get("section_citation"),
                "heading": h.section_heading,
                "chapter_title": h.chapter_title,
                "title_number": h.title_number,
                "text": h.text[:600],
            }
            for h in hits
        ],
    }


@app.get("/api/context")
async def context(lat: float, lon: float, tree_radius: int = 30) -> dict:
    """Zoning, overlay districts, ECA flags, and street trees at a point,
    from the City of Seattle's authoritative GIS layers."""
    if not (47.2 < lat < 47.9 and -122.6 < lon < -121.9):
        raise HTTPException(status_code=422, detail="Point is not in the Seattle area.")
    tree_radius = max(5, min(tree_radius, 150))
    return await point_context(lat, lon, tree_radius_m=tree_radius)


@app.get("/api/citation/{citation}")
def citation_lookup(citation: str) -> dict:
    """Exact lookup of a citation's code text (section, subsection, or chapter)."""
    retriever = get_retriever()
    citation = citation.strip()
    matches = []
    for meta in retriever.metadata.values():
        section = str(meta.get("section_citation") or "")
        chapter = str(meta.get("chapter_citation") or "")
        if (
            section == citation
            or (section and section.startswith(citation + "."))
            or (section and citation.startswith(section + "."))
            or chapter == citation
        ):
            matches.append(meta)
        if len(matches) >= 5:
            break
    return {
        "citation": citation,
        "results": [
            {
                "chunk_id": meta.get("chunk_id"),
                "citation": meta.get("full_citation") or meta.get("section_citation"),
                "heading": meta.get("section_heading"),
                "chapter_title": meta.get("chapter_title"),
                "text": str(meta.get("text", ""))[:600],
            }
            for meta in matches
        ],
    }


# King County's ArcGIS endpoint does not send CORS headers, so the browser
# cannot query it directly. This proxies the /query call against a FIXED
# upstream URL — only query parameters pass through, never a caller-supplied
# host or path.
KC_PARCEL_QUERY_URL = (
    "https://gismaps.kingcounty.gov/arcgis/rest/services/Property/"
    "KingCo_PropertyInfo/MapServer/2/query"
)


@app.get("/api/parcel/query")
def parcel_query(request: Request) -> Response:
    try:
        upstream = httpx.get(
            KC_PARCEL_QUERY_URL,
            params=dict(request.query_params),
            timeout=15.0,
            follow_redirects=True,
        )
    except httpx.HTTPError as err:
        raise HTTPException(status_code=502, detail=f"King County GIS unreachable: {err}")
    return Response(content=upstream.content, media_type="application/json")


@app.post("/api/report")
def report(payload: ReportRequest, request: Request) -> dict:
    _rate_limit(request.client.host if request.client else "unknown")
    agent = get_agent()

    address_profile = dict(payload.address_profile)
    if payload.context:
        ctx = payload.context
        trees = ctx.get("trees") or {}
        address_profile["city_gis_facts"] = {
            "zoning_layer": ctx.get("zoning"),
            "overlay_districts": [
                {k: o.get(k) for k in ("name", "type", "chapter")}
                for o in (ctx.get("overlays") or [])
                if isinstance(o, dict)
            ],
            "environmentally_critical_areas": ctx.get("eca") or [],
            "street_trees_nearby": {
                "count": trees.get("count", 0),
                "radius_m": trees.get("radius_m"),
                "largest": (trees.get("largest") or [])[:3],
            },
        }

    bundle = agent.generate_report(
        address_profile=address_profile,
        user_inputs={
            "project": payload.project_description,
            "questions": payload.questions,
        },
        evidence_requests=_evidence_requests(payload),
    )
    return {
        "report": bundle["report"],
        "citation_audit": bundle["citation_audit"],
        "grounded_ratio": bundle["grounded_ratio"],
        "evidence": _trim_evidence(bundle["evidence"]),
        "model": agent.model,
    }


# In production the built frontend is served from the same process; in dev the
# Vite server proxies /api here instead.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="app")
