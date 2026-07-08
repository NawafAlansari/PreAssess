# PreAssess

**Ask what the Seattle Municipal Code requires for your project — and get an
answer whose every citation is verified, not just generated.**

Enter a Seattle address and a project ("add a 600 sq ft detached ADU"). PreAssess
pulls the real parcel from King County GIS, resolves which overlay districts,
environmentally critical areas, and street trees apply from the City of Seattle's
own GIS layers (shown on a live parcel map), retrieves the governing code sections
from an ingested corpus of the Seattle Municipal Code, drafts a plain-language
compliance report — and then **audits every SMC citation in that report against
the code that was actually retrieved**. Citations render green (verified in
evidence), yellow (exists in code, but wasn't shown to the model), or red (not
found — likely fabricated). Click any citation to read the code text itself.

3rd place, PACT-Athon 2025 (City of Seattle + AI House).
[![Watch the demo](docs/loom-thumb.png)](https://www.loom.com/share/d4af27ac8770436d9edee1bc32035834?sid=6d392f9e-9d8e-4652-ae2b-bf69d861f378 "Watch the demo on Loom")

## Why the verification layer exists

Legal-information RAG has a specific failure mode: the model writes a confident
requirement with a plausible-looking citation that doesn't exist. Asking the
model to cite is not a guarantee; checking the citations is. PreAssess treats
"citation-backed" as a property to *verify*, not a prompt instruction:

```
citation audit for a generated report:
  23.42.022  -> grounded                    (matches retrieved evidence)
  22.801     -> in_corpus_not_retrieved     (real code, but not shown to the model)
  99.99.999  -> unknown                     (appears nowhere in the ingested code)
grounded_ratio: 0.33
```

## Architecture

```
                    browser (React + Vite + Leaflet parcel map)
                      │ address + project              citations clicked
                      ▼                                ▼
 King County GIS ◄─ /api/parcel/query    /api/report  │  /api/citation
 Seattle GIS     ◄─ /api/context             │        │
 (zoning, overlays,      │                   ▼        ▼
  ECA, street trees)     └──► overlay chapters as evidence requests
                                             │
                       FastAPI  ──►  GroundedRetriever ──► LLM (server-side key,
                                     dense + BM25, RRF      any OpenAI-compatible
                                     8,066 SMC chunks       endpoint; Groq default)
                                     (Titles 22, 23, 25)        │
                                                                ▼
                                                          citation_check
                                                          audits the report
```

The overlay resolution matters: the city's master overlay layer carries the SMC
chapter that governs each district, so when a parcel sits in (say) the Columbia
City Landmark District, the report retrieves and cites SMC 25.20 by itself —
instead of asking the resident to go find out.

- **Retrieval**: sentence-transformer embeddings + SQLite FTS5, fused with
  reciprocal-rank fusion. The corpus: 8,066 chunks, 2,376 sections across
  Titles 22, 23, and 25, extracted from the City's PDF code by the ingestion
  pipeline in `data_processing/`.
- **The browser holds no secrets**: the Groq key lives on the server; King
  County GIS (which sends no CORS headers) is proxied against a fixed URL.

## Retrieval quality is measured, not assumed

39 hand-labeled resident-style queries (`eval/queries.jsonl`), scored with
recall@k and MRR (`python -m eval.retrieval_eval`):

| config | recall@1 | recall@3 | recall@5 | MRR@5 |
|--------|----------|----------|----------|-------|
| dense | 0.28 | 0.56 | 0.62 | 0.42 |
| fts (BM25) | 0.31 | 0.49 | 0.56 | 0.40 |
| prefilter (old design) | 0.26 | 0.54 | 0.59 | 0.40 |
| **rrf-fused (shipped)** | **0.38** | 0.56 | **0.67** | **0.49** |

The eval earned its keep immediately: it caught the original hybrid mode
returning an *arbitrary* candidate subset (missing `ORDER BY rank` before
`LIMIT`) — 0.07 recall@5 — and then showed that rank fusion beats the
prefilter design entirely. `eval/results.md` has the current numbers and the
open misses.

## Quickstart

Prerequisites: Python 3.11+, Node 20+, and optionally a Groq API key (retrieval
and the checklist work without one; report generation needs it).

```bash
# API (from the repo root; corpus artifacts are committed)
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # 63 tests, offline, no keys needed
GROQ_API_KEY=... uvicorn api.main:app --port 8000

# Frontend (dev)
npm ci
npm run dev                          # Vite proxies /api to :8000
```

Or as one container:

```bash
GROQ_API_KEY=... docker compose up --build   # serves app + API on :8000
```

Rebuilding the corpus from a new code supplement PDF:
`./data_processing/run_pipeline.sh /path/to/MunicipalCode.pdf`

## API

| Endpoint | What |
|----------|------|
| `POST /api/report` | Property + project → report, citation audit, grounded ratio, evidence (rate-limited) |
| `GET /api/search?q=` | Fused retrieval over the corpus |
| `GET /api/citation/{smc}` | Exact code text for a citation (section, subsection, or chapter) |
| `GET /api/context` | Zoning, overlay districts, ECA flags, street trees at a point (Seattle GIS) |
| `GET /api/parcel/query` | King County GIS proxy (fixed upstream) |
| `GET /api/health`, `GET /api/stats` | Corpus + config introspection |

## Limitations

- **Titles 22, 23, and 25 of ~28.** Street use (Title 15) and utilities
  (Title 21), for example, are not yet ingested; citations into missing titles
  are honestly reported as "not found in the ingested code" rather than guessed
  at, and the retriever warns loudly when a filter targets an unloaded title.
  The pipeline ingests any title from the City's code PDF in one command.
- **Not legal advice.** The report is a research aid over the code text; the
  instant checklist uses a small static table of simplified reference values
  (clearly scoped in `src/`), and permits are decided by the City, not a model.
- **Retrieval is embedding + BM25, no cross-encoder reranker.** recall@1 of
  0.38 on hard resident phrasing has clear headroom; the eval harness exists so
  improvements are measured, not vibes.
- **grounded_ratio measures citation discipline, not correctness** — a report
  can cite real, retrieved code and still reason about it imperfectly.

## License

MIT
