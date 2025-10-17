# Seattle Checklist

[![Seattle Checklist overview](https://cdn.loom.com/sessions/thumbnails/d4af27ac8770436d9edee1bc32035834-with-play.png)](https://www.loom.com/share/d4af27ac8770436d9edee1bc32035834)  
_Update the Loom link if you publish a new walkthrough._

Seattle Checklist is a community project that turns Seattle’s maze of municipal
code into a friendly guide for homeowners, tenants, and small businesses. The
app pulls together zoning rules, tree protections, and permitting steps so
residents can understand what the City expects before they remodel, plant, or
build.

## Why we built it

Seattle publishes thousands of pages of land-use rules across PDFs, web portals,
and agency memos. There is no official database that answers a simple question
like “What applies to my lot?” We want to close that gap by:

- Translating legal text into plain-language checklists.
- Showing every requirement with a real Seattle Municipal Code citation.
- Highlighting missing data so residents know when they need to call the City.

## How it works

1. **Upload** the latest municipal code supplement PDF.
2. **Extract** the chapters we care about (Titles 22 & 23 today) into structured
   JSON.
3. **Chunk & embed** each section so the AI can search the relevant snippets
   instead of the entire PDF.
4. **Retrieve** the right passages when someone asks about their address, and
   let our Groq-powered agent write a report using only those verified chunks.

The result is a “grounded” answer: every sentence in the report comes from a
specific chunk of the municipal code, and the citation stays with it.

## Quick start (app)

```bash
cd seattle-checker
npm install
npm run dev
```

The Vite development server will print a local URL. Open it in your browser and
walk through the checklist for any Seattle address.

### Requirements

- Node 18+ or the latest LTS release.
- Modern browser (Chrome, Edge, Firefox, Safari).

## Refreshing municipal code data

When Seattle releases a new code supplement, run the data pipeline once to keep
the ground truth up to date.

```bash
cd seattle-checker
chmod +x data_processing/run_pipeline.sh
./data_processing/run_pipeline.sh /path/to/MunicipalCode.pdf
```

The script will:

1. Extract Titles 22 and 23 into `data/title22.json`, `data/title23.json`.
2. Create chunked records with citations (`data/processed/smc_chunks.jsonl`).
3. Build a searchable SQLite database (`data/processed/smc_ground_truth.db`).
4. Generate dense embeddings (`data/processed/smc_embeddings.npz`) for semantic
   retrieval.

> **Python dependencies**: `PyPDF2`, `numpy`, `torch`, and `transformers`. Install
> them with `pip install PyPDF2 numpy torch transformers`.

## Using the AI report agent

The agent ties everything together: it retrieves the right code sections and
asks Groq’s LLM to draft a resident-friendly report with citations.

```bash
export GROQ_API_KEY=your_api_key_here
export PYTHONPATH="$(pwd)/seattle-checker"
python -m smc_agents.report_agent
```

The sample run fetches zoning, tree, and permit requirements for a mock address.
Adapt the `EvidenceRequest` entries in `smc_agents/report_agent.py` to match the
topics you want in production or wire the retriever into your own backend.

## Architectural overview

- **Front-end**: React + Vite experience that guides residents through a
  checklist and surfaces citations inline.
- **Ground-truth store**: JSON chunks, SQLite FTS index, and embeddings created
  from the municipal code PDF.
- **Retriever** (`smc_agents/retriever.py`): Hybrid search that filters by title
  or citation, applies dense similarity scoring, and returns the most relevant
  text with metadata.
- **Groq agent** (`smc_agents/report_agent.py`): Composes property data, user
  questions, and retrieved evidence into a prompt; produces a report that cites
  the Seattle Municipal Code.
