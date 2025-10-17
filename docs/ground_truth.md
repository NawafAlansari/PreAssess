# Seattle Checker Ground Truth Workflow

This project now includes a reproducible pipeline for turning municipal code
supplements into LLM-ready building blocks. The flow is:

1. **OCR / parsing**  
   `data_processing/extract_title.py` (existing) splits the PDF supplements into
   structured JSON (`data/title22.json`, `data/title23.json`, ...).

2. **Chunking**  
   `data_processing/build_ground_truth.py` normalizes the JSON hierarchy and
   emits compact, citation-carrying snippets.

3. **Indexing**  
   `data_processing/build_sqlite_index.py` writes those snippets into a SQLite
   database with an FTS5 search index so you can retrieve the most relevant
   passages for a prompt.

4. **Dense embeddings (semantic recall)**  
   `data_processing/build_embeddings.py` encodes each chunk with a
   transformer-based embedding model and stores the vectors in
   `data/processed/smc_embeddings.npz`.

5. **Retrieval / debugging**  
   `data_processing/query_ground_truth.py` provides a quick command-line helper
   to inspect what the retriever or agent will surface before prompting.

6. **Agent layer**  
   `smc_agents/retriever.py` + `smc_agents/report_agent.py` combine semantic
   search with Groq’s chat models so your LLM can call into the ground-truth
   store like a tool.

## Generate the chunk file

```bash
python data_processing/build_ground_truth.py \
  data/title22.json data/title23.json \
  --output data/processed/smc_chunks.jsonl
```

The JSONL file contains one chunk per line with text, hierarchy metadata, and
canonical citations (e.g., `SMC 23.44.014`).

## Build the SQLite index

```bash
python data_processing/build_sqlite_index.py \
  data/processed/smc_chunks.jsonl \
  --database data/processed/smc_ground_truth.db
```

The resulting database has:

- `chunks`: metadata-packed rows keyed by `chunk_id`
- `chunks_fts`: FTS5 virtual table for fast text lookup

Use the DB from any environment (Python, Node, Go, etc.) as the authoritative
store for SMC snippets. At runtime you can filter by `title_number`,
`section_citation`, or any other metadata before submitting the text to an LLM.

## Generate embeddings (semantic recall)

```bash
pip install torch transformers  # if not already available
python data_processing/build_embeddings.py \
  --chunks data/processed/smc_chunks.jsonl \
  --output data/processed/smc_embeddings.npz \
  --model sentence-transformers/all-MiniLM-L6-v2
```

This produces a normalized 384-d embedding vector for every unique chunk.
Because the vectors live in a NumPy archive, they can be loaded quickly into
memory for hybrid (keyword + semantic) search.

## Smoke-test retrieval

```bash
python data_processing/query_ground_truth.py "rear yard setback" --title 23 --limit 3
```

The helper prints the top hits with their citations so you can verify context
before wiring the database into your app or agent.

## Integrate with an LLM

1. Ensure the SQLite DB and embeddings archive exist (`smc_ground_truth.db`,
   `smc_embeddings.npz`, `smc_chunks.jsonl`).
2. From the repo root, run Python with the app folder on the path:
   ```bash
   export PYTHONPATH="$(pwd)/seattle-checker"
   ```
3. Instantiate the retriever and agent:
   ```python
   from smc_agents.retriever import GroundedRetriever
   from smc_agents.report_agent import SeattleReportAgent, EvidenceRequest

   retriever = GroundedRetriever(
       embeddings_path=Path("seattle-checker/data/processed/smc_embeddings.npz"),
       chunks_path=Path("seattle-checker/data/processed/smc_chunks.jsonl"),
       sqlite_path=Path("seattle-checker/data/processed/smc_ground_truth.db"),
   )
   agent = SeattleReportAgent(retriever=retriever, model="llama-3.1-70b-versatile")
   ```
4. Tell the agent what evidence to fetch (queries + filters), pass in parcel/user
   context, and call `generate_report`. The agent will:
   - Run semantic search over the embeddings.
   - (Optionally) constrain results with citation filters or FTS queries.
   - Send the combined evidence + context to Groq’s chat model.
   - Return the generated report along with the underlying chunk metadata for
     audit/citation display.

Example CLI demo:

```bash
GROQ_API_KEY=... PYTHONPATH="$(pwd)/seattle-checker" \
python -m smc_agents.report_agent
```

The agent composes a resident-friendly summary with inline SMC citations,
grounded exclusively in the retrieved snippets.

Because the pipeline is deterministic, you can rerun it whenever the City
updates the PDF supplements, refresh the index, and keep the checklist grounded
on the latest legal language.
