# Retrieval evaluation

39 hand-labeled queries (eval/queries.jsonl), top-5 retrieval.

| config | recall@1 | recall@3 | recall@5 | MRR@5 |
|--------|----------|----------|----------|-------|
| dense | 0.28 | 0.54 | 0.62 | 0.42 |
| fts | 0.03 | 0.15 | 0.23 | 0.09 |
| prefilter | 0.23 | 0.51 | 0.56 | 0.37 |
| rrf-fused | 0.36 | 0.51 | 0.64 | 0.46 |
| rrf+ce | 0.59 | 0.64 | 0.64 | 0.61 |

The app uses rrf-fused (reciprocal-rank fusion of dense + BM25).
`prefilter` is the previous design (FTS candidate filter + dense
rerank), kept for comparison.
`rrf+ce` re-scores the fused top-30 with a cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2); run with EVAL_RERANK=1.

rrf-fused misses (14):
- who enforces the building code and can adopt rules
- city forcing repair or demolition of a dangerous building
- penalty for violating mobile home park rules
- where can parking be located for apartment buildings
- bicycle parking requirements for new buildings
- what is the purpose of the sign standards
- view corridor requirements near the waterfront
- what must a major institution master plan contain
- building in potentially hazardous or flood prone locations
- setback requirements in single family zones
- seattle mixed north rainier zone special provisions
- screening requirements for parking in structures
- deadline to respond to an administrative appeal
- purpose of the tree protection code
