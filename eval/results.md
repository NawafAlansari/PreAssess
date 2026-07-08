# Retrieval evaluation

39 hand-labeled queries (eval/queries.jsonl), top-5 retrieval.

| config | recall@1 | recall@3 | recall@5 | MRR@5 |
|--------|----------|----------|----------|-------|
| dense | 0.28 | 0.56 | 0.62 | 0.42 |
| fts | 0.31 | 0.49 | 0.56 | 0.40 |
| prefilter | 0.26 | 0.54 | 0.59 | 0.40 |
| rrf-fused | 0.38 | 0.56 | 0.67 | 0.49 |

The app uses rrf-fused (reciprocal-rank fusion of dense + BM25).
`prefilter` is the previous design (FTS candidate filter + dense
rerank), kept for comparison.

rrf-fused misses (13):
- who enforces the building code and can adopt rules
- city forcing repair or demolition of a dangerous building
- penalty for violating mobile home park rules
- where can parking be located for apartment buildings
- bicycle parking requirements for new buildings
- what is the purpose of the sign standards
- view corridor requirements near the waterfront
- building in potentially hazardous or flood prone locations
- setback requirements in single family zones
- seattle mixed north rainier zone special provisions
- screening requirements for parking in structures
- deadline to respond to an administrative appeal
- purpose of the tree protection code
