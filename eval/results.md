# Retrieval evaluation

30 hand-labeled queries (eval/queries.jsonl), top-5 retrieval.

| config | recall@1 | recall@3 | recall@5 | MRR@5 |
|--------|----------|----------|----------|-------|
| dense | 0.23 | 0.53 | 0.60 | 0.39 |
| fts | 0.27 | 0.43 | 0.50 | 0.36 |
| prefilter | 0.20 | 0.53 | 0.57 | 0.36 |
| rrf-fused | 0.33 | 0.50 | 0.60 | 0.43 |

The app uses rrf-fused (reciprocal-rank fusion of dense + BM25).
`prefilter` is the previous design (FTS candidate filter + dense
rerank), kept for comparison.

rrf-fused misses (12):
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
