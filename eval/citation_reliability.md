# Citation reliability: bare LLM vs PreAssess pipeline

Model: llama-3.3-70b-versatile. 15 resident questions. Audited against
the full ingested corpus (21 SMC titles). 'Fabricated' = cited section
appears nowhere in the code.

| system | citations/answer | cited section exists | fabricated | verified against retrieved evidence |
|--------|-----------------|----------------------|------------|--------------------------------------|
| bare LLM (no retrieval) | 3.7 | 67% | 33% | — (no evidence to verify against) |
| PreAssess pipeline | 2.5 | 100% | 0% | 92% |

Fabricated-citation examples from the bare LLM:
- "Can I run a small food business out of my home kitchen?" -> cited SMC 23.84a.038 (does not exist)
- "What are the noise rules for construction near my house, and what hours apply?" -> cited SMC 18.02.040 (does not exist)
- "What are the noise rules for construction near my house, and what hours apply?" -> cited SMC 18.02.050 (does not exist)
- "What are the noise rules for construction near my house, and what hours apply?" -> cited SMC 18.02.030 (does not exist)
- "Are short term rentals like Airbnb regulated in Seattle?" -> cited SMC 23.45A.020 (does not exist)
- "Are short term rentals like Airbnb regulated in Seattle?" -> cited SMC 23.45A.030 (does not exist)
- "Are short term rentals like Airbnb regulated in Seattle?" -> cited SMC 23.45A.040 (does not exist)
- "Are short term rentals like Airbnb regulated in Seattle?" -> cited SMC 23.45A.050 (does not exist)
