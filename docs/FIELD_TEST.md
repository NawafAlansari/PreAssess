# Field test: PreAssess in city-staff scenarios, measured against alternatives

Date: 2026-07-08. Everything below was produced by running the system live
(full 21-title corpus, llama-3.3-70b-versatile via Groq) and is reproducible
with the committed eval scripts.

## 1. Quantified comparisons

### 1a. Resident-phrased questions vs keyword search

Municipal code portals (Municode/MunicodeNEXT hosts Seattle's code) offer
keyword/boolean search. Our BM25-over-FTS5 configuration is a faithful proxy
for that interaction: exact-word matching against the same corpus. On 39
hand-labeled resident-phrased queries (`eval/queries.jsonl`, scored by whether
the governing section is returned):

| system | recall@1 | recall@5 |
|--------|----------|----------|
| keyword search (BM25 proxy) | 0.03 | 0.23 |
| PreAssess fused retrieval | **0.36** | **0.64** |

Residents don't speak in code vocabulary ("can I rent out my garage" contains
no useful keywords for SMC 23.42.048). Keyword search collapses on the full
code — common permit words match all 21 titles. Labeled honestly: this is a
proxy measured on our corpus, not a scrape of Municode's ranking; the failure
mode it demonstrates (vocabulary mismatch) is inherent to keyword search.

### 1b. Citation reliability: bare LLM vs the pipeline (the headline)

Same model, same 15 resident questions, with and without the retrieval +
audit pipeline (`eval/citation_reliability.py`, results in
`eval/citation_reliability.md`):

| system | citations/answer | cited section exists | fabricated | verified vs retrieved evidence |
|--------|-----------------|----------------------|------------|-------------------------------|
| bare LLM (chatbot-style) | 3.7 | 67% | **33%** | — |
| PreAssess pipeline | 2.5 | **100%** | **0%** | **92%** |

One in three SMC citations from the bare model does not exist — it invented an
entire noise chapter (18.02.030–050) and an entire short-term-rental chapter
(23.45A.020–050), fluently. Existence is checked against the ingested corpus;
Titles 12/13/19 are absent from it, so a bare citation into those would be
unfairly scored — the headline examples were spot-checked and are genuinely
fabricated (Seattle's noise code is 25.08; short-term rentals are Title 6).

### 1c. Positioning vs commercial tools (qualitative, sourced)

| | scope | parcel-aware | verifies its own citations | open / measured |
|---|---|---|---|---|
| **PreAssess** | Seattle municipal code, residents + staff | yes (King County + Seattle GIS, live) | **yes — per-citation audit + grounded ratio** | MIT, eval published |
| Municode/CivicPlus | hosts most US codes | no | n/a (keyword search) | closed |
| UpCodes Copilot | building codes, AEC professionals | no | cites sources; no adversarial audit of generated citations | closed, self-reported 93% accuracy |
| Symbium | CA zoning/permits (rule-encoded) | yes | n/a (deterministic rules, costly per jurisdiction) | closed |

The defensible claim: no tool in this space audits the citations its own model
generates and shows the user the verdict. That is PreAssess's contribution.

## 2. City-staff scenarios (run live)

**Permit counter (SDCI front desk).** Resident: "convert my detached garage
into a rental unit" at 325 9th Ave. One lookup returned: parcel zoned
MIO-240-HR — the Major Institution Overlay (Harborview), auto-resolved from
the city's GIS — with the report citing the MIO chapter [SMC 23.69.022,
23.69.024] alongside ADU size limits [23.42.022] and unit standards
[23.42.048]. The overlay changes the whole answer, and staff didn't have to
know to check for it.

**Code enforcement (tree complaint).** "Owner cut down several large trees on
a steep slope without permits." The report assembled the enforcement chain:
steep-slope erosion hazard area [25.09.012], tree removal in ECAs
[25.09.065.C], tree replacement [25.11.090], and correctly noted the pruning
exemption [25.09.065.D] does not apply. Grounded ratio 0.78 with the two
unverified citations flagged, not hidden.

**Hearing prep (clerk).** `GET /api/citation/25.08.425` → the exact text of
"Sounds created by construction and maintenance equipment," with heading, in
one call.

## 3. Honest limits observed in testing

- Retrieval headroom: "dumpster in the street" surfaced load-dumping and
  litter sections before Street Use (Title 15) permits — recall@1 0.36 means
  roughly one miss in three at the top slot; the report's multi-evidence
  design and the audit's yellow/red flags are the mitigation.
- grounded_ratio measures citation discipline, not legal correctness.
- The staff scenarios above used the same resident-facing UI; a staff-specific
  mode (complaint queue integration, PDF export for case files) is future work.
