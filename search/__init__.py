"""
Search and ranking over the Seattle Municipal Code: the evaluated PreAssess system.

A resident's complaint goes through four stages:

    complaint ─► query understanding (LLM rewrite + hypothetical passage)
              ─► 5 retrieval lists (dense ×3 query versions, BM25 ×2), top 100 chunks each
              ─► weighted reciprocal rank fusion → top 100 chunks → top 30 sections
              ─► LambdaMART over 41 features per section
              ─► (optional) an LLM reorders the top 10

This package is a consolidated, readable version of the research code that produced
the reported numbers. `search.config` holds every setting; each module documents the
step it implements.
"""
