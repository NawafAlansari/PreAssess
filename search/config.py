"""
Every setting, in one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = DATA_DIR / "smc_chunks.jsonl"
# The chunk-id order of this file defines row order everywhere (vectors, ties).
IDS_PATH = DATA_DIR / "smc_embeddings.npz"
INDEX_DIR = ROOT / "data" / "search"          # built by `python -m search build`

# Dense encoder. A version fine-tuned on complaint data works better; the public
# base model is the default.
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_MAX_LENGTH = 512
ENCODER_BATCH = 64

# First stage.
LIST_DEPTH = 100                 # chunks per retrieval list
BM25_FIELD_WEIGHTS = (2.0, 5.0, 1.0)   # path, heading, body (FTS5 bm25 column weights)
RRF_K = 60
# (list name, weight): dense on the complaint's words / the rewrite / the hypothetical
# passage, BM25 on the words / the rewrite.
LIST_WEIGHTS = (
    ("dense_words", 3.0),
    ("dense_rewrite", 2.0),
    ("dense_hyde", 2.0),
    ("bm25_words", 1.0),
    ("bm25_rewrite", 1.0),
)
POOL_CHUNKS = 100                # fused chunks kept
CANDIDATE_SECTIONS = 30          # sections passed to the ranker
ABSENT_RANK = 101                # section rank when a list doesn't contain it

# Ranker features.
PRIOR_ALPHA = 5.0                # smoothing toward the global relevance rate
RELEVANT_GRADE = 2               # grades are 0-3; relevant = 2 or 3
FACETS = ("rule", "definition", "penalty", "enforcement", "exceptions")

# LambdaMART (LightGBM lambdarank). Fixed in advance, not tuned.
LGBM_PARAMS = dict(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    lambdarank_truncation_level=10,
    random_state=0,
    verbose=-1,
)

# Cascade: the LLM reorders this many of the ranker's top sections.
CASCADE_TOP = 10
