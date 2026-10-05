"""
Command line.

    python -m search build                       # fielded FTS5 index + chunk vectors (one-time, a few minutes on CPU)
    python -m search query "dog barks all night"  # first-stage sections for a complaint (no LLM)
    python -m search query "…" --ranker data/search/ranker.txt
                                                  # + LambdaMART, run in a separate process (see search/ranker.py)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from search import config


def cmd_build(args) -> None:
    import numpy as np

    from search import lexical
    from search.corpus import Corpus
    from search.dense import Encoder
    from search.pipeline import index_paths

    corpus = Corpus()
    fts, vec = index_paths(Path(args.index_dir), args.encoder)
    lexical.build(corpus, fts)
    print(f"keyword index: {fts} ({len(corpus)} chunks)")
    vectors = Encoder(args.encoder).encode([corpus.context_text(c) for c in corpus.ids])
    vec.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec, vectors)
    print(f"vectors: {vec} {vectors.shape}")


def cmd_query(args) -> None:
    import numpy as np

    from search.pipeline import Searcher

    searcher = Searcher.open(Path(args.index_dir), args.encoder)
    res = searcher.search(args.complaint)
    order = res.candidates
    if args.ranker:
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "X.npy", res.X)
            out = subprocess.run([sys.executable, "-m", "search", "_score", args.ranker, str(Path(tmp) / "X.npy")],
                                 capture_output=True, text=True, check=True)
            scores = json.loads(out.stdout)
        order = [order[j] for j in np.argsort(-np.asarray(scores), kind="stable")]
    for r, s in enumerate(order[: args.top], start=1):
        print(f"{r:2d}. SMC {s}  {searcher.corpus.section_text(s)['heading'] or ''}")


def cmd_score(args) -> None:
    """LightGBM-only process: score a saved feature matrix."""
    import numpy as np

    from search import ranker

    print(json.dumps(ranker.load(Path(args.model)).predict(np.load(args.X)).tolist()))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m search")
    p.add_argument("--index-dir", default=str(config.INDEX_DIR))
    p.add_argument("--encoder", default=config.ENCODER)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("complaint")
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--ranker")
    s = sub.add_parser("_score")
    s.add_argument("model")
    s.add_argument("X")
    args = p.parse_args(argv)
    {"build": cmd_build, "query": cmd_query, "_score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
