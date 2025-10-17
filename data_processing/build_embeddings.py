#!/usr/bin/env python3
"""
Generate dense embeddings for Seattle Municipal Code chunks.

This script reads the JSONL file created by `build_ground_truth.py`, encodes
each chunk with a sentence-transformer model, and saves the normalized
embeddings to an `.npz` file for fast semantic retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_CHUNKS = Path("seattle-checker/data/processed/smc_chunks.jsonl")
DEFAULT_OUTPUT = Path("seattle-checker/data/processed/smc_embeddings.npz")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help=f"Path to chunk JSONL (default: {DEFAULT_CHUNKS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output .npz file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size (default: %(default)s).",
    )
    return parser.parse_args(argv)


def read_unique_chunks(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    ids: List[str] = []
    seen: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            chunk_id = payload.get("chunk_id")
            text = payload.get("text")
            if not chunk_id or not text:
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ids.append(chunk_id)
            texts.append(text)

    return ids, texts


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)


def encode_chunks(
    model_name: str,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return embeddings


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    chunk_ids, texts = read_unique_chunks(args.chunks)
    if not chunk_ids:
        raise SystemExit(f"No chunks found in {args.chunks}")

    print(f"Embedding {len(chunk_ids)} chunks with {args.model}...")
    embeddings = encode_chunks(args.model, texts, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, chunk_ids=np.array(chunk_ids), embeddings=embeddings)

    print(f"Wrote embeddings to {args.output} (shape={embeddings.shape}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
