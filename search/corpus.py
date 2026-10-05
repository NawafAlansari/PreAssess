"""
The corpus: SMC chunks, the sections they belong to, and the text that gets indexed.

The retrieval unit is a chunk (paragraphs packed up to ~900 characters by
`data_processing/build_ground_truth.py`); the unit we rank and return is a section,
identified by its 3-part citation ("23.54.015"). Every chunk is indexed with a
breadcrumb prefix ("Title 23 > Chapter 23.54 … > Section 23.54.015 …") so a passage
cut from the middle of a long section still carries its topic words.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np

from search import config

# Content words used by the word-overlap features.
STOPWORDS = set(
    "the a an and or of to in on for at by with from is are was were be been it its this that there their "
    "they them as not no but if into out over under about than then so such can will would should could may "
    "might must have has had do does did i my me we our you your he she his her who what when where which "
    "while all any some more most other only own same very just also near next".split()
)


def section_of(citation: str | None) -> str:
    """'23.47A.004.D.3' -> '23.47A.004' (sub-units roll up to their section)."""
    return ".".join((citation or "").split(".")[:3])


def chapter_of(section: str) -> str:
    return ".".join(section.split(".")[:2])


def content_words(text: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z]+", (text or "").lower()) if w not in STOPWORDS and len(w) > 2}


class Corpus:
    """Chunk records, keyed by chunk id, in the canonical row order."""

    def __init__(self, chunks_path: Path = config.CHUNKS_PATH, ids_path: Path = config.IDS_PATH):
        self.chunks: dict[str, dict] = {}
        with Path(chunks_path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("chunk_id"):
                        # Later duplicate ids overwrite earlier ones (the ids are not
                        # unique in the extracted data; ~3% of records are affected).
                        self.chunks[rec["chunk_id"]] = rec
        self.ids: list[str] = [str(i) for i in np.load(ids_path)["chunk_ids"]]
        self._heading_freq = Counter(c.get("section_heading") for c in self.chunks.values() if c.get("section_heading"))

    def __len__(self) -> int:
        return len(self.ids)

    def chunk(self, cid: str) -> dict:
        return self.chunks[cid]

    def section(self, cid: str) -> str:
        return section_of(self.chunks.get(cid, {}).get("section_citation"))

    # ---------------------------------------------------------------- indexed text
    def clean_heading(self, heading: str | None) -> str:
        """Drop headings that are page-header noise (very short, or repeated > 40 times)."""
        if not heading or len(heading) < 4 or self._heading_freq[heading] > 40:
            return ""
        return heading.strip()

    @staticmethod
    def clean_chapter_title(title: str | None) -> str:
        """Keep chapter titles only when they look like real titles (mostly upper case)."""
        if not title:
            return ""
        letters = [ch for ch in title if ch.isalpha()]
        return title.strip() if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.8 else ""

    def context_text(self, cid: str) -> str:
        """Breadcrumb + chunk text: what the dense index embeds."""
        c = self.chunks[cid]
        sec = c.get("section_citation") or ""
        chapter = c.get("chapter_citation") or chapter_of(sec)
        parts = [f"Title {c.get('title_number')}", f"Chapter {chapter} {self.clean_chapter_title(c.get('chapter_title'))}".strip()]
        if sec:
            parts.append(f"Section {sec} {self.clean_heading(c.get('section_heading'))}".strip())
        return " > ".join(parts) + "\n" + str(c.get("text", ""))

    # ---------------------------------------------------------------- sections
    def sections(self, chunk_ids) -> list[str]:
        """Distinct sections of a ranked chunk list, in first-appearance order."""
        out: list[str] = []
        for cid in chunk_ids:
            s = self.section(cid)
            if s and s not in out:
                out.append(s)
        return out

    @cached_property
    def section_rows(self) -> dict[str, list[int]]:
        """Section -> row indices of all its chunks."""
        rows: dict[str, list[int]] = defaultdict(list)
        for j, cid in enumerate(self.ids):
            rows[self.section(cid)].append(j)
        return rows

    @cached_property
    def section_words(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Section -> content words of its heading, and of its full text."""
        head: dict[str, str] = {}
        body: dict[str, str] = defaultdict(str)
        for cid in self.ids:
            c = self.chunks[cid]
            s = self.section(cid)
            head.setdefault(s, str(c.get("section_heading") or ""))
            body[s] += " " + str(c.get("text", ""))
        return {s: content_words(h) for s, h in head.items()}, {s: content_words(b) for s, b in body.items()}

    def section_chars(self, section: str) -> int:
        return sum(len(str(self.chunks[self.ids[j]].get("text", ""))) for j in self.section_rows[section])

    def section_text(self, section: str, limit: int = 600) -> dict:
        """Citation, chapter title, heading and opening text of a section (for the LLM reorder)."""
        cids = [self.ids[j] for j in self.section_rows[section]]
        first = self.chunks[cids[0]]
        return {
            "citation": section,
            "chapter_title": first.get("chapter_title"),
            "heading": first.get("section_heading"),
            "text": " ".join(str(self.chunks[c].get("text", "")) for c in cids)[:limit],
        }
