#!/usr/bin/env python3
"""
Normalize Seattle Municipal Code JSON exports and emit LLM-ready chunks.

The loader accepts one or more title JSON files (as produced by
`extract_title.py`) and outputs a JSONL file where each line represents a
chunk with rich metadata (title → subtitle → division → chapter → section).
Chunks are sized to play nicely with downstream embedding and retrieval
pipelines, while preserving exact source citations for traceable responses.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


MAX_CHARS_DEFAULT = 900


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_type: str
    text: str
    source_path: str
    title_number: int
    title_label: str
    subtitle_number: Optional[str]
    subtitle_title: Optional[str]
    division_number: Optional[str]
    division_title: Optional[str]
    chapter_citation: str
    chapter_title: str
    section_citation: Optional[str]
    section_heading: Optional[str]
    start_page: Optional[int]
    end_page: Optional[int]

    def to_json(self) -> str:
        payload = {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "source_path": self.source_path,
            "title_number": self.title_number,
            "title_label": self.title_label,
            "subtitle_number": self.subtitle_number,
            "subtitle_title": self.subtitle_title,
            "division_number": self.division_number,
            "division_title": self.division_title,
            "chapter_citation": self.chapter_citation,
            "chapter_title": self.chapter_title,
            "section_citation": self.section_citation,
            "section_heading": self.section_heading,
            "full_citation": build_full_citation(
                self.title_number, self.section_citation
            ),
            "start_page": self.start_page,
            "end_page": self.end_page,
            "char_count": len(self.text),
        }
        return json.dumps(payload, ensure_ascii=True)


def build_full_citation(title_number: int, section_citation: Optional[str]) -> Optional[str]:
    if not section_citation:
        return None
    citation = section_citation.strip()
    if citation.lower().startswith(("smc", "title", "chapter")):
        return citation
    normalized = re.sub(r"^0+", "", citation)
    return f"SMC {normalized}"


def load_title(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_sections(
    title_data: Dict[str, object],
    path: Path,
    max_chars: int,
) -> Iterator[ChunkRecord]:
    title_number = int(title_data.get("number", 0))
    title_label = str(title_data.get("label", "")).strip() or f"Title {title_number}"

    for subtitle in title_data.get("subtitles", []) or []:
        subtitle_number = subtitle.get("number")
        subtitle_title = subtitle.get("title")
        # Chapters can appear directly under subtitles or inside divisions.
        for chapter in subtitle.get("chapters", []) or []:
            yield from chapter_to_chunks(
                title_number=title_number,
                title_label=title_label,
                subtitle_number=subtitle_number,
                subtitle_title=subtitle_title,
                division_number=None,
                division_title=None,
                chapter=chapter,
                source_path=path,
                max_chars=max_chars,
            )

        for division in subtitle.get("divisions", []) or []:
            division_number = division.get("number")
            division_title = division.get("title")
            for chapter in division.get("chapters", []) or []:
                yield from chapter_to_chunks(
                    title_number=title_number,
                    title_label=title_label,
                    subtitle_number=subtitle_number,
                    subtitle_title=subtitle_title,
                    division_number=division_number,
                    division_title=division_title,
                    chapter=chapter,
                    source_path=path,
                    max_chars=max_chars,
                )


def chapter_to_chunks(
    *,
    title_number: int,
    title_label: str,
    subtitle_number: Optional[str],
    subtitle_title: Optional[str],
    division_number: Optional[str],
    division_title: Optional[str],
    chapter: Dict[str, object],
    source_path: Path,
    max_chars: int,
) -> Iterator[ChunkRecord]:
    chapter_citation = chapter.get("citation")
    chapter_title = chapter.get("title")
    start_page = chapter.get("start_page")
    end_page = chapter.get("end_page")
    intro_text = (chapter.get("intro") or "").strip()
    if intro_text:
        for idx, chunk_text in enumerate(split_text(intro_text, max_chars)):
            yield ChunkRecord(
                chunk_id=build_chunk_id(
                    title_number, chapter_citation, suffix=f"chapter-intro-{idx}"
                ),
                chunk_type="chapter_intro",
                text=chunk_text,
                source_path=str(source_path),
                title_number=title_number,
                title_label=title_label,
                subtitle_number=subtitle_number,
                subtitle_title=subtitle_title,
                division_number=division_number,
                division_title=division_title,
                chapter_citation=chapter_citation,
                chapter_title=chapter_title,
                section_citation=None,
                section_heading=None,
                start_page=start_page,
                end_page=end_page,
            )

    for section in chapter.get("sections", []) or []:
        section_text = (section.get("text") or "").strip()
        if not section_text:
            continue
        section_citation = section.get("citation")
        section_heading = section.get("heading")
        section_start = section.get("start_page")
        section_end = section.get("end_page")
        for idx, chunk_text in enumerate(split_text(section_text, max_chars)):
            suffix = f"section-{sanitize_citation(section_citation)}-{idx}"
            yield ChunkRecord(
                chunk_id=build_chunk_id(title_number, chapter_citation, suffix=suffix),
                chunk_type="section",
                text=chunk_text,
                source_path=str(source_path),
                title_number=title_number,
                title_label=title_label,
                subtitle_number=subtitle_number,
                subtitle_title=subtitle_title,
                division_number=division_number,
                division_title=division_title,
                chapter_citation=chapter_citation,
                chapter_title=chapter_title,
                section_citation=section_citation,
                section_heading=section_heading,
                start_page=section_start,
                end_page=section_end,
            )


def sanitize_citation(citation: Optional[str]) -> str:
    if not citation:
        return "unknown"
    return re.sub(r"[^0-9a-zA-Z]+", "-", citation).strip("-").lower() or "unknown"


def build_chunk_id(title_number: int, chapter_citation: Optional[str], *, suffix: str) -> str:
    chapter_part = sanitize_citation(chapter_citation) if chapter_citation else "unknown"
    return f"smc-{title_number}-{chapter_part}-{suffix}"


def split_text(text: str, max_chars: int) -> List[str]:
    paragraphs = [
        para.strip()
        for para in re.split(r"\n\s*\n", text)
        if para and para.strip()
    ]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(split_long_paragraph(paragraph, max_chars))
            continue

        projected = current_len + (2 if current else 0) + len(paragraph)
        if projected <= max_chars:
            current.append(paragraph)
            current_len = projected
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    words = paragraph.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        additional = len(word) + (1 if current else 0)
        if current_len + additional > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += additional

    if current:
        chunks.append(" ".join(current))

    return chunks


def ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more Seattle Municipal Code title JSON files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "data/processed/smc_chunks.jsonl",
        help="Destination JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS_DEFAULT,
        help="Maximum characters per chunk (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_output_dir(args.output)

    total_chunks = 0
    total_sections = 0

    with args.output.open("w", encoding="utf-8") as writer:
        for input_path in args.inputs:
            title_data = load_title(input_path)
            for chunk in chunk_sections(title_data, input_path, args.max_chars):
                writer.write(chunk.to_json())
                writer.write("\n")
                total_chunks += 1
                if chunk.chunk_type == "section":
                    total_sections += 1

    print(
        f"Generated {total_chunks} chunks "
        f"({total_sections} section chunks) → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
