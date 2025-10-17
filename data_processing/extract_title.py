#!/usr/bin/env python3
"""
Extract and structure a Seattle Municipal Code title from the PDF supplement.

The script identifies the requested title (e.g., Title 22) by scanning page
headers, then parses the text into a hierarchical representation:
title → subtitle → division → chapter → section. The output is JSON intended to
support downstream validation and LLM-ready chunking.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PyPDF2 import PdfReader


@dataclass
class Section:
    citation: str
    heading: str
    start_page: int
    end_page: int
    body_lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "citation": self.citation,
            "heading": self.heading,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "text": "\n".join(self.body_lines).strip(),
        }

    def has_body(self) -> bool:
        return any(line.strip() for line in self.body_lines)


@dataclass
class Chapter:
    citation: str
    title: str
    start_page: int
    end_page: int
    intro_lines: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "citation": self.citation,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "intro": "\n".join(self.intro_lines).strip(),
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass
class Division:
    number: str
    title: str
    start_page: int
    end_page: int
    intro_lines: List[str] = field(default_factory=list)
    chapters: List[Chapter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "intro": "\n".join(self.intro_lines).strip(),
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }


@dataclass
class Subtitle:
    number: str
    title: str
    start_page: int
    end_page: int
    intro_lines: List[str] = field(default_factory=list)
    divisions: List[Division] = field(default_factory=list)
    chapters: List[Chapter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "intro": "\n".join(self.intro_lines).strip(),
            "divisions": [division.to_dict() for division in self.divisions],
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }


@dataclass
class TitleData:
    label: str
    number: int
    start_page: int
    end_page: int
    preface_lines: List[str] = field(default_factory=list)
    subtitles: List[Subtitle] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "number": self.number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "preface": "\n".join(self.preface_lines).strip(),
            "subtitles": [subtitle.to_dict() for subtitle in self.subtitles],
        }


DIVISION_PATTERN = re.compile(r"^Division\s+(\d+)\s+(.*)$", re.IGNORECASE)
FOOTER_PATTERN = re.compile(r"^\d{2}-\d+(?:\.\d+)?\s*\(Seattle\s+\d+-\d+\)$")


def parse_subtitle_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a subtitle header (e.g., 'Subtitle I Construction Codes').
    Returns (subtitle_number, subtitle_title) or None if the line is not a subtitle.
    """
    if not line.lower().startswith("subtitle"):
        return None
    parts = line.split(None, 2)
    if len(parts) < 2:
        return None
    number_token = re.sub(r"[.\-]+$", "", parts[1])
    if not re.fullmatch(r"[IVXLCDM]+(?:[A-Z])?", number_token, re.IGNORECASE):
        return None
    subtitle_number = f"Subtitle {number_token.upper()}"
    subtitle_title = parts[2].strip() if len(parts) == 3 else ""
    if subtitle_title and subtitle_title[0].islower():
        return None
    return subtitle_number, subtitle_title


def compile_heading_patterns(
    title_number: int,
) -> Tuple[re.Pattern, re.Pattern, re.Pattern]:
    """
    Build regex patterns for chapter and section headings using the supplied title
    number (e.g., 22 → Chapter 22.xx; Section 22.xx.xxx).
    """
    prefix = re.escape(str(title_number))
    chapter_full = re.compile(
        rf"^\s*Chapter\s+({prefix}\.[0-9A-Za-z]+)\s+(.*)$", re.IGNORECASE
    )
    chapter_short = re.compile(rf"^\s*Chapter\s+({prefix}\.[0-9A-Za-z]+)\s*$")
    section_pattern = re.compile(
        rf"^\s*({prefix}\.[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)+)\s+(.*)$"
    )
    return chapter_full, chapter_short, section_pattern


def normalize_page_text(raw: str) -> str:
    """
    Light-touch normalization for reliable parsing:
    - drop carriage returns
    - mend hyphenated line breaks (e.g., \"regula-\\ntion\" → \"regulation\")
    """
    if not raw:
        return ""
    text = raw.replace("\r", "")
    return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)


def iter_lines_with_pages(
    pages: Iterable[Tuple[int, str]]
) -> Iterator[Tuple[Optional[str], int]]:
    """
    Yield lines paired with their page number, keeping blank lines (as None) to
    preserve paragraph breaks. Page footers are filtered out.
    """
    for page_number, raw_text in pages:
        normalized = normalize_page_text(raw_text)
        for raw_line in normalized.split("\n"):
            if not raw_line.strip():
                yield None, page_number
                continue
            line = raw_line.rstrip()
            if FOOTER_PATTERN.match(line.strip()):
                continue
            yield line, page_number


def build_title_structure(
    lines: Iterable[Tuple[Optional[str], int]],
    title_label: str,
    title_number: int,
    start_page: int,
    end_page: int,
) -> TitleData:
    chapter_full_pattern, chapter_short_pattern, section_pattern = (
        compile_heading_patterns(title_number)
    )

    title = TitleData(
        label=title_label,
        number=title_number,
        start_page=start_page,
        end_page=end_page,
    )

    current_subtitle: Optional[Subtitle] = None
    current_division: Optional[Division] = None
    current_chapter: Optional[Chapter] = None
    current_section: Optional[Section] = None

    def finalize_section() -> None:
        nonlocal current_section, current_chapter
        if current_section and current_chapter:
            current_chapter.sections.append(current_section)
        current_section = None

    def finalize_chapter() -> None:
        nonlocal current_chapter, current_division, current_subtitle
        finalize_section()
        if current_chapter:
            if current_division is not None:
                current_division.chapters.append(current_chapter)
            elif current_subtitle is not None:
                current_subtitle.chapters.append(current_chapter)
            else:
                title.preface_lines.extend(current_chapter.intro_lines)
        current_chapter = None

    def finalize_division() -> None:
        nonlocal current_division, current_subtitle
        finalize_chapter()
        if current_division and current_subtitle:
            current_subtitle.divisions.append(current_division)
        current_division = None

    def finalize_subtitle() -> None:
        nonlocal current_subtitle
        finalize_division()
        if current_subtitle:
            title.subtitles.append(current_subtitle)
        current_subtitle = None

    for raw_line, page in lines:
        if raw_line is None:
            target = current_section or current_chapter or current_division or current_subtitle
            if isinstance(target, Section):
                target.body_lines.append("")
                target.end_page = page
            elif isinstance(target, Chapter):
                target.intro_lines.append("")
                target.end_page = page
            elif isinstance(target, Division):
                target.intro_lines.append("")
                target.end_page = page
            elif isinstance(target, Subtitle):
                target.intro_lines.append("")
                target.end_page = page
            else:
                title.preface_lines.append("")
            continue

        line = raw_line.strip()
        condensed = re.sub(r"\s+", " ", line)

        subtitle_info = parse_subtitle_line(condensed)
        if subtitle_info:
            finalize_subtitle()
            subtitle_number, subtitle_title = subtitle_info
            current_subtitle = Subtitle(
                number=subtitle_number,
                title=subtitle_title,
                start_page=page,
                end_page=page,
            )
            current_division = None
            current_chapter = None
            current_section = None
            continue

        division_match = DIVISION_PATTERN.match(condensed)
        if division_match and current_subtitle is not None:
            finalize_division()
            division_number = division_match.group(1)
            division_title = division_match.group(2).strip()
            current_division = Division(
                number=division_number,
                title=division_title,
                start_page=page,
                end_page=page,
            )
            current_chapter = None
            current_section = None
            continue

        chapter_match = chapter_full_pattern.match(condensed)
        chapter_short_match = chapter_short_pattern.match(condensed)
        if chapter_match or chapter_short_match:
            finalize_chapter()
            match = chapter_match or chapter_short_match
            assert match is not None
            chapter_citation = match.group(1)
            chapter_title = (
                chapter_match.group(2).strip() if chapter_match else ""
            )
            current_chapter = Chapter(
                citation=chapter_citation,
                title=chapter_title,
                start_page=page,
                end_page=page,
            )
            current_section = None
            continue

        section_match = section_pattern.match(condensed)
        if section_match and current_chapter is not None:
            finalize_section()
            section_citation = section_match.group(1)
            section_heading = section_match.group(2).strip()
            current_section = Section(
                citation=section_citation,
                heading=section_heading,
                start_page=page,
                end_page=page,
            )
            continue

        if (
            current_subtitle
            and not current_subtitle.title
            and current_chapter is None
            and not re.match(r"^Subtitle\b", line, re.IGNORECASE)
            and not line.lower().startswith("chapter ")
        ):
            if line.lower() != "sections:":
                current_subtitle.title = line
                current_subtitle.end_page = max(current_subtitle.end_page, page)
                continue

        if (
            current_chapter
            and not current_chapter.title
            and current_section is None
            and line.lower() != "sections:"
        ):
            current_chapter.title = line
            current_chapter.end_page = max(current_chapter.end_page, page)
            continue

        target = current_section or current_chapter or current_division or current_subtitle
        if isinstance(target, Section):
            target.body_lines.append(line)
            target.end_page = max(target.end_page, page)
        elif isinstance(target, Chapter):
            target.intro_lines.append(line)
            target.end_page = max(target.end_page, page)
        elif isinstance(target, Division):
            target.intro_lines.append(line)
            target.end_page = max(target.end_page, page)
        elif isinstance(target, Subtitle):
            target.intro_lines.append(line)
            target.end_page = max(target.end_page, page)
        else:
            title.preface_lines.append(line)

    finalize_subtitle()
    clean_title(title)
    return title


def clean_title(title: TitleData) -> None:
    """
    Deduplicate subtitles and sections and normalize inferred titles.
    """

    def clean_chapter(chapter: Chapter) -> None:
        deduped: List[Section] = []
        index_map: Dict[str, int] = {}

        for section in chapter.sections:
            key = section.citation
            if key in index_map:
                idx = index_map[key]
                existing = deduped[idx]
                if should_replace_section(existing, section):
                    deduped[idx] = section
            else:
                index_map[key] = len(deduped)
                deduped.append(section)
        chapter.sections = deduped

    def clean_subtitle(subtitle: Subtitle) -> None:
        for division in subtitle.divisions:
            for chapter in division.chapters:
                clean_chapter(chapter)
        for chapter in subtitle.chapters:
            clean_chapter(chapter)

    cleaned: List[Subtitle] = []
    subtitle_map: Dict[str, int] = {}

    for subtitle in title.subtitles:
        clean_subtitle(subtitle)
        key = subtitle.number.lower()
        if key in subtitle_map:
            idx = subtitle_map[key]
            existing = cleaned[idx]
            if should_replace_subtitle(existing, subtitle):
                cleaned[idx] = subtitle
        else:
            subtitle_map[key] = len(cleaned)
            cleaned.append(subtitle)

    cleaned.sort(key=lambda s: s.start_page)
    title.subtitles = cleaned


def should_replace_section(existing: Section, new: Section) -> bool:
    """
    Prefer sections that contain body text; otherwise, prefer the later section.
    """
    existing_has_body = existing.has_body()
    new_has_body = new.has_body()
    if new_has_body and not existing_has_body:
        return True
    if new_has_body == existing_has_body:
        if len(new.body_lines) > len(existing.body_lines):
            return True
        if new.end_page >= existing.end_page:
            return True
    return False


def should_replace_subtitle(existing: Subtitle, new: Subtitle) -> bool:
    """
    Prefer subtitles that carry more structure (chapters/divisions) or span later
    pages.
    """
    existing_weight = (
        len(existing.divisions)
        + len(existing.chapters)
        + sum(len(div.chapters) for div in existing.divisions)
    )
    new_weight = (
        len(new.divisions)
        + len(new.chapters)
        + sum(len(div.chapters) for div in new.divisions)
    )
    if new_weight > existing_weight:
        return True
    if new_weight == existing_weight and new.end_page >= existing.end_page:
        return True
    return False


def locate_title_range(
    reader: PdfReader,
    title_number: int,
    next_title_number: Optional[int],
) -> Tuple[int, int]:
    """
    Return (start_index, end_index) for the requested title. The end index is the
    first page of the next title, or the document end if no higher title is found.
    """
    pattern = re.compile(rf"^Title\s+{title_number}\b", re.IGNORECASE)
    if next_title_number is None:
        next_title_pattern = None
    else:
        next_title_pattern = re.compile(
            rf"^Title\s+{next_title_number}\b", re.IGNORECASE
        )

    start_index = None
    end_index = len(reader.pages)

    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if start_index is None and pattern.match(text):
            start_index = index
            continue
        if (
            start_index is not None
            and next_title_pattern is not None
            and next_title_pattern.match(text)
        ):
            end_index = index
            break

    if start_index is None:
        raise ValueError(f"Could not locate Title {title_number} in the PDF.")

    return start_index, end_index


def extract_pages(reader: PdfReader, start: int, end: int) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    for index in range(start, end):
        page_number = index + 1  # human-readable numbering
        text = reader.pages[index].extract_text() or ""
        pages.append((page_number, text))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a Seattle Municipal Code title into structured JSON."
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        nargs="?",
        default=Path("Seattle, WA Municipal Code.pdf"),
        help="Path to the municipal code supplement PDF.",
    )
    parser.add_argument(
        "--title",
        type=int,
        default=23,
        help="Title number to extract (e.g., 22, 23).",
    )
    parser.add_argument(
        "--next-title",
        type=int,
        default=None,
        help="Next title number; used to determine the end page. Defaults to title+1.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the title (defaults to 'Title {n}').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path (defaults to 'title{n}.json').",
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    title_number: int = args.title
    next_title_number: Optional[int] = (
        args.next_title if args.next_title is not None else title_number + 1
    )

    reader = PdfReader(str(pdf_path))
    start_idx, end_idx = locate_title_range(reader, title_number, next_title_number)
    pages = extract_pages(reader, start_idx, end_idx)

    label = args.label or f"Title {title_number}"
    output_path = args.output or Path(f"title{title_number}.json")

    line_iter = iter_lines_with_pages(pages)
    title = build_title_structure(
        line_iter,
        title_label=label,
        title_number=title_number,
        start_page=pages[0][0],
        end_page=pages[-1][0],
    )
    data = title.to_dict()

    output_path.write_text(json.dumps(data, indent=2))
    print(
        f"Extracted {label}: pages {data['start_page']}–{data['end_page']} -> {output_path}"
    )


if __name__ == "__main__":
    main()
