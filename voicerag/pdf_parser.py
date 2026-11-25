"""PDF -> ordered, typed blocks (headings, paragraphs, tables) per page.

Why not just `page.extract_text()`? Plain text extraction scrambles tables
(cells of different columns end up on the same line) and loses which lines are
headings. Here we:

1. find tables with pdfplumber and extract them as rows,
2. extract the remaining text as lines *with font information*,
3. classify lines into headings / paragraphs using font size and weight,
4. drop boilerplate (the repeated page header/footer) detected automatically.

Nothing in this module knows about the Kestrel document: the heading rules are
relative to the document's own body font size.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# --- layout heuristics (relative, so they work for other documents too) ------
HEADING1_RATIO = 1.5        # line is a chapter heading if size >= 1.5 x body size
HEADING2_MIN_DELTA = 1.0    # bold and >= body + 1pt => section heading
PARAGRAPH_GAP_RATIO = 0.6   # vertical gap above 0.6 x font size starts a new paragraph
MARGIN_FRACTION = 0.08      # top/bottom 8% of a page counts as header/footer area
BOILERPLATE_SHARE = 0.5     # repeated on >= 50% of pages (and >= 3 pages) => boilerplate


@dataclass
class Block:
    """A unit of page content, in reading order."""

    kind: str                                   # "heading1" | "heading2" | "paragraph" | "table"
    text: str = ""                              # heading / paragraph text
    rows: list[list[str]] = field(default_factory=list)   # table rows (first row = header)
    italic: bool = False                        # paragraphs only; notes are usually italic


@dataclass
class ParsedPage:
    """All blocks of one page. `number` is the 1-based PDF page index."""

    number: int
    blocks: list[Block]


@dataclass
class _Line:
    """A single text line with the font facts we classify on."""

    text: str
    top: float
    bottom: float
    size: float
    bold: bool
    italic: bool


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _clean(text: str) -> str:
    """Collapse all whitespace (including newlines inside table cells)."""
    return re.sub(r"\s+", " ", text or "").strip()


def _normalise_for_repeat_check(text: str) -> str:
    """Make 'Page 3' and 'Page 4' compare equal by masking digits."""
    return re.sub(r"\d+", "#", text.strip().lower())


def _inside(char: dict, boxes: list[tuple]) -> bool:
    """True if the centre of `char` lies inside any of the bounding boxes."""
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _extract_lines(page: "pdfplumber.page.Page", boxes: list[tuple]) -> list[_Line]:
    """Text lines of a page, excluding anything inside a table box."""

    def outside_tables(obj: dict) -> bool:
        return obj.get("object_type") != "char" or not _inside(obj, boxes)

    lines: list[_Line] = []
    for raw in page.filter(outside_tables).extract_text_lines():
        text = _clean(raw["text"])
        if not text:
            continue
        first = raw["chars"][0]
        font = first["fontname"].lower()
        lines.append(
            _Line(
                text=text,
                top=raw["top"],
                bottom=raw["bottom"],
                size=round(first["size"], 1),
                bold="bold" in font,
                italic="italic" in font or "oblique" in font,
            )
        )
    return lines


def _find_boilerplate(per_page_lines: list[list[_Line]], page_height: float) -> set[str]:
    """Return normalised texts of lines that repeat in the page margins.

    This finds the running header/footer ("The Kestrel Islands · ... Page N")
    without hard-coding it. Keeping it would put the word "Kestrel" into every
    chunk and pollute BM25 and the embeddings.
    """
    counts: Counter[str] = Counter()
    for lines in per_page_lines:
        seen_on_page = {
            _normalise_for_repeat_check(ln.text)
            for ln in lines
            if ln.top < page_height * MARGIN_FRACTION or ln.bottom > page_height * (1 - MARGIN_FRACTION)
        }
        counts.update(seen_on_page)
    threshold = max(3, BOILERPLATE_SHARE * len(per_page_lines))
    return {text for text, n in counts.items() if n >= threshold}


def _body_font_size(per_page_lines: list[list[_Line]]) -> float:
    """Most common font size, weighted by characters = the body text size."""
    weights: Counter[float] = Counter()
    for lines in per_page_lines:
        for ln in lines:
            weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 11.0


def _classify(line: _Line, body_size: float) -> str:
    """Decide whether a line is a chapter heading, section heading or body text."""
    if line.size >= body_size * HEADING1_RATIO:
        return "heading1"
    if line.bold and line.size >= body_size + HEADING2_MIN_DELTA:
        return "heading2"
    return "paragraph"


def _table_rows(table) -> list[list[str]]:
    """Extract a pdfplumber table as cleaned rows, dropping empty rows."""
    rows = [[_clean(cell or "") for cell in row] for row in table.extract()]
    return [row for row in rows if any(row)]


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def parse_pdf(path: str | Path) -> list[ParsedPage]:
    """Parse a PDF into pages of ordered blocks.

    Returns one `ParsedPage` per PDF page. Page numbers come from the PDF page
    index (not from text printed on the page), so citations always match the
    page the learner sees in a PDF viewer.
    """
    with pdfplumber.open(str(path)) as pdf:
        # ---- pass 1: collect lines and tables of every page ----
        page_lines: list[list[_Line]] = []
        page_tables: list[list[tuple[float, list[list[str]]]]] = []   # (top, rows)
        for page in pdf.pages:
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]
            page_lines.append(_extract_lines(page, boxes))
            page_tables.append([(t.bbox[1], _table_rows(t)) for t in tables])
        height = float(pdf.pages[0].height) if pdf.pages else 842.0

    boilerplate = _find_boilerplate(page_lines, height)
    page_lines = [
        [ln for ln in lines if _normalise_for_repeat_check(ln.text) not in boilerplate]
        for lines in page_lines
    ]
    body_size = _body_font_size(page_lines)

    # ---- pass 2: turn lines + tables into ordered blocks ----
    parsed: list[ParsedPage] = []
    for index, (lines, tables) in enumerate(zip(page_lines, page_tables), start=1):
        # Merge lines and tables into one list ordered by vertical position.
        items: list[tuple[float, str, object]] = [(ln.top, "line", ln) for ln in lines]
        items += [(top, "table", rows) for top, rows in tables if rows]
        items.sort(key=lambda it: it[0])
        parsed.append(ParsedPage(number=index, blocks=_group_items(items, body_size)))
    return parsed


def _group_items(items: list[tuple[float, str, object]], body_size: float) -> list[Block]:
    """Group consecutive lines into blocks; tables become their own block."""
    blocks: list[Block] = []
    current: list[_Line] = []
    current_kind = ""

    def flush() -> None:
        nonlocal current, current_kind
        if current:
            blocks.append(
                Block(
                    kind=current_kind,
                    text=" ".join(ln.text for ln in current),
                    italic=all(ln.italic for ln in current),
                )
            )
        current, current_kind = [], ""

    for _top, what, payload in items:
        if what == "table":
            flush()
            blocks.append(Block(kind="table", rows=payload))  # type: ignore[arg-type]
            continue
        line: _Line = payload  # type: ignore[assignment]
        kind = _classify(line, body_size)
        if current:
            prev = current[-1]
            gap = line.top - prev.bottom
            # Headings may wrap over several lines (up to one font size apart);
            # body paragraphs break at a larger-than-line-spacing gap.
            max_gap = line.size * (1.0 if kind.startswith("heading") else PARAGRAPH_GAP_RATIO)
            if kind != current_kind or gap > max_gap or line.italic != prev.italic:
                flush()
        current_kind = kind
        current.append(line)
    flush()
    return blocks
