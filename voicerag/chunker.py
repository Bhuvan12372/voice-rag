"""Parsed pages -> retrievable `Chunk`s with metadata.

Rules (each one exists to fix a concrete retrieval problem):

* A chunk never crosses a page boundary, so every citation is a single page.
* Prose is packed by paragraph up to `chunk_max_chars`; a chunk never crosses a
  heading, so each chunk is about one topic.
* A table becomes its own chunk(s) as *markdown with the header row*. Long
  tables are split by rows and the header is repeated in every piece.
* The short note right under a table ("see Chapter 9") stays in the table's
  chunk, because it is a pointer the answer may need.
* For search (embeddings / BM25 / reranker) tables are described row-by-row as
  sentences, while the LLM still sees the original markdown.
* Chunks under an "updates/errata" chapter are flagged `is_update=True`, and
  "see Chapter N" references are stored in `see_also`.
"""
from __future__ import annotations

import re

from .config import Settings
from .models import Chunk
from .pdf_parser import Block, ParsedPage

# A paragraph directly under a table is treated as that table's note if it is
# italic or short (notes are one line; body paragraphs are much longer).
NOTE_MAX_CHARS = 200

_CHAPTER_HEADING = re.compile(r"^(\d+)\.\s+(.*)$")
# Matches "Chapter 9", "chapters 3 and 6", "Chapters 2, 4 & 7". Numbers are limited
# to 1-2 digits so that a year in "see Chapter 9, 2025 Updates" is not read as a chapter.
_CHAPTER_REF = re.compile(
    r"\bChapters?\s+(\d{1,2}(?!\d)(?:\s*(?:,|and|&)\s*\d{1,2}(?!\d))*)", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------
def split_chapter_heading(text: str) -> tuple[int | None, str]:
    """'3. Getting There' -> (3, 'Getting There'); unnumbered -> (None, text)."""
    match = _CHAPTER_HEADING.match(text.strip())
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, text.strip()


def find_chapter_refs(text: str, own_chapter: int | None) -> list[int]:
    """Chapter numbers mentioned in `text` ('see Chapter 9'), excluding the chunk's own."""
    refs = {int(n) for group in _CHAPTER_REF.findall(text) for n in re.findall(r"\d+", group)}
    refs.discard(own_chapter if own_chapter is not None else -1)
    return sorted(refs)


def pack_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedily pack paragraphs into pieces of at most `max_chars`.

    A single paragraph longer than the limit is split at sentence boundaries
    (as a last resort), so no text is ever dropped.
    """
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = [paragraph]
        if len(paragraph) > max_chars:
            parts = _split_long(paragraph, max_chars)
        for part in parts:
            candidate = f"{current}\n\n{part}" if current else part
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph into sentence groups under `max_chars`."""
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    return out


# ----------------------------------------------------------------------------
# table helpers
# ----------------------------------------------------------------------------
def _md_cell(cell: str) -> str:
    return cell.replace("|", "\\|")


def table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Render header + rows as a markdown table."""
    width = len(header)
    lines = ["| " + " | ".join(_md_cell(c) for c in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        padded = (row + [""] * width)[:width]          # tolerate ragged rows
        lines.append("| " + " | ".join(_md_cell(c) for c in padded) + " |")
    return "\n".join(lines)


def table_to_sentences(header: list[str], rows: list[list[str]]) -> str:
    """One line per row: 'Season: Summer; Days: Daily; Depart Carrow Haven: 08:30 ...'.

    Used only for searching: cross-encoders and embedding models are trained on
    prose, so this scores much better than pipes and dashes.
    """
    lines = []
    for row in rows:
        cells = (row + [""] * len(header))[: len(header)]
        lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# the chunker
# ----------------------------------------------------------------------------
class _ChunkBuilder:
    """Walks the blocks in reading order, tracking chapter / section state."""

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings
        self.update_re = re.compile(settings.update_chapter_pattern, re.IGNORECASE)
        self.chunks: list[Chunk] = []
        self.chapter_number: int | None = None
        self.chapter_title = ""
        self.section_title = ""
        self.buffer: list[str] = []          # paragraphs waiting to be packed
        self.page = 0
        self._counter = 0
        self._last_table: list[Chunk] = []   # chunks of the table directly above

    # ----- chunk creation -----
    def _new_chunk(self, text: str, kind: str, **extra) -> Chunk:
        self._counter += 1
        return Chunk(
            chunk_id=f"p{self.page:02d}-{self._counter:02d}",
            text=text,
            page=self.page,
            chapter_number=self.chapter_number,
            chapter_title=self.chapter_title,
            section_title=self.section_title,
            kind=kind,
            is_update=bool(self.update_re.search(self.chapter_title)),
            see_also=find_chapter_refs(text, self.chapter_number),
            **extra,
        )

    def flush_prose(self) -> None:
        """Turn the buffered paragraphs into one or more text chunks."""
        if self.buffer:
            for piece in pack_paragraphs(self.buffer, self.cfg.chunk_max_chars):
                self.chunks.append(self._new_chunk(piece, "text"))
        self.buffer = []

    def add_table(self, rows: list[list[str]]) -> None:
        """Add a table as markdown chunk(s); long tables repeat the header."""
        header, body = rows[0], rows[1:]
        step = self.cfg.table_max_rows
        groups = [body[i : i + step] for i in range(0, len(body), step)] or [[]]
        self._last_table = []
        for group in groups:
            chunk = self._new_chunk(
                table_to_markdown(header, group),
                "table",
                table_columns=", ".join(header),
                search_body=table_to_sentences(header, group),
            )
            self.chunks.append(chunk)
            self._last_table.append(chunk)

    def attach_note(self, note: str) -> None:
        """Append the note under a table to the table chunk(s), text and search body."""
        for chunk in self._last_table:
            chunk.text = f"{chunk.text}\n\n{note}"
            chunk.search_body = f"{chunk.search_body}\n{note}"
            chunk.see_also = find_chapter_refs(chunk.text, chunk.chapter_number)
        self._last_table = []

    # ----- block dispatch -----
    def feed(self, block: Block, previous_was_table: bool) -> None:
        if block.kind == "heading1":
            self.flush_prose()
            self.chapter_number, self.chapter_title = split_chapter_heading(block.text)
            self.section_title = ""
        elif block.kind == "heading2":
            self.flush_prose()
            self.section_title = block.text
        elif block.kind == "table":
            self.flush_prose()
            if block.rows:
                self.add_table(block.rows)
        else:  # paragraph
            is_note = previous_was_table and (block.italic or len(block.text) <= NOTE_MAX_CHARS)
            if is_note and self._last_table:
                self.attach_note(block.text)
            else:
                self.buffer.append(block.text)

    def start_page(self, number: int) -> None:
        """Chunks never cross pages: flush whatever is pending from the last page."""
        self.flush_prose()
        self.page = number
        self._last_table = []


def build_chunks(pages: list[ParsedPage], settings: Settings) -> list[Chunk]:
    """Create all chunks for a parsed document, in reading order."""
    builder = _ChunkBuilder(settings)
    for page in pages:
        builder.start_page(page.number)
        previous_kind = ""
        for block in page.blocks:
            builder.feed(block, previous_was_table=(previous_kind == "table"))
            previous_kind = block.kind
    builder.flush_prose()
    return builder.chunks
