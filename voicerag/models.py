"""Plain data containers shared across the pipeline.

Kept free of heavy logic so every other module can import them without
pulling in ML libraries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from .cost_tracker import QuestionMetrics

# What the learner sees when the answer is not in the document.
NOT_FOUND_MESSAGE = "This isn't in the document."


@dataclass
class Chunk:
    """One retrievable piece of the document, always from a single page."""

    chunk_id: str
    text: str                       # exactly what the answer LLM reads (markdown for tables)
    page: int                       # 1-based PDF page: this is what we cite
    chapter_number: Optional[int] = None
    chapter_title: str = ""
    section_title: str = ""
    kind: str = "text"              # "text" | "table"
    is_update: bool = False         # True if it sits in an updates/errata chapter
    see_also: list[int] = field(default_factory=list)  # chapters this chunk points to
    table_columns: str = ""         # for tables: "Season, Days, ..." (debug / metadata only)
    # Text used for searching instead of `text`. For tables this is one sentence
    # per row ("Season: Summer; Days: Daily; ..."), because search models are
    # trained on prose and score markdown tables poorly. Empty = use `text`.
    search_body: str = ""

    # ----- derived views -----------------------------------------------------
    @property
    def context_line(self) -> str:
        """Breadcrumb such as 'Chapter 3: Getting There | The mainland ferry'.

        Tables carry no title of their own, so this line is prepended to the
        text we embed. It is NOT part of the evidence shown to the LLM.
        """
        parts: list[str] = []
        if self.chapter_number is not None:
            parts.append(f"Chapter {self.chapter_number}: {self.chapter_title}")
        elif self.chapter_title:
            parts.append(self.chapter_title)
        if self.section_title:
            parts.append(self.section_title)
        return " | ".join(parts)

    @property
    def embed_text(self) -> str:
        """Text used for embeddings, BM25 and reranking (breadcrumb + body)."""
        body = self.search_body or self.text
        ctx = self.context_line
        return f"{ctx}\n{body}" if ctx else body

    @property
    def prompt_label(self) -> str:
        """Header line placed before the chunk in the LLM prompt.

        Metadata only reaches the LLM if it is written into the prompt, so the
        page, chapter and 'overrides' marker are all spelled out here.
        """
        parts = [f"Page {self.page}"]
        if self.chapter_number is not None:
            parts.append(f"Ch.{self.chapter_number} {self.chapter_title}")
        elif self.chapter_title:
            parts.append(self.chapter_title)
        if self.section_title:
            parts.append(self.section_title)
        if self.kind == "table":
            parts.append("table")
        if self.is_update:
            parts.append("LATER UPDATE - overrides conflicting earlier chapters")
        return "[" + " | ".join(parts) + "]"

    # ----- (de)serialisation for chunks.json ---------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ScoredChunk:
    """A chunk together with its reranker score and how it got into the context."""

    chunk: Chunk
    score: float
    origin: str = "retrieved"       # "retrieved" | "override"


@dataclass
class Turn:
    """One finished question/answer pair kept for follow-up questions."""

    question: str
    answer: str


@dataclass
class Understanding:
    """Output of the query-understanding step."""

    standalone_question: str
    sub_questions: list[str] = field(default_factory=list)
    intent: str = "new_question"    # "new_question" | "rephrase_previous"


@dataclass
class RawAnswer:
    """The answer LLM's JSON output after type-coercion, before guardrails."""

    found: bool
    answer: str = ""
    pages: list[int] = field(default_factory=list)
    missing: str = ""               # what the sources lacked (used to guide a retry)


@dataclass
class FinalAnswer:
    """Everything the UI / eval needs about one answered question."""

    question: str                   # as the learner said it
    found: bool
    answer: str
    pages: list[int]                # verified citations only
    sources: list[ScoredChunk]      # the chunks the LLM was shown
    understanding: Optional[Understanding]
    path: list[str]                 # human-readable trace of what the pipeline did
    warnings: list[str]
    metrics: QuestionMetrics
    best_score: Optional[float] = None   # best reranker score of the first retrieval
