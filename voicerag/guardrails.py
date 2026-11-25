"""Checks applied to the model's answer before the learner sees it.

1. `found` must be true AND the answer must cite at least one real page.
2. Cited pages must be a subset of the pages the LLM was actually shown, so
   the model cannot invent a citation. Invented pages are dropped; if nothing
   verifiable remains, the answer is rejected as unsupported.
3. A very low best reranker score means the question is unrelated to the
   document, so we can answer "not in the document" without calling the LLM.

The retrieval score is only ever used to *skip* work (rule 3), never to turn a
"not found" into "found": related text is not the same as an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .models import RawAnswer, ScoredChunk


@dataclass
class GuardrailResult:
    """Verdict on one raw answer."""

    found: bool
    answer: str
    pages: list[int]
    missing: str = ""
    warnings: list[str] = field(default_factory=list)


def split_citations(cited: list[int], context: list[ScoredChunk]) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) cited pages, comparing against the pages shown to the LLM."""
    allowed = {sc.chunk.page for sc in context}
    valid = sorted({p for p in cited if p in allowed})
    invalid = sorted({p for p in cited if p not in allowed})
    return valid, invalid


def apply_guardrails(raw: RawAnswer, context: list[ScoredChunk]) -> GuardrailResult:
    """Decide whether `raw` may be shown as a found, cited answer."""
    if not raw.found:
        return GuardrailResult(False, "", [], missing=raw.missing)
    if not raw.answer:
        return GuardrailResult(False, "", [], missing="empty answer", warnings=["Model said found but gave no answer."])

    valid, invalid = split_citations(raw.pages, context)
    warnings: list[str] = []
    if invalid:
        warnings.append(f"Dropped citations to pages that were not retrieved: {invalid}")
    if not valid:
        warnings.append("Rejected: the answer had no verifiable page citation.")
        return GuardrailResult(False, "", [], missing="no verifiable citation", warnings=warnings)
    return GuardrailResult(True, raw.answer, valid, warnings=warnings)


def below_relevance_floor(best_score: float, settings: Settings) -> bool:
    """True when even the best chunk is so irrelevant that the LLM call can be skipped."""
    return best_score < settings.rerank_min_score
