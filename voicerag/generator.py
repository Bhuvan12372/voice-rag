"""Grounded answer generation: the LLM sees only the retrieved chunks."""
from __future__ import annotations

import re

from .config import Settings
from .cost_tracker import QuestionMetrics
from .llm_client import JSONLLM, LLMOutputError
from .models import RawAnswer, ScoredChunk
from .prompts import ANSWER_SYSTEM


def format_context(context: list[ScoredChunk]) -> str:
    """Render chunks as labelled sources.

    Chunk metadata only reaches the LLM if it is written into the prompt, so
    each source starts with its label: [Page 5 | Ch.3 ... | table]. A chunk from
    an updates chapter is additionally marked "LATER UPDATE".
    """
    return "\n\n".join(f"{sc.chunk.prompt_label}\n{sc.chunk.text}" for sc in context)


def _to_bool(value) -> bool:
    """Accept true/false and the strings models sometimes emit instead."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _to_pages(value) -> list[int]:
    """Coerce ['5', 'Page 11', 7] -> [5, 11, 7] (unique, order kept)."""
    if not isinstance(value, list):
        return []
    pages: list[int] = []
    for item in value:
        match = re.search(r"\d+", str(item))
        if match and int(match.group()) not in pages:
            pages.append(int(match.group()))
    return pages


def coerce_answer(data: dict) -> RawAnswer:
    """Validate/normalise the model's JSON into a `RawAnswer`."""
    return RawAnswer(
        found=_to_bool(data.get("found")),
        answer=str(data.get("answer") or "").strip(),
        pages=_to_pages(data.get("pages")),
        missing=str(data.get("missing") or "").strip(),
    )


class AnswerGenerator:
    """Builds the grounded prompt and parses the JSON reply."""

    def __init__(self, llm: JSONLLM, settings: Settings) -> None:
        self._llm = llm
        self._cfg = settings

    def generate(
        self,
        question: str,
        context: list[ScoredChunk],
        metrics: QuestionMetrics,
        *,
        stage: str = "answer",
        previous_answer: str = "",
    ) -> RawAnswer:
        """Ask the LLM; retry once if it returns unusable JSON, else raise `LLMOutputError`."""
        user = f"SOURCES:\n{format_context(context)}\n\nQUESTION: {question}"
        if previous_answer:
            user += (
                f"\n\n(The learner wants your previous answer said again in other words. "
                f"Previous answer: {previous_answer})"
            )
        last_error: LLMOutputError | None = None
        for _attempt in range(2):
            try:
                data = self._llm.complete_json(
                    stage=stage,
                    system=ANSWER_SYSTEM,
                    user=user,
                    metrics=metrics,
                    max_tokens=self._cfg.max_answer_tokens,
                )
                return coerce_answer(data)
            except LLMOutputError as exc:
                last_error = exc
        raise last_error  # type: ignore[misc]
