"""Turn the spoken question into something a search engine can use.

One cheap LLM call does three jobs at once (so it adds one call, not three):

* resolve follow-ups: "Why is that?" -> "Why is Whitcombe Rock closed?"
* split multi-fact questions into sub-questions (total cost = fare + levy)
* detect "say that again more simply" requests, which need no new retrieval

A second method, `decompose`, is used only as a retry when the first answer was
"not found".
"""
from __future__ import annotations

from .config import Settings
from .cost_tracker import QuestionMetrics
from .llm_client import JSONLLM, LLMOutputError
from .models import Turn, Understanding
from .prompts import DECOMPOSE_SYSTEM, UNDERSTAND_SYSTEM

_INTENTS = {"new_question", "rephrase_previous"}


def format_history(history: list[Turn], answer_chars: int) -> str:
    """Render recent turns as text. Answers are included because follow-ups refer to them."""
    if not history:
        return "(no previous conversation)"
    lines = []
    for turn in history:
        answer = turn.answer if len(turn.answer) <= answer_chars else turn.answer[:answer_chars] + "..."
        lines.append(f"Learner: {turn.question}\nAssistant: {answer}")
    return "\n".join(lines)


def _clean_questions(value, limit: int) -> list[str]:
    """Keep only non-empty strings, de-duplicated, at most `limit`."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out[:limit]


class QueryUnderstanding:
    """Rewrites / decomposes the learner's question using the recent history."""

    def __init__(self, llm: JSONLLM, settings: Settings) -> None:
        self._llm = llm
        self._cfg = settings

    def understand(self, question: str, history: list[Turn], metrics: QuestionMetrics) -> Understanding:
        """Standalone question + sub-questions + intent. Falls back to the raw question on bad output."""
        if not history and not self._cfg.understand_always:
            return Understanding(standalone_question=question)

        user = (
            f"CONVERSATION SO FAR:\n{format_history(history, self._cfg.history_answer_chars)}\n\n"
            f"NEW QUESTION: {question}"
        )
        try:
            data = self._llm.complete_json(
                stage="understand", system=UNDERSTAND_SYSTEM, user=user, metrics=metrics, max_tokens=250
            )
        except LLMOutputError:
            return Understanding(standalone_question=question)   # safe fallback: search the raw text

        standalone = str(data.get("standalone_question") or "").strip() or question
        intent = data.get("intent") if data.get("intent") in _INTENTS else "new_question"
        subs = _clean_questions(data.get("sub_questions"), self._cfg.max_sub_questions)
        # A "sub-question" identical to the main question adds nothing.
        subs = [s for s in subs if s.lower() != standalone.lower()]
        return Understanding(standalone_question=standalone, sub_questions=subs, intent=intent)

    def decompose(self, question: str, missing: str, metrics: QuestionMetrics) -> list[str]:
        """Retry helper: break a question that failed into single-fact search queries."""
        user = f"QUESTION: {question}\nWHAT WAS MISSING: {missing or '(not stated)'}"
        try:
            data = self._llm.complete_json(
                stage="retry_decompose", system=DECOMPOSE_SYSTEM, user=user, metrics=metrics, max_tokens=200
            )
        except LLMOutputError:
            return []
        return _clean_questions(data.get("sub_questions"), self._cfg.max_sub_questions)
