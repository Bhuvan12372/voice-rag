"""Cost and latency accounting.

* `Pricing` turns token counts into rupees.
* `QuestionMetrics` collects the LLM calls and per-stage timings of ONE question.
* `SessionLedger` keeps the running totals for a whole session.

Only LLM calls are included in this cost tracker. Speech recognition uses
Google's service and embedding/reranker models run locally; neither is included
in the LLM cost or stage timings.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Pricing:
    """Token prices in USD per million tokens plus the FX rate."""

    input_usd_per_m: float
    output_usd_per_m: float
    usd_to_inr: float

    def cost_inr(self, prompt_tokens: int, completion_tokens: int) -> float:
        usd = (prompt_tokens * self.input_usd_per_m + completion_tokens * self.output_usd_per_m) / 1_000_000
        return usd * self.usd_to_inr


@dataclass
class LLMCall:
    """One request to the LLM."""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cost_inr: float
    estimated: bool = False         # True when the provider returned no usage and we guessed


@dataclass
class QuestionMetrics:
    """Everything measured while answering one question."""

    calls: list[LLMCall] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _finished: float | None = field(default=None, repr=False)

    # ----- recording -----
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block of work. Repeated stage names accumulate."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + (time.perf_counter() - t0)

    def add_call(self, call: LLMCall) -> None:
        self.calls.append(call)

    def finish(self) -> None:
        """Stop the wall clock (called once when the answer is ready)."""
        self._finished = time.perf_counter()

    # ----- derived numbers -----
    @property
    def total_seconds(self) -> float:
        end = self._finished if self._finished is not None else time.perf_counter()
        return end - self._started

    @property
    def llm_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    @property
    def local_seconds(self) -> float:
        """Time not spent waiting for the LLM (embedding, search, reranking, glue)."""
        return max(0.0, self.total_seconds - self.llm_seconds)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def cost_inr(self) -> float:
        return sum(c.cost_inr for c in self.calls)


@dataclass
class SessionLedger:
    """Running totals across the questions of one session."""

    questions: list[QuestionMetrics] = field(default_factory=list)

    def add(self, metrics: QuestionMetrics) -> None:
        self.questions.append(metrics)

    @property
    def count(self) -> int:
        return len(self.questions)

    @property
    def total_cost_inr(self) -> float:
        return sum(q.cost_inr for q in self.questions)

    @property
    def average_cost_inr(self) -> float:
        return self.total_cost_inr / self.count if self.count else 0.0

    @property
    def average_seconds(self) -> float:
        return sum(q.total_seconds for q in self.questions) / self.count if self.count else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(q.prompt_tokens + q.completion_tokens for q in self.questions)
