"""Thin wrapper around an OpenAI-compatible chat API that returns parsed JSON.

Works with Gemini (via Google's OpenAI-compatible endpoint), Groq, OpenAI and
others: only the base URL, key and model name change. Every call is recorded in
the `QuestionMetrics` it is given, using the token counts the provider reports.
"""
from __future__ import annotations

import json
import re
import time
from typing import Protocol

from .config import Settings
from .cost_tracker import LLMCall, Pricing, QuestionMetrics


class LLMOutputError(ValueError):
    """The model replied, but not with usable JSON."""


class JSONLLM(Protocol):
    """Interface the rest of the app depends on (tests supply a fake)."""

    def complete_json(
        self, *, stage: str, system: str, user: str, metrics: QuestionMetrics, max_tokens: int
    ) -> dict:
        """Run one call and return the parsed JSON object."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and chatter."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError(f"No JSON object in model output: {text[:120]!r}") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Invalid JSON in model output: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMOutputError("Model output is JSON but not an object.")
    return value


class LLMClient:
    """`JSONLLM` implementation using the `openai` SDK."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import

        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
        self._cfg = settings
        self._pricing = Pricing(
            settings.price_input_usd_per_m, settings.price_output_usd_per_m, settings.usd_to_inr
        )
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=1,
        )

    def complete_json(
        self, *, stage: str, system: str, user: str, metrics: QuestionMetrics, max_tokens: int
    ) -> dict:
        """One chat call at temperature 0; records usage, cost and latency."""
        kwargs: dict = dict(
            model=self._cfg.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        if self._cfg.llm_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        seconds = time.perf_counter() - started

        text = response.choices[0].message.content or ""
        usage = response.usage
        estimated = usage is None
        if usage is not None:
            prompt_tokens, completion_tokens = usage.prompt_tokens, usage.completion_tokens
        else:  # provider sent no usage block: ~4 characters per token, flagged as an estimate
            prompt_tokens, completion_tokens = (len(system) + len(user)) // 4, len(text) // 4

        metrics.add_call(
            LLMCall(
                stage=stage,
                model=self._cfg.llm_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                seconds=seconds,
                cost_inr=self._pricing.cost_inr(prompt_tokens, completion_tokens),
                estimated=estimated,
            )
        )
        return parse_json_object(text)
