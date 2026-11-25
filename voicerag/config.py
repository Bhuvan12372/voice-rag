"""Central configuration.

Every tunable number lives here so that the rest of the code has no magic
constants. Values can be overridden with environment variables (or a `.env`
file); see `.env.example`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bump this whenever chunking logic changes. It is part of the cache key, so an
# old index is never silently reused after the chunking rules change.
CHUNKING_VERSION = 1

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default, cast=str):
    """Read an environment variable, falling back to `default` when unset/empty."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    """All settings for one run of the app. Immutable so it is safe to share."""

    # ---- Paths ----
    pdf_path: Path = PROJECT_ROOT / "data" / "Kestrel_Islands_Field_Guide.pdf"
    cache_dir: Path = PROJECT_ROOT / "cache"

    # ---- Local models ----
    embed_model: str = "BAAI/bge-base-en-v1.5"
    # bge models are trained with this instruction in front of *queries* (not documents).
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Chunking ----
    chunk_max_chars: int = 900          # prose is split at paragraph boundaries above this size
    table_max_rows: int = 20            # long tables are split by rows (header repeated)
    # Chapters whose title matches this regex are flagged is_update=True at ingest.
    # Document-specific heuristic (see README, "Known limitations").
    update_chapter_pattern: str = r"update|errata|revision|correction"

    # ---- Retrieval ----
    dense_k: int = 20                   # candidates from vector search
    lexical_k: int = 20                 # candidates from BM25
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    candidate_k: int = 12             # fused candidates sent to the reranker
    rerank_top_n: int = 5               # chunks kept for a single-part question
    sub_query_top_n: int = 3            # chunks kept per sub-question for multi-part questions
    max_context_chunks: int = 7         # hard cap on chunks sent to the answer LLM
    override_n: int = 2                 # max "update" chunks forced into the context
    update_candidates: int = 10         # update chunks fetched (filtered search) before reranking
    max_sub_questions: int = 3          # cap on decomposition fan-out

    # ---- Guardrail thresholds (scores are in the reranker's output scale) ----
    # Below this the app short-circuits to "not in the document" with no LLM call.
    # Deliberately low: it only catches clearly unrelated questions. The LLM's own
    # `found` flag plus the citation check do the real work. Tune using eval output.
    rerank_min_score: float = 0.01
    # Per-chunk minimum for inclusion in the answer context (not the best-score guard).
    context_min_rerank_score: float = 0.001
    # An update chunk is forced into the context only if it scores at least this.
    update_min_score: float = 0.05

    # ---- Conversation ----
    history_turns: int = 3
    history_answer_chars: int = 500     # previous answers are truncated to this in the rewrite prompt
    understand_always: bool = True      # run the rewrite/decompose call even with no history
    retry_on_not_found: bool = True     # one decomposition retry when the answer is "not found"

    # ---- LLM ----
    llm_api_key: str = field(default="", repr=False)   # repr=False keeps the key out of logs
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_json_mode: bool = True
    llm_timeout_s: float = 30.0
    max_answer_tokens: int = 400

    # ---- Speech transcription ----
    groq_api_key: str = field(default="", repr=False)
    whisper_model: str = "whisper-large-v3"

    # ---- Pricing (for the notional cost tracker) ----
    price_input_usd_per_m: float = 0.10
    price_output_usd_per_m: float = 0.40
    usd_to_inr: float = 88.0
    pricing_note: str = "placeholder prices - update before reporting"


def load_settings() -> Settings:
    """Build Settings from defaults overridden by environment variables."""
    d = Settings()
    return Settings(
        pdf_path=Path(_env("PDF_PATH", d.pdf_path)),
        cache_dir=Path(_env("CACHE_DIR", d.cache_dir)),
        embed_model=_env("EMBED_MODEL", d.embed_model),
        rerank_model=_env("RERANK_MODEL", d.rerank_model),
        rerank_min_score=_env("RERANK_MIN_SCORE", d.rerank_min_score, float),
        context_min_rerank_score=_env(
            "CONTEXT_MIN_RERANK_SCORE", d.context_min_rerank_score, float
        ),
        update_min_score=_env("UPDATE_MIN_SCORE", d.update_min_score, float),
        history_turns=_env("HISTORY_TURNS", d.history_turns, int),
        understand_always=_env("UNDERSTAND_ALWAYS", d.understand_always, bool),
        retry_on_not_found=_env("RETRY_ON_NOT_FOUND", d.retry_on_not_found, bool),
        llm_api_key=_env("LLM_API_KEY", ""),
        llm_base_url=_env("LLM_BASE_URL", d.llm_base_url),
        llm_model=_env("LLM_MODEL", d.llm_model),
        llm_json_mode=_env("LLM_JSON_MODE", d.llm_json_mode, bool),
        groq_api_key=_env("GROQ_API_KEY", ""),
        whisper_model=_env("WHISPER_MODEL", d.whisper_model),
        price_input_usd_per_m=_env("PRICE_INPUT_USD_PER_M", d.price_input_usd_per_m, float),
        price_output_usd_per_m=_env("PRICE_OUTPUT_USD_PER_M", d.price_output_usd_per_m, float),
        usd_to_inr=_env("USD_TO_INR", d.usd_to_inr, float),
        pricing_note=_env("PRICING_NOTE", d.pricing_note),
    )
