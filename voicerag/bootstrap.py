"""Wires the real components together (models, index, retriever, LLM).

Kept separate from `pipeline.py` so the pipeline itself only depends on small
interfaces and can be unit-tested with fakes, without PyTorch or an API key.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .embedder import Embedder
from .indexer import Index, load_or_build_index
from .llm_client import LLMClient
from .pipeline import VoiceRAGPipeline
from .reranker import Reranker
from .retriever import ChromaVectorStore, HybridRetriever


@dataclass
class App:
    """Everything the UI or the eval script needs."""

    pipeline: VoiceRAGPipeline
    index: Index
    settings: Settings


def build_app(settings: Settings, pdf_path: Path | None = None) -> App:
    """Load models, ingest the PDF if needed (cached by hash), and build the pipeline."""
    pdf = Path(pdf_path or settings.pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf}. Put the document at data/ or set PDF_PATH in .env."
        )
    embedder = Embedder(settings.embed_model, settings.embed_query_prefix)
    index = load_or_build_index(pdf, settings, embedder)

    updates = sum(1 for c in index.chunks if c.is_update)
    store = ChromaVectorStore(index.collection, total=len(index.chunks), updates=updates)
    retriever = HybridRetriever(index.chunks, store, embedder, settings)
    pipeline = VoiceRAGPipeline(settings, retriever, Reranker(settings.rerank_model), LLMClient(settings))
    return App(pipeline=pipeline, index=index, settings=settings)
