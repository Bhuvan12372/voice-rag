"""Hybrid retrieval: dense vectors + BM25, merged with Reciprocal Rank Fusion.

Dense search finds paraphrases ("when does the boat leave" ~ "departs"); BM25
finds exact tokens (names such as "Whitcombe", times such as "09:15"). Their
scores live on different scales, so we fuse by *rank* (RRF) instead of
weighting raw scores.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .config import Settings
from .models import Chunk

_TOKEN = re.compile(r"\d+[:.]\d+|\w+")      # keeps times like 08:30 as one token


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; times such as '09:15' stay intact."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of documents (small, dependency-free).

    Held in memory and rebuilt from chunks.json at load time. Cost grows
    linearly with the number of chunks, which is milliseconds for thousands of
    chunks; see the README for the upgrade path at larger scale.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(doc)) for doc in documents]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documents)
        # "+1" inside the log keeps IDF non-negative even for very common terms.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for `query`."""
        terms = tokenize(query)
        out = []
        for tf, length in zip(self._tf, self._len):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = freq + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
                score += self._idf.get(term, 0.0) * freq * (self.k1 + 1) / norm
            out.append(score)
        return out

    def top(self, query: str, k: int) -> list[int]:
        """Indices of the best `k` documents with a positive score, best first."""
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:k] if scores[i] > 0]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion.

    Each ranked list gives an item `1 / (k + rank)` (rank starts at 1); scores
    are summed over lists. Items ranked well by several lists win. Ties keep the
    order of first appearance, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))


class VectorStore(Protocol):
    """What the retriever needs from a vector database."""

    def query(self, embedding: list[float], n: int, only_updates: bool = False) -> list[str]:
        """Chunk ids nearest to `embedding`, best first."""


class ChromaVectorStore:
    """`VectorStore` backed by a Chroma collection (which also holds metadata)."""

    def __init__(self, collection, total: int, updates: int) -> None:
        self._collection = collection
        self._total = total
        self._updates = updates

    def query(self, embedding: list[float], n: int, only_updates: bool = False) -> list[str]:
        available = self._updates if only_updates else self._total
        n = min(n, available)
        if n <= 0:
            return []
        kwargs = {"where": {"is_update": True}} if only_updates else {}
        result = self._collection.query(query_embeddings=[embedding], n_results=n, **kwargs)
        return list(result["ids"][0])


class HybridRetriever:
    """Dense + BM25 search over the chunks, fused with RRF."""

    def __init__(self, chunks: list[Chunk], store: VectorStore, embedder, settings: Settings) -> None:
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._store = store
        self._embedder = embedder
        self._cfg = settings
        # BM25 sees exactly the same text as the embeddings (breadcrumb + body).
        self._bm25 = BM25Index([c.embed_text for c in chunks])

    def search(self, query: str) -> list[Chunk]:
        """Top `candidate_k` chunks for `query` by hybrid search (unordered quality; rerank next)."""
        dense_ids = self._store.query(self._embedder.embed_query(query), self._cfg.dense_k)
        lexical_ids = [self.chunks[i].chunk_id for i in self._bm25.top(query, self._cfg.lexical_k)]
        fused = rrf_fuse([dense_ids, lexical_ids], k=self._cfg.rrf_k)
        return [self._by_id[cid] for cid in fused[: self._cfg.candidate_k]]

    def search_updates(self, query: str, n: int) -> list[Chunk]:
        """Nearest chunks among those flagged `is_update` (metadata-filtered vector search)."""
        ids = self._store.query(self._embedder.embed_query(query), n, only_updates=True)
        return [self._by_id[cid] for cid in ids]

    def update_chapters(self) -> set[int]:
        """Chapter numbers that contain update chunks."""
        return {c.chapter_number for c in self.chunks if c.is_update and c.chapter_number is not None}

    def update_chunks_in(self, chapters: set[int]) -> list[Chunk]:
        """All update chunks belonging to the given chapters (follows 'see Chapter N')."""
        return [c for c in self.chunks if c.is_update and c.chapter_number in chapters]
