"""Cross-encoder reranker.

A bi-encoder (the embedding model) scores query and chunk separately; a
cross-encoder reads them *together*, which is slower but much better at
deciding "does this chunk actually answer this question?". We only run it on
the ~20 fused candidates, so it costs roughly 100 ms on a CPU.
"""
from __future__ import annotations

import math

from .models import Chunk, ScoredChunk


def sigmoid(x: float) -> float:
    """Squash a raw logit into 0..1 (numerically safe for large |x|)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker:
    """Scores (query, chunk) pairs with a MS MARCO cross-encoder.

    Scores are returned as probabilities in 0..1. Thresholds in `config.py`
    refer to this scale and were chosen from the dev questions, not guessed.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy: pulls in PyTorch

        self.model_name = model_name
        self._model = self._load(CrossEncoder, model_name)

    @staticmethod
    def _load(cross_encoder_cls, model_name: str):
        """Load the model so that predict() returns RAW logits in every library version.

        The activation argument was renamed between sentence-transformers
        releases, so we try the new name first. We apply the sigmoid ourselves,
        which makes the score scale independent of the model's saved config.
        """
        import torch

        try:
            return cross_encoder_cls(model_name, activation_fn=torch.nn.Identity())
        except TypeError:
            return cross_encoder_cls(model_name, default_activation_function=torch.nn.Identity())

    def rerank(self, query: str, chunks: list[Chunk]) -> list[ScoredChunk]:
        """All `chunks` scored against `query`, best first."""
        if not chunks:
            return []
        raw = self._model.predict([(query, c.embed_text) for c in chunks], show_progress_bar=False)
        scored = [ScoredChunk(chunk=c, score=sigmoid(float(s))) for c, s in zip(chunks, raw)]
        return sorted(scored, key=lambda sc: sc.score, reverse=True)
