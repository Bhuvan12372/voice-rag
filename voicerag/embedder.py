"""Local sentence-embedding model (BGE) behind a tiny interface.

The model runs on the CPU, so embedding costs no API money; its time is still
measured as part of the "retrieve" stage.
"""
from __future__ import annotations


class Embedder:
    """Wraps a sentence-transformers model.

    BGE models are trained so that *queries* carry an instruction prefix while
    *documents* do not. Both kinds of vector are L2-normalised, so cosine
    similarity and Euclidean distance rank results identically.
    """

    def __init__(self, model_name: str, query_prefix: str = "") -> None:
        # Imported here so that modules which only need the data classes (and the
        # unit tests) do not pay the cost of loading PyTorch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed document chunks (no prefix)."""
        vectors = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query (with the BGE instruction prefix)."""
        vector = self._model.encode(
            self.query_prefix + query, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()
