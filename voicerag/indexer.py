"""Ingest the PDF once and cache the result on disk.

Cache layout::

    cache/<key>/
        chunks.json     all chunks (also used to rebuild BM25 at load time)
        meta.json       what the key was made from (human readable)
        chroma/         Chroma persistent directory (embeddings + metadata)
        READY           written last; a folder without it is a failed build

`<key>` is a hash of the PDF bytes **and** every setting that changes the
index (embedding model, chunking version, chunking parameters). Changing any of
them produces a new key, so a stale index can never be silently reused.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import build_chunks
from .config import CHUNKING_VERSION, Settings
from .models import Chunk
from .pdf_parser import parse_pdf

COLLECTION_NAME = "chunks"
_ADD_BATCH = 100


@dataclass
class Index:
    """A ready-to-query index."""

    chunks: list[Chunk]
    collection: Any             # chromadb Collection
    key: str
    directory: Path
    from_cache: bool            # True = loaded from disk, False = freshly built
    seconds: float              # time to load or build


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in blocks (works for big PDFs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_cache_key(pdf_path: Path, settings: Settings) -> tuple[str, dict]:
    """Return (key, ingredients). The ingredients are saved to meta.json."""
    ingredients = {
        "pdf_sha256": file_sha256(pdf_path),
        "embed_model": settings.embed_model,
        "chunking_version": CHUNKING_VERSION,
        "chunk_max_chars": settings.chunk_max_chars,
        "table_max_rows": settings.table_max_rows,
        "update_chapter_pattern": settings.update_chapter_pattern,
    }
    blob = json.dumps(ingredients, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], ingredients


def _open_collection(directory: Path):
    """Open (or create) the persistent Chroma collection inside `directory`."""
    import chromadb  # imported lazily: heavy

    client = chromadb.PersistentClient(path=str(directory / "chroma"))
    # embedding_function=None: we always pass our own vectors, so Chroma must not
    # try to download its default model.
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=None)


def _build(pdf_path: Path, directory: Path, ingredients: dict, settings: Settings, embedder) -> list[Chunk]:
    """Parse -> chunk -> embed -> write everything into `directory`."""
    chunks = build_chunks(parse_pdf(pdf_path), settings)
    if not chunks:
        raise ValueError(f"No text could be extracted from {pdf_path} (scanned PDF? OCR is not supported).")

    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    collection = _open_collection(directory)
    for start in range(0, len(chunks), _ADD_BATCH):
        batch = chunks[start : start + _ADD_BATCH]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[start : start + _ADD_BATCH],
            documents=[c.embed_text for c in batch],
            # Chroma metadata must be scalar values; this is what filtered search uses.
            metadatas=[
                {
                    "page": c.page,
                    "chapter_number": -1 if c.chapter_number is None else c.chapter_number,
                    "kind": c.kind,
                    "is_update": c.is_update,
                }
                for c in batch
            ],
        )
    (directory / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (directory / "meta.json").write_text(json.dumps(ingredients, indent=1), encoding="utf-8")
    (directory / "READY").write_text("ok")        # last: marks the build as complete
    return chunks


def load_or_build_index(pdf_path: Path, settings: Settings, embedder) -> Index:
    """Return the index for `pdf_path`, building it only if no valid cache exists."""
    started = time.perf_counter()
    key, ingredients = compute_cache_key(pdf_path, settings)
    directory = settings.cache_dir / key

    if (directory / "READY").exists():
        chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        return Index(chunks, _open_collection(directory), key, directory, True, time.perf_counter() - started)

    if directory.exists():                       # leftover of an interrupted build
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    chunks = _build(pdf_path, directory, ingredients, settings, embedder)
    return Index(chunks, _open_collection(directory), key, directory, False, time.perf_counter() - started)
