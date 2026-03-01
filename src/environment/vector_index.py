"""Vector search index using embeddings and numpy cosine similarity."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from src.config import settings, PROJECT_ROOT
from src.environment.document_store import Chunk
from src.gemini_client import embed

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "embeddings_cache"


class VectorIndex:
    """Embedding-based vector search with disk caching."""

    def __init__(self, top_k: int | None = None, batch_size: int = 50):
        self.top_k = top_k or settings.vector_top_k
        self.batch_size = batch_size
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def build(self, chunks: list[Chunk], cache_key: str = "") -> None:
        self._chunks = chunks

        # Try loading from cache
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None and cached.shape[0] == len(chunks):
                self._embeddings = cached
                logger.info("Loaded embeddings from cache (%s)", cache_key)
                return

        # Compute embeddings in batches
        texts = [c.text for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = embed(batch)
            all_embeddings.extend(vecs)

        self._embeddings = np.array(all_embeddings, dtype=np.float32)

        # Save to cache
        if cache_key:
            self._save_cache(cache_key, self._embeddings)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Return (chunk_id, score) pairs sorted by cosine similarity."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        k = top_k or self.top_k
        query_vec = np.array(embed([query])[0], dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normed = self._embeddings / norms

        query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-10)
        scores = normed @ query_norm

        top_indices = np.argsort(scores)[::-1][:k]
        results = [
            (self._chunks[idx].chunk_id, float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return CACHE_DIR / f"{h}.npy"

    def _load_cache(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if path.exists():
            return np.load(path)
        return None

    def _save_cache(self, key: str, arr: np.ndarray) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path(key), arr)
