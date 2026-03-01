"""BM25 search index over document chunks."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.config import settings
from src.environment.document_store import Chunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 search over document chunks."""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.bm25_top_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Return (chunk_id, score) pairs sorted by relevance."""
        if self._bm25 is None:
            return []

        k = top_k or self.top_k
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = [
            (self._chunks[idx].chunk_id, score)
            for idx, score in ranked[:k]
            if score > 0
        ]
        return results
