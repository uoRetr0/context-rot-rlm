"""Hybrid retrieval: BM25 + vector search fused via Reciprocal Rank Fusion."""

from __future__ import annotations

from src.config import settings
from src.environment.bm25_index import BM25Index
from src.environment.document_store import Chunk, DocumentStore
from src.environment.vector_index import VectorIndex


class HybridRetriever:
    """Combines BM25 and vector search using RRF."""

    def __init__(
        self,
        store: DocumentStore,
        top_k: int | None = None,
        rrf_k: int | None = None,
        cache_key: str = "",
    ):
        self.store = store
        self.top_k = top_k or settings.hybrid_top_k
        self.rrf_k = rrf_k or settings.rrf_k

        self.bm25 = BM25Index()
        self.vector = VectorIndex()

        self.bm25.build(store.chunks)
        self.vector.build(store.chunks, cache_key=cache_key)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Return (chunk_id, rrf_score) pairs."""
        k = top_k or self.top_k

        bm25_results = self.bm25.search(query)
        vec_results = self.vector.search(query)

        return self._rrf_fuse(bm25_results, vec_results, k)

    def _rrf_fuse(
        self,
        *result_lists: list[tuple[int, float]],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion."""
        k = top_k or self.top_k
        scores: dict[int, float] = {}

        for results in result_lists:
            for rank, (chunk_id, _score) in enumerate(results):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search_chunks(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return actual Chunk objects."""
        results = self.search(query, top_k)
        chunk_ids = [cid for cid, _ in results]
        return self.store.get_chunks(chunk_ids)
