"""SEARCH tool: query → ranked chunk IDs."""

from __future__ import annotations

from src.environment.hybrid_retriever import HybridRetriever


class SearchTool:
    """Retrieves the most relevant chunk IDs for a query."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def __call__(self, query: str, top_k: int | None = None) -> list[int]:
        """Return chunk IDs ranked by relevance."""
        results = self.retriever.search(query, top_k=top_k)
        return [chunk_id for chunk_id, _score in results]
