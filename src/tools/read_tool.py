"""READ tool: chunk IDs → concatenated text."""

from __future__ import annotations

from src.environment.document_store import DocumentStore


class ReadTool:
    """Reads the text content of specified chunks."""

    def __init__(self, store: DocumentStore):
        self.store = store

    def __call__(self, chunk_ids: list[int]) -> str:
        """Return concatenated text of the given chunks."""
        return self.store.get_chunks_text(chunk_ids)

    def read_span(self, start: int, end: int) -> str:
        """Read raw character span from the document."""
        return self.store.get_span(start, end)
