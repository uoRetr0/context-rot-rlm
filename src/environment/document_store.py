"""Document chunking and span-addressed storage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import settings


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        return (self.start_char, self.end_char)


class DocumentStore:
    """Stores a document as overlapping chunks with span addressing."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size
        self.chunks: list[Chunk] = []
        self.full_text: str = ""
        self.doc_id: str = ""

    def ingest(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        """Chunk a document and store it."""
        self.full_text = text
        self.doc_id = doc_id
        self.chunks = self._chunk_text(text)
        return self.chunks

    def _chunk_text(self, text: str) -> list[Chunk]:
        """Split text into word-based overlapping chunks."""
        words = text.split()
        chunks = []
        idx = 0
        chunk_id = 0

        while idx < len(words):
            end = min(idx + self.chunk_size, len(words))
            chunk_words = words[idx:end]

            if len(chunk_words) < self.min_chunk_size and chunks:
                # Merge small trailing chunk into previous
                prev = chunks[-1]
                merged_text = prev.text + " " + " ".join(chunk_words)
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    text=merged_text,
                    start_char=prev.start_char,
                    end_char=self._find_end_char(text, words, end),
                )
                break

            chunk_text = " ".join(chunk_words)
            start_char = self._find_start_char(text, words, idx)
            end_char = self._find_end_char(text, words, end)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
            ))
            chunk_id += 1
            idx += self.chunk_size - self.chunk_overlap

        return chunks

    def _find_start_char(self, text: str, words: list[str], word_idx: int) -> int:
        """Find character offset for a word index."""
        pos = 0
        for i in range(word_idx):
            pos = text.find(words[i], pos) + len(words[i])
        if word_idx < len(words):
            pos = text.find(words[word_idx], pos)
        return max(0, pos)

    def _find_end_char(self, text: str, words: list[str], word_idx: int) -> int:
        """Find character offset for end of word range."""
        if word_idx >= len(words):
            return len(text)
        pos = 0
        for i in range(word_idx):
            pos = text.find(words[i], pos) + len(words[i])
        return pos

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        if 0 <= chunk_id < len(self.chunks):
            return self.chunks[chunk_id]
        return None

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        return [c for cid in chunk_ids if (c := self.get_chunk(cid)) is not None]

    def get_span(self, start_char: int, end_char: int) -> str:
        return self.full_text[start_char:end_char]

    def get_chunks_text(self, chunk_ids: list[int]) -> str:
        chunks = self.get_chunks(chunk_ids)
        return "\n\n---\n\n".join(
            f"[Chunk {c.chunk_id}] {c.text}" for c in chunks
        )

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def doc_length(self) -> int:
        return len(self.full_text)
