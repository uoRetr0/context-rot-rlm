"""Tests for BM25 index."""

import pytest
from src.environment.bm25_index import BM25Index
from src.environment.document_store import Chunk


def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=i, text=t, start_char=0, end_char=len(t))
        for i, t in enumerate(texts)
    ]


def test_bm25_basic_search():
    chunks = _make_chunks([
        "The cat sat on the mat",
        "Dogs are loyal animals",
        "Python is a programming language",
        "The quick brown fox jumps over the lazy dog",
    ])

    index = BM25Index(top_k=3)
    index.build(chunks)

    results = index.search("programming language")
    assert len(results) > 0
    # Chunk 2 (Python) should be most relevant
    assert results[0][0] == 2


def test_bm25_empty_query():
    chunks = _make_chunks(["Hello world"])
    index = BM25Index()
    index.build(chunks)

    results = index.search("")
    assert isinstance(results, list)


def test_bm25_returns_scores():
    chunks = _make_chunks(["apple banana", "cherry date", "apple pie"])
    index = BM25Index()
    index.build(chunks)

    results = index.search("apple")
    for chunk_id, score in results:
        assert isinstance(chunk_id, int)
        assert isinstance(score, float)
        assert score > 0
