"""Tests for document store chunking."""

import pytest
from src.environment.document_store import DocumentStore


def test_ingest_creates_chunks():
    store = DocumentStore(chunk_size=10, chunk_overlap=2, min_chunk_size=3)
    text = " ".join(f"word{i}" for i in range(50))
    chunks = store.ingest(text)

    assert len(chunks) > 0
    assert store.num_chunks == len(chunks)
    assert store.full_text == text


def test_chunk_ids_sequential():
    store = DocumentStore(chunk_size=10, chunk_overlap=2, min_chunk_size=3)
    text = " ".join(f"word{i}" for i in range(50))
    chunks = store.ingest(text)

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == i


def test_get_chunk():
    store = DocumentStore(chunk_size=10, chunk_overlap=2, min_chunk_size=3)
    text = " ".join(f"word{i}" for i in range(50))
    store.ingest(text)

    chunk = store.get_chunk(0)
    assert chunk is not None
    assert chunk.chunk_id == 0

    assert store.get_chunk(-1) is None
    assert store.get_chunk(9999) is None


def test_get_chunks_text():
    store = DocumentStore(chunk_size=10, chunk_overlap=2, min_chunk_size=3)
    text = " ".join(f"word{i}" for i in range(50))
    store.ingest(text)

    result = store.get_chunks_text([0, 1])
    assert "[Chunk 0]" in result
    assert "[Chunk 1]" in result


def test_small_document():
    store = DocumentStore(chunk_size=100, chunk_overlap=10)
    text = "This is a small document."
    chunks = store.ingest(text)

    assert len(chunks) == 1
    assert "small document" in chunks[0].text
