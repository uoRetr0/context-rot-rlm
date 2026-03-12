"""Tests for the synthetic multi-hop benchmark generator."""

import random

from benchmarks.multihop_synthetic import _generate_2hop, _generate_3hop


def test_generate_2hop_contains_aliases_and_distractors():
    sample = _generate_2hop(random.Random(42), doc_length=300, seed=42)

    assert "Internal briefs shorten" in sample.document
    assert "satellite field office" in sample.document
    assert len(sample.bridge_facts) == 2
    assert sample.answer in sample.document


def test_generate_3hop_contains_distractor_chain():
    sample = _generate_3hop(random.Random(7), doc_length=400, seed=7)

    assert "archival filings abbreviate" in sample.document
    assert "satellite lab" in sample.document
    assert "collaborated with" in sample.document
    assert len(sample.bridge_facts) == 3
