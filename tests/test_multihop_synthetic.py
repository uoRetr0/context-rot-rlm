"""Tests for the synthetic multi-hop benchmark generator."""

import random

from benchmarks.multihop_synthetic import _generate_2hop, _generate_3hop, generate_benchmark


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


def test_generate_benchmark_supports_500k_lengths():
    samples = generate_benchmark(num_samples=1, hops_list=[2, 3], doc_lengths=[500000])

    assert len(samples) == 2
    assert sorted((sample.hops, sample.doc_length) for sample in samples) == [
        (2, 500000),
        (3, 500000),
    ]
