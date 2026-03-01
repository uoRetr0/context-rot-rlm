"""Tests for needle-in-haystack benchmark generator."""

import pytest
from benchmarks.needle_haystack import generate_needle_haystack, NeedleSample


def test_generate_sample():
    sample = generate_needle_haystack(haystack_length=1000, needle_position=0.5, seed=42)
    assert isinstance(sample, NeedleSample)
    assert len(sample.document.split()) >= 900  # approximately 1000 words
    assert sample.question
    assert sample.answer


def test_needle_is_in_document():
    sample = generate_needle_haystack(haystack_length=1000, needle_position=0.5, seed=42)
    assert sample.needle_text in sample.document


def test_different_positions():
    s1 = generate_needle_haystack(haystack_length=1000, needle_position=0.0, seed=42)
    s2 = generate_needle_haystack(haystack_length=1000, needle_position=1.0, seed=42)

    # Needle should appear earlier in s1 than s2
    pos1 = s1.document.find(s1.needle_text)
    pos2 = s2.document.find(s2.needle_text)
    assert pos1 < pos2


def test_different_lengths():
    s_short = generate_needle_haystack(haystack_length=500, needle_position=0.5, seed=42)
    s_long = generate_needle_haystack(haystack_length=5000, needle_position=0.5, seed=42)
    assert len(s_short.document) < len(s_long.document)
