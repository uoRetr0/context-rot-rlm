"""Tests for evaluation metrics."""

import pytest
from benchmarks.metrics import exact_match, token_f1, rouge_l, compute_all_metrics


def test_exact_match_identical():
    assert exact_match("hello world", "hello world") == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("Hello World", "hello world") == 1.0


def test_exact_match_different():
    assert exact_match("hello", "world") == 0.0


def test_exact_match_punctuation():
    assert exact_match("hello, world!", "hello world") == 1.0


def test_f1_identical():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_f1_partial():
    score = token_f1("the cat", "the cat sat on mat")
    assert 0 < score < 1


def test_f1_no_overlap():
    assert token_f1("apple banana", "cherry date") == 0.0


def test_f1_empty():
    assert token_f1("", "") == 1.0
    assert token_f1("hello", "") == 0.0


def test_rouge_l_identical():
    score = rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.99


def test_rouge_l_partial():
    score = rouge_l("the cat sat", "the cat sat on the mat")
    assert 0 < score <= 1


def test_compute_all():
    result = compute_all_metrics("hello world", "hello world")
    assert "exact_match" in result
    assert "f1" in result
    assert "rouge_l" in result
    assert all(v == 1.0 for v in result.values())
