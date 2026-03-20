"""Tests for needle-in-haystack benchmark generator."""

from benchmarks.needle_haystack import (
    NeedleSample,
    generate_challenging_benchmark,
    generate_needle_haystack,
    generate_stress_benchmark,
)


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


def test_adversarial_mode_includes_distractors():
    sample = generate_needle_haystack(
        haystack_length=5000,
        needle_position=0.5,
        seed=42,
        difficulty="adversarial",
        distractor_count=4,
    )

    assert sample.difficulty == "adversarial"
    assert len(sample.distractor_texts) == 4
    for distractor in sample.distractor_texts:
        assert distractor in sample.document
        assert sample.answer not in distractor


def test_generate_stress_benchmark_uses_stress_configuration():
    samples = generate_stress_benchmark(num_samples=1)

    assert samples
    assert all(sample.difficulty == "adversarial" for sample in samples)
    assert all(len(sample.distractor_texts) > 0 for sample in samples)


def test_challenging_mode_repeats_confusing_alternatives():
    sample = generate_needle_haystack(
        haystack_length=100000,
        needle_position=0.9,
        seed=42,
        difficulty="challenging",
        distractor_count=4,
        distractor_repeat_factor=3,
        confusion_block_repeats=2,
    )

    assert sample.difficulty == "challenging"
    assert sample.requested_position == 0.9
    assert 0.0 <= sample.needle_position <= 1.0
    assert sample.document.count("conflicting alternatives") >= 1
    repeated_hits = sum(sample.document.count(distractor) for distractor in sample.distractor_texts)
    assert repeated_hits >= len(sample.distractor_texts) * 3


def test_generate_challenging_benchmark_uses_challenge_configuration():
    samples = generate_challenging_benchmark(num_samples=1)

    assert samples
    assert all(sample.difficulty == "challenging" for sample in samples)
    assert all(sample.requested_position in {0.1, 0.5, 0.9} for sample in samples)
