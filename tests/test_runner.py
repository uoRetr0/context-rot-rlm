"""Tests for experiment runner utilities."""

from types import SimpleNamespace

import pytest

from experiments import runner
from src.controllers.fullcontext import FullContextController


def test_run_single_skips_retriever_for_fullcontext(monkeypatch):
    controller = FullContextController()

    def _should_not_build(*args, **kwargs):
        raise AssertionError("HybridRetriever should not be built for fullcontext")

    monkeypatch.setattr(runner, "HybridRetriever", _should_not_build)
    monkeypatch.setattr(
        "src.controllers.fullcontext.generate_json",
        lambda *args, **kwargs: {"answer": "ok", "confidence": 1.0, "reasoning": "stub"},
    )

    result, elapsed = runner._run_single(
        controller=controller,
        document="alpha beta gamma",
        question="What is the answer?",
        sample_id="sample-1",
    )

    assert result.answer == "ok"
    assert elapsed >= 0


def test_run_needle_haystack_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.settings, "google_api_key", "")
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        runner.run_needle_haystack(
            methods=["fullcontext"],
            max_samples=1,
            save_as="needle_auth_guard",
        )


def test_run_needle_haystack_can_skip_completed_methods_without_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.settings, "google_api_key", "")
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_load_partial", lambda filename: ([], {"fullcontext"}))

    results = runner.run_needle_haystack(
        methods=["fullcontext"],
        max_samples=1,
        save_as="needle_resume_guard",
    )

    assert results == []


def test_run_multihop_keeps_2hop_samples_for_fullcontext_and_mapreduce(monkeypatch, tmp_path):
    samples = [
        SimpleNamespace(document="doc-2", question="q2", answer="a2", hops=2, doc_length=500000),
        SimpleNamespace(document="doc-3", question="q3", answer="a3", hops=3, doc_length=500000),
    ]

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "gen_multihop", lambda: samples)
    monkeypatch.setattr(runner, "_load_partial", lambda filename: ([], set()))
    monkeypatch.setattr(runner, "_require_google_api_key", lambda: None)
    monkeypatch.setattr(runner, "save_results", lambda results, save_as: None)
    monkeypatch.setattr(runner, "get_controller", lambda method, model=None: object())
    monkeypatch.setattr(
        runner,
        "_run_single",
        lambda controller, document, question, sample_id, trace_dir=None: (
            SimpleNamespace(answer="stub", confidence=1.0),
            0.01,
        ),
    )
    monkeypatch.setattr(
        runner,
        "compute_all_metrics",
        lambda predicted, reference: {"exact_match": 0.0, "f1": 0.0, "rouge_l": 0.0},
    )

    results = runner.run_multihop(
        methods=["fullcontext", "mapreduce"],
        save_as="multihop_test",
    )

    observed = sorted((row.method, row.metadata["hops"], row.metadata["doc_length"]) for row in results)
    assert observed == [
        ("fullcontext", 2, 500000),
        ("fullcontext", 3, 500000),
        ("mapreduce", 2, 500000),
        ("mapreduce", 3, 500000),
    ]
