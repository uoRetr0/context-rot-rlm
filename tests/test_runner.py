"""Tests for experiment runner utilities."""

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
