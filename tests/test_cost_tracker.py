"""Tests for cost tracker."""

import pytest
from src.cost_tracker import CostTracker, BudgetExceededError


def test_record_cost():
    ct = CostTracker(max_dollars=100, warn_at=80)
    cost = ct.record("gemini-2.0-flash", 1_000_000, 100_000)

    # input: 1M * 0.10/1M = 0.10, output: 100k * 0.40/1M = 0.04
    assert abs(cost - 0.14) < 0.001
    assert abs(ct.total_cost - 0.14) < 0.001


def test_budget_exceeded():
    ct = CostTracker(max_dollars=0.001, warn_at=0.0005)
    ct.record("gemini-2.0-flash", 1_000_000, 1_000_000)

    with pytest.raises(BudgetExceededError):
        ct.check_budget()


def test_multiple_models():
    ct = CostTracker(max_dollars=100, warn_at=80)
    ct.record("gemini-2.0-flash", 100, 50)
    ct.record("gemini-1.5-pro", 100, 50)

    assert len(ct.usage) == 2
    assert ct.total_cost > 0


def test_summary():
    ct = CostTracker(max_dollars=100, warn_at=80)
    ct.record("gemini-2.0-flash", 1000, 500)
    summary = ct.summary()

    assert "gemini-2.0-flash" in summary
    assert "TOTAL" in summary
