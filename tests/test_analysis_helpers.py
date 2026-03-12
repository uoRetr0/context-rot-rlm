"""Tests for analysis helper tables."""

import pandas as pd

from analysis.analyze import _common_cap_table


def test_common_cap_table_uses_cheapest_mean_cap():
    df = pd.DataFrame([
        {"method": "cheap", "exact_match": 1.0, "f1": 1.0, "rouge_l": 1.0, "cost_usd": 0.5},
        {"method": "cheap", "exact_match": 0.5, "f1": 0.5, "rouge_l": 0.5, "cost_usd": 0.5},
        {"method": "expensive", "exact_match": 1.0, "f1": 1.0, "rouge_l": 1.0, "cost_usd": 0.4},
        {"method": "expensive", "exact_match": 0.0, "f1": 0.0, "rouge_l": 0.0, "cost_usd": 0.9},
    ])

    summary = _common_cap_table(df, "cost_usd")

    assert summary.loc["cheap", "cost_usd_cap"] == 0.5
    assert summary.loc["cheap", "coverage"] == 1.0
    assert summary.loc["expensive", "coverage"] == 0.5
    assert summary.loc["expensive", "f1"] == 1.0
