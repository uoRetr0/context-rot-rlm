"""Aggregate and analyze experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / settings.output_dir


def load_results(filename: str) -> pd.DataFrame:
    """Load results JSON into a DataFrame."""
    path = RESULTS_DIR / f"{filename}.json"
    with open(path) as f:
        data = json.load(f)

    rows = []
    for r in data:
        row = {
            "sample_id": r["sample_id"],
            "method": r["method"],
            "question": r["question"],
            "predicted": r["predicted"],
            "reference": r["reference"],
            "confidence": r["confidence"],
            "duration_s": r["duration_s"],
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "llm_calls": r.get("llm_calls", 0),
            "cost_usd": r.get("cost_usd", 0.0),
            "status": r.get(
                "status",
                "error" if r.get("predicted") == "ERROR" or r.get("metadata", {}).get("error") else "ok",
            ),
            **r["metrics"],
            **r.get("metadata", {}),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _scoreable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only completed samples when aggregating benchmark quality."""
    if "status" not in df.columns:
        return df
    return df[df["status"] == "ok"].copy()


def _efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average cost, latency, and LLM calls per completed sample."""
    cols = [c for c in ["cost_usd", "duration_s", "llm_calls", "input_tokens", "output_tokens"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df.groupby("method")[cols].mean()


def _tradeoff_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize accuracy alongside efficiency ratios."""
    cols = [c for c in ["f1", "rouge_l", "cost_usd", "duration_s", "llm_calls"] if c in df.columns]
    if not cols:
        return pd.DataFrame()

    summary = df.groupby("method")[cols].mean()
    if "cost_usd" in summary.columns:
        summary["f1_per_dollar"] = summary["f1"] / summary["cost_usd"].replace(0, np.nan)
    if "duration_s" in summary.columns:
        summary["f1_per_second"] = summary["f1"] / summary["duration_s"].replace(0, np.nan)
    if "llm_calls" in summary.columns:
        summary["f1_per_call"] = summary["f1"] / summary["llm_calls"].replace(0, np.nan)
    return summary


def _bootstrap_intervals(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = ("exact_match", "f1", "rouge_l"),
    n_boot: int = 1000,
    seed: int = settings.seed,
) -> pd.DataFrame:
    """Bootstrap 95% confidence intervals for metric means by method."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    rng = np.random.default_rng(seed)
    for method, group in df.groupby("method"):
        row: dict[str, float | str] = {"method": method}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            boots = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                sample = rng.choice(values, size=len(values), replace=True)
                boots[i] = sample.mean()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.percentile(boots, 2.5))
            row[f"{metric}_ci_high"] = float(np.percentile(boots, 97.5))
        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


def _common_cap_table(df: pd.DataFrame, cap_col: str) -> pd.DataFrame:
    """Compare methods under the cheapest method's average per-sample cap."""
    required = {"exact_match", "f1", "rouge_l", cap_col, "method"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    cap_value = df.groupby("method")[cap_col].mean().min()
    capped = df[df[cap_col] <= cap_value].copy()
    if capped.empty:
        return pd.DataFrame()

    summary = capped.groupby("method")[["exact_match", "f1", "rouge_l", cap_col]].mean()
    summary["samples_kept"] = capped.groupby("method").size()
    summary["coverage"] = summary["samples_kept"] / df.groupby("method").size()
    summary[f"{cap_col}_cap"] = cap_value
    return summary


def needle_analysis() -> dict[str, pd.DataFrame]:
    """Analyze needle-in-haystack results."""
    df = _scoreable(load_results("needle_haystack"))

    # Per-method overall scores
    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()

    # Accuracy by haystack length
    by_length = df.groupby(["method", "haystack_length"])[["f1"]].mean().unstack(0)

    # Accuracy by needle position
    by_position = df.groupby(["method", "needle_position"])[["f1"]].mean().unstack(0)

    # Context rot: difference between best and worst position per method×length
    rot = df.groupby(["method", "haystack_length", "needle_position"])["f1"].mean()
    rot = rot.unstack("needle_position")
    rot_diff = rot.max(axis=1) - rot.min(axis=1)
    rot_diff = rot_diff.unstack("method")

    return {
        "overall": overall,
        "confidence_intervals": _bootstrap_intervals(df),
        "efficiency": _efficiency_table(df),
        "tradeoffs": _tradeoff_table(df),
        "within_common_cost_cap": _common_cap_table(df, "cost_usd"),
        "within_common_call_cap": _common_cap_table(df, "llm_calls"),
        "by_length": by_length,
        "by_position": by_position,
        "context_rot": rot_diff,
    }


def multihop_analysis() -> dict[str, pd.DataFrame]:
    """Analyze multi-hop results."""
    df = _scoreable(load_results("multihop"))

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_hops = df.groupby(["method", "hops"])[["f1"]].mean().unstack(0)

    tables: dict[str, pd.DataFrame] = {
        "overall": overall,
        "confidence_intervals": _bootstrap_intervals(df),
        "efficiency": _efficiency_table(df),
        "tradeoffs": _tradeoff_table(df),
        "within_common_cost_cap": _common_cap_table(df, "cost_usd"),
        "within_common_call_cap": _common_cap_table(df, "llm_calls"),
        "by_hops": by_hops,
    }

    if "doc_length" in df.columns:
        tables["by_doc_length"] = df.groupby(["method", "doc_length"])[["f1"]].mean().unstack(0)
        tables["by_condition"] = (
            df.groupby(["hops", "doc_length", "method"])[["f1", "rouge_l"]]
            .mean()
            .unstack("method")
        )

    return tables


def longbench_analysis() -> dict[str, pd.DataFrame]:
    """Analyze LongBench results."""
    df = _scoreable(load_results("longbench"))

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_dataset = df.groupby(["method", "dataset"])[["f1", "rouge_l"]].mean().unstack(0)

    return {
        "overall": overall,
        "confidence_intervals": _bootstrap_intervals(df),
        "efficiency": _efficiency_table(df),
        "tradeoffs": _tradeoff_table(df),
        "within_common_cost_cap": _common_cap_table(df, "cost_usd"),
        "within_common_call_cap": _common_cap_table(df, "llm_calls"),
        "by_dataset": by_dataset,
    }


def musique_analysis() -> dict[str, pd.DataFrame]:
    """Analyze MuSiQue multi-hop results."""
    df = _scoreable(load_results("musique"))

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_hops = df.groupby(["method", "hops"])[["f1"]].mean().unstack(0)
    by_doc_length = df.groupby(["method", "doc_length"])[["f1"]].mean().unstack(0)
    by_condition = df.groupby(["hops", "doc_length", "method"])[["f1", "rouge_l"]].mean().unstack("method")

    return {
        "overall": overall,
        "confidence_intervals": _bootstrap_intervals(df),
        "efficiency": _efficiency_table(df),
        "tradeoffs": _tradeoff_table(df),
        "within_common_cost_cap": _common_cap_table(df, "cost_usd"),
        "within_common_call_cap": _common_cap_table(df, "llm_calls"),
        "by_hops": by_hops,
        "by_doc_length": by_doc_length,
        "by_condition": by_condition,
    }


def pro_musique_analysis() -> dict[str, pd.DataFrame]:
    """Analyze Pro model MuSiQue results (phaseA + phaseB combined)."""
    frames = []
    for name in ["musique_pro_phaseA", "musique_pro_phaseB"]:
        try:
            frames.append(load_results(name))
        except FileNotFoundError:
            pass

    if not frames:
        raise FileNotFoundError("No Pro MuSiQue results found")

    df = _scoreable(pd.concat(frames, ignore_index=True))

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_hops = df.groupby(["method", "hops"])[["f1"]].mean().unstack(0)
    by_doc_length = df.groupby(["method", "doc_length"])[["f1"]].mean().unstack(0)
    by_condition = (
        df.groupby(["hops", "doc_length", "method"])[["f1", "rouge_l"]]
        .mean()
        .unstack("method")
    )

    return {
        "overall": overall,
        "confidence_intervals": _bootstrap_intervals(df),
        "efficiency": _efficiency_table(df),
        "tradeoffs": _tradeoff_table(df),
        "by_hops": by_hops,
        "by_doc_length": by_doc_length,
        "by_condition": by_condition,
    }


def full_analysis() -> dict[str, Any]:
    """Run all analyses and return tables."""
    results = {}

    try:
        results["needle"] = needle_analysis()
        logger.info("Needle analysis complete")
    except FileNotFoundError:
        logger.warning("No needle results found")

    try:
        results["multihop"] = multihop_analysis()
        logger.info("Multi-hop analysis complete")
    except FileNotFoundError:
        logger.warning("No multihop results found")

    try:
        results["longbench"] = longbench_analysis()
        logger.info("LongBench analysis complete")
    except FileNotFoundError:
        logger.warning("No longbench results found")

    try:
        results["musique"] = musique_analysis()
        logger.info("MuSiQue analysis complete")
    except FileNotFoundError:
        logger.warning("No MuSiQue results found")

    try:
        results["pro_musique"] = pro_musique_analysis()
        logger.info("Pro MuSiQue analysis complete")
    except FileNotFoundError:
        logger.warning("No Pro MuSiQue results found")

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a text summary of all results."""
    for bench_name, tables in results.items():
        print(f"\n{'='*60}")
        print(f"  {bench_name.upper()}")
        print(f"{'='*60}")
        for table_name, df in tables.items():
            print(f"\n--- {table_name} ---")
            print(df.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = full_analysis()
    print_summary(results)
