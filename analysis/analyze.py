"""Aggregate and analyze experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
            **r["metrics"],
            **r.get("metadata", {}),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def needle_analysis() -> dict[str, pd.DataFrame]:
    """Analyze needle-in-haystack results."""
    df = load_results("needle_haystack")

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
        "by_length": by_length,
        "by_position": by_position,
        "context_rot": rot_diff,
    }


def multihop_analysis() -> dict[str, pd.DataFrame]:
    """Analyze multi-hop results."""
    df = load_results("multihop")

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_hops = df.groupby(["method", "hops"])[["f1"]].mean().unstack(0)

    return {"overall": overall, "by_hops": by_hops}


def longbench_analysis() -> dict[str, pd.DataFrame]:
    """Analyze LongBench results."""
    df = load_results("longbench")

    overall = df.groupby("method")[["exact_match", "f1", "rouge_l"]].mean()
    by_dataset = df.groupby(["method", "dataset"])[["f1", "rouge_l"]].mean().unstack(0)

    return {"overall": overall, "by_dataset": by_dataset}


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
