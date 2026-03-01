"""Visualization: matplotlib/seaborn plots for experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / settings.output_dir
PLOTS_DIR = RESULTS_DIR / "plots"

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)
METHOD_COLORS = {
    "fullcontext": "#4C72B0",
    "rag": "#55A868",
    "mapreduce": "#C44E52",
    "rlm": "#8172B2",
}
METHOD_LABELS = {
    "fullcontext": "Full Context",
    "rag": "RAG",
    "mapreduce": "Map-Reduce",
    "rlm": "RLM (Ours)",
}


def _load(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{filename}.json"
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data:
        rows.append({
            "sample_id": r["sample_id"],
            "method": r["method"],
            "confidence": r["confidence"],
            "duration_s": r["duration_s"],
            **r["metrics"],
            **r.get("metadata", {}),
        })
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_needle_heatmap() -> None:
    """Heatmap: F1 by haystack_length × needle_position for each method."""
    df = _load("needle_haystack")

    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        pivot = sub.pivot_table(
            values="f1", index="needle_position", columns="haystack_length",
            aggfunc="mean",
        )
        sns.heatmap(
            pivot, ax=ax, vmin=0, vmax=1, cmap="RdYlGn",
            annot=True, fmt=".2f", cbar=method == methods[-1],
        )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_ylabel("Needle Position" if method == methods[0] else "")
        ax.set_xlabel("Haystack Length (words)")

    fig.suptitle("Needle-in-Haystack: F1 Score", fontsize=14, y=1.02)
    _save(fig, "needle_heatmap")


def plot_needle_by_length() -> None:
    """Line plot: F1 vs haystack length for each method."""
    df = _load("needle_haystack")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        grouped = sub.groupby("haystack_length")["f1"].mean()
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Haystack Length (words)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Context Rot: F1 vs Document Length")
    ax.set_xscale("log")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "needle_by_length")


def plot_needle_by_position() -> None:
    """Line plot: F1 vs needle position for each method."""
    df = _load("needle_haystack")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        grouped = sub.groupby("needle_position")["f1"].mean()
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Needle Position (0=start, 1=end)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Positional Bias: F1 vs Needle Placement")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "needle_by_position")


def plot_multihop_comparison() -> None:
    """Bar chart: F1 by method and hop count."""
    df = _load("multihop")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df.pivot_table(values="f1", index="hops", columns="method", aggfunc="mean")

    x = np.arange(len(pivot.index))
    width = 0.18
    for i, method in enumerate(sorted(pivot.columns)):
        vals = pivot[method].values
        ax.bar(
            x + i * width, vals, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Number of Hops")
    ax.set_ylabel("F1 Score")
    ax.set_title("Multi-Hop QA: F1 by Method and Hop Count")
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels([f"{h}-hop" for h in pivot.index])
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "multihop_comparison")


def plot_longbench_comparison() -> None:
    """Bar chart: F1 and ROUGE-L by method for LongBench datasets."""
    df = _load("longbench")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, ["f1", "rouge_l"]):
        pivot = df.pivot_table(values=metric, index="dataset", columns="method", aggfunc="mean")
        pivot.plot(
            kind="bar", ax=ax,
            color=[METHOD_COLORS.get(m) for m in pivot.columns],
        )
        ax.set_title(f"LongBench: {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.legend([METHOD_LABELS.get(m, m) for m in pivot.columns])
        ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    _save(fig, "longbench_comparison")


def plot_overall_summary() -> None:
    """Summary bar chart across all benchmarks."""
    dfs = {}
    for name in ["needle_haystack", "multihop", "longbench"]:
        try:
            dfs[name] = _load(name)
        except FileNotFoundError:
            continue

    if not dfs:
        return

    all_df = pd.concat(dfs.values(), ignore_index=True)
    summary = all_df.groupby("method")[["f1", "rouge_l"]].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary.index))
    width = 0.3

    sorted_methods = sorted(summary.index)
    f1_vals = [summary.loc[m, "f1"] for m in sorted_methods]
    rl_vals = [summary.loc[m, "rouge_l"] for m in sorted_methods]

    ax.bar(x - width / 2, f1_vals, width, label="F1", color="#4C72B0")
    ax.bar(x + width / 2, rl_vals, width, label="ROUGE-L", color="#55A868")

    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance: F1 and ROUGE-L")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in sorted_methods])
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "overall_summary")


def generate_all_plots() -> None:
    """Generate all plots."""
    logging.basicConfig(level=logging.INFO)

    plot_fns = [
        plot_needle_heatmap,
        plot_needle_by_length,
        plot_needle_by_position,
        plot_multihop_comparison,
        plot_longbench_comparison,
        plot_overall_summary,
    ]

    for fn in plot_fns:
        try:
            fn()
        except Exception as e:
            logger.warning("Failed to generate %s: %s", fn.__name__, e)


if __name__ == "__main__":
    generate_all_plots()
