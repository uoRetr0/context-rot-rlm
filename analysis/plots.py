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
            "cost_usd": r.get("cost_usd", 0.0),
            "llm_calls": r.get("llm_calls", 0),
            "status": r.get(
                "status",
                "error" if r.get("predicted") == "ERROR" or r.get("metadata", {}).get("error") else "ok",
            ),
            **r["metrics"],
            **r.get("metadata", {}),
        })
    df = pd.DataFrame(rows)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


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
    """Bar chart: F1 by method, grouped by hops x doc_length when multi-length data present."""
    df = _load("multihop")

    has_multi_length = "doc_length" in df.columns and df["doc_length"].nunique() > 1

    if has_multi_length:
        # Grouped bar by condition (hops x doc_length)
        df["group"] = df.apply(
            lambda r: f"{r['hops']}-hop\n{r['doc_length'] // 1000}K", axis=1
        )
        df["sort_key"] = df["hops"] * 1_000_000 + df["doc_length"]
        group_order = (
            df[["group", "sort_key"]]
            .drop_duplicates()
            .sort_values("sort_key")["group"]
            .tolist()
        )
        pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
        pivot = pivot.reindex(group_order)
    else:
        pivot = df.pivot_table(values="f1", index="hops", columns="method", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 2), 6))
    methods = sorted(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        vals = pivot[method].values
        ax.bar(
            x + i * width, vals, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Condition (Hops × Document Length)" if has_multi_length else "Number of Hops")
    ax.set_ylabel("F1 Score")
    ax.set_title("Multi-Hop QA: F1 by Method and Condition")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    if has_multi_length:
        ax.set_xticklabels(pivot.index)
    else:
        ax.set_xticklabels([f"{h}-hop" for h in pivot.index])
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "multihop_comparison")


def plot_multihop_heatmap() -> None:
    """3-panel heatmap: RAG F1 / RLM F1 / delta by hops × doc_length."""
    df = _load("multihop")
    if "doc_length" not in df.columns or df["doc_length"].nunique() <= 1:
        return

    df["doc_length_k"] = df["doc_length"] // 1000

    panels = []
    for method in ["rag", "rlm"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        pivot = sub.pivot_table(values="f1", index="doc_length_k", columns="hops", aggfunc="mean")
        panels.append((METHOD_LABELS.get(method, method), pivot))

    if len(panels) < 2:
        return

    # Compute delta (RLM - RAG)
    delta = panels[1][1] - panels[0][1]
    panels.append(("RLM − RAG (Δ F1)", delta))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (title, pivot) in zip(axes, panels):
        vmin, vmax = (-0.3, 0.3) if "Δ" in title else (0, 1)
        cmap = "RdBu_r" if "Δ" in title else "RdYlGn"
        sns.heatmap(
            pivot, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
            annot=True, fmt=".2f", cbar=True,
        )
        ax.set_title(title)
        ax.set_ylabel("Document Length (K words)" if ax == axes[0] else "")
        ax.set_xlabel("Hops")

    fig.suptitle("Multi-Hop: F1 by Document Length × Hops", fontsize=14, y=1.02)
    _save(fig, "multihop_heatmap")


def plot_pro_musique_comparison() -> None:
    """Grouped bar chart: Pro model MuSiQue results by condition."""
    frames = []
    for name in ["musique_pro_phaseA", "musique_pro_phaseB"]:
        try:
            frames.append(_load(name))
        except FileNotFoundError:
            continue

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)

    df["group"] = df.apply(
        lambda r: f"{r['hops']}-hop\n{r['doc_length'] // 1000}K", axis=1
    )
    df["sort_key"] = df["hops"] * 1_000_000 + df["doc_length"]
    group_order = (
        df[["group", "sort_key"]]
        .drop_duplicates()
        .sort_values("sort_key")["group"]
        .tolist()
    )

    pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
    pivot = pivot.reindex(group_order)

    methods = sorted(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 2), 6))
    x = np.arange(len(pivot.index))
    width = 0.35

    for i, method in enumerate(methods):
        vals = pivot[method].values
        ax.bar(
            x + i * width - width / 2, vals, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Condition (Hops × Document Length)")
    ax.set_ylabel("F1 Score")
    ax.set_title("MuSiQue (Pro Model): RAG vs RLM by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "pro_musique_comparison")


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


def plot_musique_comparison() -> None:
    """Grouped bar chart: F1 by (hops x doc_length) for RAG vs RLM."""
    df = _load("musique")

    # Create a combined group label like "2-hop\n10K"
    df["group"] = df.apply(
        lambda r: f"{r['hops']}-hop\n{r['doc_length'] // 1000}K", axis=1
    )

    # Sort groups logically: by hops then doc_length
    df["sort_key"] = df["hops"] * 1_000_000 + df["doc_length"]
    group_order = (
        df[["group", "sort_key"]]
        .drop_duplicates()
        .sort_values("sort_key")["group"]
        .tolist()
    )

    pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
    pivot = pivot.reindex(group_order)

    methods = sorted(pivot.columns)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.35

    for i, method in enumerate(methods):
        vals = pivot[method].values
        ax.bar(
            x + i * width - width / 2, vals, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xlabel("Condition (Hops x Document Length)")
    ax.set_ylabel("F1 Score")
    ax.set_title("MuSiQue: RAG vs RLM by Hop Count and Document Length")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, "musique_comparison")


def plot_overall_summary() -> None:
    """Summary bar chart across all benchmarks."""
    dfs = {}
    for name in ["needle_haystack", "multihop", "longbench", "musique"]:
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


def plot_efficiency_frontier() -> None:
    """Scatter plot: mean cost vs mean F1 for each benchmark."""
    benchmarks = ["needle_haystack", "multihop", "longbench", "musique"]
    frames: list[tuple[str, pd.DataFrame]] = []
    for name in benchmarks:
        try:
            df = _load(name)
        except FileNotFoundError:
            continue
        if not df.empty:
            frames.append((name, df))

    if not frames:
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5), squeeze=False)
    flat_axes = axes[0]

    for ax, (name, df) in zip(flat_axes, frames):
        summary = df.groupby("method")[["f1", "cost_usd", "duration_s"]].mean().reset_index()
        for _, row in summary.iterrows():
            method = row["method"]
            ax.scatter(
                row["cost_usd"],
                row["f1"],
                s=max(80, row["duration_s"] * 8),
                color=METHOD_COLORS.get(method),
                alpha=0.85,
            )
            ax.annotate(
                METHOD_LABELS.get(method, method),
                (row["cost_usd"], row["f1"]),
                textcoords="offset points",
                xytext=(6, 4),
            )

        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Mean Cost per Sample (USD)")
        ax.set_ylabel("Mean F1")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")

    fig.suptitle("Accuracy vs Cost Frontier (marker size = latency)", fontsize=14, y=1.02)
    _save(fig, "efficiency_frontier")


def generate_all_plots() -> None:
    """Generate all plots."""
    logging.basicConfig(level=logging.INFO)

    plot_fns = [
        plot_needle_heatmap,
        plot_needle_by_length,
        plot_needle_by_position,
        plot_multihop_comparison,
        plot_multihop_heatmap,
        plot_longbench_comparison,
        plot_musique_comparison,
        plot_pro_musique_comparison,
        plot_overall_summary,
        plot_efficiency_frontier,
    ]

    for fn in plot_fns:
        try:
            fn()
        except Exception as e:
            logger.warning("Failed to generate %s: %s", fn.__name__, e)


if __name__ == "__main__":
    generate_all_plots()
