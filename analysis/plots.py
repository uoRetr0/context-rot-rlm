"""Visualization: matplotlib/seaborn plots for experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / settings.output_dir
PLOTS_DIR = RESULTS_DIR / "plots"

sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)

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
METHOD_ORDER = ["fullcontext", "rag", "mapreduce", "rlm"]
METHOD_LINE_STYLES = {
    "fullcontext": {
        "linestyle": "--",
        "linewidth": 2.6,
        "marker": "^",
        "markersize": 7.5,
        "markeredgewidth": 1.0,
        "zorder": 6,
        "alpha": 1.0,
    },
    "rag": {
        "linestyle": "-",
        "linewidth": 2.2,
        "marker": "o",
        "markersize": 5.5,
        "markeredgewidth": 0.8,
        "zorder": 4,
        "alpha": 0.95,
    },
    "mapreduce": {
        "linestyle": "-.",
        "linewidth": 2.2,
        "marker": "s",
        "markersize": 5.5,
        "markeredgewidth": 0.8,
        "zorder": 3,
        "alpha": 0.95,
    },
    "rlm": {
        "linestyle": "-",
        "linewidth": 2.4,
        "marker": "D",
        "markersize": 5.5,
        "markeredgewidth": 0.8,
        "zorder": 5,
        "alpha": 0.95,
    },
}
BENCHMARK_LABELS = {
    "needle_haystack": "Needle",
    "multihop": "Synthetic Multi-Hop",
    "longbench": "LongBench",
    "musique": "MuSiQue",
    "musique_pro_phaseA": "MuSiQue (Pro)",
    "musique_pro_phaseB": "MuSiQue (Pro)",
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
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _ordered_methods(methods: list[str] | pd.Index) -> list[str]:
    seen = set(methods)
    ordered = [method for method in METHOD_ORDER if method in seen]
    ordered.extend(sorted(method for method in seen if method not in METHOD_ORDER))
    return ordered


def _benchmark_label(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name.replace("_", " ").title())


def _condition_order(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    df = df.copy()
    if "doc_length" in df.columns:
        df["group"] = df.apply(lambda r: f"{r['hops']}-hop / {r['doc_length'] // 1000}K", axis=1)
        df["sort_key"] = df["hops"] * 1_000_000 + df["doc_length"]
        order = (
            df[["group", "sort_key"]]
            .drop_duplicates()
            .sort_values("sort_key")["group"]
            .tolist()
        )
    else:
        df["group"] = df["hops"].map(lambda value: f"{value}-hop")
        order = sorted(df["group"].unique(), key=lambda value: int(value.split("-")[0]))
    return order, df


def plot_needle_heatmap() -> None:
    """Heatmap: F1 by haystack length and needle position for each method."""
    df = _load("needle_haystack")

    methods = _ordered_methods(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.3 * len(methods), 5.8), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        pivot = sub.pivot_table(values="f1", index="needle_position", columns="haystack_length", aggfunc="mean")
        sns.heatmap(
            pivot,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar=method == methods[-1],
            cbar_kws={"label": "Mean F1"} if method == methods[-1] else None,
        )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_ylabel("Needle Position" if method == methods[0] else "")
        ax.set_xlabel("Haystack Length (words)")

    fig.suptitle("Needle-in-Haystack: Mean F1 Heatmap", fontsize=16, y=1.02)
    fig.tight_layout()
    _save(fig, "needle_heatmap")


def plot_needle_by_length() -> None:
    """Line plot: F1 vs haystack length for each method."""
    df = _load("needle_haystack")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in _ordered_methods(df["method"].unique()):
        sub = df[df["method"] == method]
        grouped = sub.groupby("haystack_length")["f1"].mean().sort_index()
        style = METHOD_LINE_STYLES.get(method, {})
        ax.plot(
            grouped.index,
            grouped.values,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            markeredgecolor=METHOD_COLORS.get(method),
            **style,
        )

    ax.set_xlabel("Haystack Length (words)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Needle-in-Haystack: Mean F1 by Document Length")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "needle_by_length")


def plot_needle_by_position() -> None:
    """Line plot: F1 vs needle position for each method."""
    df = _load("needle_haystack")
    positions = sorted(df["needle_position"].unique())
    grouped_df = (
        df.groupby(["method", "needle_position"])["f1"]
        .mean()
        .reset_index()
    )
    y_min = grouped_df["f1"].min()
    y_max = grouped_df["f1"].max()
    pad = 0.015
    lower = max(0.0, y_min - pad)
    upper = min(1.0, y_max + pad)
    if upper - lower < 0.10:
        center = (upper + lower) / 2
        half_span = 0.05
        lower = max(0.0, center - half_span)
        upper = min(1.0, center + half_span)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for method in _ordered_methods(df["method"].unique()):
        sub = df[df["method"] == method]
        grouped = sub.groupby("needle_position")["f1"].mean().reindex(positions)
        style = METHOD_LINE_STYLES.get(method, {})
        ax.plot(
            grouped.index,
            grouped.values,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            markeredgecolor=METHOD_COLORS.get(method),
            **style,
        )

    ax.set_xlabel("Needle Position (0=start, 1=end)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Needle-in-Haystack: Mean F1 by Needle Position")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{pos:.2f}" for pos in positions])
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    _save(fig, "needle_by_position")


def plot_challenging_needle_by_position() -> None:
    """Line plot: F1 vs requested position for the 500k challenging slice."""
    try:
        df = _load("needle_haystack_challenging_500k")
    except FileNotFoundError:
        return

    position_col = "needle_position_requested" if "needle_position_requested" in df.columns else "needle_position"
    if position_col not in df.columns:
        return

    df = df.copy()
    df[position_col] = df[position_col].round(1)
    positions = sorted(df[position_col].dropna().unique())
    if not positions:
        return

    grouped_df = df.groupby(["method", position_col])["f1"].mean().reset_index()
    y_min = grouped_df["f1"].min()
    y_max = grouped_df["f1"].max()
    pad = 0.05
    lower = max(0.0, y_min - pad)
    upper = min(1.0, y_max + pad)
    if upper - lower < 0.25:
        center = (upper + lower) / 2
        half_span = 0.125
        lower = max(0.0, center - half_span)
        upper = min(1.0, center + half_span)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for method in _ordered_methods(df["method"].unique()):
        grouped = (
            df[df["method"] == method]
            .groupby(position_col)["f1"]
            .mean()
            .reindex(positions)
        )
        style = METHOD_LINE_STYLES.get(method, {})
        ax.plot(
            grouped.index,
            grouped.values,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            markeredgecolor=METHOD_COLORS.get(method),
            **style,
        )

    ax.set_xlabel("Requested Needle Position (0=start, 1=end)")
    ax.set_ylabel("Mean F1")
    ax.set_title("500K Challenging Needle: Mean F1 by Position")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{pos:.1f}" for pos in positions])
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    _save(fig, "needle_challenging_500k")


def plot_multihop_comparison() -> None:
    """Heatmap: synthetic multi-hop F1 by condition and method."""
    df = _load("multihop")
    order, df = _condition_order(df)
    pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
    pivot = pivot.reindex(index=order, columns=_ordered_methods(pivot.columns))

    fig_height = max(5.5, 0.7 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    sns.heatmap(
        pivot,
        ax=ax,
        norm=PowerNorm(gamma=0.6, vmin=0, vmax=1),
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 12, "weight": "semibold"},
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("Synthetic Multi-Hop: Mean F1 by Method and Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Condition")
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.columns], rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "multihop_comparison")


def plot_multihop_heatmap() -> None:
    """Three-panel heatmap: RAG, RLM, and RLM-RAG delta."""
    df = _load("multihop")
    if "doc_length" not in df.columns or df["doc_length"].nunique() <= 1:
        return

    df = df.copy()
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

    delta = panels[1][1] - panels[0][1]
    panels.append(("RLM - RAG (Delta F1)", delta))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (title, pivot) in zip(axes, panels):
        is_delta = "Delta" in title
        sns.heatmap(
            pivot,
            ax=ax,
            vmin=-0.3 if is_delta else 0,
            vmax=0.3 if is_delta else 1,
            cmap="RdBu_r" if is_delta else "YlGnBu",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Delta F1" if is_delta else "Mean F1"},
        )
        ax.set_title(title)
        ax.set_ylabel("Document Length (K words)" if ax == axes[0] else "")
        ax.set_xlabel("Hops")

    fig.suptitle("Synthetic Multi-Hop: Retrieval and Recursive Performance", fontsize=16, y=1.03)
    fig.tight_layout()
    _save(fig, "multihop_heatmap")


def plot_pro_musique_comparison() -> None:
    """Heatmap: Pro MuSiQue F1 by condition and method."""
    frames = []
    for name in ["musique_pro_phaseA", "musique_pro_phaseB"]:
        try:
            frames.append(_load(name))
        except FileNotFoundError:
            continue

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    order, df = _condition_order(df)
    pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
    pivot = pivot.reindex(index=order, columns=_ordered_methods(pivot.columns))

    fig_height = max(5.0, 0.7 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    sns.heatmap(
        pivot,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("MuSiQue (Pro): Mean F1 by Method and Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Condition")
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.columns], rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "pro_musique_comparison")


def plot_longbench_comparison() -> None:
    """Two-panel heatmap: LongBench F1 and ROUGE-L by dataset and method."""
    df = _load("longbench")
    methods = _ordered_methods(df["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    for ax, metric, title in zip(axes, ["f1", "rouge_l"], ["F1", "ROUGE-L"]):
        pivot = df.pivot_table(values=metric, index="dataset", columns="method", aggfunc="mean")
        pivot = pivot.reindex(columns=methods)
        sns.heatmap(
            pivot,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": title},
        )
        ax.set_title(f"LongBench: {title}")
        ax.set_xlabel("")
        ax.set_ylabel("Dataset" if ax == axes[0] else "")
        ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.columns], rotation=20, ha="right")

    fig.tight_layout()
    _save(fig, "longbench_comparison")


def plot_musique_comparison() -> None:
    """Heatmap: MuSiQue F1 by condition and method."""
    df = _load("musique")
    order, df = _condition_order(df)
    pivot = df.pivot_table(values="f1", index="group", columns="method", aggfunc="mean")
    pivot = pivot.reindex(index=order, columns=_ordered_methods(pivot.columns))

    fig_height = max(5.5, 0.7 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    sns.heatmap(
        pivot,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("MuSiQue: Mean F1 by Method and Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Condition")
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.columns], rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "musique_comparison")


def plot_overall_summary() -> None:
    """Heatmap: benchmark-by-method summary of mean F1."""
    summaries = []
    for name in ["needle_haystack", "multihop", "longbench", "musique"]:
        try:
            df = _load(name)
        except FileNotFoundError:
            continue
        summary = df.groupby("method")["f1"].mean()
        for method, value in summary.items():
            summaries.append({
                "benchmark": _benchmark_label(name),
                "method": method,
                "f1": value,
            })

    if not summaries:
        return

    summary_df = pd.DataFrame(summaries)
    pivot = summary_df.pivot_table(values="f1", index="benchmark", columns="method", aggfunc="mean")
    pivot = pivot.reindex(columns=_ordered_methods(pivot.columns))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(
        pivot,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("Benchmark-by-Method Summary (Mean F1)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.columns], rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "overall_summary")


def plot_efficiency_frontier() -> None:
    """Heatmap panel: benchmark-by-method accuracy, cost, and call count."""
    frames: list[tuple[str, pd.DataFrame]] = []
    for name in ["needle_haystack", "multihop", "longbench", "musique"]:
        try:
            df = _load(name)
        except FileNotFoundError:
            continue
        if not df.empty:
            frames.append((name, df))

    if not frames:
        return

    rows = []
    for name, df in frames:
        summary = df.groupby("method")[["f1", "cost_usd", "llm_calls"]].mean().reset_index()
        for _, row in summary.iterrows():
            rows.append({
                "benchmark": _benchmark_label(name),
                "method": row["method"],
                "f1": row["f1"],
                "cost_usd": row["cost_usd"],
                "llm_calls": row["llm_calls"],
            })

    summary_df = pd.DataFrame(rows)
    methods = _ordered_methods(summary_df["method"].unique())
    benchmarks = [_benchmark_label(name) for name, _ in frames]

    f1_pivot = summary_df.pivot_table(values="f1", index="benchmark", columns="method", aggfunc="mean")
    f1_pivot = f1_pivot.reindex(index=benchmarks, columns=methods)

    cost_pivot = summary_df.pivot_table(values="cost_usd", index="benchmark", columns="method", aggfunc="mean")
    cost_pivot = cost_pivot.reindex(index=benchmarks, columns=methods)

    calls_pivot = summary_df.pivot_table(values="llm_calls", index="benchmark", columns="method", aggfunc="mean")
    calls_pivot = calls_pivot.reindex(index=benchmarks, columns=methods)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), sharey=True)

    sns.heatmap(
        f1_pivot,
        ax=axes[0],
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 12, "weight": "semibold"},
        cbar_kws={"label": "Mean F1"},
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        cost_pivot,
        ax=axes[1],
        cmap="YlOrRd",
        annot=True,
        fmt=".4f",
        linewidths=0.5,
        annot_kws={"size": 11, "weight": "semibold"},
        cbar_kws={"label": "Mean Cost per Sample (USD)"},
    )
    axes[1].set_title("Cost")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    sns.heatmap(
        calls_pivot,
        ax=axes[2],
        cmap="OrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        annot_kws={"size": 12, "weight": "semibold"},
        cbar_kws={"label": "Mean LLM Calls"},
    )
    axes[2].set_title("LLM Calls")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in methods], rotation=20, ha="right")
        ax.tick_params(labelsize=11)

    fig.suptitle("Benchmark-by-Method Efficiency Summary", fontsize=18, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "efficiency_frontier")


def generate_all_plots() -> None:
    """Generate all plots."""
    logging.basicConfig(level=logging.INFO)

    plot_fns = [
        plot_needle_heatmap,
        plot_needle_by_length,
        plot_needle_by_position,
        plot_challenging_needle_by_position,
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
        except Exception as exc:
            logger.warning("Failed to generate %s: %s", fn.__name__, exc)


if __name__ == "__main__":
    generate_all_plots()
