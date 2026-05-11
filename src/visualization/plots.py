"""Visualization utilities for all experiments.

All functions save figures to config.FIGURES_DIR and also return the
matplotlib Figure object for inline display in notebooks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import config

_STYLE = "seaborn-v0_8-whitegrid"


def plot_layer_auroc_heatmap(
    aurocs: list[float],
    model_name: str = "LLaMA-3.1-8B",
    save: bool = True,
) -> plt.Figure:
    """Exp 1: per-layer AUROC as a horizontal heatmap."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 2))
        data = np.array(aurocs).reshape(1, -1)
        sns.heatmap(
            data, ax=ax, cmap="RdYlGn", vmin=0.5, vmax=1.0,
            annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "AUROC"},
        )
        ax.set_xlabel("Layer Index")
        ax.set_yticks([])
        ax.set_title(f"Per-Layer Hallucination Detection AUROC ({model_name})")
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp1_layer_auroc_heatmap.pdf", bbox_inches="tight")
    return fig


def plot_strategy_comparison(
    results: dict[str, dict],
    save: bool = True,
) -> plt.Figure:
    """Exp 2: bar chart of Accuracy / Token Cost / Latency per strategy.

    Args:
        results: {strategy_name: {"accuracy": float, "token_cost": float, "latency_ms": float}}
    """
    strategies = list(results.keys())
    metrics = ["accuracy", "token_cost", "latency_ms"]
    labels  = ["Factual Accuracy (%)", "Avg Token Cost", "Avg Latency (ms)"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric, label in zip(axes, metrics, labels):
            values = [results[s][metric] for s in strategies]
            bars = ax.bar(strategies, values, color=sns.color_palette("Set2", len(strategies)))
            ax.set_title(label)
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=15, ha="right")
            if metric == "accuracy":
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp2_strategy_comparison.pdf", bbox_inches="tight")
    return fig


def plot_mid_ab_comparison(
    mid_a: dict, mid_b: dict, save: bool = True
) -> plt.Figure:
    """Exp 3: MID-A vs MID-B accuracy / latency comparison."""
    metrics = ["accuracy", "latency_ms"]
    labels  = ["Factual Accuracy", "Latency (ms)"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, metric, label in zip(axes, metrics, labels):
            ax.bar(["MID-A (top-p)", "MID-B (DoLa)"],
                   [mid_a[metric], mid_b[metric]],
                   color=["#4C72B0", "#DD8452"])
            ax.set_title(label)
        fig.suptitle("MID Strategy: A vs B")
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp3_mid_ab.pdf", bbox_inches="tight")
    return fig


def plot_tsne(
    hidden_3b: np.ndarray, labels_3b: np.ndarray,
    hidden_8b: np.ndarray, labels_8b: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Exp 4: t-SNE of hallucination vs truthful hidden states, 3B vs 8B."""
    from sklearn.manifold import TSNE

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, hs, lb, title in zip(
            axes,
            [hidden_3b, hidden_8b],
            [labels_3b, labels_8b],
            ["LLaMA-3.2 3B", "LLaMA-3.1 8B"],
        ):
            emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(hs)
            for label, color, name in [(0, "#4C72B0", "Truthful"), (1, "#DD8452", "Hallucination")]:
                mask = lb == label
                ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=name, alpha=0.5, s=10)
            ax.set_title(title)
            ax.legend()
        fig.suptitle("t-SNE: Hidden State Separability by Model Size")
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp4_tsne.pdf", bbox_inches="tight")
    return fig


def plot_domain_heatmap(
    domain_aurocs: dict[str, list[float]],
    save: bool = True,
) -> plt.Figure:
    """Exp 5: per-domain, per-layer AUROC heatmap.

    Args:
        domain_aurocs: {domain_name: [auroc_layer_0, auroc_layer_1, ...]}
    """
    domains = list(domain_aurocs.keys())
    data = np.array([domain_aurocs[d] for d in domains])  # (n_domains, n_layers)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, len(domains) * 1.5 + 1))
        sns.heatmap(
            data, ax=ax, cmap="RdYlGn", vmin=0.5, vmax=1.0,
            annot=False, linewidths=0.3,
            xticklabels=[str(i) for i in range(data.shape[1])],
            yticklabels=domains,
            cbar_kws={"label": "AUROC"},
        )
        ax.set_xlabel("Layer Index")
        ax.set_title("Domain-wise Hallucination Detection AUROC per Layer")
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp5_domain_heatmap.pdf", bbox_inches="tight")
    return fig


def plot_baseline_comparison(
    baselines: dict[str, dict],
    save: bool = True,
) -> plt.Figure:
    """Exp 6: comprehensive comparison table as a styled bar chart.

    Args:
        baselines: {method_name: {"accuracy": float, "token_cost": float, "latency_ms": float, "efficiency": float}}
    """
    methods  = list(baselines.keys())
    metrics  = ["accuracy", "token_cost", "latency_ms", "efficiency"]
    labels   = ["Factual Accuracy", "Token Cost", "Latency (ms)", "Efficiency Score"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        palette = sns.color_palette("Set1", len(methods))
        for ax, metric, label in zip(axes, metrics, labels):
            values = [baselines[m][metric] for m in methods]
            ax.bar(methods, values, color=palette)
            ax.set_title(label)
            ax.set_xticklabels(methods, rotation=20, ha="right")
            if metric == "accuracy":
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        fig.suptitle("Baseline Comparison: Ours vs. Baselines", fontsize=13, fontweight="bold")
        fig.tight_layout()
    if save:
        fig.savefig(config.FIGURES_DIR / "exp6_baseline_comparison.pdf", bbox_inches="tight")
    return fig
