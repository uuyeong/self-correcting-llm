"""Exp 4: Model size comparison — LLaMA-3.2 3B vs LLaMA-3.1 8B.

Trains probing classifiers on both models and compares:
    - Per-layer AUROC
    - t-SNE separability of hidden states

Outputs:
    - results/exp4_size_comparison.json
    - results/figures/exp4_tsne.pdf
"""

import json
import pickle

import numpy as np

import config
from src.data.dataset_loader import load_truthfulqa
from src.models.llm_wrapper import LLMWrapper
from src.models.probing_classifier import train_probing_classifiers
from src.visualization.plots import plot_layer_auroc_heatmap, plot_tsne


def run_for_model(model_name: str, questions, answers, labels, cache_tag: str):
    cache = config.CACHE_DIR / f"exp4_{cache_tag}_hidden_states.npy"
    if cache.exists():
        hs = np.load(cache)
    else:
        wrapper = LLMWrapper(model_name=model_name)
        hs = wrapper.extract_hidden_states(questions, answers)
        np.save(cache, hs)

    result = train_probing_classifiers(hs, labels)
    return hs, result


def main():
    print("=== Exp 4: Model Size Comparison ===")

    records   = load_truthfulqa(split=config.TRUTHFULQA_SPLIT)
    questions = [r["question"] for r in records]
    answers   = [r["answer"]   for r in records]
    labels    = np.array([r["label"] for r in records])

    print("Running on LLaMA-3.2 3B...")
    hs_3b, result_3b = run_for_model(config.SMALL_MODEL,   questions, answers, labels, "3b")

    print("Running on LLaMA-3.1 8B...")
    hs_8b, result_8b = run_for_model(config.PRIMARY_MODEL, questions, answers, labels, "8b")

    # ── Save AUROC summary ─────────────────────────────────────────────────
    summary = {
        "3B": {"best_layer": result_3b.best_layer, "best_auroc": result_3b.best_auroc,
               "aurocs": result_3b.aurocs},
        "8B": {"best_layer": result_8b.best_layer, "best_auroc": result_8b.best_auroc,
               "aurocs": result_8b.aurocs},
    }
    with open(config.RESULTS_DIR / "exp4_size_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n3B best: layer={result_3b.best_layer}, AUROC={result_3b.best_auroc:.4f}")
    print(f"8B best: layer={result_8b.best_layer}, AUROC={result_8b.best_auroc:.4f}")

    # ── AUROC heatmaps ─────────────────────────────────────────────────────
    plot_layer_auroc_heatmap(result_3b.aurocs, model_name="LLaMA-3.2 3B", save=False
                             ).savefig(config.FIGURES_DIR / "exp4_auroc_3b.pdf", bbox_inches="tight")
    plot_layer_auroc_heatmap(result_8b.aurocs, model_name="LLaMA-3.1 8B", save=False
                             ).savefig(config.FIGURES_DIR / "exp4_auroc_8b.pdf", bbox_inches="tight")

    # ── t-SNE on best layer ────────────────────────────────────────────────
    best_3b_hs = hs_3b[:, result_3b.best_layer, :]
    best_8b_hs = hs_8b[:, result_8b.best_layer, :]

    # Subsample for t-SNE speed
    n = min(500, len(labels))
    idx = np.random.choice(len(labels), n, replace=False)
    plot_tsne(best_3b_hs[idx], labels[idx], best_8b_hs[idx], labels[idx])
    print(f"\nFigures saved to {config.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
