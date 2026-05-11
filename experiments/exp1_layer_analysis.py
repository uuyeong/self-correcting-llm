"""Exp 1: Per-layer AUROC analysis using Probing Classifiers.

Outputs:
    - results/exp1_aurocs.npy          : array of AUROC per layer
    - results/figures/exp1_layer_auroc_heatmap.pdf
    - Prints best layer and AUROC to stdout
    - Updates config.BEST_LAYER (written to config.py is manual)

Run on Colab:
    !python experiments/exp1_layer_analysis.py
"""

import numpy as np
import pickle
from pathlib import Path

import config
from src.data.dataset_loader import load_truthfulqa
from src.models.llm_wrapper import LLMWrapper
from src.models.probing_classifier import train_probing_classifiers
from src.visualization.plots import plot_layer_auroc_heatmap


def main():
    print("=== Exp 1: Layer-wise AUROC Analysis ===")

    # ── 1. Load dataset ──────────────────────────────────────────────────
    print("Loading TruthfulQA...")
    records = load_truthfulqa(split=config.TRUTHFULQA_SPLIT)
    questions = [r["question"] for r in records]
    answers   = [r["answer"]   for r in records]
    labels    = np.array([r["label"] for r in records])
    print(f"  {len(records)} records (truthful + hallucinated)")

    # ── 2. Extract hidden states ─────────────────────────────────────────
    cache_path = config.CACHE_DIR / "exp1_hidden_states.npy"
    if cache_path.exists():
        print(f"Loading cached hidden states from {cache_path}")
        hidden_states = np.load(cache_path)
    else:
        print("Extracting hidden states (this may take a while)...")
        wrapper = LLMWrapper(model_name=config.PRIMARY_MODEL)
        hidden_states = wrapper.extract_hidden_states(questions, answers)
        np.save(cache_path, hidden_states)
        print(f"Saved to {cache_path}  shape={hidden_states.shape}")

    # ── 3. Train probing classifiers ─────────────────────────────────────
    print("Training per-layer probing classifiers...")
    result = train_probing_classifiers(hidden_states, labels)

    # ── 4. Save results ──────────────────────────────────────────────────
    np.save(config.RESULTS_DIR / "exp1_aurocs.npy", np.array(result.aurocs))
    with open(config.RESULTS_DIR / "exp1_probing_result.pkl", "wb") as f:
        pickle.dump(result, f)

    print(f"\nBest layer: {result.best_layer}  |  AUROC: {result.best_auroc:.4f}")
    print("Top-5 layers by AUROC:")
    top5 = sorted(enumerate(result.aurocs), key=lambda x: -x[1])[:5]
    for layer, auroc in top5:
        print(f"  Layer {layer:2d}: {auroc:.4f}")

    # ── 5. Plot ───────────────────────────────────────────────────────────
    plot_layer_auroc_heatmap(result.aurocs, model_name="LLaMA-3.1-8B")
    print(f"\nFigure saved to {config.FIGURES_DIR}/exp1_layer_auroc_heatmap.pdf")
    print(f"\n⚠️  Set BEST_LAYER = {result.best_layer} in config.py before running Exp 2+")


if __name__ == "__main__":
    main()
