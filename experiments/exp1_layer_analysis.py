"""Exp 1: Per-layer AUROC analysis using Probing Classifiers.

Outputs:
    - results/exp1_aurocs.json          : AUROC per layer (git-tracked)
    - results/figures/exp1_layer_auroc_heatmap.pdf
    - {HS_CACHE}/exp1_probing_result.pkl : saved to Drive for Week 3+ reuse
    - Prints best layer and AUROC to stdout

Run on Colab:
    %run experiments/exp1_layer_analysis.py
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

import config
from src.models.probing_classifier import train_probing_classifiers
from src.visualization.plots import plot_layer_auroc_heatmap


def _hs_cache_dir() -> Path:
    """Return Drive HS cache path if available, else local CACHE_DIR."""
    env = os.environ.get("HS_CACHE")
    if env:
        return Path(env)
    return config.CACHE_DIR


def main():
    print("=== Exp 1: Layer-wise AUROC Analysis ===")

    hs_dir = _hs_cache_dir()

    # ── 1. Load hidden states (from Drive cache) ─────────────────────────
    hs_path    = hs_dir / "exp1_hidden_states.npy"
    label_path = hs_dir / "exp1_labels.npy"

    if not hs_path.exists():
        raise FileNotFoundError(
            f"Hidden states not found at {hs_path}\n"
            "Week 1을 먼저 실행하여 exp1_hidden_states.npy를 Drive에 저장하세요."
        )

    print(f"Loading hidden states from {hs_path}")
    hidden_states = np.load(hs_path)
    labels        = np.load(label_path)
    print(f"  shape={hidden_states.shape}, labels={labels.shape}")

    # ── 2. Train probing classifiers ─────────────────────────────────────
    print("Training per-layer probing classifiers...")
    result = train_probing_classifiers(hidden_states, labels)

    # ── 3. Save results ──────────────────────────────────────────────────
    # JSON for git tracking
    auroc_json = config.RESULTS_DIR / "exp1_aurocs.json"
    with open(auroc_json, "w") as f:
        json.dump({"aurocs": result.aurocs, "best_layer": result.best_layer,
                   "best_auroc": result.best_auroc}, f, indent=2)
    print(f"AUROC JSON saved: {auroc_json}")

    # pkl to Drive so Week 3+ can reload without re-training
    pkl_drive = hs_dir / "exp1_probing_result.pkl"
    with open(pkl_drive, "wb") as f:
        pickle.dump(result, f)
    print(f"Probing result saved to Drive: {pkl_drive}")

    # ── 4. Print summary ─────────────────────────────────────────────────
    print(f"\nBest layer : {result.best_layer}")
    print(f"Best AUROC : {result.best_auroc:.4f}")
    print("Top-5 layers by AUROC:")
    top5 = sorted(enumerate(result.aurocs), key=lambda x: -x[1])[:5]
    for layer, auroc in top5:
        print(f"  Layer {layer:2d}: {auroc:.4f}")

    # ── 5. Plot ───────────────────────────────────────────────────────────
    plot_layer_auroc_heatmap(result.aurocs, model_name="LLaMA-3.1-8B")
    print(f"\nFigure saved: {config.FIGURES_DIR}/exp1_layer_auroc_heatmap.pdf")
    print(f"\n>>> config.py 에서 BEST_LAYER = {result.best_layer} 로 업데이트하세요 <<<")

    return result


if __name__ == "__main__":
    main()
