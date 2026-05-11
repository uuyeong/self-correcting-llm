"""Exp 2: 3-tier regeneration strategy effect comparison.

Measures Factual Accuracy / Token Cost / Latency for each strategy level
(LOW / MID-A / HIGH) on HaluEval samples.

Outputs:
    - results/exp2_strategy_results.json
    - results/figures/exp2_strategy_comparison.pdf

Prerequisite: Run exp1 first and set config.BEST_LAYER.
"""

import json
import pickle
import time

import numpy as np
from tqdm import tqdm

import config
from src.data.dataset_loader import load_halueval
from src.models.llm_wrapper import LLMWrapper
from src.models.probing_classifier import ProbingResult
from src.pipeline.hallucination_detector import HallucinationDetector, StrategyLevel
from src.pipeline.regeneration_strategy import RegenerationStrategy
from src.visualization.plots import plot_strategy_comparison


def factual_accuracy(predictions: list[str], references: list[str]) -> float:
    """Simple EM-based factual accuracy (placeholder for BERTScore)."""
    hits = sum(
        any(ref.lower() in pred.lower() for ref in refs.split("|"))
        for pred, refs in zip(predictions, references)
    )
    return hits / len(predictions)


def main():
    assert config.BEST_LAYER is not None, "Run exp1 first and set BEST_LAYER in config.py"
    print("=== Exp 2: Strategy Comparison ===")

    # ── Load HaluEval ─────────────────────────────────────────────────────
    records = load_halueval(config.HALUEVAL_PATH)
    # Use a balanced subset for speed
    from sklearn.utils import resample
    pos = [r for r in records if r["label"] == 1][:200]
    neg = [r for r in records if r["label"] == 0][:200]
    records = pos + neg
    np.random.seed(42)
    np.random.shuffle(records)
    questions   = [r["question"] for r in records]
    answers_ref = [r["answer"]   for r in records]
    labels      = np.array([r["label"] for r in records])

    # ── Load probing result ───────────────────────────────────────────────
    with open(config.RESULTS_DIR / "exp1_probing_result.pkl", "rb") as f:
        probing_result: ProbingResult = pickle.load(f)

    wrapper   = LLMWrapper(model_name=config.PRIMARY_MODEL)
    detector  = HallucinationDetector(probing_result, layer=config.BEST_LAYER)

    # ── Extract hidden states ─────────────────────────────────────────────
    cache = config.CACHE_DIR / "exp2_hidden_states.npy"
    if cache.exists():
        hs = np.load(cache)
    else:
        hs = wrapper.extract_hidden_states(questions, answers_ref)
        np.save(cache, hs)

    layer_hs = hs[:, config.BEST_LAYER, :]           # (N, H)
    classifications = detector.classify_batch(layer_hs)  # [(level, score), ...]

    # ── Run strategies ─────────────────────────────────────────────────────
    strategy_results: dict[str, dict] = {
        "LOW":    {"preds": [], "costs": [], "latencies": []},
        "MID-A":  {"preds": [], "costs": [], "latencies": []},
        "HIGH":   {"preds": [], "costs": [], "latencies": []},
    }

    strat_low  = RegenerationStrategy(wrapper, mid_variant="A")
    strat_mida = RegenerationStrategy(wrapper, mid_variant="A")
    strat_high = RegenerationStrategy(wrapper, mid_variant="A")

    for i, (question, answer, (level, score)) in enumerate(
        tqdm(zip(questions, answers_ref, classifications), total=len(questions))
    ):
        prompt = f"Q: {question}\nA:"

        for strategy_name, forced_level in [
            ("LOW",  StrategyLevel.LOW),
            ("MID-A", StrategyLevel.MID),
            ("HIGH",  StrategyLevel.HIGH),
        ]:
            out = strat_low.apply(prompt, answer, forced_level)
            strategy_results[strategy_name]["preds"].append(out["text"])
            strategy_results[strategy_name]["costs"].append(out["token_cost"])
            strategy_results[strategy_name]["latencies"].append(out["latency_ms"])

    # ── Compute metrics ───────────────────────────────────────────────────
    summary = {}
    for name, data in strategy_results.items():
        acc = factual_accuracy(data["preds"], answers_ref)
        summary[name] = {
            "accuracy":    acc,
            "token_cost":  float(np.mean(data["costs"])),
            "latency_ms":  float(np.mean(data["latencies"])),
            "efficiency":  acc / max(np.mean(data["costs"]) * np.mean(data["latencies"]), 1e-9),
        }

    with open(config.RESULTS_DIR / "exp2_strategy_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults:")
    for name, metrics in summary.items():
        print(f"  {name}: acc={metrics['accuracy']:.3f}, "
              f"cost={metrics['token_cost']:.1f}, lat={metrics['latency_ms']:.1f}ms")

    plot_strategy_comparison(summary)
    print(f"\nFigure saved to {config.FIGURES_DIR}/exp2_strategy_comparison.pdf")


if __name__ == "__main__":
    main()
