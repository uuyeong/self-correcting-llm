"""Exp 6: Comprehensive Baseline Comparison.

Compares three methods on the full HaluEval test set:
    Baseline 1: No detection, direct generation
    Baseline 2: Always full regeneration (HIGH strategy)
    Ours:       Signal-strength-based 3-tier strategy

Outputs:
    - results/exp6_baseline_comparison.json
    - results/figures/exp6_baseline_comparison.pdf
"""

import json
import pickle

import numpy as np
from tqdm import tqdm

import config
from src.data.dataset_loader import load_halueval
from src.models.llm_wrapper import LLMWrapper
from src.models.probing_classifier import ProbingResult
from src.pipeline.hallucination_detector import HallucinationDetector, StrategyLevel
from src.pipeline.regeneration_strategy import RegenerationStrategy
from src.visualization.plots import plot_baseline_comparison


def factual_accuracy(predictions: list[str], references: list[str]) -> float:
    hits = sum(
        any(ref.lower() in pred.lower() for ref in refs.split("|"))
        for pred, refs in zip(predictions, references)
    )
    return hits / len(predictions)


def run_baseline(
    name: str,
    questions: list[str],
    answers_ref: list[str],
    wrapper: LLMWrapper,
    detector: HallucinationDetector | None,
    layer_hs: np.ndarray | None,
    strategy: RegenerationStrategy,
    forced_level: StrategyLevel | None = None,
) -> dict:
    preds, costs, latencies = [], [], []
    for i, (q, a) in enumerate(tqdm(zip(questions, answers_ref), total=len(questions), desc=name)):
        prompt = f"Q: {q}\nA:"
        if forced_level is not None:
            level = forced_level
        else:
            level, _ = detector.classify(layer_hs[i])
        out = strategy.apply(prompt, a, level)
        preds.append(out["text"])
        costs.append(out["token_cost"])
        latencies.append(out["latency_ms"])

    acc = factual_accuracy(preds, answers_ref)
    avg_cost = float(np.mean(costs))
    avg_lat  = float(np.mean(latencies))
    eff = acc / max(avg_cost * avg_lat, 1e-9)
    return {"accuracy": acc, "token_cost": avg_cost, "latency_ms": avg_lat, "efficiency": eff}


def main():
    assert config.BEST_LAYER is not None, "Run exp1 first and set BEST_LAYER in config.py"
    print("=== Exp 6: Baseline Comparison ===")

    records = load_halueval(config.HALUEVAL_PATH)
    records = records[:400]  # balanced subset
    questions   = [r["question"] for r in records]
    answers_ref = [r["answer"]   for r in records]

    with open(config.RESULTS_DIR / "exp1_probing_result.pkl", "rb") as f:
        probing_result: ProbingResult = pickle.load(f)

    wrapper  = LLMWrapper(model_name=config.PRIMARY_MODEL)
    detector = HallucinationDetector(probing_result, layer=config.BEST_LAYER)
    strategy = RegenerationStrategy(wrapper, mid_variant="A")

    cache = config.CACHE_DIR / "exp6_hidden_states.npy"
    if cache.exists():
        hs = np.load(cache)
    else:
        hs = wrapper.extract_hidden_states(questions, answers_ref)
        np.save(cache, hs)
    layer_hs = hs[:, config.BEST_LAYER, :]

    results = {
        "Baseline 1\n(No detect)":   run_baseline("Baseline1", questions, answers_ref,
                                                    wrapper, None, None, strategy,
                                                    forced_level=StrategyLevel.LOW),
        "Baseline 2\n(Always regen)": run_baseline("Baseline2", questions, answers_ref,
                                                    wrapper, None, None, strategy,
                                                    forced_level=StrategyLevel.HIGH),
        "Ours\n(3-tier)":             run_baseline("Ours", questions, answers_ref,
                                                    wrapper, detector, layer_hs, strategy),
    }

    with open(config.RESULTS_DIR / "exp6_baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    for name, metrics in results.items():
        print(f"\n{name.replace(chr(10), ' ')}: acc={metrics['accuracy']:.3f}, "
              f"cost={metrics['token_cost']:.1f}, lat={metrics['latency_ms']:.1f}ms, "
              f"eff={metrics['efficiency']:.6f}")

    plot_baseline_comparison(results)
    print(f"\nFigure saved to {config.FIGURES_DIR}/exp6_baseline_comparison.pdf")


if __name__ == "__main__":
    main()
