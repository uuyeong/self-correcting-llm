"""Exp 3: MID-A (top-p) vs MID-B (DoLa) comparison.

Both strategies applied to MID-level samples only.

Outputs:
    - results/exp3_mid_comparison.json
    - results/figures/exp3_mid_ab.pdf
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
from src.visualization.plots import plot_mid_ab_comparison


def factual_accuracy(predictions: list[str], references: list[str]) -> float:
    hits = sum(
        any(ref.lower() in pred.lower() for ref in refs.split("|"))
        for pred, refs in zip(predictions, references)
    )
    return hits / len(predictions)


def main():
    assert config.BEST_LAYER is not None, "Run exp1 first and set BEST_LAYER in config.py"
    print("=== Exp 3: MID-A vs MID-B ===")

    records = load_halueval(config.HALUEVAL_PATH)
    # Select samples that land in MID zone (will be verified by classifier)
    with open(config.RESULTS_DIR / "exp1_probing_result.pkl", "rb") as f:
        probing_result: ProbingResult = pickle.load(f)

    wrapper  = LLMWrapper(model_name=config.PRIMARY_MODEL)
    detector = HallucinationDetector(probing_result, layer=config.BEST_LAYER)

    questions   = [r["question"] for r in records[:500]]
    answers_ref = [r["answer"]   for r in records[:500]]

    cache = config.CACHE_DIR / "exp3_hidden_states.npy"
    if cache.exists():
        hs = np.load(cache)
    else:
        hs = wrapper.extract_hidden_states(questions, answers_ref)
        np.save(cache, hs)

    layer_hs       = hs[:, config.BEST_LAYER, :]
    classifications = detector.classify_batch(layer_hs)

    mid_indices = [i for i, (lv, _) in enumerate(classifications) if lv == StrategyLevel.MID]
    print(f"MID-level samples: {len(mid_indices)}")

    strat_a = RegenerationStrategy(wrapper, mid_variant="A")
    strat_b = RegenerationStrategy(
        wrapper, mid_variant="B",
        dola_high_layer=config.BEST_LAYER,
        dola_low_layer=max(0, config.BEST_LAYER - 8),
    )

    results = {"MID-A": {"preds": [], "costs": [], "latencies": []},
               "MID-B": {"preds": [], "costs": [], "latencies": []}}

    for i in tqdm(mid_indices, desc="MID samples"):
        prompt = f"Q: {questions[i]}\nA:"
        for name, strat in [("MID-A", strat_a), ("MID-B", strat_b)]:
            out = strat.apply(prompt, answers_ref[i], StrategyLevel.MID)
            results[name]["preds"].append(out["text"])
            results[name]["costs"].append(out["token_cost"])
            results[name]["latencies"].append(out["latency_ms"])

    mid_refs = [answers_ref[i] for i in mid_indices]
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "accuracy":   factual_accuracy(data["preds"], mid_refs),
            "token_cost": float(np.mean(data["costs"])),
            "latency_ms": float(np.mean(data["latencies"])),
        }

    with open(config.RESULTS_DIR / "exp3_mid_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMID-A: {summary['MID-A']}")
    print(f"MID-B: {summary['MID-B']}")

    plot_mid_ab_comparison(summary["MID-A"], summary["MID-B"])
    print(f"\nFigure saved to {config.FIGURES_DIR}/exp3_mid_ab.pdf")


if __name__ == "__main__":
    main()
