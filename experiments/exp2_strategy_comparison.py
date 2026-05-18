"""Exp 2: 3-tier regeneration strategy comparison.

Compares 4 strategies on HaluEval hallucinated samples:
  - Baseline : always pass-through (return hallucinated answer)
  - MID-A    : always apply top-p=0.7 regeneration
  - HIGH     : always apply full regeneration
  - Ours     : classifier-guided adaptive strategy

Outputs:
    - results/exp2_strategy_results.json
    - results/figures/exp2_strategy_comparison.pdf

Prerequisite: exp1 complete, BEST_LAYER set in config.py
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from src.data.dataset_loader import load_halueval
from src.models.llm_wrapper import LLMWrapper
from src.pipeline.hallucination_detector import HallucinationDetector, StrategyLevel
from src.pipeline.regeneration_strategy import RegenerationStrategy
from src.visualization.plots import plot_strategy_comparison


def _hs_cache_dir() -> Path:
    env = os.environ.get("HS_CACHE")
    return Path(env) if env else config.CACHE_DIR


def _load_pairs(path, n: int = 200) -> list[dict]:
    """Return list of {question, correct, hallucinated} from HaluEval."""
    records = load_halueval(path)
    pairs = []
    for i in range(0, len(records) - 1, 2):
        r_correct = records[i]
        r_hall    = records[i + 1]
        if r_correct["question"] == r_hall["question"]:
            pairs.append({
                "question":     r_correct["question"],
                "correct":      r_correct["answer"],
                "hallucinated": r_hall["answer"],
            })
    return pairs[:n]


def factual_accuracy(preds: list[str], correct_refs: list[str]) -> float:
    """Check if correct reference is contained in each prediction."""
    hits = sum(ref.lower() in pred.lower() for pred, ref in zip(preds, correct_refs))
    return hits / max(len(preds), 1)


def main():
    assert config.BEST_LAYER is not None, "Run exp1 first and set BEST_LAYER in config.py"
    print("=== Exp 2: Strategy Comparison ===")

    hs_dir = _hs_cache_dir()

    # ── Data ──────────────────────────────────────────────────────────────
    pairs = _load_pairs(config.HALUEVAL_PATH, n=200)
    questions  = [p["question"]     for p in pairs]
    corrects   = [p["correct"]      for p in pairs]
    hall_ans   = [p["hallucinated"] for p in pairs]
    print(f"Using {len(pairs)} HaluEval question pairs")

    # ── Probing classifier ────────────────────────────────────────────────
    pkl_path = hs_dir / "exp1_probing_result.pkl"
    with open(pkl_path, "rb") as f:
        probing_result = pickle.load(f)

    wrapper  = LLMWrapper(model_name=config.PRIMARY_MODEL)
    detector = HallucinationDetector(probing_result, layer=config.BEST_LAYER)

    # ── Hidden states for hallucinated answers ────────────────────────────
    hs_path = hs_dir / "exp2_hidden_states.npy"
    if hs_path.exists():
        print(f"Loading cached HS from {hs_path}")
        hs = np.load(hs_path)
    else:
        print("Extracting hidden states for hallucinated answers...")
        hs = wrapper.extract_hidden_states(questions, hall_ans, batch_size=8)
        np.save(hs_path, hs)
        print(f"Saved to {hs_path}")

    layer_hs        = hs[:, config.BEST_LAYER, :]
    classifications = detector.classify_batch(layer_hs)

    # ── Run strategies ─────────────────────────────────────────────────────
    strat = RegenerationStrategy(wrapper, mid_variant="A")

    strategy_preds: dict[str, list] = {k: [] for k in ["Baseline", "MID-A", "HIGH", "Ours"]}
    strategy_costs: dict[str, list] = {k: [] for k in ["Baseline", "MID-A", "HIGH", "Ours"]}
    strategy_lats:  dict[str, list] = {k: [] for k in ["Baseline", "MID-A", "HIGH", "Ours"]}

    for i, (question, hall, (level, _)) in enumerate(
        tqdm(zip(questions, hall_ans, classifications), total=len(questions), desc="Running strategies")
    ):
        prompt = f"Q: {question}\nA:"

        for name, forced_level in [
            ("Baseline", StrategyLevel.LOW),
            ("MID-A",    StrategyLevel.MID),
            ("HIGH",     StrategyLevel.HIGH),
        ]:
            out = strat.apply(prompt, hall, forced_level)
            strategy_preds[name].append(out["text"] if name != "Baseline" else hall)
            strategy_costs[name].append(out["token_cost"])
            strategy_lats[name].append(out["latency_ms"])

        # Ours: classifier-guided
        out = strat.apply(prompt, hall, level)
        strategy_preds["Ours"].append(out["text"] if level != StrategyLevel.LOW else hall)
        strategy_costs["Ours"].append(out["token_cost"])
        strategy_lats["Ours"].append(out["latency_ms"])

    # ── Metrics ───────────────────────────────────────────────────────────
    summary = {}
    for name in ["Baseline", "MID-A", "HIGH", "Ours"]:
        acc = factual_accuracy(strategy_preds[name], corrects)
        summary[name] = {
            "accuracy":   acc,
            "token_cost": float(np.mean(strategy_costs[name])),
            "latency_ms": float(np.mean(strategy_lats[name])),
            "efficiency": acc / max(np.mean(strategy_costs[name]) * np.mean(strategy_lats[name]), 1e-9),
        }

    level_counts = {lv.value: sum(1 for l, _ in classifications if l == lv)
                    for lv in StrategyLevel}
    summary["_meta"] = {"level_counts": level_counts, "n_samples": len(pairs)}

    out_path = config.RESULTS_DIR / "exp2_strategy_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults:")
    for name in ["Baseline", "MID-A", "HIGH", "Ours"]:
        m = summary[name]
        print(f"  {name:8s}: acc={m['accuracy']:.3f}, "
              f"cost={m['token_cost']:.1f} tok, lat={m['latency_ms']:.1f} ms")
    print(f"\nLevel distribution: {level_counts}")

    plot_strategy_comparison({k: summary[k] for k in ["Baseline", "MID-A", "HIGH", "Ours"]})
    print(f"Figure saved: {config.FIGURES_DIR}/exp2_strategy_comparison.pdf")


if __name__ == "__main__":
    main()
