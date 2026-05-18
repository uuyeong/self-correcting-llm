"""Exp 3: MID-A (top-p) vs MID-B (DoLa) comparison.

Applied only to samples the classifier assigns MID level.

Outputs:
    - results/exp3_mid_comparison.json
    - results/figures/exp3_mid_ab.pdf
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
from src.visualization.plots import plot_mid_ab_comparison


def _hs_cache_dir() -> Path:
    env = os.environ.get("HS_CACHE")
    return Path(env) if env else config.CACHE_DIR


def _halueval_path() -> str:
    return os.environ.get("HALUEVAL_PATH") or str(config.HALUEVAL_PATH)


def factual_accuracy(preds: list[str], correct_refs: list[str]) -> float:
    hits = sum(ref.lower() in pred.lower() for pred, ref in zip(preds, correct_refs))
    return hits / max(len(preds), 1)


def main():
    assert config.BEST_LAYER is not None, "Run exp1 first and set BEST_LAYER in config.py"
    print("=== Exp 3: MID-A vs MID-B ===")

    hs_dir = _hs_cache_dir()

    # ── Data: correct + hallucinated pairs ───────────────────────────────
    records = load_halueval(_halueval_path())
    pairs = []
    for i in range(0, len(records) - 1, 2):
        rc, rh = records[i], records[i + 1]
        if rc["question"] == rh["question"]:
            pairs.append({"question": rc["question"],
                          "correct": rc["answer"],
                          "hallucinated": rh["answer"]})
    pairs = pairs[:500]

    questions = [p["question"]     for p in pairs]
    corrects  = [p["correct"]      for p in pairs]
    hall_ans  = [p["hallucinated"] for p in pairs]

    # ── Probing classifier ────────────────────────────────────────────────
    pkl_path = hs_dir / "exp1_probing_result.pkl"
    with open(pkl_path, "rb") as f:
        probing_result = pickle.load(f)

    wrapper  = LLMWrapper(model_name=config.PRIMARY_MODEL)
    detector = HallucinationDetector(probing_result, layer=config.BEST_LAYER)

    # ── Hidden states ─────────────────────────────────────────────────────
    hs_path = hs_dir / "exp3_hidden_states.npy"
    if hs_path.exists():
        print(f"Loading cached HS from {hs_path}")
        hs = np.load(hs_path)
    else:
        print("Extracting hidden states...")
        hs = wrapper.extract_hidden_states(questions, hall_ans, batch_size=8)
        np.save(hs_path, hs)

    layer_hs        = hs[:, config.BEST_LAYER, :]
    classifications = detector.classify_batch(layer_hs)

    mid_indices = [i for i, (lv, _) in enumerate(classifications) if lv == StrategyLevel.MID]
    print(f"MID-level samples: {len(mid_indices)} / {len(pairs)}")

    if not mid_indices:
        print("No MID samples found — adjust thresholds in config.py")
        return

    # ── MID-A vs MID-B ────────────────────────────────────────────────────
    strat_a = RegenerationStrategy(wrapper, mid_variant="A")
    strat_b = RegenerationStrategy(
        wrapper,
        mid_variant="B",
        dola_high_layer=config.BEST_LAYER,
        dola_low_layer=max(0, config.BEST_LAYER - 8),
    )

    res = {name: {"preds": [], "costs": [], "latencies": []}
           for name in ["MID-A", "MID-B"]}

    for i in tqdm(mid_indices, desc="MID samples"):
        prompt = f"Q: {questions[i]}\nA:"
        for name, strat in [("MID-A", strat_a), ("MID-B", strat_b)]:
            out = strat.apply(prompt, hall_ans[i], StrategyLevel.MID)
            res[name]["preds"].append(out["text"])
            res[name]["costs"].append(out["token_cost"])
            res[name]["latencies"].append(out["latency_ms"])

    mid_corrects = [corrects[i] for i in mid_indices]
    summary = {}
    for name, data in res.items():
        summary[name] = {
            "accuracy":   factual_accuracy(data["preds"], mid_corrects),
            "token_cost": float(np.mean(data["costs"])),
            "latency_ms": float(np.mean(data["latencies"])),
        }
    summary["_meta"] = {"n_mid_samples": len(mid_indices)}

    out_path = config.RESULTS_DIR / "exp3_mid_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMID-A: acc={summary['MID-A']['accuracy']:.3f}, "
          f"lat={summary['MID-A']['latency_ms']:.1f} ms")
    print(f"MID-B: acc={summary['MID-B']['accuracy']:.3f}, "
          f"lat={summary['MID-B']['latency_ms']:.1f} ms")

    plot_mid_ab_comparison(summary["MID-A"], summary["MID-B"])
    print(f"Figure saved: {config.FIGURES_DIR}/exp3_mid_ab.pdf")


if __name__ == "__main__":
    main()
