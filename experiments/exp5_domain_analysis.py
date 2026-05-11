"""Exp 5: Domain-wise optimal layer analysis using PopQA.

Groups samples by domain (general / medical / legal) and trains
per-layer probing classifiers for each domain.

Outputs:
    - results/exp5_domain_aurocs.json
    - results/figures/exp5_domain_heatmap.pdf

Note: PopQA only has correct answers (label=0). We augment with
hallucinated answers by prompting the model with misleading context.
For the scaffold, we use random label assignment as a placeholder.
"""

import json
import numpy as np

import config
from src.data.dataset_loader import load_popqa, get_domain_split
from src.models.llm_wrapper import LLMWrapper
from src.models.probing_classifier import train_probing_classifiers
from src.visualization.plots import plot_domain_heatmap


def main():
    print("=== Exp 5: Domain Analysis ===")

    records = load_popqa(split="test", max_samples=1500)
    domain_splits = get_domain_split(records)
    print(f"Domain distribution: { {k: len(v) for k, v in domain_splits.items()} }")

    wrapper = LLMWrapper(model_name=config.PRIMARY_MODEL)
    domain_aurocs: dict[str, list[float]] = {}

    for domain, recs in domain_splits.items():
        if len(recs) < 50:
            print(f"  Skipping {domain}: too few samples ({len(recs)})")
            continue
        print(f"\nProcessing domain: {domain} ({len(recs)} samples)")

        questions = [r["question"] for r in recs]
        answers   = [r["answer"]   for r in recs]

        # PopQA has only correct answers; generate synthetic hallucinated ones
        # by pairing each question with a random other sample's answer
        shuffled_answers = answers.copy()
        np.random.seed(42)
        np.random.shuffle(shuffled_answers)

        all_qs  = questions + questions
        all_ans = answers   + shuffled_answers
        all_lbs = np.array([0] * len(questions) + [1] * len(questions))

        cache = config.CACHE_DIR / f"exp5_{domain}_hidden_states.npy"
        if cache.exists():
            hs = np.load(cache)
        else:
            hs = wrapper.extract_hidden_states(all_qs, all_ans)
            np.save(cache, hs)

        result = train_probing_classifiers(hs, all_lbs)
        domain_aurocs[domain] = result.aurocs
        print(f"  Best layer: {result.best_layer}  AUROC: {result.best_auroc:.4f}")

    with open(config.RESULTS_DIR / "exp5_domain_aurocs.json", "w") as f:
        json.dump(domain_aurocs, f, indent=2)

    plot_domain_heatmap(domain_aurocs)
    print(f"\nFigure saved to {config.FIGURES_DIR}/exp5_domain_heatmap.pdf")


if __name__ == "__main__":
    main()
