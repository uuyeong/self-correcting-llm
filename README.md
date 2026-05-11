# Self-Correcting LLM
**Hallucination Detection and Reliability-Controlled Re-generation via Internal Activation Analysis**

*Generative AI (ANT5010/AATG010) Final Project*

---

## Overview

This project proposes a self-correcting LLM pipeline that:
1. Extracts per-layer **hidden states** during generation
2. Uses a **Probing Classifier** to detect hallucination signals
3. Applies a **3-tier regeneration strategy** (LOW / MID / HIGH) based on signal strength

### Key Idea
| Signal | Condition | Action |
|--------|-----------|--------|
| LOW | score < 0.40 | Pass-through (no extra cost) |
| MID | 0.40 ≤ score < 0.70 | Partial correction (top-p shrink or DoLa-style logit contrast) |
| HIGH | score ≥ 0.70 | Full regeneration |

---

## Models
| Role | Model |
|------|-------|
| Primary (large) | `meta-llama/Llama-3.1-8B-Instruct` |
| Comparison (small) | `meta-llama/Llama-3.2-3B-Instruct` |

---

## Project Structure

```
self-correcting-llm/
├── config.py                  # Global settings (model names, thresholds, paths)
├── requirements.txt
├── src/
│   ├── data/dataset_loader.py     # TruthfulQA / HaluEval / PopQA loaders
│   ├── models/
│   │   ├── llm_wrapper.py         # Hidden states extraction + generation
│   │   └── probing_classifier.py  # Per-layer LogisticRegression probing
│   ├── pipeline/
│   │   ├── hallucination_detector.py   # Score → LOW/MID/HIGH
│   │   └── regeneration_strategy.py    # 3-tier strategy execution
│   └── visualization/plots.py    # All figures (heatmaps, t-SNE, bar charts)
├── experiments/
│   ├── exp1_layer_analysis.py     # Layer-wise AUROC heatmap
│   ├── exp2_strategy_comparison.py
│   ├── exp3_mid_strategy.py       # MID-A vs MID-B
│   ├── exp4_model_size.py         # 3B vs 8B + t-SNE
│   ├── exp5_domain_analysis.py    # Domain-wise layer analysis
│   └── exp6_baseline_comparison.py
├── notebooks/
│   └── full_pipeline_demo.ipynb  # Submission notebook
└── demo/
    ├── app.py                     # Gradio UI (pre-computed, CPU-only)
    └── precomputed_results.json   # Populated after experiments
```

---

## Setup (Google Colab)

```python
# 1. Clone repo
!git clone https://github.com/<your-repo>/self-correcting-llm.git
%cd self-correcting-llm

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Login to HuggingFace (for LLaMA access)
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

---

## Running Experiments

Run experiments **in order** on Colab Pro (A100 40GB):

```bash
python experiments/exp1_layer_analysis.py   # ~30 min
# ⚠️ Set BEST_LAYER in config.py after this step
python experiments/exp2_strategy_comparison.py
python experiments/exp3_mid_strategy.py
python experiments/exp4_model_size.py
python experiments/exp5_domain_analysis.py
python experiments/exp6_baseline_comparison.py
```

All results are saved to `results/` and figures to `results/figures/`.

---

## Demo (HuggingFace Spaces)

The Gradio demo uses pre-computed results and runs on CPU:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

---

## Datasets

| Dataset | Size | Use | Source |
|---------|------|-----|--------|
| TruthfulQA | 817 | Probing Classifier train/eval | HuggingFace |
| HaluEval | 10,000 | Hallucination detection eval | GitHub |
| PopQA | 2,000 (subset) | Domain analysis | HuggingFace |

**HaluEval** must be downloaded manually from [GitHub](https://github.com/RUCAIBox/HaluEval)
and the path set in `config.py` as `HALUEVAL_PATH`.

---

## Deadline
**June 23, 2026 23:59**
