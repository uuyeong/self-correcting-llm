"""Global configuration for the Self-Correcting LLM project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "hidden_states_cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────────
PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"    # large model
SMALL_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"    # for Exp 4 size comparison

# ── Model loading defaults ─────────────────────────────────────────────────
LOAD_IN_4BIT = True       # bitsandbytes 4-bit quantization
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8

# ── Probing Classifier ─────────────────────────────────────────────────────
TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
PROBING_MAX_ITER = 1000
PROBING_C = 1.0           # LogisticRegression regularization

# ── Hallucination Detection Thresholds ────────────────────────────────────
# Probing score ∈ [0, 1]; higher = more likely hallucination
LOW_THRESHOLD  = 0.40     # below → LOW (pass through)
HIGH_THRESHOLD = 0.70     # above → HIGH (full regeneration); between → MID

# ── Regeneration Strategy ─────────────────────────────────────────────────
MID_A_TOP_P = 0.70        # MID-A: conservative top-p sampling
DOLA_ALPHA  = 1.0         # MID-B: DoLa logit contrast weight

# Optimal probing layer (updated after Exp 1)
BEST_LAYER = 16         # set after running exp1_layer_analysis.py

# ── Datasets ───────────────────────────────────────────────────────────────
TRUTHFULQA_SPLIT = "validation"
HALUEVAL_PATH    = None    # set to local path after download from GitHub
POPQA_SPLIT      = "test"

# ── Efficiency metric ──────────────────────────────────────────────────────
# Efficiency = Accuracy_Gain / (Token_Cost × Latency_ms)
# higher is better
