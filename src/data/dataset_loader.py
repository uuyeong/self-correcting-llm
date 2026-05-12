"""Load and preprocess TruthfulQA, HaluEval, and PopQA datasets.

Each loader returns a list of dicts with unified keys:
    {
        "question": str,
        "answer":   str,          # best/correct answer
        "label":    int,          # 1 = hallucination/false, 0 = truthful/true
        "domain":   str | None,   # medical / legal / general (PopQA only)
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def load_truthfulqa(split: str = "validation") -> list[dict]:
    """Load TruthfulQA generation split.

    Uses the 'best_answer' field as the reference answer and marks each
    example as truthful (label=0) or hallucination-prone (label=1) based
    on whether it appears in the 'incorrect_answers' list.
    """
    ds = load_dataset("truthful_qa", "generation", split=split)
    records = []
    for ex in ds:
        correct_answers = ex["correct_answers"]
        incorrect_answers = ex["incorrect_answers"]
        # Create one record per correct answer (label=0)
        for ans in correct_answers[:1]:
            records.append({
                "question": ex["question"],
                "answer":   ans,
                "label":    0,
                "domain":   None,
            })
        # Create one record per incorrect answer (label=1)
        for ans in incorrect_answers[:1]:
            records.append({
                "question": ex["question"],
                "answer":   ans,
                "label":    1,
                "domain":   None,
            })
    return records


def load_halueval(json_path: str | Path) -> list[dict]:
    """Load HaluEval QA subset from a local JSON file.

    HaluEval JSON format (each line):
        {"question": ..., "right_answer": ..., "hallucinated_answer": ...}

    Download from: https://github.com/RUCAIBox/HaluEval
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval file not found at {path}. "
            "Download it from https://github.com/RUCAIBox/HaluEval and set "
            "HALUEVAL_PATH in config.py."
        )
    records = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line.strip())
            records.append({
                "question": ex["question"],
                "answer":   ex["right_answer"],
                "label":    0,
                "domain":   None,
            })
            records.append({
                "question": ex["question"],
                "answer":   ex["hallucinated_answer"],
                "label":    1,
                "domain":   None,
            })
    return records


_POPQA_DOMAIN_MAP = {
    # property_id → domain label (approximate)
    "P31":   "general",
    "P17":   "general",
    "P131":  "general",
    "P106":  "general",
    "P36":   "general",
    "P27":   "general",
    "P176":  "general",
    "P101":  "general",
    "P495":  "general",
    "P30":   "general",
    "P364":  "general",
    "P407":  "general",
    "P103":  "general",
    "P20":   "general",
    "P1376": "general",
    "P159":  "general",
}

_DOMAIN_KEYWORDS = {
    "medical":  ["disease", "drug", "hospital", "doctor", "symptom", "treatment",
                 "medicine", "clinical", "patient", "diagnosis"],
    "legal":    ["law", "court", "judge", "attorney", "legal", "statute",
                 "constitutional", "verdict", "trial", "crime"],
}


def _infer_domain(question: str) -> str:
    q = question.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return domain
    return "general"


def load_popqa(split: str = "test", max_samples: int | None = 2000) -> list[dict]:
    """Load PopQA and assign domain labels via keyword heuristic."""
    ds = load_dataset("akariasai/PopQA", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    records = []
    for ex in ds:
        domain = _infer_domain(ex["question"])
        answers = ex.get("possible_answers") or []
        records.append({
            "question": ex["question"],
            "answer":   answers[0] if answers else "",
            "label":    0,              # PopQA provides correct answers only
            "domain":   domain,
        })
    return records


def get_domain_split(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by domain for Exp 5."""
    from collections import defaultdict
    splits: dict[str, list] = defaultdict(list)
    for r in records:
        splits[r["domain"] or "general"].append(r)
    return dict(splits)
