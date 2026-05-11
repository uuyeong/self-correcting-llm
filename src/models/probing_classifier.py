"""Per-layer Probing Classifier using LogisticRegression.

Trains one classifier per transformer layer and evaluates AUROC.
Returns a ProbingResult with per-layer metrics and the best classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class ProbingResult:
    aurocs: list[float]           # AUROC per layer (index = layer id)
    best_layer: int
    best_auroc: float
    classifiers: list[LogisticRegression] = field(repr=False)
    scalers: list[StandardScaler]         = field(repr=False)

    def score(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        """Return hallucination probability for samples in hidden (N, H)."""
        x = self.scalers[layer_idx].transform(hidden)
        return self.classifiers[layer_idx].predict_proba(x)[:, 1]


def train_probing_classifiers(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    train_split: float = config.TRAIN_SPLIT,
    val_split: float   = config.VAL_SPLIT,
    C: float           = config.PROBING_C,
    max_iter: int      = config.PROBING_MAX_ITER,
    seed: int          = 42,
) -> ProbingResult:
    """Train one LogisticRegression per layer.

    Args:
        hidden_states: (N, n_layers, H)
        labels:        (N,) — 1=hallucination, 0=truthful
    """
    n_samples, n_layers, _ = hidden_states.shape
    assert len(labels) == n_samples

    # Train / val+test split first, then split val/test
    idx_train, idx_temp = train_test_split(
        np.arange(n_samples), train_size=train_split, random_state=seed, stratify=labels
    )
    relative_val = val_split / (1 - train_split)
    idx_val, idx_test = train_test_split(
        idx_temp, train_size=relative_val, random_state=seed, stratify=labels[idx_temp]
    )

    aurocs = []
    classifiers = []
    scalers = []

    for layer in range(n_layers):
        X_train = hidden_states[idx_train, layer, :]
        X_val   = hidden_states[idx_val,   layer, :]
        y_train = labels[idx_train]
        y_val   = labels[idx_val]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=seed)
        clf.fit(X_train_s, y_train)

        proba = clf.predict_proba(X_val_s)[:, 1]
        auroc = roc_auc_score(y_val, proba)

        aurocs.append(float(auroc))
        classifiers.append(clf)
        scalers.append(scaler)

    best_layer = int(np.argmax(aurocs))
    return ProbingResult(
        aurocs=aurocs,
        best_layer=best_layer,
        best_auroc=aurocs[best_layer],
        classifiers=classifiers,
        scalers=scalers,
    )
