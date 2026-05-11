"""Hallucination Detector: maps probing score → signal level (LOW / MID / HIGH).

Uses a trained ProbingResult to score a hidden state vector and return a
StrategyLevel that the RegenerationStrategy can act on.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

import config
from src.models.probing_classifier import ProbingResult


class StrategyLevel(str, Enum):
    LOW  = "LOW"   # pass-through; no correction needed
    MID  = "MID"   # partial correction (top-p shrink or DoLa)
    HIGH = "HIGH"  # full regeneration


class HallucinationDetector:
    def __init__(
        self,
        probing_result: ProbingResult,
        layer: int | None = None,
        low_threshold:  float = config.LOW_THRESHOLD,
        high_threshold: float = config.HIGH_THRESHOLD,
    ):
        self.result    = probing_result
        self.layer     = layer if layer is not None else probing_result.best_layer
        self.low_th    = low_threshold
        self.high_th   = high_threshold

    def score(self, hidden: np.ndarray) -> float:
        """Return hallucination probability for a single sample.

        Args:
            hidden: (H,) hidden state vector from the chosen layer
        """
        proba = self.result.score(self.layer, hidden.reshape(1, -1))
        return float(proba[0])

    def classify(self, hidden: np.ndarray) -> tuple[StrategyLevel, float]:
        """Return (StrategyLevel, raw_score) for a single hidden state."""
        s = self.score(hidden)
        if s < self.low_th:
            level = StrategyLevel.LOW
        elif s < self.high_th:
            level = StrategyLevel.MID
        else:
            level = StrategyLevel.HIGH
        return level, s

    def classify_batch(
        self, hidden_states: np.ndarray
    ) -> list[tuple[StrategyLevel, float]]:
        """Classify a batch of hidden states.

        Args:
            hidden_states: (N, H) array for the chosen layer
        """
        probas = self.result.score(self.layer, hidden_states)
        results = []
        for p in probas:
            p = float(p)
            if p < self.low_th:
                level = StrategyLevel.LOW
            elif p < self.high_th:
                level = StrategyLevel.MID
            else:
                level = StrategyLevel.HIGH
            results.append((level, p))
        return results
