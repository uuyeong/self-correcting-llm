"""Three-tier regeneration strategy.

    LOW  → pass-through (return original answer)
    MID-A → conservative top-p sampling (top_p 0.9 → 0.7)
    MID-B → DoLa-style logit contrast via LLMWrapper.generate_with_dola
    HIGH  → full regeneration with default sampling

Each method returns a dict with:
    {
        "text":        str,        # final answer text
        "strategy":    str,        # "LOW" / "MID-A" / "MID-B" / "HIGH"
        "token_cost":  int,        # extra tokens generated (0 for LOW)
        "latency_ms":  float,      # wall time in ms (0 for LOW)
    }
"""

from __future__ import annotations

import time

import config
from src.models.llm_wrapper import LLMWrapper
from src.pipeline.hallucination_detector import StrategyLevel


class RegenerationStrategy:
    def __init__(
        self,
        wrapper: LLMWrapper,
        mid_variant: str = "A",          # "A" = top-p, "B" = DoLa
        dola_high_layer: int | None = None,
        dola_low_layer:  int = 0,
        mid_a_top_p: float  = config.MID_A_TOP_P,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ):
        self.wrapper        = wrapper
        self.mid_variant    = mid_variant
        self.dola_high      = dola_high_layer or wrapper.n_layers - 1
        self.dola_low       = dola_low_layer
        self.mid_a_top_p    = mid_a_top_p
        self.max_new_tokens = max_new_tokens

    def apply(
        self,
        prompt: str,
        original_answer: str,
        level: StrategyLevel,
    ) -> dict:
        if level == StrategyLevel.LOW:
            return {
                "text":       original_answer,
                "strategy":   "LOW",
                "token_cost": 0,
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()

        if level == StrategyLevel.MID:
            if self.mid_variant == "A":
                text, _ = self.wrapper.generate(
                    prompt,
                    top_p=self.mid_a_top_p,
                    max_new_tokens=self.max_new_tokens,
                )
                strategy_label = "MID-A"
            else:
                text = self.wrapper.generate_with_dola(
                    prompt,
                    high_layer=self.dola_high,
                    low_layer=self.dola_low,
                    max_new_tokens=self.max_new_tokens,
                )
                strategy_label = "MID-B"
        else:  # HIGH
            text, _ = self.wrapper.generate(
                prompt,
                top_p=0.9,
                max_new_tokens=self.max_new_tokens,
            )
            strategy_label = "HIGH"

        latency_ms = (time.perf_counter() - t0) * 1000
        token_cost  = len(self.wrapper.tokenizer.encode(text))

        return {
            "text":       text,
            "strategy":   strategy_label,
            "token_cost": token_cost,
            "latency_ms": latency_ms,
        }
