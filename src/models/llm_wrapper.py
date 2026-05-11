"""LLM wrapper for loading LLaMA models and extracting per-layer hidden states.

Usage:
    wrapper = LLMWrapper(model_name=config.PRIMARY_MODEL)
    hs = wrapper.extract_hidden_states(questions, answers)
    # hs.shape: (n_samples, n_layers, hidden_dim)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

import config


class LLMWrapper:
    def __init__(
        self,
        model_name: str = config.PRIMARY_MODEL,
        load_in_4bit: bool = config.LOAD_IN_4BIT,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if load_in_4bit
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            output_hidden_states=True,
        )
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract_hidden_states(
        self,
        questions: list[str],
        answers: list[str],
        batch_size: int = config.BATCH_SIZE,
        layer_agg: str = "last_token",
    ) -> np.ndarray:
        """Return hidden states for each (question, answer) pair.

        Args:
            questions: list of question strings
            answers:   list of answer strings (same length)
            batch_size: processing batch size
            layer_agg: how to aggregate across sequence positions.
                       'last_token' uses the final token's representation.

        Returns:
            np.ndarray of shape (n_samples, n_layers, hidden_dim)
        """
        assert len(questions) == len(answers)
        all_hidden = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Extracting hidden states"):
            batch_q = questions[i : i + batch_size]
            batch_a = answers[i : i + batch_size]
            texts = [f"Q: {q}\nA: {a}" for q, a in zip(batch_q, batch_a)]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.model.device)

            outputs = self.model(**inputs, output_hidden_states=True)
            # outputs.hidden_states: tuple of (n_layers+1) tensors, each (B, T, H)
            # index 0 = embedding layer; indices 1..n_layers = transformer layers
            hidden_states = outputs.hidden_states[1:]  # skip embedding

            if layer_agg == "last_token":
                # Use the last non-padding token for each sample
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
                batch_repr = []
                for layer_hs in hidden_states:
                    # layer_hs: (B, T, H)
                    last_tok = layer_hs[torch.arange(len(batch_q)), seq_lens]  # (B, H)
                    batch_repr.append(last_tok.float().cpu().numpy())
                # batch_repr: list of n_layers arrays of shape (B, H)
                # stack to (B, n_layers, H)
                batch_arr = np.stack(batch_repr, axis=1)
            else:
                raise ValueError(f"Unknown layer_agg: {layer_agg}")

            all_hidden.append(batch_arr)

        return np.concatenate(all_hidden, axis=0)  # (N, n_layers, H)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        top_p: float = 0.9,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> tuple[str, list[torch.Tensor]]:
        """Generate text and return (generated_text, per-layer hidden states of last input token).

        Returns:
            text: generated string
            hidden_states: list of n_layers tensors, each shape (1, H)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Get hidden states of the prompt
            fwd = self.model(**inputs, output_hidden_states=True)
            prompt_hs = [hs[0, -1:].float().cpu() for hs in fwd.hidden_states[1:]]

            # Generate continuation
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, prompt_hs

    @torch.no_grad()
    def generate_with_dola(
        self,
        prompt: str,
        high_layer: int,
        low_layer: int,
        alpha: float = config.DOLA_ALPHA,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> str:
        """MID-B: DoLa-style generation contrasting high vs. low layer logits.

        Approximation: run a single forward pass, contrast logits from the
        projection of high_layer vs low_layer hidden states, then greedy-decode.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        lm_head = self.model.lm_head  # weight: (vocab_size, hidden_dim)

        generated_ids = inputs["input_ids"].clone()
        for _ in range(max_new_tokens):
            fwd = self.model(input_ids=generated_ids, output_hidden_states=True)
            # hidden_states[1:] → (n_layers,) tensors of shape (1, T, H)
            hs_high = fwd.hidden_states[high_layer + 1][:, -1, :]   # (1, H)
            hs_low  = fwd.hidden_states[low_layer  + 1][:, -1, :]   # (1, H)

            logits_high = lm_head(hs_high.to(lm_head.weight.dtype))  # (1, V)
            logits_low  = lm_head(hs_low.to(lm_head.weight.dtype))   # (1, V)

            # Contrast: amplify directions that high-layer agrees on
            contrasted = logits_high + alpha * (logits_high - logits_low)
            next_token = contrasted.argmax(dim=-1, keepdim=True)      # (1, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        new_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
