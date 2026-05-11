"""Gradio demo for Self-Correcting LLM.

Runs entirely on pre-computed results (no GPU required).
Deploy to HuggingFace Spaces with CPU tier.

Usage:
    python demo/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# ── Load pre-computed results ──────────────────────────────────────────────
_DATA_PATH = Path(__file__).parent / "precomputed_results.json"
with open(_DATA_PATH) as f:
    _DATA = json.load(f)

_SAMPLES = _DATA["samples"]
_SUMMARY = _DATA["summary"]

_QUESTION_LIST = [s["question"] for s in _SAMPLES]

# ── Colour map for strategy levels ────────────────────────────────────────
_LEVEL_COLOR = {"LOW": "#2ecc71", "MID-A": "#f39c12", "MID-B": "#e67e22", "HIGH": "#e74c3c"}


def lookup_sample(question: str):
    for s in _SAMPLES:
        if s["question"] == question:
            return s
    return None


def render_gauge(score: float) -> plt.Figure:
    """Simple half-circle gauge for hallucination score."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])

    # Background arc
    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [0.85] * 200, color="#ecf0f1", linewidth=20, alpha=0.5)

    # Score arc
    score_theta = np.linspace(0, score * np.pi, 200)
    color = "#2ecc71" if score < 0.4 else ("#f39c12" if score < 0.7 else "#e74c3c")
    ax.plot(score_theta, [0.85] * 200, color=color, linewidth=20)
    ax.text(np.pi / 2, 0.3, f"{score:.2f}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(np.pi / 2, 0.0, "Hallucination Score", ha="center", va="bottom", fontsize=9)
    fig.patch.set_alpha(0)
    return fig


def render_summary_bar() -> plt.Figure:
    methods = list(_SUMMARY.keys())
    accs   = [_SUMMARY[m]["accuracy"]    or 0.0  for m in methods]
    costs  = [_SUMMARY[m]["token_cost"]  or 0.0  for m in methods]
    lats   = [_SUMMARY[m]["latency_ms"]  or 0.0  for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    short = [m.split("\n")[0] for m in methods]
    for ax, vals, title in zip(axes, [accs, costs, lats],
                               ["Factual Accuracy", "Avg Token Cost", "Avg Latency (ms)"]):
        ax.bar(short, vals, color=["#95a5a6", "#95a5a6", "#2ecc71"])
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(short, rotation=10, fontsize=8)
    fig.suptitle("Experiment Results Summary", fontweight="bold")
    fig.tight_layout()
    return fig


def on_select(question: str):
    s = lookup_sample(question)
    if s is None:
        return "No data", "—", "—", None, None

    level     = s["strategy_applied"]
    color     = _LEVEL_COLOR.get(level, "#95a5a6")
    badge_md  = f'<span style="background:{color};color:white;padding:3px 10px;border-radius:8px;font-weight:bold">{level}</span>'

    info_md = (
        f"**Hallucination Score:** `{s['hallucination_score']:.3f}`\n\n"
        f"**Strategy Applied:** {badge_md}\n\n"
        f"**Token Cost:** {s['token_cost']} tokens\n\n"
        f"**Latency:** {s['latency_ms']:.1f} ms"
    )

    gauge_fig   = render_gauge(s["hallucination_score"])
    summary_fig = render_summary_bar()

    return (
        s["original_answer"],
        s["corrected_answer"],
        info_md,
        gauge_fig,
        summary_fig,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Self-Correcting LLM Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Self-Correcting LLM\n"
        "### Hallucination Detection & Reliability-Controlled Re-generation\n"
        "Select a sample question to see how the pipeline detects hallucinations "
        "and applies the appropriate correction strategy."
    )

    with gr.Row():
        question_dd = gr.Dropdown(
            choices=_QUESTION_LIST, label="Sample Question", interactive=True, scale=3
        )

    with gr.Row():
        with gr.Column(scale=1):
            gauge_plot   = gr.Plot(label="Hallucination Score Gauge")
            info_display = gr.Markdown(label="Detection Info")
        with gr.Column(scale=2):
            orig_box = gr.Textbox(label="Original Answer (before correction)", lines=4)
            corr_box = gr.Textbox(label="Corrected Answer", lines=4)

    summary_plot = gr.Plot(label="Experiment Results Summary")

    question_dd.change(
        fn=on_select,
        inputs=question_dd,
        outputs=[orig_box, corr_box, info_display, gauge_plot, summary_plot],
    )

    gr.Markdown(
        "---\n"
        "*Demo uses pre-computed results from experiments. "
        "See [GitHub](https://github.com/) for full code.*"
    )


if __name__ == "__main__":
    demo.launch()
