"""Gradio Web UI for Hugging Face Spaces deployment."""

import gradio as gr

from hebrew_readability.config import LEVEL_DESCRIPTIONS
from inference.engine import ReadabilityInferenceEngine

engine = ReadabilityInferenceEngine()


def analyze_hebrew_text(text: str):
    if not text or not text.strip():
        return "0.0 / 10", "None", "None", {}, "Please enter Hebrew text."

    res = engine.evaluate_text(text.strip(), print_report=False)
    score_str = f"{res['raw_readability_score']:.2f} / 10"
    core_str = ", ".join(res["core_iqr_matches"]) if res["core_iqr_matches"] else "None"
    ext_str = (
        ", ".join(res["extended_p1_p99_matches"]) if res["extended_p1_p99_matches"] else "Out of Bounds"
    )
    prob_dict = {
        f"{lvl} ({LEVEL_DESCRIPTIONS.get(lvl, lvl)})": prob / 100.0
        for lvl, prob in list(res["probabilities"].items())[:6]
    }
    return score_str, core_str, ext_str, prob_dict, res["natural_language_summary"]


with gr.Blocks(title="Hebrew Readability Classifier") as demo:
    gr.Markdown("# Continuous Hebrew Readability Assessment Engine")
    input_box = gr.Textbox(
        label="Input Hebrew Text",
        lines=5,
        rtl=True,
        placeholder="הזן טקסט בעברית להערכת רמת קריאות...",
    )
    submit_btn = gr.Button("Evaluate Readability", variant="primary")

    with gr.Row():
        score_out = gr.Textbox(label="Continuous Score (0-10)")
        core_out = gr.Textbox(label="Core IQR (Q1-Q3) Levels")
        ext_out = gr.Textbox(label="Extended (P1-P99) Levels")

    summary_out = gr.Textbox(label="Pedagogical Summary")
    probs_out = gr.Label(label="Top Pedagogical Level Probabilities")

    submit_btn.click(
        fn=analyze_hebrew_text,
        inputs=input_box,
        outputs=[score_out, core_out, ext_out, probs_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch()