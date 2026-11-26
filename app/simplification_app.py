
import html
import json
import os
import sys
from pathlib import Path

import gradio as gr

# Make project modules available when running from Cloud Run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hebrew_readability.config import LEVEL_DESCRIPTIONS, RAW_LEVEL_TO_POPULATION
from inference.engine import ReadabilityInferenceEngine
from inference.agent import HebrewSimplificationAgent
from app.gemini_client import create_gemini_client


# Initialize the heavy components only when the first request arrives.
_engine = None
_agent = None


def get_agent():
    global _engine, _agent

    if _agent is None:
        gemini_client, model_name = create_gemini_client()

        prompt_path = PROJECT_ROOT / "prompts" / "feature_informed_prompt.json"
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            prompt_cfg = json.load(prompt_file)

        _engine = ReadabilityInferenceEngine()

        _agent = HebrewSimplificationAgent(
            engine=_engine,
            feature_prompt_cfg=prompt_cfg,
            gemini_client=gemini_client,
            model_name=model_name,
            log_dir=str(PROJECT_ROOT / "artifacts"),
        )

    return _agent


def make_iteration_html(trajectory, best_iteration):
    cards = []

    for turn in trajectory:
        number = turn.get("iteration", "?")
        score = turn.get("predicted_score", "—")
        lower = turn.get("target_q1", "—")
        upper = turn.get("target_q3", "—")
        distance = turn.get("dist_to_iqr", "—")
        selected = number == best_iteration

        status = (
            "נבחר כתוצאה הטובה ביותר"
            if selected
            else ("בתוך טווח היעד" if turn.get("in_core_iqr") else "עדיין מחוץ לטווח")
        )

        draft = html.escape(str(turn.get("draft_text", "")))
        feedback = html.escape(
            str(turn.get("tool_natural_language_feedback", ""))
        )
        features = html.escape(
            ", ".join(turn.get("top_offending_features") or [])
        )

        cards.append(
            f"""
            <section class="iteration-card">
                <h3>סבב {number} — {status}</h3>
                <p><b>ציון:</b> {score} |
                   <b>טווח יעד:</b> {lower}–{upper} |
                   <b>מרחק מהטווח:</b> {distance}</p>
                <h4>הטקסט שנוצר בסבב הזה</h4>
                <div class="hebrew-text">{draft}</div>
                <h4>אבחון והנחיות להמשך</h4>
                <div class="hebrew-text">{feedback}</div>
                <p><b>מאפיינים שדרשו תיקון:</b> {features or "לא צוינו"}</p>
            </section>
            """
        )

    return "\n".join(cards)


def simplify_text(source_text, target_raw_level):
    if not source_text or not source_text.strip():
        raise gr.Error("יש להזין טקסט בעברית.")

    if not target_raw_level:
        raise gr.Error("יש לבחור רמת יעד.")

    target_population = RAW_LEVEL_TO_POPULATION[target_raw_level]

    try:
        agent = get_agent()

        result = agent.run_agentic_simplification(
            source_text=source_text.strip(),
            target_population=target_population,
            target_raw_level=target_raw_level,
            max_iters=3,
        )

        summary = f"""
### תוצאות התהליך

- **ציון הטקסט המקורי:** {result["source_score"]}
- **ציון הטקסט המותאם:** {result["final_score"]}
- **טווח היעד:** {result["target_q1"]}–{result["target_q3"]}
- **מספר סבבים שבוצעו:** {result["total_iterations_run"]}
- **הסבב שנבחר כתוצאה הטובה ביותר:** {result["best_iteration_selected"]}
- **האם הושג טווח היעד:** {"כן" if result["converged_in_core_iqr"] else "לא"}
- **זמן התהליך:** {result["total_latency_sec"]} שניות
"""

        iterations = make_iteration_html(
            result["trajectory"],
            result["best_iteration_selected"],
        )

        return (
            result["final_simplified_text"],
            summary,
            iterations,
        )

    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(
            f"אירעה שגיאה בתהליך הפישוט: {error}"
        )


level_choices = [
    (
        f"{level} — {LEVEL_DESCRIPTIONS.get(level, level)}",
        level,
    )
    for level in RAW_LEVEL_TO_POPULATION
]

with gr.Blocks(
    title="מפשט טקסטים בעברית",
    theme=gr.themes.Soft(),
    css="""
    .hebrew-text {
        direction: rtl;
        text-align: right;
        white-space: pre-wrap;
        line-height: 1.8;
        padding: 12px;
        background: var(--background-fill-secondary);
        border-radius: 8px;
    }
    .iteration-card {
        direction: rtl;
        text-align: right;
        border: 1px solid var(--border-color-primary);
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    }
    """
) as demo:

    gr.Markdown("# מפשט טקסטים בעברית")
    gr.Markdown(
        "הזיני טקסט, בחרי רמת קריאות רצויה, "
        "וצפי בתהליך הפישוט האוטונומי ובתוצאה שנבחרה."
    )

    source_input = gr.Textbox(
        label="הטקסט המקורי",
        placeholder="הדביקי כאן את הטקסט שברצונך לפשט...",
        lines=10,
        rtl=True,
    )

    target_level = gr.Dropdown(
        choices=level_choices,
        value="5",
        label="רמת הקריאות הרצויה",
    )

    simplify_button = gr.Button(
        "פשטי את הטקסט",
        variant="primary",
    )

    gr.Markdown("## התוצאה הסופית")

      final_output = gr.Textbox(
        label="הטקסט המותאם",
        lines=10,
        rtl=True,
    )

    summary_output = gr.Markdown()

    gr.Markdown("## שלבי העבודה של הסוכן")

    iterations_output = gr.HTML()

    simplify_button.click(
        fn=simplify_text,
        inputs=[source_input, target_level],
        outputs=[
            final_output,
            summary_output,
            iterations_output,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "8080")),
        show_error=True,
    )
