"""
Phase 4 (Deliverable 6): OpenAI Experimentation Pipeline & Analysis Script.

Iterates over a test dataset CSV [paragraph_id, text, target_level], executes all
prompting strategies (cot_persona, few_shot_persona, feature_informed) and the
autonomous agentic loop (agentic), evaluates outputs via ReadabilityInferenceEngine,
streams incremental results to an 8-column CSV, and prints a comparative analysis report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Ensure local src/ is on sys.path when invoked from CLI
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hebrew_readability.config import (
    LEVEL_DESCRIPTIONS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    setup_pipeline_logger,
)
from inference.agent import HebrewSimplificationAgent
from inference.engine import ReadabilityInferenceEngine

OUTPUT_COLUMNS: List[str] = [
    "paragraph_id",
    "text",
    "target_level",
    "prompt",
    "original_level",
    "simplified_text",
    "simplified_score",
    "simplified_level",
]

DEFAULT_PROMPTS: List[str] = [
    "cot_persona",
    "few_shot_persona",
    "feature_informed",
    "agentic",
]

PROMPT_FILES: Dict[str, str] = {
    "cot_persona": "prompts/persona_cot_prompt.json",
    "few_shot_persona": "prompts/few_shot_prompt.json",
    "feature_informed": "prompts/feature_informed_prompt.json",
}

# Bidirectional alias normalization between SHAP/dataset level codes and engine calibration keys
RAW_LEVEL_ALIASES: Dict[str, List[str]] = {
    "1": ["1", "2"],
    "2": ["2"],
    "3": ["3"],
    "'2-4": ["'2-4", "2-4"],
    "2-4": ["'2-4", "2-4"],
    "4": ["4"],
    "5": ["5"],
    "6": ["6"],
    "'4-9": ["'4-9", "4-9"],
    "4-9": ["'4-9", "4-9"],
    "7": ["7"],
    "8": ["8"],
    "9": ["9"],
    "10": ["10"],
    "11": ["11"],
    "12": ["12"],
    "Bagrut": ["Bagrut", "Bagrut_Olim"],
    "Bagrut_Olim": ["Bagrut_Olim", "Bagrut"],
    "hearing_impaired": ["hearing_impaired", "Bagrut_hearing_impaired"],
    "Bagrut_hearing_impaired": ["Bagrut_hearing_impaired", "hearing_impaired"],
    "intellectual_disability": ["intellectual_disability"],
    "Hadshon": ["Hadshon", "Olim_news"],
    "Olim_news": ["Olim_news", "Hadshon"],
    "Olim_ragil": ["Olim_ragil", "מבחני אולפן - רגיל"],
    "מבחני אולפן - רגיל": ["מבחני אולפן - רגיל", "Olim_ragil"],
    "Olim_mitkadmim": ["Olim_mitkadmim", "מבחני אולפן - מתקדמים"],
    "מבחני אולפן - מתקדמים": ["מבחני אולפן - מתקדמים", "Olim_mitkadmim"],
}

# Maps every raw level code (or population string) to its canonical 7-population profile key
RAW_LEVEL_TO_POPULATION: Dict[str, str] = {
    "1": "Grade 1-3 - Elementary Beginner",
    "2": "Grade 1-3 - Elementary Beginner",
    "3": "Grade 1-3 - Elementary Beginner",
    "'2-4": "Grade 1-3 - Elementary Beginner",
    "2-4": "Grade 1-3 - Elementary Beginner",
    "4": "Grade 4-6 - Elementary Intermediate",
    "5": "Grade 4-6 - Elementary Intermediate",
    "6": "Grade 4-6 - Elementary Intermediate",
    "'4-9": "Grade 7-9 - Middle School",
    "4-9": "Grade 7-9 - Middle School",
    "7": "Grade 7-9 - Middle School",
    "8": "Grade 7-9 - Middle School",
    "9": "Grade 7-9 - Middle School",
    "10": "Grade 10-12 - High School",
    "11": "Grade 10-12 - High School",
    "12": "Grade 10-12 - High School",
    "Bagrut": "Grade 10-12 - High School",
    "hearing_impaired": "Grade 10-12 - High School",
    "Bagrut_hearing_impaired": "Grade 10-12 - High School",
    "Bagrut_Olim": "Second-language learners",
    "Hadshon": "Second-language learners",
    "Olim_news": "Second-language learners",
    "Olim_ragil": "Second-language learners",
    "מבחני אולפן - רגיל": "Second-language learners",
    "Olim_mitkadmim": "Second-language learners",
    "מבחני אולפן - מתקדמים": "Second-language learners",
    "intellectual_disability": "Intellectual Disability Accessible",
}


# ---------------------------------------------------------------------------
# 1. LEVEL & FORMATTING HELPERS
# ---------------------------------------------------------------------------
def resolve_population_and_description(target_level: str) -> Tuple[str, str]:
    """Maps a raw target_level code to its 7-population profile name and exact level description."""
    clean_lvl = str(target_level).strip()
    if clean_lvl in POPULATION_SPECS:
        return clean_lvl, clean_lvl

    pop_name = (
        RAW_LEVEL_TO_POPULATION.get(clean_lvl)
        or RAW_LEVEL_TO_POPULATION.get(clean_lvl.lstrip("'"))
        or "Grade 4-6 - Elementary Intermediate"
    )
    lvl_desc = (
        LEVEL_DESCRIPTIONS.get(clean_lvl)
        or LEVEL_DESCRIPTIONS.get(f"'{clean_lvl}")
        or LEVEL_DESCRIPTIONS.get(clean_lvl.lstrip("'"))
        or f"Level {clean_lvl} ({pop_name})"
    )
    return pop_name, lvl_desc


def get_raw_level_iqr(engine: Any, target_level: str) -> Tuple[float, float, float]:
    """Returns (Q1, Q3, mu) for a specific raw level from engine.calibration_stats."""
    clean_lvl = str(target_level).strip()
    candidates = RAW_LEVEL_ALIASES.get(
        clean_lvl, [clean_lvl, f"'{clean_lvl}", clean_lvl.lstrip("'")]
    )
    cal_stats = getattr(engine, "calibration_stats", {})
    for cand in candidates:
        if cand in cal_stats:
            s = cal_stats[cand]
            return round(float(s["Q1"]), 4), round(float(s["Q3"]), 4), round(float(s["mu"]), 4)

    pop_name, _ = resolve_population_and_description(clean_lvl)
    if hasattr(engine, "get_population_iqr_bounds"):
        q1, q3 = engine.get_population_iqr_bounds(pop_name)
        return q1, q3, round((q1 + q3) / 2.0, 4)

    anchor = float(POPULATION_SPECS[pop_name]["anchor_score"])
    return round(anchor - 0.45, 4), round(anchor + 0.45, 4), anchor


def format_iqr_levels_plain(eval_res: Dict[str, Any]) -> str:
    """Formats original_level for the agentic pipeline: comma-separated raw levels in Core IQR."""
    core_matches = eval_res.get("core_iqr_matches") or []
    if core_matches:
        return ", ".join(str(m) for m in core_matches)

    probs = eval_res.get("probabilities") or {}
    if probs:
        top_lvl = next(iter(probs.keys()))
        return f"NONE_IN_IQR (Closest: {top_lvl})"
    return "NONE_IN_IQR"


def format_iqr_levels_with_probs(eval_res: Dict[str, Any]) -> str:
    """
    Formats simplified_level: comma-separated list of raw levels whose Core IQR (Q1-Q3)
    contains simplified_score, annotated with their probabilities.
    """
    core_matches = eval_res.get("core_iqr_matches") or []
    probs = eval_res.get("probabilities") or {}

    if not core_matches:
        if probs:
            top_lvl, top_prob = next(iter(probs.items()))
            return f"NONE_IN_IQR (Closest: {top_lvl} [{top_prob}%])"
        return "NONE_IN_IQR"

    # Sort matched Core IQR levels by descending probability
    ordered_matches = sorted(
        core_matches,
        key=lambda lvl: float(probs.get(lvl, 0.0)),
        reverse=True,
    )
    formatted = [
        f"{lvl} ({probs[lvl]}%)" if lvl in probs else str(lvl)
        for lvl in ordered_matches
    ]
    return ", ".join(formatted)


# ---------------------------------------------------------------------------
# 2. STATIC PROMPT BUILDER & LLM EXECUTION
# ---------------------------------------------------------------------------
def build_static_prompt_messages(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    source_text: str,
    target_level: str,
) -> List[Dict[str, str]]:
    """
    Constructs the system and user messages for cot_persona, few_shot_persona,
    or feature_informed, instructing the LLM to estimate up to 3 raw source levels
    and adapt the text to the requested target_level.
    """
    pop_name, lvl_desc = resolve_population_and_description(target_level)
    prompt_json_str = json.dumps(prompt_cfg, ensure_ascii=False, indent=2).replace(
        "{{TARGET_POPULATION}}", f"{pop_name} (Specifically Raw Level '{target_level}': {lvl_desc})"
    )

    valid_raw_levels_guide = {
        k: v for k, v in LEVEL_DESCRIPTIONS.items() if not k.startswith("'") or k in ("'2-4", "'4-9")
    }

    system_content = (
        f"You are an expert Hebrew language educator and computational linguist.\n"
        f"Follow this prompt specification:\n{prompt_json_str}\n\n"
        f"VALID RAW LEVEL CODES FOR SOURCE LEVEL ESTIMATION:\n"
        f"{json.dumps(valid_raw_levels_guide, ensure_ascii=False, indent=2)}\n\n"
        f"CRITICAL OUTPUT REQUIREMENTS:\n"
        f"1. Respond STRICTLY in valid JSON.\n"
        f"2. Include a top-level key \"estimated_original_raw_levels\" containing a list of up to 3 raw level codes "
        f"(e.g., [\"8\", \"9\", \"10\"]) from VALID RAW LEVEL CODES that best match the input text's current complexity.\n"
        f"3. Adapt the text specifically for Target Raw Level '{target_level}' ({lvl_desc}, Population: '{pop_name}') "
        f"and place the simplified Hebrew text in \"simplified_text\" (as well as inside the prompt's schema)."
    )

    user_content = json.dumps(
        {
            "requested_target_raw_level": target_level,
            "requested_target_level_description": lvl_desc,
            "requested_target_population": pop_name,
            "input_hebrew_text": source_text,
        },
        ensure_ascii=False,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_llm_static_output(
    llm_json: Dict[str, Any], fallback_text: str
) -> Tuple[str, str]:
    """
    Extracts (original_level_str, simplified_text) from any of the 3 static prompt schemas.
    Ensures original_level_str contains up to 3 estimated levels.
    """
    # 1. Extract up to 3 estimated original levels
    raw_est = llm_json.get("estimated_original_raw_levels")
    levels_list: List[str] = []

    if isinstance(raw_est, list):
        levels_list = [str(x).strip() for x in raw_est if str(x).strip()]
    elif isinstance(raw_est, str) and raw_est.strip():
        levels_list = [x.strip() for x in raw_est.split(",") if x.strip()]

    if not levels_list:
        # Fallback to schema-specific classification blocks
        if isinstance(llm_json.get("classification"), dict):
            prim = llm_json["classification"].get("primary_level")
            if prim:
                levels_list.append(str(prim).strip())
        elif isinstance(llm_json.get("step_1_source_classification"), dict):
            s1 = llm_json["step_1_source_classification"]
            if s1.get("primary_population"):
                levels_list.append(str(s1["primary_population"]).strip())
            for sec in s1.get("secondary_populations") or []:
                if str(sec).strip():
                    levels_list.append(str(sec).strip())

    original_level_str = ", ".join(levels_list[:3]) if levels_list else "Unknown"

    # 2. Extract simplified_text across all 3 prompt schemas
    candidates = [
        llm_json.get("simplified_text"),
        llm_json.get("step_3_simplification", {}).get("simplified_text")
        if isinstance(llm_json.get("step_3_simplification"), dict)
        else None,
        llm_json.get("step_2_adaptation", {}).get("simplified_text")
        if isinstance(llm_json.get("step_2_adaptation"), dict)
        else None,
        llm_json.get("adapted_text"),
    ]
    simplified_text = fallback_text
    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            simplified_text = cand.strip()
            break

    return original_level_str, simplified_text


def call_openai_with_retry(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Calls OpenAI Chat Completions with JSON mode and exponential backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_sec = 2.0 ** attempt
            time.sleep(wait_sec)
    return {}


# ---------------------------------------------------------------------------
# 3. CORE EXPERIMENTATION RUNNER
# ---------------------------------------------------------------------------
def run_experimentation_pipeline(
    input_csv: Path,
    output_csv: Path,
    engine: Any,
    openai_client: Optional[Any],
    model_name: str = "gpt-4o-mini",
    prompts_to_run: List[str] = DEFAULT_PROMPTS,
    max_iters: int = 3,
    mock_mode: bool = False,
) -> pd.DataFrame:
    """
    Iterates over input_csv [paragraph_id, text, target_level], runs all selected prompts
    and the agentic workflow, evaluates with ReadabilityInferenceEngine, and streams
    each completed row to output_csv.
    """
    logger = setup_pipeline_logger(
        name="ExperimentationPipeline",
        artifact_dir="artifacts",
        log_filename="experimentation_run.log",
    )

    df_in = pd.read_csv(input_csv, dtype=str)
    required_in_cols = {"paragraph_id", "text", "target_level"}
    missing_cols = required_in_cols - set(df_in.columns)
    if missing_cols:
        raise ValueError(
            f"Input CSV {input_csv} is missing required columns: {missing_cols}. "
            f"Found: {list(df_in.columns)}"
        )

    # Load static prompt JSONs once
    loaded_prompts: Dict[str, Dict[str, Any]] = {}
    for p_name, rel_path in PROMPT_FILES.items():
        full_p = PROJECT_ROOT / rel_path
        if full_p.exists():
            loaded_prompts[p_name] = json.loads(full_p.read_text(encoding="utf-8"))
        else:
            loaded_prompts[p_name] = {"prompt_id": p_name, "role": "Hebrew readability expert"}

    # Initialize singleton agent once
    agent = HebrewSimplificationAgent(
        engine=engine,
        feature_prompt_cfg=loaded_prompts["feature_informed"],
        openai_client=openai_client,
        model_name=model_name,
        log_dir="artifacts",
    )

    # Check existing output_csv for idempotent resume
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_pairs: Set[Tuple[str, str]] = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        df_existing = pd.read_csv(output_csv, dtype=str)
        for _, r in df_existing.iterrows():
            completed_pairs.add((str(r["paragraph_id"]), str(r["prompt"])))
        logger.info(
            "Resuming from existing output CSV (%d completed (paragraph_id, prompt) rows found).",
            len(completed_pairs),
        )
    else:
        with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()

    total_rows = len(df_in)
    for idx, row in df_in.iterrows():
        pid = str(row["paragraph_id"]).strip()
        source_text = str(row["text"]).strip()
        target_level = str(row["target_level"]).strip()
        pop_name, lvl_desc = resolve_population_and_description(target_level)

        # Evaluate source_text ONCE per paragraph and cache for the agentic baseline
        source_eval: Optional[Dict[str, Any]] = None

        for prompt_name in prompts_to_run:
            if (pid, prompt_name) in completed_pairs:
                logger.info(
                    "[%d/%d] Skipping already completed (%s, %s)",
                    idx + 1,
                    total_rows,
                    pid,
                    prompt_name,
                )
                continue

            t0 = time.time()
            logger.info(
                "[%d/%d] Running paragraph_id=%s | target_level=%s (%s) | prompt=%s",
                idx + 1,
                total_rows,
                pid,
                target_level,
                pop_name,
                prompt_name,
            )

            if prompt_name == "agentic":
                if source_eval is None:
                    source_eval = engine.evaluate_text(source_text, print_report=False)

                original_level_val = format_iqr_levels_plain(source_eval)

                mock_turns = None
                if mock_mode:
                    mock_turns = [
                        {
                            "step_3_simplification": {
                                "simplified_text": "טקסט מופשט ראשוני לבדיקה."
                            }
                        },
                        {
                            "step_3_simplification": {
                                "simplified_text": "טֶקְסְט פָּשׁוּט וּמְנֻקָּד לִבְדִיקָה."
                                if target_level in ("1", "2", "3", "'2-4", "2-4")
                                else "טקסט פשוט, קצר וברור לבדיקה."
                            }
                        },
                    ]

                agent_res = agent.run_agentic_simplification(
                    source_text=source_text,
                    target_population=pop_name,
                    paragraph_id=f"{pid}_agentic",
                    max_iters=max_iters,
                    mock_responses=mock_turns,
                )
                simplified_text = agent_res["final_simplified_text"]
                eval_simplified = engine.evaluate_text(simplified_text, print_report=False)
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified)

            else:
                # Single-turn static prompt execution (cot_persona, few_shot_persona, feature_informed)
                messages = build_static_prompt_messages(
                    prompt_name=prompt_name,
                    prompt_cfg=loaded_prompts[prompt_name],
                    source_text=source_text,
                    target_level=target_level,
                )
                if mock_mode:
                    llm_json = {
                        "estimated_original_raw_levels": ["8", "9", "10"],
                        "simplified_text": (
                            "טֶקְסְט פָּשׁוּט עִם נִיקּוּד."
                            if target_level in ("1", "2", "3", "'2-4", "2-4")
                            else "טקסט מופשט וברור המותאם לרמת היעד."
                        ),
                    }
                else:
                    llm_json = call_openai_with_retry(
                        client=openai_client,
                        model_name=model_name,
                        messages=messages,
                    )

                original_level_val, simplified_text = parse_llm_static_output(
                    llm_json=llm_json, fallback_text=source_text
                )
                eval_simplified = engine.evaluate_text(simplified_text, print_report=False)
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified)

            out_row = {
                "paragraph_id": pid,
                "text": source_text,
                "target_level": target_level,
                "prompt": prompt_name,
                "original_level": original_level_val,
                "simplified_text": simplified_text,
                "simplified_score": simplified_score,
                "simplified_level": simplified_level_val,
            }

            with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
                writer.writerow(out_row)
                f.flush()

            completed_pairs.add((pid, prompt_name))
            logger.info(
                "Completed (%s, %s) in %.2fs -> Score=%.4f | Simplified_Level=%s",
                pid,
                prompt_name,
                time.time() - t0,
                simplified_score,
                simplified_level_val,
            )

    return pd.read_csv(output_csv)


# ---------------------------------------------------------------------------
# 4. COMPARATIVE ANALYSIS & SUMMARY REPORT
# ---------------------------------------------------------------------------
def analyze_experiment_results(output_csv: Path) -> pd.DataFrame:
    """
    Computes summary metrics across prompts:
    - Exact Raw Target Level IQR Hit Rate (%)
    - Broad Pedagogical Category Hit Rate (%)
    - Mean & Std of Continuous Simplified Readability Score
    """
    df = pd.read_csv(output_csv)
    if df.empty:
        return pd.DataFrame()

    def check_exact_raw_hit(row: pd.Series) -> bool:
        target = str(row["target_level"]).strip()
        aliases = set(RAW_LEVEL_ALIASES.get(target, [target, f"'{target}", target.lstrip("'")]))
        simp_str = str(row["simplified_level"])
        if simp_str.startswith("NONE_IN_IQR"):
            return False
        matched_raw = {
            part.split("(")[0].strip() for part in simp_str.split(",") if part.strip()
        }
        return len(aliases & matched_raw) > 0

    def check_category_hit(row: pd.Series) -> bool:
        target = str(row["target_level"]).strip()
        target_cats = set(
            LEVEL_TO_CATEGORIES.get(target)
            or LEVEL_TO_CATEGORIES.get(target.lstrip("'"))
            or []
        )
        simp_str = str(row["simplified_level"])
        if simp_str.startswith("NONE_IN_IQR"):
            return False
        matched_raw = [
            part.split("(")[0].strip() for part in simp_str.split(",") if part.strip()
        ]
        matched_cats: Set[str] = set()
        for m in matched_raw:
            matched_cats.update(
                LEVEL_TO_CATEGORIES.get(m) or LEVEL_TO_CATEGORIES.get(m.lstrip("'")) or []
            )
        return len(target_cats & matched_cats) > 0

    df["exact_raw_iqr_hit"] = df.apply(check_exact_raw_hit, axis=1)
    df["broad_category_hit"] = df.apply(check_category_hit, axis=1)
    df["simplified_score"] = pd.to_numeric(df["simplified_score"], errors="coerce")

    summary = (
        df.groupby("prompt", as_index=False)
        .agg(
            n_samples=("paragraph_id", "count"),
            exact_raw_iqr_hit_pct=("exact_raw_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            broad_category_hit_pct=("broad_category_hit", lambda s: round(s.mean() * 100.0, 2)),
            mean_simplified_score=("simplified_score", lambda s: round(s.mean(), 3)),
            std_simplified_score=("simplified_score", lambda s: round(s.std(), 3)),
        )
        .sort_values("broad_category_hit_pct", ascending=False)
    )

    print("\n" + "=" * 80)
    print("DELIVERABLE 6: EXPERIMENTAL COMPARISON SUMMARY BY PROMPT MECHANISM")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")
    return summary


# ---------------------------------------------------------------------------
# 5. OFFLINE SMOKE TEST (--test) & CLI ENTRYPOINT
# ---------------------------------------------------------------------------
def _run_smoke_test() -> None:
    """Verifies the 8-column CSV contract across all 4 methods offline without GPU/API."""
    import tempfile

    class DummyCalibrationEngine:
        def __init__(self) -> None:
            self.calibration_stats = {
                "2": {"mu": 3.10, "sigma": 0.45, "Q1": 2.75, "Q3": 3.45},
                "3": {"mu": 3.55, "sigma": 0.45, "Q1": 3.20, "Q3": 3.90},
                "4": {"mu": 5.10, "sigma": 0.40, "Q1": 4.80, "Q3": 5.40},
                "8": {"mu": 5.75, "sigma": 0.35, "Q1": 5.50, "Q3": 6.00},
            }

        def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
            if "1-3" in target_population:
                return (2.75, 3.90)
            return (4.80, 5.40)

        def evaluate_text(self, text: str, print_report: bool = False) -> Dict[str, Any]:
            has_nikkud = any("\u0591" <= c <= "\u05c7" for c in text)
            if "מקור מורכב" in text:
                return {
                    "raw_readability_score": 5.82,
                    "core_iqr_matches": ["8"],
                    "core_iqr_categories": ["Middle School"],
                    "extended_p1_p99_matches": ["4", "8"],
                    "probabilities": {"8": 78.5, "4": 21.5},
                    "transformed_features": {"1_nikkud_ratio_log_inv": 1.0},
                }
            if has_nikkud:
                return {
                    "raw_readability_score": 3.25,
                    "core_iqr_matches": ["2", "3"],
                    "core_iqr_categories": ["Elementary - Beginner"],
                    "extended_p1_p99_matches": ["2", "3", "4"],
                    "probabilities": {"2": 54.2, "3": 45.8},
                    "transformed_features": {"1_nikkud_ratio_log_inv": 0.05},
                }
            return {
                "raw_readability_score": 5.05,
                "core_iqr_matches": ["4"],
                "core_iqr_categories": ["Elementary - Intermediate"],
                "extended_p1_p99_matches": ["3", "4", "8"],
                "probabilities": {"4": 81.0, "3": 10.0, "8": 9.0},
                "transformed_features": {"1_nikkud_ratio_log_inv": 0.95},
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_csv = tmp_dir / "test_in.csv"
        out_csv = tmp_dir / "test_out.csv"

        pd.DataFrame(
            [
                {"paragraph_id": "p_01", "text": "טקסט מקור מורכב מאוד.", "target_level": "2"},
                {"paragraph_id": "p_02", "text": "טקסט מקור מורכב נוסף.", "target_level": "4"},
            ]
        ).to_csv(in_csv, index=False)

        df_res = run_experimentation_pipeline(
            input_csv=in_csv,
            output_csv=out_csv,
            engine=DummyCalibrationEngine(),
            openai_client=None,
            mock_mode=True,
        )

        assert list(df_res.columns) == OUTPUT_COLUMNS, f"Invalid columns: {list(df_res.columns)}"
        assert len(df_res) == 8, f"Expected 8 rows (2 texts x 4 prompts), got {len(df_res)}"
        analyze_experiment_results(out_csv)
        print("[✓] Phase 4 Experimentation Pipeline smoke test passed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI Simplification Experiments & Analysis"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/wikipedia_simplification_test_100.csv"),
        help="Path to input CSV with columns: [paragraph_id, text, target_level]",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/experimentation_results.csv"),
        help="Path to output 8-column CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum self-correction iterations for the agentic workflow.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        choices=DEFAULT_PROMPTS,
        help="Subset of prompts to run (default: all 4).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run fast offline smoke test and exit.",
    )
    args, _ = parser.parse_known_args()

    if args.test:
        _run_smoke_test()
        return

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_LAB")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set os.environ['OPENAI_LAB'] before running."
        )

    openai_client = OpenAI(api_key=api_key)
    engine = ReadabilityInferenceEngine()

    run_experimentation_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        engine=engine,
        openai_client=openai_client,
        model_name=args.model,
        prompts_to_run=args.prompts,
        max_iters=args.max_iters,
        mock_mode=False,
    )
    analyze_experiment_results(args.output_csv)


if __name__ == "__main__":
    main()



# --- DELIVERABLE 6 PATCH: Calibrated Length Constraints & Enriched Sidecar CSV ---
from inference.agent import RAW_LEVEL_LENGTH_BUDGETS

ENRICHED_COLUMNS: List[str] = OUTPUT_COLUMNS + [
    "original_score",
    "original_tool_iqr_levels",
    "target_q1",
    "target_q3",
    "target_mu",
    "iterations_used",
]

def build_static_prompt_messages(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    source_text: str,
    target_level: str,
    target_q1: float = 4.5,
    target_q3: float = 5.5,
    target_mu: float = 5.0,
) -> List[Dict[str, str]]:
    pop_name, lvl_desc = resolve_population_and_description(target_level)
    budget_rule = RAW_LEVEL_LENGTH_BUDGETS.get(
        str(target_level).strip(),
        "Match sentence length and paragraph volume strictly to the target grade.",
    )
    prompt_json_str = json.dumps(prompt_cfg, ensure_ascii=False, indent=2).replace(
        "{{TARGET_POPULATION}}",
        f"{pop_name} (Specifically Raw Level '{target_level}': {lvl_desc})",
    )
    valid_raw_levels_guide = {
        k: v for k, v in LEVEL_DESCRIPTIONS.items() if not k.startswith("'") or k in ("'2-4", "'4-9")
    }
    nl = chr(10)
    system_content = (
        f"You are an expert Hebrew language educator and computational linguist.{nl}"
        f"Follow this prompt specification:{nl}{prompt_json_str}{nl}{nl}"
        f"VALID RAW LEVEL CODES FOR SOURCE LEVEL ESTIMATION:{nl}"
        f"{json.dumps(valid_raw_levels_guide, ensure_ascii=False, indent=2)}{nl}{nl}"
        f"CRITICAL OUTPUT & CALIBRATION REQUIREMENTS:{nl}"
        f"1. Respond STRICTLY in valid JSON.{nl}"
        f"2. Include a top-level key 'estimated_original_raw_levels' with up to 3 raw level codes from VALID RAW LEVEL CODES.{nl}"
        f"3. Adapt the text specifically for Target Raw Level '{target_level}' ({lvl_desc}, Calibrated Core IQR: [{target_q1:.2f}, {target_q3:.2f}], Anchor: {target_mu:.2f}).{nl}"
        f"4. STRICT LENGTH & SENTENCE BUDGET FOR LEVEL '{target_level}': {budget_rule}{nl}"
        f"5. Place the adapted Hebrew text in 'simplified_text'."
    )
    user_content = json.dumps(
        {
            "requested_target_raw_level": target_level,
            "requested_target_level_description": lvl_desc,
            "calibrated_target_iqr": [target_q1, target_q3],
            "length_and_sentence_budget": budget_rule,
            "input_hebrew_text": source_text,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

def run_experimentation_pipeline(
    input_csv: Path,
    output_csv: Path,
    engine: Any,
    openai_client: Optional[Any],
    model_name: str = "gpt-4o-mini",
    prompts_to_run: List[str] = DEFAULT_PROMPTS,
    max_iters: int = 3,
    mock_mode: bool = False,
) -> pd.DataFrame:
    logger = setup_pipeline_logger(
        name="ExperimentationPipeline",
        artifact_dir="artifacts",
        log_filename="experimentation_run.log",
    )
    df_in = pd.read_csv(input_csv, dtype=str)
    enriched_csv = output_csv.with_name(f"{output_csv.stem}_enriched.csv")

    loaded_prompts = {}
    for p_name, rel_path in PROMPT_FILES.items():
        full_p = PROJECT_ROOT / rel_path
        loaded_prompts[p_name] = (
            json.loads(full_p.read_text(encoding="utf-8"))
            if full_p.exists()
            else {"prompt_id": p_name, "role": "Hebrew readability expert"}
        )

    agent = HebrewSimplificationAgent(
        engine=engine,
        feature_prompt_cfg=loaded_prompts["feature_informed"],
        openai_client=openai_client,
        model_name=model_name,
        log_dir="artifacts",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_pairs: Set[Tuple[str, str]] = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        df_existing = pd.read_csv(output_csv, dtype=str)
        for _, r in df_existing.iterrows():
            completed_pairs.add((str(r["paragraph_id"]), str(r["prompt"])))
    else:
        with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL).writeheader()

    if not enriched_csv.exists() or enriched_csv.stat().st_size == 0:
        with open(enriched_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=ENRICHED_COLUMNS, quoting=csv.QUOTE_ALL).writeheader()

    total_rows = len(df_in)
    for idx, row in df_in.iterrows():
        pid = str(row["paragraph_id"]).strip()
        source_text = str(row["text"]).strip()
        target_level = str(row["target_level"]).strip()
        pop_name, _ = resolve_population_and_description(target_level)
        t_q1, t_q3, t_mu = get_raw_level_iqr(engine, target_level)

        # Evaluate source text ONCE per paragraph and reuse across all 4 prompts
        source_eval = engine.evaluate_text(source_text, print_report=False)
        orig_score = round(float(source_eval["raw_readability_score"]), 4)
        orig_tool_iqr = format_iqr_levels_plain(source_eval)

        for prompt_name in prompts_to_run:
            if (pid, prompt_name) in completed_pairs:
                continue

            if prompt_name == "agentic":
                original_level_val = orig_tool_iqr
                mock_turns = (
                    [
                        {"step_3_simplification": {"simplified_text": "טקסט מופשט ראשוני לבדיקה."}},
                        {"step_3_simplification": {"simplified_text": "טֶקְסְט פָּשׁוּט וּמְנֻקָּד לִבְדִיקָה."}},
                    ]
                    if mock_mode
                    else None
                )
                agent_res = agent.run_agentic_simplification(
                    source_text=source_text,
                    target_population=pop_name,
                    paragraph_id=f"{pid}_agentic",
                    max_iters=max_iters,
                    mock_responses=mock_turns,
                    target_raw_level=target_level,
                    cached_source_eval=source_eval,
                )
                simplified_text = agent_res["final_simplified_text"]
                eval_simplified = agent_res.get("final_eval_payload") or engine.evaluate_text(
                    simplified_text, print_report=False
                )
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified)
                iters_used = int(agent_res["total_iterations_run"])
            else:
                messages = build_static_prompt_messages(
                    prompt_name=prompt_name,
                    prompt_cfg=loaded_prompts[prompt_name],
                    source_text=source_text,
                    target_level=target_level,
                    target_q1=t_q1,
                    target_q3=t_q3,
                    target_mu=t_mu,
                )
                llm_json = (
                    {
                        "estimated_original_raw_levels": ["8", "9", "10"],
                        "simplified_text": "טקסט מופשט וברור המותאם לרמת היעד.",
                    }
                    if mock_mode
                    else call_openai_with_retry(client=openai_client, model_name=model_name, messages=messages)
                )
                original_level_val, simplified_text = parse_llm_static_output(
                    llm_json=llm_json, fallback_text=source_text
                )
                eval_simplified = engine.evaluate_text(simplified_text, print_report=False)
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified)
                iters_used = 1

            base_row = {
                "paragraph_id": pid,
                "text": source_text,
                "target_level": target_level,
                "prompt": prompt_name,
                "original_level": original_level_val,
                "simplified_text": simplified_text,
                "simplified_score": simplified_score,
                "simplified_level": simplified_level_val,
            }
            with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL).writerow(base_row)
                f.flush()

            enriched_row = {
                **base_row,
                "original_score": orig_score,
                "original_tool_iqr_levels": orig_tool_iqr,
                "target_q1": t_q1,
                "target_q3": t_q3,
                "target_mu": t_mu,
                "iterations_used": iters_used,
            }
            with open(enriched_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=ENRICHED_COLUMNS, quoting=csv.QUOTE_ALL).writerow(enriched_row)
                f.flush()

            completed_pairs.add((pid, prompt_name))

    return pd.read_csv(output_csv)
