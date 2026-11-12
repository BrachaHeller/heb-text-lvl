"""Ablation-pure batch experimentation pipeline across cot_persona, few_shot_persona, feature_informed, and agentic using gpt-4o."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "pipelines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hebrew_readability.config import (
    DEFAULT_LLM_MODEL,
    K12_GRADE_TRACK_LEVELS,
    LEVEL_DESCRIPTIONS,
    LEVEL_SHAP_WEIGHTS,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    RAW_LEVEL_MEDIANS,
    RAW_LEVEL_TO_POPULATION,
    get_sub_paragraph_split_instruction,
    get_vocalization_policy,
    setup_pipeline_logger,
)
from inference.agent import HebrewSimplificationAgent
from pipelines.analyze_experimentation import analyze_experimentation_results

DEFAULT_PROMPTS: List[str] = [
    "cot_persona",
    "few_shot_persona",
    "feature_informed",
    "agentic",
]

REQUIRED_DELIVERABLE_COLUMNS: List[str] = [
    "paragraph_id",
    "text",
    "target_level",
    "prompt",
    "original_level",
    "simplified_text",
    "simplified_score",
    "simplified_level",
]

ENRICHED_COLUMNS: List[str] = REQUIRED_DELIVERABLE_COLUMNS + [
    "original_score",
    "original_tool_iqr_levels",
    "target_q1",
    "target_q3",
    "target_mu",
    "iterations_used",
    "top_shap_offenders",
    "word_retention_ratio",
    "k12_track_probs_json",
    "special_track_probs_json",
    "vocalization_mode",
    "stagnation_triggered",
]


def resolve_population_and_description(target_level: str) -> Tuple[str, str]:
    clean = str(target_level).strip()
    pop = RAW_LEVEL_TO_POPULATION.get(
        clean,
        RAW_LEVEL_TO_POPULATION.get(clean.lstrip("'"), "Grade 4-6 - Elementary Intermediate"),
    )
    desc = LEVEL_DESCRIPTIONS.get(
        clean, LEVEL_DESCRIPTIONS.get(clean.lstrip("'"), pop)
    )
    return pop, desc


def format_iqr_levels_plain(eval_res: Dict[str, Any]) -> str:
    matches = [str(m).strip() for m in (eval_res.get("core_iqr_matches") or [])]
    if eval_res.get("is_above_calibrated_range") and "academic/technical" not in matches:
        matches.append("academic/technical")
    if matches:
        return ", ".join(matches)
    probs = eval_res.get("probabilities") or {}
    if probs:
        top_lvl = next(iter(probs.keys()))
        return f"NONE_IN_IQR (Closest: {top_lvl})"
    return "NONE_IN_IQR"


def format_simplified_level_with_track_probs(
    eval_res: Dict[str, Any], target_level: str
) -> str:
    """Formats simplified_level with track-appropriate probabilities."""
    matches = [str(m).strip() for m in (eval_res.get("core_iqr_matches") or [])]
    clean_t = str(target_level).strip()
    is_k12 = clean_t in K12_GRADE_TRACK_LEVELS or clean_t.lstrip("'") in K12_GRADE_TRACK_LEVELS
    track_probs = (
        eval_res.get("k12_track_probabilities")
        if is_k12
        else eval_res.get("special_track_probabilities")
    ) or eval_res.get("core_iqr_probabilities") or eval_res.get("probabilities") or {}

    if matches:
        parts = []
        for m in matches:
            p = track_probs.get(m, (eval_res.get("core_iqr_probabilities") or {}).get(m))
            parts.append(f"{m} ({p:.1f}%)" if isinstance(p, (int, float)) else m)
        return ", ".join(parts)

    if track_probs:
        top_lvl, top_p = next(iter(track_probs.items()))
        return f"NONE_IN_IQR (Closest: {top_lvl} [{top_p:.1f}%])"
    return "NONE_IN_IQR"


def build_static_prompt_messages(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    source_text: str,
    target_level: str,
) -> List[Dict[str, str]]:
    """
    Strictly ablation-pure message builder for static baselines (cot_persona,
    few_shot_persona, feature_informed). Zero Evaluator bounds, zero quantitative
    SHAP/median tables, and zero dicta-onnx vocalization_plan hooks are exposed.
    """
    clean_lvl = str(target_level).strip()
    pop_name, lvl_desc = resolve_population_and_description(clean_lvl)
    prompt_json_str = json.dumps(prompt_cfg, ensure_ascii=False, indent=2).replace(
        "{{TARGET_POPULATION}}",
        f"{pop_name} (Specifically Raw Level '{clean_lvl}': {lvl_desc})",
    )
    valid_raw_levels_guide = {
        k: v
        for k, v in LEVEL_DESCRIPTIONS.items()
        if not k.startswith("'") or k in ("'2-4", "'4-9")
    }
    system_content = (
        f"Follow the specification below for prompt condition '{prompt_name}':\n"
        f"{prompt_json_str}\n\n"
        f"Valid Level Taxonomy for Classification: {json.dumps(valid_raw_levels_guide, ensure_ascii=False)}\n"
        f"Preserve 100% of core factual propositions from the input text. "
        f"Respond strictly in valid JSON."
    )
    user_content = json.dumps(
        {
            "requested_target_population": pop_name,
            "requested_target_raw_level": clean_lvl,
            "target_level_description": lvl_desc,
            "input_hebrew_text": source_text,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]



def call_static_llm_json(
    openai_client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    if mock_mode or openai_client is None:
        return {
            "classification": {"primary_level": "11"},
            "step_3_simplification": {
                "simplified_text": "זהו טקסט מופשט וברור לבדיקת המערכת.",
                "vocalization_plan": {
                    "mode": "continuous",
                    "target_diacritic_ratio": 0.30,
                    "words_to_vocalize": ["מופשט"],
                },
            },
        }
    for attempt in range(1, max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(2.0 ** attempt)
    return {}


def extract_static_original_and_simplified(
    llm_json: Dict[str, Any], fallback_text: str
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    if not isinstance(llm_json, dict):
        return "Unknown", fallback_text, None
    out_fmt = llm_json.get("output_format") if isinstance(llm_json.get("output_format"), dict) else {}
    cls_dict = (
        llm_json.get("classification")
        if isinstance(llm_json.get("classification"), dict)
        else (out_fmt.get("classification") if isinstance(out_fmt.get("classification"), dict) else {})
    )
    step1_dict = (
        llm_json.get("step_1_source_classification")
        if isinstance(llm_json.get("step_1_source_classification"), dict)
        else {}
    )
    est_list = llm_json.get("estimated_original_raw_levels")
    est_str = ", ".join(str(x).strip() for x in est_list if str(x).strip()) if isinstance(est_list, list) else None
    orig_candidates = [
        cls_dict.get("primary_level"),
        cls_dict.get("level"),
        step1_dict.get("primary_population"),
        step1_dict.get("primary_level"),
        est_str,
        llm_json.get("original_level"),
        llm_json.get("source_level"),
    ]
    orig_lvl = next(
        (str(c).strip() for c in orig_candidates if isinstance(c, (str, int, float)) and str(c).strip()),
        "Unknown",
    )
    simp_candidates = [
        llm_json.get("step_3_simplification", {}).get("simplified_text")
        if isinstance(llm_json.get("step_3_simplification"), dict)
        else None,
        llm_json.get("step_2_adaptation", {}).get("simplified_text")
        if isinstance(llm_json.get("step_2_adaptation"), dict)
        else None,
        out_fmt.get("simplified_text"),
        llm_json.get("simplified_text"),
        llm_json.get("adapted_text"),
        llm_json.get("rewritten_text"),
    ]
    simp_text = next(
        (str(c).strip() for c in simp_candidates if isinstance(c, str) and str(c).strip()),
        HebrewSimplificationAgent._extract_draft_text(llm_json, fallback_text=fallback_text),
    )
    voc_plan = None
    if isinstance(llm_json.get("step_3_simplification"), dict):
        voc_plan = llm_json["step_3_simplification"].get("vocalization_plan")
    elif isinstance(llm_json.get("vocalization_plan"), dict):
        voc_plan = llm_json.get("vocalization_plan")
    return orig_lvl, simp_text, voc_plan

def run_experimentation_pipeline(
    input_csv: Path,
    output_csv: Path,
    engine: Any,
    openai_client: Optional[Any],
    model_name: str = DEFAULT_LLM_MODEL,
    prompts_to_run: List[str] = DEFAULT_PROMPTS,
    prompts_dir: Path = PROJECT_ROOT / "prompts",
    max_iters: int = 3,
    mock_mode: bool = False,
) -> pd.DataFrame:
    """
    Runs all 4 prompt mechanisms with single-pass source evaluation caching,
    idempotent CSV checkpointing, and dual 8-column + 20-column enriched output.
    Returns the main deliverable DataFrame (`output_csv`).
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_pipeline_logger(
        name="ExperimentationPipeline",
        artifact_dir=str(output_csv.parent),
        log_filename="experimentation_run.log",
    )
    df_in = pd.read_csv(input_csv)
    enriched_csv = output_csv.with_name(output_csv.stem + "_enriched.csv")

    prompt_cfgs: Dict[str, Dict[str, Any]] = {}
    prompt_files = {
        "cot_persona": "persona_cot_prompt.json",
        "few_shot_persona": "few_shot_prompt.json",
        "feature_informed": "feature_informed_prompt.json",
    }
    for p_key, fname in prompt_files.items():
        fpath = Path(prompts_dir) / fname
        if fpath.exists():
            prompt_cfgs[p_key] = json.loads(fpath.read_text(encoding="utf-8"))
        else:
            prompt_cfgs[p_key] = {"prompt_id": p_key}

    agent = HebrewSimplificationAgent(
        engine=engine,
        feature_prompt_cfg=prompt_cfgs.get("feature_informed", {}),
        openai_client=openai_client,
        model_name=model_name,
        log_dir=str(output_csv.parent),
    )

    completed_keys: Set[Tuple[str, str]] = set()
    if output_csv.exists() and enriched_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            for _, r in existing_df.iterrows():
                completed_keys.add((str(r["paragraph_id"]).strip(), str(r["target_level"]).strip(), str(r["prompt"]).strip()))
            logger.info(f"Resuming from checkpoint: {len(completed_keys)} pairs already completed.")
        except Exception:
            completed_keys.clear()

    write_header_main = not output_csv.exists() or len(completed_keys) == 0
    write_header_enriched = not enriched_csv.exists() or len(completed_keys) == 0

    with (
        open(output_csv, "a" if not write_header_main else "w", encoding="utf-8", newline="") as f_main,
        open(enriched_csv, "a" if not write_header_enriched else "w", encoding="utf-8", newline="") as f_enr,
    ):
        writer_main = csv.DictWriter(f_main, fieldnames=REQUIRED_DELIVERABLE_COLUMNS)
        writer_enr = csv.DictWriter(f_enr, fieldnames=ENRICHED_COLUMNS)
        if write_header_main:
            writer_main.writeheader()
        if write_header_enriched:
            writer_enr.writeheader()

        for _, row in df_in.iterrows():
            pid = str(row["paragraph_id"])
            source_text = str(row["text"])
            target_level = str(row["target_level"]).strip()
            pop_name, _ = resolve_population_and_description(target_level)

            cached_source_eval = engine.evaluate_text(source_text, print_report=False)
            orig_score = round(float(cached_source_eval["raw_readability_score"]), 4)
            orig_tool_levels = format_iqr_levels_plain(cached_source_eval)
            q1_t, q3_t, mu_t = engine.get_raw_level_iqr_bounds(target_level)
            src_words = max(1, len(source_text.split()))

            for p_name in prompts_to_run:
                if (str(pid).strip(), str(target_level).strip(), str(p_name).strip()) in completed_keys:
                    continue

                if p_name == "agentic":
                    mock_turns = (
                        [
                            {
                                "step_3_simplification": {
                                    "simplified_text": "טיוטה ראשונה לבדיקה.",
                                    "vocalization_plan": {"mode": "continuous", "target_diacritic_ratio": 0.25, "words_to_vocalize": []},
                                }
                            }
                        ]
                        if mock_mode
                        else None
                    )
                    agent_res = agent.run_agentic_simplification(
                        source_text=source_text,
                        target_population=pop_name,
                        target_raw_level=target_level,
                        paragraph_id=pid,
                        max_iters=max_iters,
                        mock_responses=mock_turns,
                        cached_source_eval=cached_source_eval,
                    )
                    orig_level_str = orig_tool_levels
                    simp_text = agent_res["final_simplified_text"]
                    simp_eval = agent_res["final_eval_payload"]
                    simp_score = round(float(agent_res["final_score"]), 4)
                    iters_used = int(agent_res["total_iterations_run"])
                    offenders_str = "; ".join(agent_res.get("top_offending_features") or [])
                    voc_mode = (agent_res.get("final_vocalization_plan") or {}).get("mode", "continuous")
                    stag_flag = bool(agent_res.get("stagnation_triggered", False))
                else:
                    msgs = build_static_prompt_messages(
                        prompt_name=p_name,
                        prompt_cfg=prompt_cfgs.get(p_name, {}),
                        source_text=source_text,
                        target_level=target_level,
                    )
                    llm_json = call_static_llm_json(
                        openai_client=openai_client,
                        model_name=model_name,
                        messages=msgs,
                        mock_mode=mock_mode,
                    )
                    orig_level_str, raw_simp_text, voc_plan = extract_static_original_and_simplified(
                        llm_json, fallback_text=source_text
                    )
                    if False:  # Strict ablation lock: dicta-onnx vocalizer restricted exclusively to agentic
                        simp_text = engine.apply_llm_guided_vocalization(
                            raw_simp_text,
                            vocalization_plan=voc_plan,
                            target_raw_level=target_level,
                            target_mu=mu_t,
                        )
                        voc_mode = (voc_plan or {}).get("mode", "continuous")
                    else:
                        simp_text = raw_simp_text
                        voc_mode = "raw_llm"

                    simp_eval = engine.evaluate_text(simp_text, print_report=False)
                    simp_score = round(float(simp_eval["raw_readability_score"]), 4)
                    iters_used = 1
                    diag_static = agent.diagnose_draft_in_natural_language(
                        simp_eval, target_population=pop_name, target_raw_level=target_level
                    )
                    offenders_str = "; ".join((diag_static or {}).get("top_offending_features") or [])
                    stag_flag = False

                simp_level_str = format_simplified_level_with_track_probs(
                    simp_eval, target_level=target_level
                )
                retention = round(len(simp_text.split()) / src_words, 4)

                main_record = {
                    "paragraph_id": pid,
                    "text": source_text,
                    "target_level": target_level,
                    "prompt": p_name,
                    "original_level": orig_level_str,
                    "simplified_text": simp_text,
                    "simplified_score": simp_score,
                    "simplified_level": simp_level_str,
                }
                enriched_record = {
                    **main_record,
                    "original_score": orig_score,
                    "original_tool_iqr_levels": orig_tool_levels,
                    "target_q1": q1_t,
                    "target_q3": q3_t,
                    "target_mu": mu_t,
                    "iterations_used": iters_used,
                    "top_shap_offenders": offenders_str,
                    "word_retention_ratio": retention,
                    "k12_track_probs_json": json.dumps(
                        simp_eval.get("k12_track_probabilities", {}), ensure_ascii=False
                    ),
                    "special_track_probs_json": json.dumps(
                        simp_eval.get("special_track_probabilities", {}), ensure_ascii=False
                    ),
                    "vocalization_mode": voc_mode,
                    "stagnation_triggered": stag_flag,
                }

                writer_main.writerow(main_record)
                writer_enr.writerow(enriched_record)
                f_main.flush()
                f_enr.flush()

    return pd.read_csv(output_csv)


def analyze_experiment_results(output_csv: Path) -> pd.DataFrame:
    """Wrapper returning the 4-row method_quality_summary DataFrame."""
    res = analyze_experimentation_results(results_csv=Path(output_csv))
    return res["method_summary"]
