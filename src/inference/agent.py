"""Autonomous multi-turn simplification agent with Continuous Inverse-Median Vocalization, Sub-Paragraph Splitting, and Dynamic Stagnation Breaker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hebrew_readability.config import (
    CALIBRATION_DEAD_ZONE_BRIDGES,
    DEFAULT_LLM_MODEL,
    FEATURE_DIAGNOSTIC_RULES,
    LEVEL_DESCRIPTIONS,
    LEVEL_SHAP_WEIGHTS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    RAW_LEVEL_MEDIANS,
    RAW_LEVEL_TO_POPULATION,
    SUB_PARAGRAPH_SPLIT_LEVELS,
    compute_continuous_vocalization_target,
    get_sub_paragraph_split_instruction,
    get_vocalization_policy,
    setup_pipeline_logger,
)


class HebrewSimplificationAgent:
    """
    Autonomous multi-turn simplification agent using ReadabilityInferenceEngine,
    Hybrid Differential TreeSHAP Attribution, Continuous Inverse-Median + Zipf-Driven
    dicta-onnx Vocalization, Sub-Paragraph Chunking (\\n\\n), and a Dynamic Stagnation Breaker.
    """

    def __init__(
        self,
        engine: Any,
        feature_prompt_cfg: Dict[str, Any],
        openai_client: Optional[Any] = None,
        gemini_client: Optional[Any] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        log_dir: str = "artifacts",
    ) -> None:
        self.engine = engine
        self.feature_prompt_cfg = feature_prompt_cfg or {}
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.log_dir, "agent_trajectories.jsonl")
        self.logger = setup_pipeline_logger(
            name="HebrewSimplificationAgent",
            artifact_dir=self.log_dir,
            log_filename="agent_execution.log",
        )

    @staticmethod
    def _extract_draft_text(llm_json: Dict[str, Any], fallback_text: str = "") -> str:
        """Extracts simplified_text from nested or flat LLM JSON responses."""
        if not isinstance(llm_json, dict):
            return fallback_text
        for step_key in ("step_3_simplification", "step_2_adaptation", "simplification"):
            sub = llm_json.get(step_key)
            if isinstance(sub, dict):
                cand = sub.get("simplified_text")
                if isinstance(cand, str) and cand.strip():
                    return cand.strip()
        cand_top = llm_json.get("simplified_text")
        if isinstance(cand_top, str) and cand_top.strip():
            return cand_top.strip()
        return fallback_text

    @staticmethod
    def _extract_vocalization_plan(llm_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts vocalization_plan dictionary from LLM JSON output."""
        if not isinstance(llm_json, dict):
            return {}
        step3 = llm_json.get("step_3_simplification")
        if isinstance(step3, dict) and isinstance(step3.get("vocalization_plan"), dict):
            return step3["vocalization_plan"]
        if isinstance(llm_json.get("vocalization_plan"), dict):
            return llm_json["vocalization_plan"]
        return {}

    def _get_level_bounds_and_medians(
        self, target_population: str, target_raw_level: Optional[str]
    ) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
        clean_lvl = str(target_raw_level or "").strip()
        pop_spec = POPULATION_SPECS.get(
            target_population, POPULATION_SPECS["Grade 4-6 - Elementary Intermediate"]
        )
        q1, q3, mu = 4.80, 5.50, float(pop_spec.get("anchor_score", 5.31))

        if hasattr(self.engine, "get_raw_level_iqr_bounds") and clean_lvl:
            try:
                q1, q3, mu = self.engine.get_raw_level_iqr_bounds(clean_lvl)
            except Exception:
                pass
        elif hasattr(self.engine, "calibration_stats") and clean_lvl:
            cal = self.engine.calibration_stats
            for c in RAW_LEVEL_ALIASES.get(clean_lvl, [clean_lvl, f"'{clean_lvl}", clean_lvl.lstrip("'")]):
                if c in cal:
                    q1 = float(cal[c]["Q1"])
                    q3 = float(cal[c]["Q3"])
                    mu = float(cal[c]["mu"])
                    break

        medians = (
            RAW_LEVEL_MEDIANS.get(clean_lvl)
            or RAW_LEVEL_MEDIANS.get(f"'{clean_lvl}")
            or RAW_LEVEL_MEDIANS.get(clean_lvl.lstrip("'"))
            or pop_spec.get("medians")
            or RAW_LEVEL_MEDIANS["5"]
        )
        shap_w = (
            LEVEL_SHAP_WEIGHTS.get(clean_lvl)
            or LEVEL_SHAP_WEIGHTS.get(f"'{clean_lvl}")
            or LEVEL_SHAP_WEIGHTS.get(clean_lvl.lstrip("'"))
            or LEVEL_SHAP_WEIGHTS["5"]
        )
        return q1, q3, mu, medians, shap_w

    def diagnose_draft_in_natural_language(
        self,
        eval_payload: Dict[str, Any],
        target_population: str,
        target_raw_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates whether draft converged in Core IQR (including Dead-Zone Bridge [3.2720, 3.3898]),
        computes Hybrid Differential TreeSHAP Priority, and generates natural-language directives.
        """
        clean_lvl = str(target_raw_level or "").strip()
        q1_t, q3_t, mu_t, medians, shap_w = self._get_level_bounds_and_medians(
            target_population, clean_lvl
        )
        y_pred = float(eval_payload.get("raw_readability_score", float("nan")))
        core_matches = {str(m).strip() for m in (eval_payload.get("core_iqr_matches") or [])}
        core_matches |= {m.lstrip("'") for m in core_matches}

        # Apply Dead-Zone Bridge [3.2720, 3.3898]
        for bridge in CALIBRATION_DEAD_ZONE_BRIDGES:
            if float(bridge["lower_bound"]) <= y_pred <= float(bridge["upper_bound"]):
                nearest = (
                    str(bridge["lower_level"])
                    if y_pred <= float(bridge["midpoint"])
                    else str(bridge["upper_level"])
                )
                core_matches.add(nearest)
                for ex in bridge.get("always_include_levels", []):
                    core_matches.add(ex)
                    core_matches.add(ex.lstrip("'"))

        aliases = set(
            RAW_LEVEL_ALIASES.get(clean_lvl, [clean_lvl, f"'{clean_lvl}", clean_lvl.lstrip("'")])
        )
        in_core_iqr = bool(
            (clean_lvl and len(aliases & core_matches) > 0) or (q1_t <= y_pred <= q3_t)
        )

        if in_core_iqr:
            dist_to_iqr = 0.0
        elif y_pred > q3_t:
            dist_to_iqr = round(y_pred - q3_t, 4)
        else:
            dist_to_iqr = round(q1_t - y_pred, 4)

        score_diff = round(y_pred - mu_t, 4)
        tf = eval_payload.get("transformed_features") or {}
        local_shap = eval_payload.get("local_shap_values") or {}

        offenders: List[Tuple[float, str, str, str]] = []
        for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
            x_val = float(tf.get(feat_key, medians.get(feat_key, 0.5)))


# === V3 AGENT DIAGNOSTIC RETURN GUARANTEE PATCH ===
import math
import pandas as pd

def _safe_diagnose_draft_in_natural_language(
    self,
    eval_payload: Dict[str, Any],
    target_population: str,
    target_raw_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether draft converged in Core IQR (including Dead-Zone Bridge [3.2720, 3.3898]),
    computes Hybrid Differential TreeSHAP Priority, and generates natural-language directives.
    """
    clean_lvl = str(target_raw_level or "").strip()
    q1_t, q3_t, mu_t, medians, shap_w = self._get_level_bounds_and_medians(
        target_population, clean_lvl
    )
    y_pred = float((eval_payload or {}).get("raw_readability_score", float("nan")))
    core_matches = {str(m).strip() for m in ((eval_payload or {}).get("core_iqr_matches") or [])}
    ext_matches = {str(m).strip() for m in ((eval_payload or {}).get("extended_p1_p99_matches") or [])}
    aliases = set(RAW_LEVEL_ALIASES.get(clean_lvl, [clean_lvl, f"'{clean_lvl}", clean_lvl.lstrip("'")]))

    in_core_iqr = bool(
        (clean_lvl and len(aliases & core_matches) > 0)
        or (not math.isnan(y_pred) and (q1_t <= y_pred <= q3_t))
    )
    in_extended = bool(in_core_iqr or (clean_lvl and len(aliases & ext_matches) > 0))

    if in_core_iqr or math.isnan(y_pred):
        dist_to_iqr = 0.0
    elif y_pred > q3_t:
        dist_to_iqr = round(y_pred - q3_t, 4)
    else:
        dist_to_iqr = round(q1_t - y_pred, 4)

    score_diff = 0.0 if math.isnan(y_pred) else round(y_pred - mu_t, 4)
    tf = (eval_payload or {}).get("transformed_features") or {}
    local_shap = (eval_payload or {}).get("local_shap_values") or {}
    evidence = (eval_payload or {}).get("draft_evidence") or {}

    offenders: List[Tuple[float, str, str, str]] = []
    for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
        x_val = float(tf.get(feat_key, medians.get(feat_key, 0.5)))
        m_val = float(medians.get(feat_key, 0.5))
        delta = x_val - m_val
        w_level = float(shap_w.get(feat_key, rule.get("weight", 0.05)))
        phi_i = float(local_shap.get(feat_key, 0.0))

        if score_diff >= 0 and delta > 0.02:
            priority = max(0.0, phi_i) + abs(delta) * w_level
            instr = rule["too_high"]
            if feat_key == "18_avg_syllables_per_sentence_log" and evidence.get("longest_sentence"):
                instr += f" Longest sentence ({evidence.get('longest_sentence_words', 0)} words): «{evidence['longest_sentence']}»"
            elif feat_key == "14a_zipf_rarity_tuple_inv" and evidence.get("rare_lexical_items"):
                instr += f" Rare words detected: {', '.join(evidence['rare_lexical_items'][:4])}"
            offenders.append((priority, feat_key, rule["name"], instr))
        elif score_diff < 0 and delta < -0.03:
            priority = max(0.0, -phi_i) + abs(delta) * w_level
            offenders.append((priority, feat_key, rule["name"], rule["too_low"]))

    offenders.sort(key=lambda item: item[0], reverse=True)
    top_keys = [k for _, k, _, _ in offenders[:3]]
    top_names = [name for _, _, name, _ in offenders[:3]] or [
        "Sentence Length & Breath Load",
        "Information & Syllabic Density (Paragraph Volume)",
    ]
    top_directives = [f"- {name}: {instr}" for _, _, name, instr in offenders[:3]] or [
        "- Sentence Length & Breath Load: Adjust sentence length and vocabulary to match the target level."
    ]

    if in_core_iqr:
        feedback = (
            f"TOOL DIAGNOSTIC REPORT: Draft readability (Y_pred={y_pred:.2f}) "
            f"CONVERGED within Core IQR [{q1_t:.2f}, {q3_t:.2f}] for '{target_population}' (Level '{clean_lvl}')."
        )
    else:
        direction_str = "TOO COMPLEX" if score_diff >= 0 else "TOO SIMPLE"
        feedback = (
            f"TOOL DIAGNOSTIC REPORT: Draft (Y_pred={y_pred:.2f}, Target Core IQR=[{q1_t:.2f}, {q3_t:.2f}], "
            f"Target Anchor={mu_t:.2f}) is {direction_str} by {dist_to_iqr:.3f} points. Adjust:\n"
            + "\n".join(top_directives)
        )

    return {
        "in_core_iqr": in_core_iqr,
        "in_extended": in_extended,
        "target_q1": q1_t,
        "target_q3": q3_t,
        "target_mu": mu_t,
        "dist_to_iqr": dist_to_iqr,
        "abs_distance_to_anchor": round(abs(score_diff), 4),
        "top_offending_keys": top_keys,
        "top_offending_features": top_names,
        "natural_language_feedback": feedback,
    }

def _agent_build_initial_messages(
    self,
    source_text: str,
    target_population: str,
    target_raw_level: Optional[str] = None,
) -> List[Dict[str, str]]:
    clean_lvl = str(target_raw_level or "").strip()
    q1_t, q3_t, mu_t, medians, shap_w = self._get_level_bounds_and_medians(
        target_population, clean_lvl
    )
    lvl_desc = LEVEL_DESCRIPTIONS.get(clean_lvl, target_population)
    voc_policy = get_vocalization_policy(clean_lvl, mu_target=mu_t)
    split_instr = get_sub_paragraph_split_instruction(clean_lvl, len(source_text.split()))
    split_block = f"\nSub-Paragraph Chunking Policy: {split_instr}\n" if split_instr else ""

    prompt_str = json.dumps(self.feature_prompt_cfg, ensure_ascii=False, indent=2).replace(
        "{{TARGET_POPULATION}}",
        f"{target_population} (Specifically Raw Level '{clean_lvl}': {lvl_desc})",
    )
    system_content = (
        f"You are an autonomous Hebrew readability adaptation agent.\n"
        f"{prompt_str}\n\n"
        f"Calibrated Target Level: '{clean_lvl}' ({lvl_desc}) | Core IQR: [{q1_t:.2f}, {q3_t:.2f}] | Anchor: {mu_t:.2f}\n"
        f"Target Level Mean_Abs_SHAP Weights: {json.dumps(shap_w, ensure_ascii=False)}\n"
        f"Target Level Feature Medians: {json.dumps(medians, ensure_ascii=False)}\n"
        f"Vocalization Policy: {voc_policy['instruction']}\n"
        f"{split_block}"
        f"Include 'vocalization_plan' ('mode', 'target_diacritic_ratio', 'words_to_vocalize') "
        f"inside 'step_3_simplification'. Preserve 100% of factual propositions. Respond strictly in valid JSON."
    )
    user_content = json.dumps(
        {
            "requested_target_population": target_population,
            "requested_target_raw_level": clean_lvl,
            "target_core_iqr": [q1_t, q3_t],
            "input_hebrew_text": source_text,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _agent_call_llm_json(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    mock_responses: Optional[List[Dict[str, Any]]] = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    if mock_responses is not None and len(mock_responses) > 0:
        return mock_responses[min(iteration - 1, len(mock_responses) - 1)]

    if self.gemini_client is None and self.openai_client is None:
        return {
            "step_3_simplification": {
                "simplified_text": "זהו טקסט מופשט וברור לבדיקת המערכת.",
                "vocalization_plan": {
                    "mode": "continuous",
                    "target_diacritic_ratio": 0.25,
                    "words_to_vocalize": [],
                },
            }
        }

    for attempt in range(1, 4):
        try:
            if self.gemini_client is not None:
                from google.genai import types

                system_instruction = next(
                    (m["content"] for m in messages if m["role"] == "system"),
                    None,
                )

                contents = [
                    types.Content(
                        role="model" if m["role"] == "assistant" else "user",
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                    for m in messages
                    if m["role"] != "system"
                ]

                resp = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=temperature,
                    ),
                )

                return json.loads(resp.text)

            resp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(resp.choices[0].message.content)

        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 ** attempt)

    return {}

def _safe_run_agentic_simplification(
    self,
    source_text: str,
    target_population: str,
    target_raw_level: Optional[str] = None,
    paragraph_id: str = "sample_01",
    max_iters: int = 3,
    mock_responses: Optional[List[Dict[str, Any]]] = None,
    cached_source_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start_t = time.time()
    clean_lvl = str(target_raw_level or "").strip()
    q1_t, q3_t, mu_t, _, _ = self._get_level_bounds_and_medians(target_population, clean_lvl)

    source_eval = cached_source_eval or self.engine.evaluate_text(source_text, print_report=False)
    source_score = round(float(source_eval.get("raw_readability_score", float("nan"))), 4)

    messages = self._build_initial_messages(source_text, target_population, clean_lvl)
    trajectory: List[Dict[str, Any]] = []
    best_turn: Optional[Dict[str, Any]] = None
    prev_score: Optional[float] = None
    stagnation_triggered = False
    current_temp = 0.3

    for iteration in range(1, max_iters + 1):
        t0 = time.time()
        llm_json = self._call_llm_json(
            messages=messages,
            temperature=current_temp,
            mock_responses=mock_responses,
            iteration=iteration,
        )
        raw_draft = self._extract_draft_text(llm_json, fallback_text=source_text)
        voc_plan = self._extract_vocalization_plan(llm_json)

        if hasattr(self.engine, "apply_llm_guided_vocalization"):
            vocalized_draft = self.engine.apply_llm_guided_vocalization(
                raw_draft,
                vocalization_plan=voc_plan,
                target_raw_level=clean_lvl,
                current_score=prev_score,
                target_mu=mu_t,
            )
        else:
            vocalized_draft = raw_draft

        eval_payload = self.engine.evaluate_text(vocalized_draft, print_report=False)
        diag = self.diagnose_draft_in_natural_language(
            eval_payload,
            target_population=target_population,
            target_raw_level=clean_lvl,
        )
        y_pred = round(float(eval_payload.get("raw_readability_score", float("nan"))), 4)

        stag_this_turn = False
        if (
            prev_score is not None
            and not diag["in_core_iqr"]
            and abs(y_pred - prev_score) < 0.015
        ):
            stag_this_turn = True
            stagnation_triggered = True
            current_temp = 0.6
        else:
            current_temp = 0.3
        prev_score = y_pred

        turn_rec: Dict[str, Any] = {
            "iteration": iteration,
            "draft_text": vocalized_draft,
            "predicted_score": y_pred,
            "target_q1": q1_t,
            "target_q3": q3_t,
            "target_mu": mu_t,
            "in_core_iqr": diag["in_core_iqr"],
            "in_extended": diag["in_extended"],
            "dist_to_iqr": diag["dist_to_iqr"],
            "abs_distance_to_anchor": diag["abs_distance_to_anchor"],
            "top_offending_keys": diag["top_offending_keys"],
            "top_offending_features": diag["top_offending_features"],
            "vocalization_plan": voc_plan,
            "stagnation_triggered": stag_this_turn,
            "tool_natural_language_feedback": diag["natural_language_feedback"],
            "transformed_features": eval_payload.get("transformed_features", {}),
            "local_shap_values": eval_payload.get("local_shap_values", {}),
            "eval_payload": eval_payload,
            "latency_sec": round(time.time() - t0, 2),
        }
        trajectory.append(turn_rec)

        if (
            best_turn is None
            or (turn_rec["in_core_iqr"] and not best_turn["in_core_iqr"])
            or (
                turn_rec["in_core_iqr"] == best_turn["in_core_iqr"]
                and (
                    turn_rec["dist_to_iqr"] < best_turn["dist_to_iqr"]
                    or (
                        turn_rec["dist_to_iqr"] == best_turn["dist_to_iqr"]
                        and turn_rec["abs_distance_to_anchor"] < best_turn["abs_distance_to_anchor"]
                    )
                )
            )
        ):
            best_turn = turn_rec

        if diag["in_core_iqr"]:
            break

        if iteration < max_iters:
            ev = eval_payload.get("draft_evidence") or {}
            stag_alert = ""
            if stag_this_turn:
                stag_alert = (
                    f"\n⚠️ STAGNATION ALERT (|Y_t - Y_{{t-1}}| < 0.015): Your previous revision did not move the readability score. "
                    f"Split the longest sentence ({ev.get('longest_sentence_words', 0)} words: «{ev.get('longest_sentence', '')}») "
                    f"into 2-3 short independent sentences and replace rare lemmas ({', '.join(ev.get('rare_lexical_items', [])[:3])}).\n"
                )
            voc_upd = get_vocalization_policy(clean_lvl, y_pred=y_pred, mu_target=mu_t)
            messages.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{diag['natural_language_feedback']}\n"
                        f"{stag_alert}"
                        f"Updated Vocalization Guidance: {voc_upd['instruction']}\n"
                        f"Revise 'step_3_simplification.simplified_text' and 'vocalization_plan' so the score lands inside [{q1_t:.2f}, {q3_t:.2f}]. "
                        f"Preserve 100% of core facts. Respond strictly in valid JSON."
                    ),
                }
            )

    assert best_turn is not None
    result_payload: Dict[str, Any] = {
        "paragraph_id": paragraph_id,
        "target_population": target_population,
        "target_raw_level": clean_lvl,
        "source_text": source_text,
        "source_score": source_score,
        "final_simplified_text": best_turn["draft_text"],
        "final_score": best_turn["predicted_score"],
        "final_eval_payload": best_turn["eval_payload"],
        "final_vocalization_plan": best_turn["vocalization_plan"],
        "target_q1": q1_t,
        "target_q3": q3_t,
        "target_mu": mu_t,
        "converged_in_core_iqr": best_turn["in_core_iqr"],
        "best_iteration_selected": best_turn["iteration"],
        "total_iterations_run": len(trajectory),
        "top_offending_features": best_turn["top_offending_features"],
        "stagnation_triggered": stagnation_triggered,
        "total_latency_sec": round(time.time() - start_t, 2),
        "trajectory": [{k: v for k, v in t.items() if k != "eval_payload"} for t in trajectory],
    }
    with open(self.trajectory_path, "a", encoding="utf-8") as f_log:
        f_log.write(json.dumps({k: v for k, v in result_payload.items() if k != "final_eval_payload"}, ensure_ascii=False) + "\n")
        f_log.flush()

    return result_payload

HebrewSimplificationAgent.diagnose_draft_in_natural_language = _safe_diagnose_draft_in_natural_language
HebrewSimplificationAgent._build_initial_messages = _agent_build_initial_messages
HebrewSimplificationAgent._call_llm_json = _agent_call_llm_json
HebrewSimplificationAgent.run_agentic_simplification = _safe_run_agentic_simplification
