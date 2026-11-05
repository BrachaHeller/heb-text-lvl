
"""Autonomous agentic simplification workflow and multi-turn trajectory logging."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from hebrew_readability.config import (
    FEATURE_DIAGNOSTIC_RULES,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    setup_pipeline_logger,
)


class HebrewSimplificationAgent:
    """
    Autonomous multi-turn simplification agent that uses ReadabilityInferenceEngine
    as a feedback function to verify whether Y_pred falls within the target's Core IQR (Q1-Q3).
    """

    def __init__(
        self,
        engine: Any,
        feature_prompt_cfg: Dict[str, Any],
        openai_client: Optional[Any] = None,
        model_name: str = "gpt-4o-mini",
        log_dir: str = "artifacts",
    ) -> None:
        self.engine = engine
        self.prompt_cfg = feature_prompt_cfg
        self.client = openai_client
        self.model_name = model_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_log_path = os.path.join(self.log_dir, "agent_trajectories.jsonl")
        self.logger = setup_pipeline_logger(
            name="HebrewSimplificationAgent",
            artifact_dir=self.log_dir,
            log_filename="agent_execution.log",
        )

    @staticmethod
    def _extract_categories_and_levels(
        eval_result: Dict[str, Any], cat_key: str, match_key: str
    ) -> Tuple[Set[str], Set[str]]:
        """Extracts both mapped categories and raw level keys even if engine is an older instance."""
        raw_matches = {str(m).strip() for m in (eval_result.get(match_key) or [])}
        raw_matches |= {m.lstrip("'") for m in raw_matches}

        explicit_cats = {str(c).strip() for c in (eval_result.get(cat_key) or [])}
        derived_cats: Set[str] = set()
        for m in raw_matches:
            for c in LEVEL_TO_CATEGORIES.get(m, []):
                derived_cats.add(c)

        return explicit_cats | derived_cats, raw_matches

    def _get_target_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
        """Retrieves calibrated (Q1, Q3) bounds from engine if available, or estimates from anchor."""
        if hasattr(self.engine, "get_population_iqr_bounds"):
            return self.engine.get_population_iqr_bounds(target_population)
        anchor = float(POPULATION_SPECS[target_population]["anchor_score"])
        return round(anchor - 0.45, 4), round(anchor + 0.45, 4)

    def diagnose_draft_in_natural_language(
        self, eval_result: Dict[str, Any], target_population: str
    ) -> Dict[str, Any]:
        """
        Checks if Y_pred falls strictly within the target population's Core IQR (Q1-Q3)
        and produces SHAP-weighted diagnostic directives if it misses.
        """
        if target_population not in POPULATION_SPECS:
            raise ValueError(
                f"Unknown target_population '{target_population}'. "
                f"Valid options: {list(POPULATION_SPECS.keys())}"
            )

        spec = POPULATION_SPECS[target_population]
        target_cats = set(spec["engine_categories"])
        target_raw_levels = set(spec.get("raw_level_keys", []))

        matched_iqr_cats, matched_iqr_levels = self._extract_categories_and_levels(
            eval_result, "core_iqr_categories", "core_iqr_matches"
        )
        matched_ext_cats, matched_ext_levels = self._extract_categories_and_levels(
            eval_result, "extended_p1_p99_categories", "extended_p1_p99_matches"
        )

        y_pred = float(eval_result["raw_readability_score"])
        q1_bound, q3_bound = self._get_target_iqr_bounds(target_population)
        score_diff = y_pred - float(spec["anchor_score"])

        # Strict Core IQR verification (never conflated with P1-P99 extended matches)
        in_core_iqr = bool(
            len(target_cats & matched_iqr_cats) > 0
            or len(target_raw_levels & matched_iqr_levels) > 0
            or (q1_bound <= y_pred <= q3_bound)
        )
        in_extended = bool(
            in_core_iqr
            or len(target_cats & matched_ext_cats) > 0
            or len(target_raw_levels & matched_ext_levels) > 0
        )

        draft_feats = dict(eval_result.get("transformed_features") or {})
        if (
            "14a_zipf_rarity_tuple_inv" not in draft_feats
            and "14a_zipf_rarity_tuple_log_inv" in draft_feats
        ):
            draft_feats["14a_zipf_rarity_tuple_inv"] = draft_feats[
                "14a_zipf_rarity_tuple_log_inv"
            ]

        target_medians = spec["medians"]
        offenders: List[Tuple[float, str, str]] = []

        for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
            if feat_key in draft_feats and feat_key in target_medians:
                delta = float(draft_feats[feat_key]) - float(target_medians[feat_key])
                if score_diff >= 0 and delta > 0.03:
                    offenders.append(
                        (abs(delta) * rule["weight"], rule["name"], rule["too_high"])
                    )
                elif score_diff < 0 and delta < -0.04:
                    offenders.append(
                        (abs(delta) * rule["weight"], rule["name"], rule["too_low"])
                    )

        offenders.sort(key=lambda x: x[0], reverse=True)
        top_offending_features = [name for _, name, _ in offenders[:3]] or [
            "Sentence Length",
            "Paragraph Length & Volume",
        ]
        top_directives = [
            f"- {name}: {instr}" for _, name, instr in offenders[:3]
        ] or [
            "- Sentence Length: Shorten sentences and simplify vocabulary to match the target level."
        ]

        if in_core_iqr:
            feedback = (
                f"TOOL DIAGNOSTIC REPORT: Draft score (Y_pred={y_pred:.2f}) converged inside "
                f"Core IQR [{q1_bound:.2f}, {q3_bound:.2f}] for '{target_population}'."
            )
        else:
            overall_direction = "TOO COMPLEX" if score_diff > 0 else "TOO SIMPLE"
            feedback = (
                f"TOOL DIAGNOSTIC REPORT: Draft (Y_pred={y_pred:.2f}, Target Core IQR=[{q1_bound:.2f}, {q3_bound:.2f}]) "
                f"is {overall_direction} for '{target_population}'. Adjust:\n"
                + "\n".join(top_directives)
            )

        return {
            "in_core_iqr": in_core_iqr,
            "in_extended": in_extended,
            "target_q1": q1_bound,
            "target_q3": q3_bound,
            "abs_distance_to_anchor": round(abs(score_diff), 4),
            "top_offending_features": top_offending_features,
            "natural_language_feedback": feedback,
        }

    def _build_initial_messages(
        self, source_text: str, target_population: str
    ) -> List[Dict[str, str]]:
        """Injects feature_informed_prompt.json profile and schema into the LLM system prompt."""
        role = self.prompt_cfg.get(
            "role",
            "You are an expert Hebrew language educator and computational linguist.",
        )
        pop_profile = self.prompt_cfg.get("population_feature_profiles", {}).get(
            target_population, {}
        )
        schema = self.prompt_cfg.get(
            "required_output_schema",
            {
                "step_1_source_classification": {
                    "primary_population": "Grade 7-9 - Middle School",
                    "secondary_populations": [],
                },
                "step_2_linguistic_diagnosis": {
                    "target_population": target_population,
                    "key_features_to_modify": ["Sentence Length", "Vocabulary Rarity"],
                    "adaptation_explanation": "Brief explanation.",
                },
                "step_3_simplification": {
                    "simplified_text": "הטקסט המופשט בעברית כאן."
                },
            },
        )

        system_prompt = (
            f"{role}"
            f"Target Population: '{target_population}'"
            f"Target Population Adaptation Profile:"
            f"{json.dumps(pop_profile, ensure_ascii=False, indent=2)}"
            f"You MUST respond strictly in valid JSON following this exact schema "
            f"(place your adapted Hebrew text inside 'step_3_simplification.simplified_text'):"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

        user_prompt = json.dumps(
            {
                "requested_target_population": target_population,
                "input_hebrew_text": source_text,
            },
            ensure_ascii=False,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_draft_text(llm_json: Dict[str, Any], fallback_text: str) -> str:
        """Defensively extracts the simplified Hebrew string from LLM JSON responses."""
        candidates = [
            llm_json.get("step_3_simplification", {}).get("simplified_text")
            if isinstance(llm_json.get("step_3_simplification"), dict)
            else None,
            llm_json.get("step_2_adaptation", {}).get("simplified_text")
            if isinstance(llm_json.get("step_2_adaptation"), dict)
            else None,
            llm_json.get("simplified_text"),
            llm_json.get("adapted_text"),
            llm_json.get("text"),
        ]
        for cand in candidates:
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
        return fallback_text

    def _call_llm_step(
        self,
        messages: List[Dict[str, str]],
        mock_responses: Optional[List[Dict[str, Any]]] = None,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        if mock_responses is not None:
            return mock_responses[min(iteration - 1, len(mock_responses) - 1)]
        if self.client is None:
            raise RuntimeError("openai_client is None and no mock_responses were provided.")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)

    def _append_jsonl_log(self, record: Dict[str, Any]) -> None:
        with open(self.jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "")
            f.flush()

    def run_agentic_simplification(
        self,
        source_text: str,
        target_population: str,
        paragraph_id: str = "sample_01",
        max_iters: int = 3,
        mock_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        source_eval = self.engine.evaluate_text(source_text, print_report=False)
        source_score = round(float(source_eval["raw_readability_score"]), 4)
        source_cats, source_lvls = self._extract_categories_and_levels(
            source_eval, "core_iqr_categories", "core_iqr_matches"
        )

        self.logger.info(
            f"[{paragraph_id}] Source Baseline Score: {source_score:.4f} | "
            f"Matches: {sorted(source_lvls)} ({sorted(source_cats)}) -> Target: '{target_population}'"
        )

        messages = self._build_initial_messages(source_text, target_population)
        trajectory: List[Dict[str, Any]] = []
        best_turn: Optional[Dict[str, Any]] = None

        for iteration in range(1, max_iters + 1):
            iter_start = time.time()
            llm_json = self._call_llm_step(messages, mock_responses, iteration)
            draft_text = self._extract_draft_text(llm_json, fallback_text=source_text)

            eval_res = self.engine.evaluate_text(draft_text, print_report=False)
            diag = self.diagnose_draft_in_natural_language(eval_res, target_population)
            predicted_score = round(float(eval_res["raw_readability_score"]), 4)

            turn: Dict[str, Any] = {
                "iteration": iteration,
                "draft_text": draft_text,
                "predicted_score": predicted_score,
                "target_q1": diag["target_q1"],
                "target_q3": diag["target_q3"],
                "in_core_iqr": diag["in_core_iqr"],
                "in_extended": diag["in_extended"],
                "abs_distance_to_anchor": diag["abs_distance_to_anchor"],
                "top_offending_features": diag["top_offending_features"],
                "tool_natural_language_feedback": diag["natural_language_feedback"],
                "transformed_features": eval_res.get("transformed_features", {}),
                "latency_sec": round(time.time() - iter_start, 2),
            }
            trajectory.append(turn)

            self.logger.info(
                f"[{paragraph_id}] Iter {iteration}/{max_iters} | "
                f"Y_pred={predicted_score:.4f} | Core_IQR=[{diag['target_q1']:.2f}, {diag['target_q3']:.2f}] | "
                f"Converged={diag['in_core_iqr']} | Offenders={diag['top_offending_features']}"
            )

            if (
                best_turn is None
                or (turn["in_core_iqr"] and not best_turn["in_core_iqr"])
                or (
                    turn["in_core_iqr"] == best_turn["in_core_iqr"]
                    and turn["abs_distance_to_anchor"] < best_turn["abs_distance_to_anchor"]
                )
            ):
                best_turn = turn

            if diag["in_core_iqr"]:
                break

            if iteration < max_iters:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(llm_json, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{diag['natural_language_feedback']}"
                            f"Revise your Hebrew adaptation in 'step_3_simplification.simplified_text' "
                            f"so that its readability score falls inside [{diag['target_q1']:.2f}, {diag['target_q3']:.2f}]. "
                            f"Respond strictly in valid JSON."
                        ),
                    }
                )

        assert best_turn is not None
        final_result: Dict[str, Any] = {
            "paragraph_id": paragraph_id,
            "target_population": target_population,
            "source_text": source_text,
            "source_score": source_score,
            "final_simplified_text": best_turn["draft_text"],
            "final_score": best_turn["predicted_score"],
            "target_q1": best_turn["target_q1"],
            "target_q3": best_turn["target_q3"],
            "converged_in_core_iqr": best_turn["in_core_iqr"],
            "best_iteration_selected": best_turn["iteration"],
            "total_iterations_run": len(trajectory),
            "total_latency_sec": round(time.time() - start_time, 2),
            "trajectory": trajectory,
        }

        self._append_jsonl_log(final_result)
        return final_result


def _run_smoke_test() -> None:
    """Self-contained smoke test verifying config, engine categories, and agent loop."""
    import tempfile

    class MockEngine:
        def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
            return (2.45, 4.15)

        def evaluate_text(self, text: str, print_report: bool = False) -> Dict[str, Any]:
            has_nikkud = any("֑" <= c <= "ׇ" for c in text)
            score = 3.15 if has_nikkud else 5.45
            raw_matches = ["2", "3", "'2-4"] if has_nikkud else ["6", "7", "8"]
            return {
                "raw_readability_score": score,
                "core_iqr_matches": raw_matches,
                "extended_p1_p99_matches": ["2", "3", "4", "5", "6", "7", "8"],
                "transformed_features": {
                    "1_nikkud_ratio_log_inv": 0.05 if has_nikkud else 0.98,
                    "16_total_syllables_log": 0.48 if has_nikkud else 0.65,
                    "18_avg_syllables_per_sentence_log": 0.45 if has_nikkud else 0.60,
                    "10_dct_ratio_adj_log": 0.18,
                    "8_stz_avg_tree_depth_per_sent_log": 0.30,
                    "14a_zipf_rarity_tuple_inv": 0.40,
                },
            }

    with tempfile.TemporaryDirectory() as tmp:
        agent = HebrewSimplificationAgent(
            engine=MockEngine(),
            feature_prompt_cfg={"role": "Smoke test"},
            log_dir=tmp,
        )
        mocks = [
            {"step_3_simplification": {"simplified_text": "טקסט ללא ניקוד ארוך מדי."}},
            {"step_3_simplification": {"simplified_text": "טֶקְסְט קָצָר עִם נִיקּוּד."}},
        ]
        res = agent.run_agentic_simplification(
            source_text="טקסט מקור מורכב.",
            target_population="Grade 1-3 - Elementary Beginner",
            paragraph_id="smoke_test_01",
            max_iters=3,
            mock_responses=mocks,
        )
        assert res["converged_in_core_iqr"] is True
        assert res["best_iteration_selected"] == 2
        assert "source_score" in res
        assert "top_offending_features" in res["trajectory"][0]
        assert os.path.exists(os.path.join(tmp, "agent_trajectories.jsonl"))
        print("[✓] Smoke test passed: config.py + engine.py + agent.py contracts verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run standalone smoke test")
    args = parser.parse_args()
    if args.test:
        _run_smoke_test()

        


# --- DELIVERABLE 6 PATCH: Exact Raw-Level IQR Convergence & Length Budgets ---
RAW_LEVEL_LENGTH_BUDGETS = {
    "1": "Max 25 words total; 4-6 words per sentence; full Nikkud.",
    "2": "Max 30 words total; 5-7 words per sentence; full Nikkud.",
    "3": "Max 32 words total; 6-8 words per sentence; full/partial Nikkud.",
    "'2-4": "Max 32 words total; 6-8 words per sentence.",
    "2-4": "Max 32 words total; 6-8 words per sentence.",
    "4": "Max 35 words total; 7-9 words per sentence (4-5 short sentences).",
    "5": "Max 40 words total; 8-10 words per sentence.",
    "6": "Max 45 words total; 9-12 words per sentence.",
    "'4-9": "Max 45 words total; 10-12 words per sentence.",
    "4-9": "Max 45 words total; 10-12 words per sentence.",
    "7": "Max 50 words total; 11-13 words per sentence.",
    "8": "Max 55 words total; 12-15 words per sentence.",
    "9": "Max 60 words total; 13-16 words per sentence.",
    "intellectual_disability": "Max 30 words total; strictly 1 clause (5-8 words) per sentence.",
    "Hadshon": "Max 35 words total; 6-9 everyday words per sentence.",
    "Olim_news": "Max 35 words total; 6-9 everyday words per sentence.",
    "Olim_ragil": "Max 40 words total; 8-10 simple words per sentence.",
    "מבחני אולפן - רגיל": "Max 40 words total; 8-10 simple words per sentence.",
    "Olim_mitkadmim": "Max 48 words total; 10-13 words per sentence.",
    "מבחני אולפן - מתקדמים": "Max 48 words total; 10-13 words per sentence.",
}

RAW_LEVEL_ALIAS_LOOKUP = {
    "1": ["1", "2"], "2": ["2"], "3": ["3"], "'2-4": ["'2-4", "2-4"], "2-4": ["'2-4", "2-4"],
    "4": ["4"], "5": ["5"], "6": ["6"], "'4-9": ["'4-9", "4-9"], "4-9": ["'4-9", "4-9"],
    "7": ["7"], "8": ["8"], "9": ["9"], "10": ["10"], "11": ["11"], "12": ["12"],
    "Bagrut": ["Bagrut", "Bagrut_Olim"], "Bagrut_Olim": ["Bagrut_Olim", "Bagrut"],
    "hearing_impaired": ["hearing_impaired", "Bagrut_hearing_impaired"],
    "Bagrut_hearing_impaired": ["Bagrut_hearing_impaired", "hearing_impaired"],
    "intellectual_disability": ["intellectual_disability"],
    "Hadshon": ["Hadshon", "Olim_news"], "Olim_news": ["Olim_news", "Hadshon"],
    "Olim_ragil": ["Olim_ragil", "מבחני אולפן - רגיל"],
    "מבחני אולפן - רגיל": ["מבחני אולפן - רגיל", "Olim_ragil"],
    "Olim_mitkadmim": ["Olim_mitkadmim", "מבחני אולפן - מתקדמים"],
    "מבחני אולפן - מתקדמים": ["מבחני אולפן - מתקדמים", "Olim_mitkadmim"],
}

def _get_raw_level_calibration(engine, target_raw_level: Optional[str], target_population: str):
    cal_stats = getattr(engine, "calibration_stats", {})
    if target_raw_level:
        clean = str(target_raw_level).strip()
        aliases = RAW_LEVEL_ALIAS_LOOKUP.get(clean, [clean, f"'{clean}", clean.lstrip("'")])
        for cand in aliases:
            if cand in cal_stats:
                s = cal_stats[cand]
                return round(float(s["Q1"]), 4), round(float(s["Q3"]), 4), round(float(s["mu"]), 4), set(aliases)
    if hasattr(engine, "get_population_iqr_bounds"):
        q1, q3 = engine.get_population_iqr_bounds(target_population)
        anchor = float(POPULATION_SPECS[target_population]["anchor_score"])
        return q1, q3, anchor, set()
    anchor = float(POPULATION_SPECS[target_population]["anchor_score"])
    return round(anchor - 0.45, 4), round(anchor + 0.45, 4), anchor, set()

def _patched_diagnose_draft(
    self, eval_result: Dict[str, Any], target_population: str, target_raw_level: Optional[str] = None
) -> Dict[str, Any]:
    spec = POPULATION_SPECS[target_population]
    matched_iqr_cats, matched_iqr_levels = self._extract_categories_and_levels(
        eval_result, "core_iqr_categories", "core_iqr_matches"
    )
    matched_ext_cats, matched_ext_levels = self._extract_categories_and_levels(
        eval_result, "extended_p1_p99_categories", "extended_p1_p99_matches"
    )

    y_pred = float(eval_result["raw_readability_score"])
    q1_bound, q3_bound, anchor_score, raw_aliases = _get_raw_level_calibration(
        self.engine, target_raw_level, target_population
    )
    score_diff = y_pred - anchor_score

    if raw_aliases:
        # Strict raw-level Core IQR verification (prevents stopping at Grade 6 when Target is Grade 4)
        in_core_iqr = bool((len(raw_aliases & matched_iqr_levels) > 0) or (q1_bound <= y_pred <= q3_bound))
        in_extended = bool(in_core_iqr or (len(raw_aliases & matched_ext_levels) > 0))
        target_label = f"Raw Level '{target_raw_level}' ({target_population})"
    else:
        target_cats = set(spec["engine_categories"])
        target_raw_set = set(spec.get("raw_level_keys", []))
        in_core_iqr = bool(
            (len(target_cats & matched_iqr_cats) > 0)
            or (len(target_raw_set & matched_iqr_levels) > 0)
            or (q1_bound <= y_pred <= q3_bound)
        )
        in_extended = bool(
            in_core_iqr
            or (len(target_cats & matched_ext_cats) > 0)
            or (len(target_raw_set & matched_ext_levels) > 0)
        )
        target_label = target_population

    draft_feats = dict(eval_result.get("transformed_features") or {})
    target_medians = spec["medians"]
    offenders = []
    for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
        if feat_key in draft_feats and feat_key in target_medians:
            delta = float(draft_feats[feat_key]) - float(target_medians[feat_key])
            if score_diff >= 0 and delta > 0.02:
                offenders.append((abs(delta) * rule["weight"], rule["name"], rule["too_high"]))
            elif score_diff < 0 and delta < -0.03:
                offenders.append((abs(delta) * rule["weight"], rule["name"], rule["too_low"]))

    offenders.sort(key=lambda x: x[0], reverse=True)
    top_offending_features = [name for _, name, _ in offenders[:3]] or ["Paragraph Length & Volume", "Sentence Length"]
    top_directives = [f"- {name}: {instr}" for _, name, instr in offenders[:3]] or [
        "- Sentence Length: Shorten sentences and reduce word count per sentence."
    ]
    if target_raw_level and str(target_raw_level) in RAW_LEVEL_LENGTH_BUDGETS and score_diff > 0:
        top_directives.insert(0, f"- Length & Sentence Budget for Level '{target_raw_level}': {RAW_LEVEL_LENGTH_BUDGETS[str(target_raw_level)]}")

    nl = chr(10)
    if in_core_iqr:
        feedback = f"TOOL DIAGNOSTIC REPORT: Draft score (Y_pred={y_pred:.2f}) converged inside Core IQR [{q1_bound:.2f}, {q3_bound:.2f}] for {target_label}."
    else:
        direction = "TOO COMPLEX" if score_diff > 0 else "TOO SIMPLE"
        feedback = (
            f"TOOL DIAGNOSTIC REPORT: Draft (Y_pred={y_pred:.2f}, Target Core IQR=[{q1_bound:.2f}, {q3_bound:.2f}], Anchor={anchor_score:.2f}) "
            f"is {direction} for {target_label}. Adjust:{nl}" + nl.join(top_directives[:4])
        )

    return {
        "in_core_iqr": in_core_iqr,
        "in_extended": in_extended,
        "target_q1": q1_bound,
        "target_q3": q3_bound,
        "target_mu": anchor_score,
        "abs_distance_to_anchor": round(abs(score_diff), 4),
        "top_offending_features": top_offending_features,
        "natural_language_feedback": feedback,
    }

def _patched_run_agentic_simplification(
    self,
    source_text: str,
    target_population: str,
    paragraph_id: str = "sample_01",
    max_iters: int = 3,
    mock_responses: Optional[List[Dict[str, Any]]] = None,
    target_raw_level: Optional[str] = None,
    cached_source_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    source_eval = cached_source_eval or self.engine.evaluate_text(source_text, print_report=False)
    source_score = round(float(source_eval["raw_readability_score"]), 4)
    q1_bound, q3_bound, anchor_score, _ = _get_raw_level_calibration(
        self.engine, target_raw_level, target_population
    )

    messages = self._build_initial_messages(source_text, target_population)
    if target_raw_level:
        budget_note = RAW_LEVEL_LENGTH_BUDGETS.get(str(target_raw_level), "Keep sentences concise.")
        messages[0]["content"] += (
            f"{chr(10)}{chr(10)}EXACT TARGET RAW LEVEL: '{target_raw_level}' "
            f"(Calibrated Core IQR: [{q1_bound:.2f}, {q3_bound:.2f}], Target Score Anchor: {anchor_score:.2f})."
            f"{chr(10)}STRICT LENGTH & STRUCTURE BUDGET: {budget_note}"
        )

    trajectory = []
    best_turn = None
    nl = chr(10)

    for iteration in range(1, max_iters + 1):
        iter_start = time.time()
        llm_json = self._call_llm_step(messages, mock_responses, iteration)
        draft_text = self._extract_draft_text(llm_json, fallback_text=source_text)

        eval_res = self.engine.evaluate_text(draft_text, print_report=False)
        diag = self.diagnose_draft_in_natural_language(
            eval_res, target_population, target_raw_level=target_raw_level
        )
        predicted_score = round(float(eval_res["raw_readability_score"]), 4)

        turn = {
            "iteration": iteration,
            "draft_text": draft_text,
            "predicted_score": predicted_score,
            "target_q1": diag["target_q1"],
            "target_q3": diag["target_q3"],
            "target_mu": diag["target_mu"],
            "in_core_iqr": diag["in_core_iqr"],
            "in_extended": diag["in_extended"],
            "abs_distance_to_anchor": diag["abs_distance_to_anchor"],
            "top_offending_features": diag["top_offending_features"],
            "tool_natural_language_feedback": diag["natural_language_feedback"],
            "transformed_features": eval_res.get("transformed_features", {}),
            "eval_payload": eval_res,
            "latency_sec": round(time.time() - iter_start, 2),
        }
        trajectory.append(turn)

        self.logger.info(
            f"[{paragraph_id}] Iter {iteration}/{max_iters} | RawTarget={target_raw_level or target_population} | "
            f"Y_pred={predicted_score:.4f} | IQR=[{diag['target_q1']:.2f}, {diag['target_q3']:.2f}] | "
            f"Converged={diag['in_core_iqr']} | Offenders={diag['top_offending_features']}"
        )

        if (
            best_turn is None
            or (turn["in_core_iqr"] and not best_turn["in_core_iqr"])
            or (
                turn["in_core_iqr"] == best_turn["in_core_iqr"]
                and turn["abs_distance_to_anchor"] < best_turn["abs_distance_to_anchor"]
            )
        ):
            best_turn = turn

        if diag["in_core_iqr"]:
            break

        if iteration < max_iters:
            messages.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{diag['natural_language_feedback']}{nl}{nl}"
                        f"Rewrite your Hebrew adaptation in 'step_3_simplification.simplified_text' "
                        f"so its readability score lands inside [{diag['target_q1']:.2f}, {diag['target_q3']:.2f}]. "
                        f"Respond strictly in valid JSON."
                    ),
                }
            )

    final_result = {
        "paragraph_id": paragraph_id,
        "target_population": target_population,
        "target_raw_level": target_raw_level,
        "source_text": source_text,
        "source_score": source_score,
        "final_simplified_text": best_turn["draft_text"],
        "final_score": best_turn["predicted_score"],
        "final_eval_payload": best_turn["eval_payload"],
        "target_q1": best_turn["target_q1"],
        "target_q3": best_turn["target_q3"],
        "target_mu": best_turn["target_mu"],
        "converged_in_core_iqr": best_turn["in_core_iqr"],
        "best_iteration_selected": best_turn["iteration"],
        "total_iterations_run": len(trajectory),
        "total_latency_sec": round(time.time() - start_time, 2),
        "trajectory": [{k: v for k, v in t.items() if k != "eval_payload"} for t in trajectory],
    }
    self._append_jsonl_log({k: v for k, v in final_result.items() if k != "final_eval_payload"})
    return final_result

HebrewSimplificationAgent.diagnose_draft_in_natural_language = _patched_diagnose_draft
HebrewSimplificationAgent.run_agentic_simplification = _patched_run_agentic_simplification
