"""Autonomous agentic simplification workflow and multi-turn trajectory logging."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd

POPULATION_SPECS: Dict[str, Dict[str, Any]] = {
    "Grade 1-3 - Elementary Beginner": {
        "engine_categories": ["Elementary - Beginner"],
        "anchor_score": 3.21,
        "medians": {"1_nikkud_ratio_log_inv": 0.00, "16_total_syllables_log": 0.485, "18_avg_syllables_per_sentence_log": 0.468, "8_stz_avg_tree_depth_per_sent_log": 0.300, "14a_zipf_rarity_tuple_log_inv": 0.403}
    },
    "Grade 4-6 - Elementary Intermediate": {
        "engine_categories": ["Elementary - Intermediate"],
        "anchor_score": 5.31,
        "medians": {"1_nikkud_ratio_log_inv": 0.949, "16_total_syllables_log": 0.567, "18_avg_syllables_per_sentence_log": 0.512, "8_stz_avg_tree_depth_per_sent_log": 0.316, "14a_zipf_rarity_tuple_log_inv": 0.425}
    },
    "Grade 7-9 - Middle School": {
        "engine_categories": ["Middle School"],
        "anchor_score": 5.69,
        "medians": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.621, "18_avg_syllables_per_sentence_log": 0.556, "8_stz_avg_tree_depth_per_sent_log": 0.351, "14a_zipf_rarity_tuple_log_inv": 0.435}
    },
    "Grade 10-12 - High School": {
        "engine_categories": ["High School"],
        "anchor_score": 5.95,
        "medians": {"1_nikkud_ratio_log_inv": 0.996, "16_total_syllables_log": 0.670, "18_avg_syllables_per_sentence_log": 0.587, "8_stz_avg_tree_depth_per_sent_log": 0.367, "14a_zipf_rarity_tuple_log_inv": 0.457}
    },
    "Technical/Academic": {
        "engine_categories": ["Technical/Academic"],
        "anchor_score": 6.80,
        "medians": {"1_nikkud_ratio_log_inv": 1.000, "16_total_syllables_log": 0.720, "18_avg_syllables_per_sentence_log": 0.640, "8_stz_avg_tree_depth_per_sent_log": 0.400, "14a_zipf_rarity_tuple_log_inv": 0.520}
    },
    "Second-language learners": {
        "engine_categories": ["Second-language - Beginner", "Second-language - Advanced"],
        "anchor_score": 5.33,
        "medians": {"1_nikkud_ratio_log_inv": 0.900, "16_total_syllables_log": 0.609, "18_avg_syllables_per_sentence_log": 0.528, "8_stz_avg_tree_depth_per_sent_log": 0.327, "14a_zipf_rarity_tuple_log_inv": 0.474}
    },
    "Intellectual Disability Accessible": {
        "engine_categories": ["Intellectual Disability"],
        "anchor_score": 4.96,
        "medians": {"1_nikkud_ratio_log_inv": 1.000, "16_total_syllables_log": 0.427, "18_avg_syllables_per_sentence_log": 0.464, "8_stz_avg_tree_depth_per_sent_log": 0.316, "14a_zipf_rarity_tuple_log_inv": 0.475}
    }
}

FEATURE_DIAGNOSTIC_RULES = {
    "1_nikkud_ratio_log_inv": {"name": "Vocalization (Nikkud)", "weight": 0.200, "too_high": "Add full Hebrew vowel diacritics (ניקוד).", "too_low": "Remove Nikkud vowel marks."},
    "16_total_syllables_log": {"name": "Paragraph Length & Volume", "weight": 0.274, "too_high": "Condense paragraph length.", "too_low": "Expand paragraph length slightly."},
    "18_avg_syllables_per_sentence_log": {"name": "Sentence Length", "weight": 0.147, "too_high": "Shorten sentences significantly with periods.", "too_low": "Combine choppy sentences."},
    "8_stz_avg_tree_depth_per_sent_log": {"name": "Syntactic Nesting", "weight": 0.096, "too_high": "Flatten nested grammatical structures.", "too_low": "Use standard connected sentence structure."},
    "14a_zipf_rarity_tuple_log_inv": {"name": "Vocabulary Rarity", "weight": 0.085, "too_high": "Replace rare words with everyday Hebrew.", "too_low": "Use age-appropriate curricular vocabulary."}
}

class HebrewSimplificationAgent:
    def __init__(self, engine, feature_prompt_cfg: Dict[str, Any], openai_client: Optional[Any] = None, model_name: str = "gpt-4o-mini", log_dir: str = "artifacts"):
        self.engine = engine
        self.prompt_cfg = feature_prompt_cfg
        self.client = openai_client
        self.model_name = model_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_log_path = os.path.join(self.log_dir, "agent_trajectories.jsonl")

        self.logger = logging.getLogger("HebrewSimplificationAgent")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            fh = logging.FileHandler(os.path.join(self.log_dir, "agent_execution.log"), encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def diagnose_draft_in_natural_language(self, eval_result: Dict[str, Any], target_population: str) -> Dict[str, Any]:
        spec = POPULATION_SPECS[target_population]
        target_cats = set(spec["engine_categories"])
        matched_iqr_cats = set(eval_result.get("core_iqr_categories", []))
        matched_ext_cats = set(eval_result.get("extended_p1_p99_categories", []))

        y_pred = eval_result["raw_readability_score"]
        score_diff = y_pred - spec["anchor_score"]
        in_core_iqr = len(target_cats.intersection(matched_iqr_cats)) > 0
        in_extended = len(target_cats.intersection(matched_ext_cats)) > 0

        draft_feats = eval_result.get("transformed_features", {})
        target_medians = spec["medians"]

        offenders = []
        for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
            if feat_key in draft_feats and feat_key in target_medians:
                delta = float(draft_feats[feat_key]) - float(target_medians[feat_key])
                if score_diff >= 0 and delta > 0.03:
                    offenders.append((abs(delta) * rule["weight"], rule["name"], rule["too_high"]))
                elif score_diff < 0 and delta < -0.04:
                    offenders.append((abs(delta) * rule["weight"], rule["name"], rule["too_low"]))

        offenders.sort(key=lambda x: x[0], reverse=True)
        top_directives = [f"- {name}: {instr}" for _, name, instr in offenders[:3]] or ["- Sentence Length: Adjust sentence length and vocabulary."]

        overall_direction = "TOO COMPLEX" if score_diff > 0 else "TOO SIMPLE"
        feedback = f"TOOL DIAGNOSTIC REPORT: Draft is {overall_direction} for '{target_population}'. Adjust:\n" + "\n".join(top_directives)

        return {
            "in_core_iqr": in_core_iqr,
            "in_extended": in_extended,
            "abs_distance_to_anchor": round(abs(score_diff), 4),
            "natural_language_feedback": feedback
        }

    def _call_llm_step(self, messages: List[Dict[str, str]], mock_responses: Optional[List[Dict[str, Any]]] = None, iteration: int = 1) -> Dict[str, Any]:
        if mock_responses is not None:
            return mock_responses[min(iteration - 1, len(mock_responses) - 1)]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, response_format={"type": "json_object"}, temperature=0.3)
        return json.loads(response.choices[0].message.content)

    def run_agentic_simplification(self, source_text: str, target_population: str, paragraph_id: str = "sample_01", max_iters: int = 3, mock_responses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        start_time = time.time()
        source_eval = self.engine.evaluate_text(source_text, print_report=False)
        
        messages = [
            {"role": "system", "content": "You are an autonomous Hebrew readability adaptation agent. Respond strictly in valid JSON."},
            {"role": "user", "content": json.dumps({"requested_target_population": target_population, "input_hebrew_text": source_text}, ensure_ascii=False)}
        ]

        trajectory, best_turn = [], None
        for iteration in range(1, max_iters + 1):
            iter_start = time.time()
            llm_json = self._call_llm_step(messages, mock_responses, iteration)
            draft_text = (llm_json.get("step_3_simplification", {}).get("simplified_text") or llm_json.get("simplified_text", "")).strip()

            eval_res = self.engine.evaluate_text(draft_text, print_report=False)
            diag = self.diagnose_draft_in_natural_language(eval_res, target_population)

            turn = {
                "iteration": iteration,
                "draft_text": draft_text,
                "predicted_score": eval_res["raw_readability_score"],
                "in_core_iqr": diag["in_core_iqr"],
                "abs_distance_to_anchor": diag["abs_distance_to_anchor"],
                "tool_natural_language_feedback": diag["natural_language_feedback"],
                "transformed_features": eval_res["transformed_features"],
                "latency_sec": round(time.time() - iter_start, 2)
            }
            trajectory.append(turn)

            if best_turn is None or (turn["in_core_iqr"] and not best_turn["in_core_iqr"]) or (turn["in_core_iqr"] == best_turn["in_core_iqr"] and turn["abs_distance_to_anchor"] < best_turn["abs_distance_to_anchor"]):
                best_turn = turn

            if diag["in_core_iqr"]:
                break

            if iteration < max_iters:
                messages.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
                messages.append({"role": "user", "content": f"{diag['natural_language_feedback']}\nRevise your draft."})

        return {
            "paragraph_id": paragraph_id,
            "target_population": target_population,
            "final_simplified_text": best_turn["draft_text"],
            "final_score": best_turn["predicted_score"],
            "converged_in_core_iqr": best_turn["in_core_iqr"],
            "total_latency_sec": round(time.time() - start_time, 2),
            "trajectory": trajectory
        }
