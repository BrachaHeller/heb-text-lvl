"""Stateful Readability Inference & Attribution Engine with Windowed Volume Normalization and LLM-Guided dicta-onnx Vocalization."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# === 1. UPDATE IMPORTS in src/inference/engine.py ===
from hebrew_readability.config import (
    CALIBRATION_DEAD_ZONE_BRIDGES,
    CONTINUOUS_VOCALIZATION_CONFIG,
    FINAL_FEATURE_ORDER,
    HF_DICT_REPO_ID,
    HF_MODEL_REPO_ID,
    K12_GRADE_TRACK_LEVELS,
    LEVEL_DESCRIPTIONS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    SPECIAL_L2_TRACK_LEVELS,
    WINDOWED_VOLUME_CONFIG,
    compute_continuous_vocalization_target,
    get_vocalization_policy,
    invert_nikkud_minmax_log,
    SUB_PARAGRAPH_SPLIT_CONFIG,
)

try:
    from features.extractor import FeatureExtractor
    from features.preprocessor import HebrewReadabilityPreprocessor
except ImportError:
    try:
        from hebrew_readability.features.extractor import FeatureExtractor
        from hebrew_readability.features.preprocessor import HebrewReadabilityPreprocessor
    except ImportError:
        FeatureExtractor = None  # type: ignore
        HebrewReadabilityPreprocessor = None  # type: ignore

try:
    from features.typographical_volume import _get_dicta_onnx, _vocalize_safely
except ImportError:
    from hebrew_readability.features.typographical_volume import (
        _get_dicta_onnx,
        _vocalize_safely,
    )

logger = logging.getLogger(__name__)

NIKKUD_RE = re.compile(r"[\u0591-\u05C7]")
HEBREW_WORD_TOKEN_RE = re.compile(r"[\u0590-\u05FF\"׳']+")


class ReadabilityInferenceEngine:
    """
    Evaluates Hebrew strings or feature DataFrames against calibrated levels,
    computes C++ TreeSHAP attributions, normalizes passage volume via Option A
    windowing, separates K-12 vs. Special/L2 probability tracks, and executes
    LLM-guided dicta-onnx vocalization plans.
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        model_repo_id: str = HF_MODEL_REPO_ID,
        dict_repo_id: str = HF_DICT_REPO_ID,
        enable_windowed_volume_norm: bool = True,
    ) -> None:
        self.artifact_dir = artifact_dir or "artifacts"
        self.model_repo_id = model_repo_id
        self.dict_repo_id = dict_repo_id
        self.enable_windowed_volume_norm = enable_windowed_volume_norm

        os.makedirs(self.artifact_dir, exist_ok=True)
        self._load_frozen_artifacts()
        self._init_nlp_extractor()

    def _resolve_artifact_path(self, filename: str, repo_id: str) -> str:
        local_candidate = os.path.join(self.artifact_dir, filename)
        if os.path.exists(local_candidate):
            return local_candidate
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _load_frozen_artifacts(self) -> None:
        scaler_path = self._resolve_artifact_path("minmax_scaler.pkl", self.model_repo_id)
        model_path = self._resolve_artifact_path(
            "xgboost_readability_regressor.pkl", self.model_repo_id
        )
        cal_path = self._resolve_artifact_path("calibration_stats.json", self.model_repo_id)

        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        with open(cal_path, "r", encoding="utf-8") as f:
            self.calibration_stats: Dict[str, Dict[str, float]] = json.load(f)

        self.max_q3 = max(
            (float(s["Q3"]) for s in self.calibration_stats.values() if "Q3" in s),
            default=6.05,
        )
        self.max_p99 = max(
            (float(s["P99"]) for s in self.calibration_stats.values() if "P99" in s),
            default=6.80,
        )
        self.min_q1 = min(
            (float(s["Q1"]) for s in self.calibration_stats.values() if "Q1" in s),
            default=2.00,
        )

        if HebrewReadabilityPreprocessor is not None:
            self.preprocessor = HebrewReadabilityPreprocessor()
            self.preprocessor.scaler = self.scaler
        else:
            self.preprocessor = None

    def _init_nlp_extractor(self) -> None:
        self.extractor = None
        if FeatureExtractor is not None:
            try:
                self.extractor = FeatureExtractor(
                    artifact_dir=self.artifact_dir,
                    dict_repo_id=self.dict_repo_id,
                )
            except TypeError:
                try:
                    self.extractor = FeatureExtractor(dict_repo_id=self.dict_repo_id)
                except TypeError:
                    self.extractor = FeatureExtractor()

            # Alias `extract_features_for_new_text` -> `extract_single_text` if needed
            if self.extractor is not None and not hasattr(self.extractor, "extract_features_for_new_text"):
                if hasattr(self.extractor, "extract_single_text"):
                    self.extractor.extract_features_for_new_text = self.extractor.extract_single_text

    @staticmethod
    def strip_nikkud(text: str) -> str:
        """Removes all Hebrew diacritical marks (U+0591 to U+05C7)."""
        return NIKKUD_RE.sub("", text or "")

    @staticmethod
    def compute_word_nikkud_ratio(text: str) -> float:
        """Computes the proportion of Hebrew words carrying at least one Nikkud mark."""
        words = HEBREW_WORD_TOKEN_RE.findall(text or "")
        if not words:
            return 0.0
        vocalized = sum(1 for w in words if bool(NIKKUD_RE.search(w)))
        return round(vocalized / len(words), 4)

    def _diacritize_full_context_onnx(self, clean_text: str) -> str:
        """
        Contextually vocalizes a Hebrew string using the real dicta-1.0.int8.onnx
        singleton from features.typographical_volume. Never injects synthetic vowels.
        """
        if not clean_text or not clean_text.strip():
            return clean_text
        bare_text = self.strip_nikkud(clean_text)
        dicta_model = _get_dicta_onnx()
        vocalized = _vocalize_safely(bare_text, dicta_model)
        if not isinstance(vocalized, str) or not vocalized.strip():
            raise RuntimeError("dicta-onnx returned an empty vocalization result.")
        return vocalized

    # Keep alias so apply_llm_guided_vocalization works regardless of which name it calls
    _vocalize_full_text_dicta = _diacritize_full_context_onnx

    def _lookup_word_zipf(self, clean_word: str) -> float:
        """Looks up approximate Zipf frequency from the engine's loaded frequency dictionary."""
        w = self.strip_nikkud(clean_word).strip("\"'׳״.,;:!?()-")
        if len(w) <= 2:
            return 6.5
        ext = getattr(self, "extractor", None)
        if ext is not None:
            for dict_attr in ("freq_dict", "lemma_freq_dict", "cascading_dict", "zipf_dict"):
                d = getattr(ext, dict_attr, None)
                if isinstance(d, dict):
                    if w in d and isinstance(d[w], (int, float)):
                        return float(d[w])
                    if (w, "NOUN") in d and isinstance(d[(w, "NOUN")], (int, float)):
                        return float(d[(w, "NOUN")])
                    if (w, "VERB") in d and isinstance(d[(w, "VERB")], (int, float)):
                        return float(d[(w, "VERB")])
        # Heuristic fallback: longer words tend to have lower Zipf frequency
        return max(1.5, 6.2 - 0.45 * len(w))

    def _get_nikkud_scaler_bounds(self) -> Tuple[float, float]:
        """Extracts exact (s_min, s_max) for 1_nikkud_ratio_log_inv from self.scaler if available."""
        s_min = CONTINUOUS_VOCALIZATION_CONFIG["default_s_min"]
        s_max = CONTINUOUS_VOCALIZATION_CONFIG["default_s_max"]
        try:
            if hasattr(self, "scaler") and self.scaler is not None:
                if hasattr(self.scaler, "data_min_") and hasattr(self.scaler, "data_max_"):
                    idx = 0
                    if hasattr(self.scaler, "feature_names_in_"):
                        names = list(self.scaler.feature_names_in_)
                        for cand in ("1_nikkud_ratio_log_inv", "1_nikkud_ratio_log", "1_nikkud_ratio"):
                            if cand in names:
                                idx = names.index(cand)
                                break
                    s_min = float(self.scaler.data_min_[idx])
                    s_max = float(self.scaler.data_max_[idx])
        except Exception:
            pass
        return s_min, s_max

    def _lookup_word_zipf(self, bare_word: str) -> float:
        """Returns the contextual or surface Zipf frequency of a bare Hebrew token (default 3.20 if OOV)."""
        clean_w = re.sub(r"[^\u05D0-\u05EA]", "", bare_word)
        if not clean_w:
            return 7.0
        for attr in ("lemma_freq_dict", "freq_dict", "surface_freq_dict", "zipf_lookup"):
            d = getattr(self, attr, None)
            if isinstance(d, dict) and clean_w in d:
                val = d[clean_w]
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, dict) and "zipf" in val:
                    return float(val["zipf"])
        # Heuristic fallback by word length when dictionary key is inflected
        return max(1.8, 5.8 - 0.45 * len(clean_w))

    def apply_llm_guided_vocalization(
        self,
        text: str,
        vocalization_plan: Optional[Dict[str, Any]] = None,
        target_raw_level: Optional[str] = None,
        current_score: Optional[float] = None,
        target_mu: Optional[float] = None,
    ) -> str:
        """
        Part A Continuous Inverse-Median & Zipf-Rarity Vocalizer:
          1. Computes r_base(L) and score-deficit-adjusted r* continuously via inverse scaler.
          2. Determines K = round(N_words * r*) words to vocalize.
          3. Prioritizes LLM-flagged homographs in `words_to_vocalize`, then lowest-Zipf rare content words.
        """
        if not text or not text.strip():
            return text

        s_min, s_max = self._get_nikkud_scaler_bounds()
        if target_mu is None and target_raw_level:
            _, _, target_mu = self.get_raw_level_iqr_bounds(str(target_raw_level))

        cont_spec = compute_continuous_vocalization_target(
            target_raw_level=target_raw_level,
            y_pred=current_score,
            mu_target=target_mu,
            s_min=s_min,
            s_max=s_max,
        )
        r_star = float(cont_spec["default_ratio"])
        zipf_thresh = float(cont_spec["zipf_familiarity_threshold"])

        plan = vocalization_plan or {}
        llm_ratio = plan.get("target_diacritic_ratio")
        if isinstance(llm_ratio, (int, float)) and current_score is None:
            # Blend empirical inverse-median r* with LLM plan within continuous bounds
            lo, hi = cont_spec["target_ratio_range"]
            r_star = float(np.clip(0.7 * r_star + 0.3 * float(llm_ratio), lo, hi))

        strip_re = re.compile(r"[\u0591-\u05C7]")
        bare_text = strip_re.sub("", text)

        # Preserve paragraph breaks (\n\n) while tokenizing whitespace words
        tokens = re.split(r"(\s+)", bare_text)
        word_indices = [i for i, tok in enumerate(tokens) if tok.strip() and re.search(r"[\u05D0-\u05EA]", tok)]
        n_words = len(word_indices)
        if n_words == 0:
            return text

        k_target = int(round(n_words * r_star))
        if k_target <= 0 and not plan.get("words_to_vocalize"):
            return bare_text

        # Run full text through dicta-onnx once to obtain contextually disambiguated diacritics
        full_vocalized = self._vocalize_full_text_dicta(bare_text)
        voc_tokens = re.split(r"(\s+)", full_vocalized)
        if len(voc_tokens) != len(tokens):
            # Fallback alignment by whitespace words
            voc_words_only = full_vocalized.split()
            if len(voc_words_only) == n_words:
                voc_tokens = list(tokens)
                for w_i, tok_idx in enumerate(word_indices):
                    voc_tokens[tok_idx] = voc_words_only[w_i]
            else:
                return full_vocalized if r_star >= 0.5 else bare_text

        if k_target >= n_words or r_star >= 0.92:
            return "".join(voc_tokens)

        flagged_raw = [
            strip_re.sub("", str(w).strip())
            for w in (plan.get("words_to_vocalize") or [])
            if str(w).strip()
        ]
        flagged_clean = {re.sub(r"[^\u05D0-\u05EA]", "", w) for w in flagged_raw if w}

        # Score each word position: (priority_tier, zipf_score, -word_len, token_idx)
        # Tier 0: explicitly flagged by LLM in words_to_vocalize
        # Tier 1: content word with Zipf <= level lexical familiarity threshold
        # Tier 2: remaining content words ordered from rarest (lowest Zipf) to most common
        ranked_positions: List[Tuple[int, float, int, int]] = []
        for tok_idx in word_indices:
            raw_tok = tokens[tok_idx]
            heb_only = re.sub(r"[^\u05D0-\u05EA]", "", raw_tok)
            zipf_val = self._lookup_word_zipf(heb_only)
            is_flagged = any(
                heb_only == f or (len(f) >= 3 and (heb_only.endswith(f) or f in heb_only))
                for f in flagged_clean
            )
            if is_flagged:
                tier = 0
            elif zipf_val <= zipf_thresh and len(heb_only) >= 3:
                tier = 1
            else:
                tier = 2
            ranked_positions.append((tier, zipf_val, -len(heb_only), tok_idx))

        ranked_positions.sort(key=lambda item: (item[0], item[1], item[2]))
        # Always allow at least up to len(flagged_clean) if selective homographs were explicitly requested
        effective_k = max(k_target, min(len(flagged_clean), max(2, int(round(0.06 * n_words))))) if r_star > 0.0 else min(len(flagged_clean), 2)
        selected_indices = {
            tok_idx for (tier, _, _, tok_idx) in ranked_positions[:effective_k]
            if k_target > 0 or tier == 0
        }

        out_tokens = [
            voc_tokens[i] if i in selected_indices else tokens[i]
            for i in range(len(tokens))
        ]
        return "".join(out_tokens)

    # === 3. ADD Part B Contextual Nikkud Damping inside _apply_windowed_volume_normalization in src/inference/engine.py ===
    def _apply_windowed_volume_normalization(
        self, X_raw: pd.DataFrame, text: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies:
          1. Option A Windowed Volume Normalization on `16_total_syllables`
          2. Part B Contextual Nikkud Damping on `1_nikkud_ratio` when partial/selective
             diacritics (0.03 <= r_raw <= 0.65) co-occur with post-Grade-2 syntax
             (avg_words_per_sent > 7.5), preventing XGBoost step-function cliffs.
        """
        X_adj = X_raw.copy()
        meta: Dict[str, Any] = {"windowed_norm_applied": False, "nikkud_damping_applied": False}
        if not getattr(self, "enable_windowed_volume_norm", True) or X_adj.empty:
            return X_adj, meta

        words = [w for w in text.split() if w.strip()]
        n_words = len(words)
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        n_sents = max(1, len(sents))
        avg_words_per_sent = n_words / float(n_sents)

        # --- 1. Windowed Volume Normalization (16_total_syllables) ---
        ref_win = float(WINDOWED_VOLUME_CONFIG.get("reference_word_window", 35.0))
        if n_words > ref_win and "16_total_syllables" in X_adj.columns:
            raw_total_syl = float(X_adj["16_total_syllables"].iloc[0])
            avg_syl_per_word = float(
                X_adj["17_avg_syllables_per_word"].iloc[0]
                if "17_avg_syllables_per_word" in X_adj.columns
                else raw_total_syl / max(1.0, float(n_words))
            )
            avg_syl_per_sent = float(
                X_adj["18_avg_syllables_per_sentence"].iloc[0]
                if "18_avg_syllables_per_sentence" in X_adj.columns
                else raw_total_syl / float(n_sents)
            )
            ref_syllables = ref_win * avg_syl_per_word
            if raw_total_syl > ref_syllables:
                is_simple_syntax = (
                    avg_words_per_sent <= float(WINDOWED_VOLUME_CONFIG["simple_sentence_word_threshold"])
                    and avg_syl_per_sent <= float(WINDOWED_VOLUME_CONFIG["simple_sentence_syllable_threshold"])
                )
                alpha = float(
                    WINDOWED_VOLUME_CONFIG["alpha_simple_syntax"]
                    if is_simple_syntax
                    else WINDOWED_VOLUME_CONFIG["alpha_complex_syntax"]
                )
                eff_syllables = ref_syllables + alpha * (raw_total_syl - ref_syllables)
                X_adj["16_total_syllables"] = X_adj["16_total_syllables"].astype(float)
                X_adj.loc[X_adj.index[0], "16_total_syllables"] = float(eff_syllables)
                meta.update(
                    {
                        "windowed_norm_applied": True,
                        "raw_total_syllables": round(raw_total_syl, 2),
                        "effective_total_syllables": round(eff_syllables, 2),
                        "alpha_used": alpha,
                    }
                )

        # --- 2. Part B: Contextual Nikkud Damping (1_nikkud_ratio) ---
        if "1_nikkud_ratio" in X_adj.columns:
            r_raw = float(X_adj["1_nikkud_ratio"].iloc[0])
            r_min = float(CONTINUOUS_VOCALIZATION_CONFIG["damping_min_raw_ratio"])
            r_max = float(CONTINUOUS_VOCALIZATION_CONFIG["damping_max_raw_ratio"])
            min_wps = float(CONTINUOUS_VOCALIZATION_CONFIG["damping_min_words_per_sent"])

            if r_min <= r_raw <= r_max and avg_words_per_sent > min_wps:
                ref_wps = float(CONTINUOUS_VOCALIZATION_CONFIG["damping_syntax_ref_words"])
                strength = float(CONTINUOUS_VOCALIZATION_CONFIG["damping_strength"])
                # Smooth syntactic maturity factor in (0, 1]
                maturity_ratio = min(1.0, (avg_words_per_sent - ref_wps) / 8.0)
                damping_factor = 1.0 - strength * maturity_ratio
                r_eff = r_raw * damping_factor
                X_adj.loc[X_adj.index[0], "1_nikkud_ratio"] = r_eff
                meta.update(
                    {
                        "nikkud_damping_applied": True,
                        "raw_nikkud_ratio": round(r_raw, 4),
                        "effective_nikkud_ratio": round(r_eff, 4),
                        "avg_words_per_sent": round(avg_words_per_sent, 2),
                    }
                )

        return X_adj, meta

    def _compute_local_shap(
        self, X_scaled: pd.DataFrame
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """Extracts exact C++ instance-level TreeSHAP contributions and dynamic base value phi_0."""
        try:
            booster = (
                self.model.get_booster()
                if hasattr(self.model, "get_booster")
                else self.model
            )
            dmat = xgb.DMatrix(
                X_scaled[FINAL_FEATURE_ORDER], feature_names=FINAL_FEATURE_ORDER
            )
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            local_shap = {
                feat: round(float(contribs[idx]), 4)
                for idx, feat in enumerate(FINAL_FEATURE_ORDER)
            }
            # Dynamically read base value phi_0 from the trained XGBoost booster
            base_val = round(float(contribs[-1]), 4)
            return local_shap, base_val
        except Exception as exc:
            self.logger.debug(f"Fallback from C++ TreeSHAP: {exc}")
            return {}, None

    def _extract_draft_evidence(
        self, raw_text: Optional[str], df_raw: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Extracts the longest sentence and lowest-Zipf lexical items from the candidate draft."""
        evidence: Dict[str, Any] = {
            "longest_sentence": "",
            "longest_sentence_words": 0,
            "rare_lexical_items": [],
        }
        if raw_text and raw_text.strip():
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text.strip()) if s.strip()]
            if sents:
                longest = max(sents, key=lambda s: len(HEBREW_WORD_TOKEN_RE.findall(s)))
                evidence["longest_sentence"] = longest
                evidence["longest_sentence_words"] = len(HEBREW_WORD_TOKEN_RE.findall(longest))

        if df_raw is not None and "19_debug_lexical_tuples" in df_raw.columns:
            raw_tuples = str(df_raw.iloc[0]["19_debug_lexical_tuples"] or "")
            matches = re.findall(r"([א-ת\"׳']+)(?:[^0-9-]*?)([1-3]\.\d+)", raw_tuples)
            if matches:
                matches_sorted = sorted(matches, key=lambda x: float(x[1]))
                seen: List[str] = []
                seen_words: Set[str] = set()
                for word, z_val in matches_sorted:
                    if word not in seen_words and len(word) > 2:
                        seen_words.add(word)
                        seen.append(f"{word} (Zipf={float(z_val):.2f})")
                    if len(seen) >= 4:
                        break
                evidence["rare_lexical_items"] = seen

        return evidence

    @staticmethod
    def _levels_to_categories(levels: List[str]) -> List[str]:
        cats: Set[str] = set()
        for lvl in levels:
            clean = str(lvl).strip()
            for c in (
                LEVEL_TO_CATEGORIES.get(clean)
                or LEVEL_TO_CATEGORIES.get(clean.lstrip("'"))
                or []
            ):
                cats.add(c)
        return sorted(cats)

    @staticmethod
    def _normalize_z_proximities(
        z_proximities: Dict[str, float], subset_levels: List[str]
    ) -> Dict[str, float]:
        sub = {lvl: z_proximities[lvl] for lvl in subset_levels if lvl in z_proximities}
        total = sum(sub.values())
        if total <= 0:
            return {}
        return {
            lvl: round((val / total) * 100.0, 1)
            for lvl, val in sorted(sub.items(), key=lambda x: x[1], reverse=True)
        }

    def get_raw_level_iqr_bounds(self, target_raw_level: str) -> Tuple[float, float, float]:
        """Returns (Q1, Q3, mu) for a specific raw level or 'academic/technical'."""
        clean = str(target_raw_level).strip()
        if clean.lower() in ("academic/technical", "technical/academic"):
            return (
                round(self.max_q3, 4),
                round(max(self.max_p99, self.max_q3 + 1.5), 4),
                round(self.max_q3 + 0.45, 4),
            )
        candidates = RAW_LEVEL_ALIASES.get(clean, [clean, f"'{clean}", clean.lstrip("'")])
        for cand in candidates:
            if cand in self.calibration_stats:
                s = self.calibration_stats[cand]
                return round(float(s["Q1"]), 4), round(float(s["Q3"]), 4), round(float(s["mu"]), 4)

        pop_name = POPULATION_SPECS.get(clean)
        if pop_name:
            q1, q3 = self.get_population_iqr_bounds(clean)
            return q1, q3, round(float(pop_name["anchor_score"]), 4)
        return 4.80, 5.45, 5.15

    def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
        """Returns union (min_Q1, max_Q3) across a broad population's raw levels."""
        spec = POPULATION_SPECS.get(
            target_population, POPULATION_SPECS["Grade 4-6 - Elementary Intermediate"]
        )
        candidate_keys = set(spec.get("raw_level_keys", []))
        target_cats = set(spec.get("engine_categories", []))
        q1_vals: List[float] = []
        q3_vals: List[float] = []

        for lvl, stats in self.calibration_stats.items():
            norm_lvl = str(lvl).strip()
            lvl_cats = set(
                LEVEL_TO_CATEGORIES.get(norm_lvl)
                or LEVEL_TO_CATEGORIES.get(norm_lvl.lstrip("'"))
                or []
            )
            if (
                norm_lvl in candidate_keys
                or norm_lvl.lstrip("'") in candidate_keys
                or (target_cats & lvl_cats)
            ):
                if "Q1" in stats and "Q3" in stats:
                    q1_vals.append(float(stats["Q1"]))
                    q3_vals.append(float(stats["Q3"]))

        if q1_vals and q3_vals:
            return round(min(q1_vals), 4), round(max(q3_vals), 4)
        anchor = float(spec.get("anchor_score", 5.0))
        return round(anchor - 0.45, 4), round(anchor + 0.45, 4)

    def evaluate_features_df(
        self,
        df_raw: pd.DataFrame,
        print_report: bool = False,
        raw_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transforms raw features (with Option A Windowed Volume Normalization),
        predicts continuous readability score Y_pred, applies Calibration Dead-Zone Bridges,
        and computes Track-Separated Probability Distributions (K-12 vs. Special/L2).
        """
        df_vol_norm, vol_meta = self._apply_windowed_volume_normalization(df_raw, raw_text=raw_text)

        if self.preprocessor is not None:
            X_trans = self._transform_df_safely(df_vol_norm)
        else:
            X_trans = df_vol_norm.copy()

        # Standardize 14a_zipf_rarity_tuple_inv key if needed
        if (
            "14a_zipf_rarity_tuple_inv" not in X_trans.columns
            and "14a_zipf_rarity_tuple_log_inv" in X_trans.columns
        ):
            X_trans["14a_zipf_rarity_tuple_inv"] = X_trans["14a_zipf_rarity_tuple_log_inv"]

        X_ordered = X_trans[FINAL_FEATURE_ORDER]
        X_for_model = self._align_features_for_model(X_trans)
        y_pred = float(self.model.predict(X_for_model)[0])
        local_shap_values, shap_base_value = self._compute_local_shap(X_for_model)

        core_matches: List[str] = [
            lvl
            for lvl, stats in self.calibration_stats.items()
            if float(stats["Q1"]) <= y_pred <= float(stats["Q3"])
        ]
        ext_matches: List[str] = [
            lvl
            for lvl, stats in self.calibration_stats.items()
            if float(stats.get("P1", stats.get("P10", stats["Q1"] - 0.5)))
            <= y_pred
            <= float(stats.get("P99", stats.get("P90", stats["Q3"] + 0.5)))
        ]

        # Apply Calibration Dead-Zone Bridges (e.g., [3.2720, 3.3898] between Grade 2 and Grade 3)
        dead_zone_bridged = False
        for bridge in CALIBRATION_DEAD_ZONE_BRIDGES:
            if float(bridge["lower_bound"]) <= y_pred <= float(bridge["upper_bound"]):
                dead_zone_bridged = True
                nearest_lvl = (
                    str(bridge["lower_level"])
                    if y_pred <= float(bridge["midpoint"])
                    else str(bridge["upper_level"])
                )
                if nearest_lvl not in core_matches:
                    core_matches.append(nearest_lvl)
                if nearest_lvl not in ext_matches:
                    ext_matches.append(nearest_lvl)
                for extra_lvl in bridge.get("always_include_levels", []):
                    if extra_lvl in self.calibration_stats and extra_lvl not in core_matches:
                        core_matches.append(extra_lvl)
                    if extra_lvl in self.calibration_stats and extra_lvl not in ext_matches:
                        ext_matches.append(extra_lvl)

        is_above_calibrated_range = bool(y_pred > self.max_q3)
        if is_above_calibrated_range:
            if "academic/technical" not in core_matches:
                core_matches.append("academic/technical")
            if "academic/technical" not in ext_matches:
                ext_matches.append("academic/technical")

        core_categories = self._levels_to_categories(core_matches)
        ext_categories = self._levels_to_categories(ext_matches)

        is_oob = len(ext_matches) == 0
        eval_levels = [
            lvl
            for lvl in (ext_matches if not is_oob else list(self.calibration_stats.keys()))
            if lvl in self.calibration_stats
        ]

        # Scale-invariant Z-score Gaussian proximity s_k(y) = exp(-0.5 * ((y - mu)/sigma)^2)
        all_z_proximities: Dict[str, float] = {
            lvl: math.exp(
                -0.5
                * (
                    (y_pred - float(self.calibration_stats[lvl]["mu"]))
                    / max(float(self.calibration_stats[lvl]["sigma"]), 1e-4)
                )
                ** 2
            )
            for lvl in self.calibration_stats
        }

        sorted_probs = self._normalize_z_proximities(all_z_proximities, eval_levels)
        core_cal_levels = [lvl for lvl in core_matches if lvl in self.calibration_stats]
        core_iqr_probabilities = (
            self._normalize_z_proximities(all_z_proximities, core_cal_levels)
            if core_cal_levels
            else sorted_probs
        )

        # Track-Separated Probability Normalization (K-12 Track vs. Special/L2 Track)
        k12_core = [
            lvl
            for lvl in core_cal_levels
            if lvl in K12_GRADE_TRACK_LEVELS or lvl.lstrip("'") in K12_GRADE_TRACK_LEVELS
        ]
        k12_ext = [
            lvl
            for lvl in eval_levels
            if lvl in K12_GRADE_TRACK_LEVELS or lvl.lstrip("'") in K12_GRADE_TRACK_LEVELS
        ]
        k12_all = [
            lvl
            for lvl in self.calibration_stats
            if lvl in K12_GRADE_TRACK_LEVELS or lvl.lstrip("'") in K12_GRADE_TRACK_LEVELS
        ]
        k12_track_probabilities = self._normalize_z_proximities(
            all_z_proximities, k12_core or k12_ext or k12_all
        )

        special_core = [
            lvl
            for lvl in core_cal_levels
            if lvl in SPECIAL_L2_TRACK_LEVELS or lvl.lstrip("'") in SPECIAL_L2_TRACK_LEVELS
        ]
        special_ext = [
            lvl
            for lvl in eval_levels
            if lvl in SPECIAL_L2_TRACK_LEVELS or lvl.lstrip("'") in SPECIAL_L2_TRACK_LEVELS
        ]
        special_all = [
            lvl
            for lvl in self.calibration_stats
            if lvl in SPECIAL_L2_TRACK_LEVELS or lvl.lstrip("'") in SPECIAL_L2_TRACK_LEVELS
        ]
        special_track_probabilities = self._normalize_z_proximities(
            all_z_proximities, special_core or special_ext or special_all
        )

        transformed_features = {
            feat: round(float(X_ordered.iloc[0][feat]), 4) for feat in FINAL_FEATURE_ORDER
        }
        draft_evidence = self._extract_draft_evidence(raw_text, df_raw)

        if print_report:
            print(f"\n[+] Raw Continuous Readability Score: {y_pred:.4f}")
            print(f"[+] Core IQR (Q1-Q3) Matches:         {', '.join(core_matches) if core_matches else 'None'}")
            print(f"[+] Core IQR Categories:              {', '.join(core_categories) if core_categories else 'None'}")
            print(f"[+] K-12 Track Probabilities:         {k12_track_probabilities}")

        return {
            "raw_readability_score": round(y_pred, 4),
            "core_iqr_matches": core_matches,
            "core_iqr_categories": core_categories,
            "extended_p1_p99_matches": ext_matches,
            "extended_p1_p99_categories": ext_categories,
            "probabilities": sorted_probs,
            "core_iqr_probabilities": core_iqr_probabilities,
            "k12_track_probabilities": k12_track_probabilities,
            "special_track_probabilities": special_track_probabilities,
            "is_above_calibrated_range": is_above_calibrated_range,
            "dead_zone_bridged": dead_zone_bridged,
            "windowed_volume_meta": vol_meta,
            "transformed_features": transformed_features,
            "local_shap_values": local_shap_values,
            "shap_base_value": shap_base_value,
            "draft_evidence": draft_evidence,
        }

    def evaluate_text(self, text: str, print_report: bool = False) -> Dict[str, Any]:
        # 1. Empty-text guard (with Change 4: shap_base_value = None)
        if not text or not text.strip():
            return {
                "raw_readability_score": float("nan"),
                "core_iqr_matches": [],
                "core_iqr_categories": [],
                "extended_p1_p99_matches": [],
                "extended_p1_p99_categories": [],
                "probabilities": {},
                "core_iqr_probabilities": {},
                "k12_track_probabilities": {},
                "special_track_probabilities": {},
                "is_above_calibrated_range": False,
                "dead_zone_bridged": False,
                "windowed_volume_meta": {},
                "transformed_features": {},
                "local_shap_values": {},
                "shap_base_value": None,
                "draft_evidence": {},
                "error": "Empty or whitespace-only text provided.",
            }

        # 2. Sub-paragraph chunk scoring for multi-paragraph strings separated by \n\n
        min_chunk_words = int(SUB_PARAGRAPH_SPLIT_CONFIG.get("min_sub_paragraph_words", 15))
        raw_chunks = [c.strip() for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
        valid_chunks = [c for c in raw_chunks if len(c.split()) >= min_chunk_words]

        if len(raw_chunks) > 1 and len(valid_chunks) >= 2:
            # Each chunk `c` has no \n\n, so self.evaluate_text(c) runs the single-chunk path below
            chunk_evals = [self.evaluate_text(c, print_report=False) for c in valid_chunks]
            weights = [max(1, len(c.split())) for c in valid_chunks]
            total_w = float(sum(weights))

            agg_score = round(
                sum(float(ev["raw_readability_score"]) * w for ev, w in zip(chunk_evals, weights))
                / total_w,
                4,
            )
            agg_features = {
                feat: round(
                    sum(float(ev.get("transformed_features", {}).get(feat, 0.0)) * w for ev, w in zip(chunk_evals, weights))
                    / total_w,
                    4,
                )
                for feat in FINAL_FEATURE_ORDER
            }
            agg_shap = {
                feat: round(
                    sum(float(ev.get("local_shap_values", {}).get(feat, 0.0)) * w for ev, w in zip(chunk_evals, weights))
                    / total_w,
                    4,
                )
                for feat in FINAL_FEATURE_ORDER
                if any(feat in ev.get("local_shap_values", {}) for ev in chunk_evals)
            }

            # Pick the sub-paragraph whose score is closest to agg_score for level/track dicts,
            # or re-match agg_score against calibration_stats:
            closest_ev = min(
                chunk_evals,
                key=lambda ev: abs(float(ev["raw_readability_score"]) - agg_score),
            )
            merged_res = dict(closest_ev)
            merged_res["raw_readability_score"] = agg_score
            merged_res["transformed_features"] = agg_features
            merged_res["local_shap_values"] = agg_shap
            merged_res["shap_base_value"] = chunk_evals[0].get("shap_base_value")
            merged_res["draft_evidence"] = self._extract_draft_evidence(text)
            merged_res["sub_paragraph_meta"] = {
                "num_sub_paragraphs": len(valid_chunks),
                "sub_paragraph_words": weights,
                "sub_paragraph_scores": [round(float(ev["raw_readability_score"]), 4) for ev in chunk_evals],
            }
            return merged_res

        if self.extractor is None:
            raise RuntimeError("FeatureExtractor is not initialized.")

        if hasattr(self.extractor, "extract_single_text"):
            df_raw = self.extractor.extract_single_text(text)
        else:
            df_raw = self.extractor.extract_features_for_new_text(text)
        return self.evaluate_features_df(df_raw, print_report=print_report, raw_text=text)


# === V3 REAL ONNX & CASCADING ZIPF PATCH ===
try:
    from features.typographical_volume import _get_dicta_onnx, _vocalize_safely
except ImportError:
    from hebrew_readability.features.typographical_volume import _get_dicta_onnx, _vocalize_safely

try:
    from hebrew_readability.hf_hub import download_model_artifacts
except ImportError:
    download_model_artifacts = None

_orig_load_frozen_artifacts = ReadabilityInferenceEngine._load_frozen_artifacts

def _ensure_fitted_preprocessor(self) -> None:
    """Loads the fitted HebrewReadabilityPreprocessor or binds scaler_ from self.scaler."""
    if download_model_artifacts is not None:
        try:
            paths = download_model_artifacts(repo_id=self.model_repo_id, local_dir=self.artifact_dir)
            if "preprocessor" in paths and os.path.exists(paths["preprocessor"]):
                loaded_prep = joblib.load(paths["preprocessor"])
                if hasattr(loaded_prep, "scaler_") and loaded_prep.scaler_ is not None:
                    self.preprocessor = loaded_prep
                    self.scaler = loaded_prep.scaler_
                    return
        except Exception as exc:
            logger.debug("download_model_artifacts preprocessor check: %s", exc)

    if self.preprocessor is not None and self.scaler is not None:
        self.preprocessor.scaler = self.scaler
        self.preprocessor.scaler_ = self.scaler

def _patched_load_frozen_artifacts(self) -> None:
    _orig_load_frozen_artifacts(self)
    self._ensure_fitted_preprocessor()

@staticmethod
def _populate_canonical_feature_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all 11 FINAL_FEATURE_ORDER column names (e.g., '10_dct_ratio_adj_log')
    and their non-log/log counterparts exist on the transformed DataFrame by matching
    each feature's unique numeric prefix ('1_', '7_', ..., '18_').
    """
    df_out = df_in.copy()
    prefix_to_col = {str(c).split("_")[0] + "_": c for c in df_out.columns if "_" in str(c)}
    for canon_col in FINAL_FEATURE_ORDER:
        if canon_col not in df_out.columns:
            pref = canon_col.split("_")[0] + "_"
            if pref in prefix_to_col:
                df_out[canon_col] = df_out[prefix_to_col[pref]]
    return df_out

def _transform_df_safely(self, df_vol_norm: pd.DataFrame) -> pd.DataFrame:
    """Runs preprocessor.transform() and guarantees all FINAL_FEATURE_ORDER keys exist."""
    if getattr(self, "preprocessor", None) is not None:
        if getattr(self.preprocessor, "scaler_", None) is None:
            self._ensure_fitted_preprocessor()
        try:
            X_tr = self.preprocessor.transform(df_vol_norm)
            return self._populate_canonical_feature_aliases(X_tr)
        except Exception as exc:
            logger.debug("Direct scaler fallback triggered (%s)", exc)

    df_t = df_vol_norm.copy()
    scaler_cols = list(getattr(self.scaler, "feature_names_in_", []))
    for col in [c[:-4] for c in scaler_cols if c.endswith("_log")]:
        if col in df_t.columns:
            df_t[f"{col}_log"] = np.log1p(df_t[col].astype(float).clip(lower=0.0))
    if scaler_cols:
        for c in scaler_cols:
            if c not in df_t.columns:
                df_t[c] = 0.0
        df_t[scaler_cols] = self.scaler.transform(df_t[scaler_cols])
    for inv_col in ("1_nikkud_ratio_log", "14a_zipf_rarity_tuple", "14a_zipf_rarity_tuple_log"):
        if inv_col in df_t.columns:
            df_t[f"{inv_col}_inv"] = 1.0 - df_t[inv_col].astype(float)
    return self._populate_canonical_feature_aliases(df_t)

def _align_features_for_model(self, X_df: pd.DataFrame) -> pd.DataFrame:
    """Aligns DataFrame column names and order to the exact schema expected by self.model."""
    df_aligned = self._populate_canonical_feature_aliases(X_df)
    booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
    expected_cols = (
        list(getattr(booster, "feature_names", None) or [])
        or list(getattr(self.model, "feature_names_in_", None) or [])
        or list(FINAL_FEATURE_ORDER)
    )
    prefix_to_col = {str(c).split("_")[0] + "_": c for c in df_aligned.columns if "_" in str(c)}
    for exp_c in expected_cols:
        if exp_c not in df_aligned.columns:
            pref = str(exp_c).split("_")[0] + "_"
            if pref in prefix_to_col:
                df_aligned[exp_c] = df_aligned[prefix_to_col[pref]]
    return df_aligned[expected_cols]

def _patched_compute_local_shap(
    self, X_scaled: pd.DataFrame
) -> Tuple[Dict[str, float], Optional[float]]:
    """Computes C++ TreeSHAP on model-aligned features and maps keys to FINAL_FEATURE_ORDER."""
    try:
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        X_for_model = self._align_features_for_model(X_scaled)
        dmat = xgb.DMatrix(X_for_model, feature_names=list(X_for_model.columns))
        contribs = booster.predict(dmat, pred_contribs=True)[0]
        prefix_to_canon = {f.split("_")[0] + "_": f for f in FINAL_FEATURE_ORDER}
        local_shap: Dict[str, float] = {}
        for idx, col_name in enumerate(X_for_model.columns):
            pref = str(col_name).split("_")[0] + "_"
            canon_key = prefix_to_canon.get(pref, str(col_name))
            local_shap[canon_key] = round(float(contribs[idx]), 4)
        base_val = round(float(contribs[-1]), 4)
        return local_shap, base_val
    except Exception as exc:
        logger.debug("Fallback from C++ TreeSHAP: %s", exc)
        return {}, None

_orig_windowed_vol_norm = ReadabilityInferenceEngine._apply_windowed_volume_normalization

def _compat_apply_windowed_volume_normalization(
    self,
    X_raw: pd.DataFrame,
    text: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    resolved_text = str(text if text is not None else (raw_text or ""))
    return _orig_windowed_vol_norm(self, X_raw, resolved_text)

def _real_diacritize_full_context_onnx(self, clean_text: str) -> str:
    if not clean_text or not clean_text.strip():
        return clean_text
    bare_text = self.strip_nikkud(clean_text)
    dicta_model = _get_dicta_onnx()
    vocalized = _vocalize_safely(bare_text, dicta_model)
    if not isinstance(vocalized, str) or not vocalized.strip():
        raise RuntimeError("dicta-onnx returned an empty vocalization string.")
    return vocalized

def _build_lemma_zipf_index(self) -> dict:
    if getattr(self, "_cached_lemma_zipf_index", None) is not None:
        return self._cached_lemma_zipf_index

    idx = {}
    ext = getattr(self, "extractor", None)
    if ext is not None:
        for attr_val in vars(ext).values():
            if isinstance(attr_val, dict):
                for k, v in attr_val.items():
                    if isinstance(v, dict) and k in ("tuple", "lemma", "pos", "dict_tuple", "dict_lemma"):
                        for sub_k, sub_v in v.items():
                            w_key = sub_k[0] if isinstance(sub_k, tuple) else str(sub_k)
                            z = sub_v.get("zipf") if isinstance(sub_v, dict) else sub_v
                            if isinstance(z, (int, float)):
                                idx[w_key] = max(idx.get(w_key, 0.0), float(z))
                    else:
                        w_key = k[0] if isinstance(k, tuple) else str(k)
                        z = v.get("zipf") if isinstance(v, dict) else v
                        if isinstance(z, (int, float)):
                            idx[w_key] = max(idx.get(w_key, 0.0), float(z))
            elif isinstance(attr_val, pd.DataFrame) and "lemma" in attr_val.columns:
                z_col = next((c for c in ("zipf", "zipf_freq", "zipf_score") if c in attr_val.columns), None)
                if z_col:
                    for lem, z_val in zip(attr_val["lemma"].astype(str), attr_val[z_col]):
                        if pd.notna(z_val):
                            idx[lem] = max(idx.get(lem, 0.0), float(z_val))

    self._cached_lemma_zipf_index = idx
    return idx

_orig_extract_draft_evidence = ReadabilityInferenceEngine._extract_draft_evidence

def _real_extract_draft_evidence(self, raw_text=None, df_raw=None):
    if df_raw is not None and "19_debug_lexical_tuples" in df_raw.columns:
        raw_tuples = str(df_raw.iloc[0]["19_debug_lexical_tuples"] or "")
        matches = re.findall(r"([\u05D0-\u05EA\u05F3\u05F4']+)(?:[^0-9-]*?)([0-9]+\.[0-9]+)", raw_tuples)
        if matches:
            self._last_tuple_zipf_map = {
                re.sub(r"[^\u05D0-\u05EA]", "", w): float(z)
                for w, z in matches
                if re.sub(r"[^\u05D0-\u05EA]", "", w)
            }
    return _orig_extract_draft_evidence(self, raw_text=raw_text, df_raw=df_raw)

def _ensure_dictabert_lexical_parse(self, bare_text: str) -> None:
    if not bare_text or not bare_text.strip():
        return
    if getattr(self, "_last_parsed_bare_text", None) == bare_text:
        return
    if self.extractor is not None:
        if hasattr(self.extractor, "extract_single_text"):
            df_raw = self.extractor.extract_single_text(bare_text)
        else:
            df_raw = self.extractor.extract_features_for_new_text(bare_text)
        self._last_parsed_bare_text = bare_text
        self._extract_draft_evidence(raw_text=bare_text, df_raw=df_raw)

def _dictabert_lookup_word_zipf(self, bare_word: str) -> float:
    clean_w = re.sub(r"[^\u05D0-\u05EA]", "", bare_word or "")
    if not clean_w:
        return 7.0
    if len(clean_w) <= 2:
        return 6.5

    tuple_map = getattr(self, "_last_tuple_zipf_map", None) or {}
    if clean_w in tuple_map:
        return float(tuple_map[clean_w])
    for parsed_lemma, z_val in tuple_map.items():
        if len(parsed_lemma) >= 3 and parsed_lemma in clean_w:
            return float(z_val)

    lemma_idx = self._build_lemma_zipf_index()
    if clean_w in lemma_idx:
        return float(lemma_idx[clean_w])

    return 3.0

_orig_apply_voc = ReadabilityInferenceEngine.apply_llm_guided_vocalization

def _dictabert_apply_llm_guided_vocalization(self, text: str, *args, **kwargs) -> str:
    if not getattr(self, "use_dicta_onnx_for_agent", True):
        return text
    if text and text.strip():
        self._ensure_dictabert_lexical_parse(self.strip_nikkud(text))
    return _orig_apply_voc(self, text, *args, **kwargs)

ReadabilityInferenceEngine._ensure_fitted_preprocessor = _ensure_fitted_preprocessor
ReadabilityInferenceEngine._load_frozen_artifacts = _patched_load_frozen_artifacts
ReadabilityInferenceEngine._populate_canonical_feature_aliases = _populate_canonical_feature_aliases
ReadabilityInferenceEngine._transform_df_safely = _transform_df_safely
ReadabilityInferenceEngine._align_features_for_model = _align_features_for_model
ReadabilityInferenceEngine._compute_local_shap = _patched_compute_local_shap
ReadabilityInferenceEngine._apply_windowed_volume_normalization = _compat_apply_windowed_volume_normalization
ReadabilityInferenceEngine._diacritize_full_context_onnx = _real_diacritize_full_context_onnx
ReadabilityInferenceEngine._vocalize_full_text_dicta = _real_diacritize_full_context_onnx
ReadabilityInferenceEngine._build_lemma_zipf_index = _build_lemma_zipf_index
ReadabilityInferenceEngine._extract_draft_evidence = _real_extract_draft_evidence
ReadabilityInferenceEngine._ensure_dictabert_lexical_parse = _ensure_dictabert_lexical_parse
ReadabilityInferenceEngine._lookup_word_zipf = _dictabert_lookup_word_zipf
ReadabilityInferenceEngine._get_word_zipf = _dictabert_lookup_word_zipf
ReadabilityInferenceEngine.apply_llm_guided_vocalization = _dictabert_apply_llm_guided_vocalization
