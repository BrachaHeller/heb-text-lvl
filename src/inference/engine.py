
"""ReadabilityInferenceEngine wrapping feature extraction and frozen HF artifacts."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from scipy.stats import norm

from hebrew_readability.config import (
    FINAL_FEATURE_ORDER,
    HF_DICT_REPO_ID,
    HF_MODEL_REPO_ID,
    LEVEL_DESCRIPTIONS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    read_table,
)
from hebrew_readability.hf_hub import download_model_artifacts

try:
    from features.extractor import FeatureExtractor
    from features.preprocessor import HebrewReadabilityPreprocessor
except ImportError:
    from hebrew_readability.features.extractor import FeatureExtractor
    from hebrew_readability.features.preprocessor import HebrewReadabilityPreprocessor

logger = logging.getLogger(__name__)


class ReadabilityInferenceEngine:
    """Evaluates raw Hebrew strings or pre-extracted DataFrames against calibrated levels."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        model_repo_id: str = HF_MODEL_REPO_ID,
        dict_repo_id: str = HF_DICT_REPO_ID,
        feature_extractor_fn: Optional[Callable[[str], pd.DataFrame]] = None,
        lazy_nlp: bool = True,
    ):
        paths = download_model_artifacts(repo_id=model_repo_id, local_dir=artifact_dir)

        self.preprocessor = joblib.load(paths["preprocessor"])
        self.model = joblib.load(paths["model"])

        calib_path = paths.get("calibration_table") or paths.get("calibration_csv")
        if not calib_path:
            raise KeyError(f"Calibration path missing from downloaded artifacts: {paths}")

        calib_df = read_table(
            calib_path,
            dtype={"pedagogical_level": str},
        )
        self.calibration_stats: Dict[str, Dict[str, float]] = calib_df.set_index(
            "pedagogical_level"
        ).to_dict(orient="index")

        self.dict_repo_id = dict_repo_id
        self._extractor: Optional[FeatureExtractor] = None
        self.feature_extractor_fn = feature_extractor_fn

        if not lazy_nlp and self.feature_extractor_fn is None:
            self._extractor = FeatureExtractor()
            self.feature_extractor_fn = self._extractor.extract_single_text

    def _ensure_extractor(self) -> Callable[[str], pd.DataFrame]:
        if self.feature_extractor_fn is None:
            logger.info("Initializing NLP FeatureExtractor on first text evaluation...")
            self._extractor = FeatureExtractor()
            self.feature_extractor_fn = self._extractor.extract_single_text
        return self.feature_extractor_fn

    @staticmethod
    def _levels_to_categories(levels: List[str]) -> List[str]:
        """Maps raw calibration level keys to unique pedagogical categories."""
        cats: List[str] = []
        for lvl in levels:
            norm_lvl = str(lvl).strip()
            mapped = (
                LEVEL_TO_CATEGORIES.get(norm_lvl)
                or LEVEL_TO_CATEGORIES.get(norm_lvl.lstrip("'"))
                or []
            )
            for c in mapped:
                if c not in cats:
                    cats.append(c)
        return cats

    def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
        """
        Computes the exact calibrated Core IQR (Q1, Q3) span for a target population
        by aggregating its constituent levels in self.calibration_stats.
        """
        spec = POPULATION_SPECS.get(target_population, {})
        candidate_keys = set(spec.get("raw_level_keys", []))
        target_cats = set(spec.get("engine_categories", []))

        q1_vals, q3_vals = [], []
        for lvl, stats in self.calibration_stats.items():
            norm_lvl = str(lvl).strip()
            lvl_cats = set(
                LEVEL_TO_CATEGORIES.get(norm_lvl)
                or LEVEL_TO_CATEGORIES.get(norm_lvl.lstrip("'"))
                or []
            )
            if norm_lvl in candidate_keys or norm_lvl.lstrip("'") in candidate_keys or (target_cats & lvl_cats):
                if "Q1" in stats and "Q3" in stats:
                    q1_vals.append(float(stats["Q1"]))
                    q3_vals.append(float(stats["Q3"]))

        if q1_vals and q3_vals:
            return round(min(q1_vals), 4), round(max(q3_vals), 4)

        anchor = float(spec.get("anchor_score", 5.0))
        return round(anchor - 0.45, 4), round(anchor + 0.45, 4)

    def evaluate_features_df(self, df_raw: pd.DataFrame, print_report: bool = True) -> Dict[str, Any]:
        """Scores a 1-row DataFrame of raw features and returns calibrated probabilities."""
        X_infer = self.preprocessor.transform(df_raw)
        y_pred = float(self.model.predict(X_infer)[0])

        core_matches = [
            lvl for lvl, s in self.calibration_stats.items() if s["Q1"] <= y_pred <= s["Q3"]
        ]
        ext_matches = [
            lvl for lvl, s in self.calibration_stats.items() if s["P1"] <= y_pred <= s["P99"]
        ]

        core_categories = self._levels_to_categories(core_matches)
        ext_categories = self._levels_to_categories(ext_matches)

        is_oob = len(ext_matches) == 0
        eval_levels = ext_matches if not is_oob else list(self.calibration_stats.keys())

        pdf_scores = {
            lvl: norm.pdf(
                y_pred, self.calibration_stats[lvl]["mu"], self.calibration_stats[lvl]["sigma"]
            )
            for lvl in eval_levels
        }
        total_pdf = sum(pdf_scores.values())
        sorted_probs = (
            {
                lvl: round((score / total_pdf) * 100.0, 1)
                for lvl, score in sorted(pdf_scores.items(), key=lambda x: x[1], reverse=True)
            }
            if total_pdf > 0
            else {}
        )

        top_3_natural = [
            f"{LEVEL_DESCRIPTIONS.get(lvl, lvl)} ({prob}%)"
            for lvl, prob in list(sorted_probs.items())[:3]
        ]
        if is_oob:
            summary_text = (
                f"Readability Score: {y_pred:.2f}/10 (Outside P1-P99 bounds). "
                f"Nearest pedagogical matches: {', '.join(top_3_natural)}."
            )
        else:
            summary_text = (
                f"Readability Score: {y_pred:.2f}/10. "
                f"Most suitable for: {', '.join(top_3_natural)}."
            )

        feat_dict = {
            k: round(float(v), 6) for k, v in X_infer.iloc[0].to_dict().items()
        }

        payload = {
            "raw_readability_score": round(y_pred, 4),
            "core_iqr_matches": core_matches,
            "core_iqr_categories": core_categories,
            "extended_p1_p99_matches": ext_matches,
            "extended_p1_p99_categories": ext_categories,
            "probabilities": sorted_probs,
            "is_out_of_bounds": is_oob,
            "natural_language_summary": summary_text,
            "transformed_features": feat_dict,
        }

        if print_report:
            print(f"\n[+] Raw Continuous Readability Score: {y_pred:.4f}")
            print(f"[+] Core IQR (Q1-Q3) Matches:         {', '.join(core_matches) if core_matches else 'None'}")
            print(f"[+] Core IQR Categories:              {', '.join(core_categories) if core_categories else 'None'}")
            print(f"[+] Extended (P1-P99) Matches:        {', '.join(ext_matches) if ext_matches else 'None'}")
            print("PROBABILISTIC LEVEL MAPPING")
            print("=" * 65)
            for lvl, prob in sorted_probs.items():
                desc = LEVEL_DESCRIPTIONS.get(lvl, lvl)
                print(f"▶ {lvl:<24} ({desc:<42}) | {prob:>5.1f}%")
            print(f"[Summary] {summary_text}")

        return payload

    def evaluate_text(self, raw_hebrew_text: str, print_report: bool = True) -> Dict[str, Any]:
        """Extracts linguistic features from a raw Hebrew string and predicts readability."""
        if not raw_hebrew_text or not str(raw_hebrew_text).strip():
            raise ValueError("Cannot evaluate empty or whitespace-only Hebrew text.")
        extractor_fn = self._ensure_extractor()
        df_raw = extractor_fn(str(raw_hebrew_text).strip())
        return self.evaluate_features_df(df_raw, print_report=print_report)

    def evaluate_txt_file(self, txt_filepath: str, print_report: bool = True) -> Dict[str, Any]:
        """Reads a UTF-8 .txt file and evaluates its readability."""
        if not os.path.exists(txt_filepath):
            raise FileNotFoundError(f"Input file not found: {txt_filepath}")
        with open(txt_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        return self.evaluate_text(raw_text, print_report=print_report)

        