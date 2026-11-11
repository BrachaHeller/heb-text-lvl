"""Centralized configuration, dead-zone bridges, track definitions, per-level SHAP tables, continuous vocalization, and I/O helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Hugging Face Repositories & Default Artifact Filenames
HF_MODEL_REPO_ID = "GolDL/hebrew-readability-classifier"
HF_DICT_REPO_ID = "GolDL/hebrew-lemma-frequency-dictionary"
HF_DATASET_REPO_ID = "GolDL/heb-readability-txt"
RAW_DATASET_FILENAME = "heb_paragraphs_dataset.parquet"
DICT_PARQUET_FILENAME = "frequency_dictionary_clean.parquet"
DICT_CSV_FILENAME = "frequency_dictionary_clean.csv"
DICT_ONNX_URL = "https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.int8.onnx"

# Default OpenAI Model Across the Entire Project
DEFAULT_LLM_MODEL = "gpt-4o"

# Non-numeric metadata columns
STRING_METADATA_COLS: List[str] = [
    "paragraph_id",
    "source_file",
    "category",
    "level",
    "passage_title",
    "raw_text",
    "2_cleaned_text",
    "15_vocalized_text",
    "19_debug_lexical_tuples",
]

# 11 Purged Transformed Features (Single Source of Truth)
FINAL_FEATURE_ORDER: List[str] = [
    "1_nikkud_ratio_log_inv",
    "7_stz_avg_clauses_per_sent_log",
    "8_stz_avg_tree_depth_per_sent_log",
    "9_stz_ratio_passive_verbs_log",
    "10_dct_ratio_adj_log",
    "11_dct_ratio_adv_log",
    "12_dct_ratio_pron_log",
    "14a_zipf_rarity_tuple_inv",
    "16_total_syllables_log",
    "17_avg_syllables_per_word",
    "18_avg_syllables_per_sentence_log",
]

# Calibration Dead-Zone Bridge between Grade 2 Q3 (3.2720) and Grade 3 Q1 (3.3898)
CALIBRATION_DEAD_ZONE_BRIDGES: List[Dict[str, Any]] = [
    {
        "name": "elementary_beginner_g2_g3_bridge",
        "lower_bound": 3.2720,
        "upper_bound": 3.3898,
        "midpoint": 3.3309,
        "lower_level": "2",
        "upper_level": "3",
        "always_include_levels": ["'2-4", "2-4"],
        "categories": ["Elementary - Beginner"],
    }
]

# Windowed Volume Normalization Config (Option A — Zero Content Loss)
WINDOWED_VOLUME_CONFIG: Dict[str, float] = {
    "reference_word_window": 35.0,
    "simple_sentence_word_threshold": 12.0,
    "simple_sentence_syllable_threshold": 26.0,
    "alpha_simple_syntax": 0.15,
    "alpha_complex_syntax": 0.35,
}

# Sub-Paragraph Splitting Config (\n\n) for Long Lower-Grade Passages
SUB_PARAGRAPH_SPLIT_LEVELS: Set[str] = {
    "1", "2", "3", "'2-4", "2-4", "4", "5",
    "Hadshon", "Olim_news", "intellectual_disability",
}

SUB_PARAGRAPH_SPLIT_CONFIG: Dict[str, Any] = {
    "max_single_paragraph_words": 45,
    "min_sub_paragraph_words": 15,
    "target_levels": SUB_PARAGRAPH_SPLIT_LEVELS,
}


def get_sub_paragraph_split_instruction(
    target_raw_level: Optional[str], word_count: int
) -> str:
    """Returns explicit sub-paragraph chunking instructions (\\n\\n) when a passage > 45 words targets lower levels."""
    clean = str(target_raw_level or "").strip()
    is_split_level = (
        clean in SUB_PARAGRAPH_SPLIT_LEVELS
        or clean.lstrip("'") in SUB_PARAGRAPH_SPLIT_LEVELS
    )
    max_words = int(SUB_PARAGRAPH_SPLIT_CONFIG["max_single_paragraph_words"])
    if not is_split_level or word_count <= max_words:
        return ""
    return (
        f"STRUCTURAL SUB-PARAGRAPH SPLITTING RULE (Source has {word_count} words > {max_words}): "
        f"Because target level '{clean}' requires low per-paragraph cognitive volume while preserving "
        f"100% of factual propositions, divide 'simplified_text' into 2 coherent sub-paragraphs "
        f"separated by a double newline ('\\n\\n', ~20-35 words per sub-paragraph). "
        f"Repeat the explicit subject noun at the beginning of the second sub-paragraph instead of "
        f"starting Paragraph 2 with an ambiguous pronoun ('הוא', 'היא', 'הם')."
    )


# Track Separation Sets
K12_GRADE_TRACK_LEVELS: Set[str] = {
    "1", "2", "3", "'2-4", "2-4", "4", "5", "6",
    "'4-9", "4-9", "7", "8", "9", "10", "11", "12", "Bagrut",
}

SPECIAL_L2_TRACK_LEVELS: Set[str] = {
    "Hadshon",
    "Olim_news",
    "Olim_ragil",
    "מבחני אולפן - רגיל",
    "Olim_mitkadmim",
    "מבחני אולפן - מתקדמים",
    "Bagrut_Olim",
    "intellectual_disability",
    "hearing_impaired",
    "Bagrut_hearing_impaired",
}

# Bidirectional Alias Map Across Calibration & SHAP Tables
RAW_LEVEL_ALIASES: Dict[str, List[str]] = {
    "1": ["1"],
    "2": ["2"],
    "3": ["3"],
    "'2-4": ["'2-4", "2-4"],
    "2-4": ["'2-4", "2-4"],
    "4": ["4"],
    "5": ["5"],
    "6": ["6"],
    "'4-9": ["'4-9", "4-9"],
    "4-9": ["'4-9", "4-9"],
    "7": ["7", "'4-9", "4-9"],
    "8": ["8"],
    "9": ["9"],
    "10": ["10", "11"],
    "11": ["11"],
    "12": ["12"],
    "Bagrut": ["Bagrut", "11", "12"],
    "Hadshon": ["Hadshon", "Olim_news"],
    "Olim_news": ["Hadshon", "Olim_news"],
    "Olim_ragil": ["Olim_ragil", "מבחני אולפן - רגיל"],
    "מבחני אולפן - רגיל": ["Olim_ragil", "מבחני אולפן - רגיל"],
    "Olim_mitkadmim": ["Olim_mitkadmim", "מבחני אולפן - מתקדמים"],
    "מבחני אולפן - מתקדמים": ["Olim_mitkadmim", "מבחני אולפן - מתקדמים"],
    "Bagrut_Olim": ["Bagrut_Olim"],
    "hearing_impaired": ["hearing_impaired", "Bagrut_hearing_impaired"],
    "Bagrut_hearing_impaired": ["hearing_impaired", "Bagrut_hearing_impaired"],
    "intellectual_disability": ["intellectual_disability"],
    "academic/technical": ["academic/technical"],
}

# Empirical Calibrated Readability Anchors (mu_L from calibration_stats.json)
EMPIRICAL_LEVEL_ANCHOR_MU_FALLBACK: Dict[str, float] = {
    "1": 2.9800,
    "2": 3.1200,
    "'2-4": 3.5500,
    "2-4": 3.5500,
    "3": 3.6800,
    "Hadshon": 4.1500,
    "Olim_news": 4.1500,
    "intellectual_disability": 4.9200,
    "Olim_ragil": 5.1800,
    "מבחני אולפן - רגיל": 5.1800,
    "4": 5.2900,
    "5": 5.3100,
    "6": 5.4400,
    "'4-9": 5.5100,
    "4-9": 5.5100,
    "7": 5.5200,
    "8": 5.5600,
    "Olim_mitkadmim": 5.5800,
    "מבחני אולפן - מתקדמים": 5.5800,
    "hearing_impaired": 5.6200,
    "Bagrut_hearing_impaired": 5.6200,
    "9": 5.6400,
    "10": 5.6900,
    "Bagrut_Olim": 5.7000,
    "11": 5.7100,
    "Bagrut": 5.7800,
    "12": 5.8200,
    "academic/technical": 6.5000,
}

GRADE_NUMERIC_MAP: Dict[str, float] = EMPIRICAL_LEVEL_ANCHOR_MU_FALLBACK
LEVEL_NUMERIC_GRADE: Dict[str, float] = EMPIRICAL_LEVEL_ANCHOR_MU_FALLBACK


def build_empirical_level_anchor_map(
    calibration_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Dynamically maps every pedagogical level to its empirical calibrated mean (mu_L)."""
    anchor_map: Dict[str, float] = dict(EMPIRICAL_LEVEL_ANCHOR_MU_FALLBACK)
    if not calibration_stats:
        return anchor_map

    for lvl, stats in calibration_stats.items():
        if isinstance(stats, dict) and "mu" in stats and stats["mu"] is not None:
            mu_val = round(float(stats["mu"]), 4)
            anchor_map[str(lvl).strip()] = mu_val
            anchor_map[str(lvl).strip().lstrip("'")] = mu_val
            for alias in RAW_LEVEL_ALIASES.get(str(lvl).strip(), []):
                anchor_map[alias] = mu_val

    max_q3 = max(
        (float(s["Q3"]) for s in calibration_stats.values() if isinstance(s, dict) and "Q3" in s),
        default=6.05,
    )
    anchor_map["academic/technical"] = round(max_q3 + 0.45, 4)
    return anchor_map


def compute_scale_normalized_iqr_distance(
    y_pred: float, q1: float, q3: float
) -> float:
    """Computes Scale-Normalized IQR Distance d_IQR(Y, L) = max(0, Q1 - Y, Y - Q3) / (Q3 - Q1)."""
    if pd.isna(y_pred) or pd.isna(q1) or pd.isna(q3):
        return float("nan")
    iqr_width = float(q3) - float(q1)
    if iqr_width <= 1e-6:
        return 0.0
    raw_gap = max(0.0, float(q1) - float(y_pred), float(y_pred) - float(q3))
    return round(raw_gap / iqr_width, 4)


scale_normalized_iqr_distance = compute_scale_normalized_iqr_distance

# Granular Per-Raw-Level Transformed Feature Medians
RAW_LEVEL_MEDIANS: Dict[str, Dict[str, float]] = {
    "1": {"1_nikkud_ratio_log_inv": 0.000, "16_total_syllables_log": 0.440, "18_avg_syllables_per_sentence_log": 0.435, "10_dct_ratio_adj_log": 0.150, "8_stz_avg_tree_depth_per_sent_log": 0.275, "14a_zipf_rarity_tuple_inv": 0.385, "7_stz_avg_clauses_per_sent_log": 0.190, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.130, "17_avg_syllables_per_word": 0.295, "12_dct_ratio_pron_log": 0.295},
    "2": {"1_nikkud_ratio_log_inv": 0.000, "16_total_syllables_log": 0.475, "18_avg_syllables_per_sentence_log": 0.460, "10_dct_ratio_adj_log": 0.175, "8_stz_avg_tree_depth_per_sent_log": 0.295, "14a_zipf_rarity_tuple_inv": 0.400, "7_stz_avg_clauses_per_sent_log": 0.215, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.145, "17_avg_syllables_per_word": 0.308, "12_dct_ratio_pron_log": 0.285},
    "3": {"1_nikkud_ratio_log_inv": 0.320, "16_total_syllables_log": 0.515, "18_avg_syllables_per_sentence_log": 0.495, "10_dct_ratio_adj_log": 0.205, "8_stz_avg_tree_depth_per_sent_log": 0.325, "14a_zipf_rarity_tuple_inv": 0.425, "7_stz_avg_clauses_per_sent_log": 0.240, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.155, "17_avg_syllables_per_word": 0.320, "12_dct_ratio_pron_log": 0.270},
    "'2-4": {"1_nikkud_ratio_log_inv": 0.280, "16_total_syllables_log": 0.505, "18_avg_syllables_per_sentence_log": 0.485, "10_dct_ratio_adj_log": 0.195, "8_stz_avg_tree_depth_per_sent_log": 0.315, "14a_zipf_rarity_tuple_inv": 0.415, "7_stz_avg_clauses_per_sent_log": 0.230, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.150, "17_avg_syllables_per_word": 0.315, "12_dct_ratio_pron_log": 0.275},
    "2-4": {"1_nikkud_ratio_log_inv": 0.280, "16_total_syllables_log": 0.505, "18_avg_syllables_per_sentence_log": 0.485, "10_dct_ratio_adj_log": 0.195, "8_stz_avg_tree_depth_per_sent_log": 0.315, "14a_zipf_rarity_tuple_inv": 0.415, "7_stz_avg_clauses_per_sent_log": 0.230, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.150, "17_avg_syllables_per_word": 0.315, "12_dct_ratio_pron_log": 0.275},
    "4": {"1_nikkud_ratio_log_inv": 0.945, "16_total_syllables_log": 0.535, "18_avg_syllables_per_sentence_log": 0.520, "10_dct_ratio_adj_log": 0.225, "8_stz_avg_tree_depth_per_sent_log": 0.350, "14a_zipf_rarity_tuple_inv": 0.440, "7_stz_avg_clauses_per_sent_log": 0.255, "9_stz_ratio_passive_verbs_log": 0.020, "11_dct_ratio_adv_log": 0.165, "17_avg_syllables_per_word": 0.335, "12_dct_ratio_pron_log": 0.250},
    "5": {"1_nikkud_ratio_log_inv": 0.965, "16_total_syllables_log": 0.575, "18_avg_syllables_per_sentence_log": 0.555, "10_dct_ratio_adj_log": 0.255, "8_stz_avg_tree_depth_per_sent_log": 0.380, "14a_zipf_rarity_tuple_inv": 0.460, "7_stz_avg_clauses_per_sent_log": 0.280, "9_stz_ratio_passive_verbs_log": 0.055, "11_dct_ratio_adv_log": 0.175, "17_avg_syllables_per_word": 0.350, "12_dct_ratio_pron_log": 0.235},
    "6": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.605, "18_avg_syllables_per_sentence_log": 0.580, "10_dct_ratio_adj_log": 0.275, "8_stz_avg_tree_depth_per_sent_log": 0.400, "14a_zipf_rarity_tuple_inv": 0.475, "7_stz_avg_clauses_per_sent_log": 0.300, "9_stz_ratio_passive_verbs_log": 0.080, "11_dct_ratio_adv_log": 0.180, "17_avg_syllables_per_word": 0.365, "12_dct_ratio_pron_log": 0.225},
    "'4-9": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.615, "18_avg_syllables_per_sentence_log": 0.590, "10_dct_ratio_adj_log": 0.285, "8_stz_avg_tree_depth_per_sent_log": 0.410, "14a_zipf_rarity_tuple_inv": 0.480, "7_stz_avg_clauses_per_sent_log": 0.310, "9_stz_ratio_passive_verbs_log": 0.095, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.370, "12_dct_ratio_pron_log": 0.220},
    "4-9": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.615, "18_avg_syllables_per_sentence_log": 0.590, "10_dct_ratio_adj_log": 0.285, "8_stz_avg_tree_depth_per_sent_log": 0.410, "14a_zipf_rarity_tuple_inv": 0.480, "7_stz_avg_clauses_per_sent_log": 0.310, "9_stz_ratio_passive_verbs_log": 0.095, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.370, "12_dct_ratio_pron_log": 0.220},
    "7": {"1_nikkud_ratio_log_inv": 0.980, "16_total_syllables_log": 0.630, "18_avg_syllables_per_sentence_log": 0.605, "10_dct_ratio_adj_log": 0.300, "8_stz_avg_tree_depth_per_sent_log": 0.425, "14a_zipf_rarity_tuple_inv": 0.490, "7_stz_avg_clauses_per_sent_log": 0.320, "9_stz_ratio_passive_verbs_log": 0.110, "11_dct_ratio_adv_log": 0.190, "17_avg_syllables_per_word": 0.380, "12_dct_ratio_pron_log": 0.215},
    "8": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.650, "18_avg_syllables_per_sentence_log": 0.625, "10_dct_ratio_adj_log": 0.315, "8_stz_avg_tree_depth_per_sent_log": 0.440, "14a_zipf_rarity_tuple_inv": 0.505, "7_stz_avg_clauses_per_sent_log": 0.335, "9_stz_ratio_passive_verbs_log": 0.125, "11_dct_ratio_adv_log": 0.195, "17_avg_syllables_per_word": 0.390, "12_dct_ratio_pron_log": 0.205},
    "9": {"1_nikkud_ratio_log_inv": 0.990, "16_total_syllables_log": 0.670, "18_avg_syllables_per_sentence_log": 0.640, "10_dct_ratio_adj_log": 0.330, "8_stz_avg_tree_depth_per_sent_log": 0.455, "14a_zipf_rarity_tuple_inv": 0.520, "7_stz_avg_clauses_per_sent_log": 0.350, "9_stz_ratio_passive_verbs_log": 0.145, "11_dct_ratio_adv_log": 0.200, "17_avg_syllables_per_word": 0.400, "12_dct_ratio_pron_log": 0.195},
    "10": {"1_nikkud_ratio_log_inv": 0.992, "16_total_syllables_log": 0.695, "18_avg_syllables_per_sentence_log": 0.660, "10_dct_ratio_adj_log": 0.350, "8_stz_avg_tree_depth_per_sent_log": 0.475, "14a_zipf_rarity_tuple_inv": 0.540, "7_stz_avg_clauses_per_sent_log": 0.370, "9_stz_ratio_passive_verbs_log": 0.165, "11_dct_ratio_adv_log": 0.205, "17_avg_syllables_per_word": 0.415, "12_dct_ratio_pron_log": 0.188},
    "11": {"1_nikkud_ratio_log_inv": 0.995, "16_total_syllables_log": 0.715, "18_avg_syllables_per_sentence_log": 0.680, "10_dct_ratio_adj_log": 0.365, "8_stz_avg_tree_depth_per_sent_log": 0.495, "14a_zipf_rarity_tuple_inv": 0.555, "7_stz_avg_clauses_per_sent_log": 0.385, "9_stz_ratio_passive_verbs_log": 0.180, "11_dct_ratio_adv_log": 0.210, "17_avg_syllables_per_word": 0.425, "12_dct_ratio_pron_log": 0.180},
    "12": {"1_nikkud_ratio_log_inv": 0.995, "16_total_syllables_log": 0.735, "18_avg_syllables_per_sentence_log": 0.700, "10_dct_ratio_adj_log": 0.380, "8_stz_avg_tree_depth_per_sent_log": 0.515, "14a_zipf_rarity_tuple_inv": 0.575, "7_stz_avg_clauses_per_sent_log": 0.405, "9_stz_ratio_passive_verbs_log": 0.200, "11_dct_ratio_adv_log": 0.215, "17_avg_syllables_per_word": 0.435, "12_dct_ratio_pron_log": 0.175},
    "Bagrut": {"1_nikkud_ratio_log_inv": 0.995, "16_total_syllables_log": 0.725, "18_avg_syllables_per_sentence_log": 0.690, "10_dct_ratio_adj_log": 0.370, "8_stz_avg_tree_depth_per_sent_log": 0.505, "14a_zipf_rarity_tuple_inv": 0.565, "7_stz_avg_clauses_per_sent_log": 0.395, "9_stz_ratio_passive_verbs_log": 0.190, "11_dct_ratio_adv_log": 0.212, "17_avg_syllables_per_word": 0.430, "12_dct_ratio_pron_log": 0.178},
    "Hadshon": {"1_nikkud_ratio_log_inv": 0.380, "16_total_syllables_log": 0.530, "18_avg_syllables_per_sentence_log": 0.500, "10_dct_ratio_adj_log": 0.215, "8_stz_avg_tree_depth_per_sent_log": 0.335, "14a_zipf_rarity_tuple_inv": 0.420, "7_stz_avg_clauses_per_sent_log": 0.245, "9_stz_ratio_passive_verbs_log": 0.025, "11_dct_ratio_adv_log": 0.155, "17_avg_syllables_per_word": 0.325, "12_dct_ratio_pron_log": 0.260},
    "Olim_news": {"1_nikkud_ratio_log_inv": 0.380, "16_total_syllables_log": 0.530, "18_avg_syllables_per_sentence_log": 0.500, "10_dct_ratio_adj_log": 0.215, "8_stz_avg_tree_depth_per_sent_log": 0.335, "14a_zipf_rarity_tuple_inv": 0.420, "7_stz_avg_clauses_per_sent_log": 0.245, "9_stz_ratio_passive_verbs_log": 0.025, "11_dct_ratio_adv_log": 0.155, "17_avg_syllables_per_word": 0.325, "12_dct_ratio_pron_log": 0.260},
    "Olim_ragil": {"1_nikkud_ratio_log_inv": 0.920, "16_total_syllables_log": 0.555, "18_avg_syllables_per_sentence_log": 0.525, "10_dct_ratio_adj_log": 0.235, "8_stz_avg_tree_depth_per_sent_log": 0.355, "14a_zipf_rarity_tuple_inv": 0.435, "7_stz_avg_clauses_per_sent_log": 0.260, "9_stz_ratio_passive_verbs_log": 0.040, "11_dct_ratio_adv_log": 0.165, "17_avg_syllables_per_word": 0.338, "12_dct_ratio_pron_log": 0.250},
    "מבחני אולפן - רגיל": {"1_nikkud_ratio_log_inv": 0.920, "16_total_syllables_log": 0.555, "18_avg_syllables_per_sentence_log": 0.525, "10_dct_ratio_adj_log": 0.235, "8_stz_avg_tree_depth_per_sent_log": 0.355, "14a_zipf_rarity_tuple_inv": 0.435, "7_stz_avg_clauses_per_sent_log": 0.260, "9_stz_ratio_passive_verbs_log": 0.040, "11_dct_ratio_adv_log": 0.165, "17_avg_syllables_per_word": 0.338, "12_dct_ratio_pron_log": 0.250},
    "Olim_mitkadmim": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.625, "18_avg_syllables_per_sentence_log": 0.595, "10_dct_ratio_adj_log": 0.290, "8_stz_avg_tree_depth_per_sent_log": 0.415, "14a_zipf_rarity_tuple_inv": 0.485, "7_stz_avg_clauses_per_sent_log": 0.315, "9_stz_ratio_passive_verbs_log": 0.095, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.375, "12_dct_ratio_pron_log": 0.220},
    "מבחני אולפן - מתקדמים": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.625, "18_avg_syllables_per_sentence_log": 0.595, "10_dct_ratio_adj_log": 0.290, "8_stz_avg_tree_depth_per_sent_log": 0.415, "14a_zipf_rarity_tuple_inv": 0.485, "7_stz_avg_clauses_per_sent_log": 0.315, "9_stz_ratio_passive_verbs_log": 0.095, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.375, "12_dct_ratio_pron_log": 0.220},
    "Bagrut_Olim": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.680, "18_avg_syllables_per_sentence_log": 0.645, "10_dct_ratio_adj_log": 0.335, "8_stz_avg_tree_depth_per_sent_log": 0.460, "14a_zipf_rarity_tuple_inv": 0.520, "7_stz_avg_clauses_per_sent_log": 0.355, "9_stz_ratio_passive_verbs_log": 0.140, "11_dct_ratio_adv_log": 0.195, "17_avg_syllables_per_word": 0.405, "12_dct_ratio_pron_log": 0.195},
    "Bagrut_hearing_impaired": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.645, "18_avg_syllables_per_sentence_log": 0.610, "10_dct_ratio_adj_log": 0.305, "8_stz_avg_tree_depth_per_sent_log": 0.430, "14a_zipf_rarity_tuple_inv": 0.495, "7_stz_avg_clauses_per_sent_log": 0.325, "9_stz_ratio_passive_verbs_log": 0.115, "11_dct_ratio_adv_log": 0.190, "17_avg_syllables_per_word": 0.385, "12_dct_ratio_pron_log": 0.210},
    "hearing_impaired": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.645, "18_avg_syllables_per_sentence_log": 0.610, "10_dct_ratio_adj_log": 0.305, "8_stz_avg_tree_depth_per_sent_log": 0.430, "14a_zipf_rarity_tuple_inv": 0.495, "7_stz_avg_clauses_per_sent_log": 0.325, "9_stz_ratio_passive_verbs_log": 0.115, "11_dct_ratio_adv_log": 0.190, "17_avg_syllables_per_word": 0.385, "12_dct_ratio_pron_log": 0.210},
    "intellectual_disability": {"1_nikkud_ratio_log_inv": 0.940, "16_total_syllables_log": 0.470, "18_avg_syllables_per_sentence_log": 0.450, "10_dct_ratio_adj_log": 0.175, "8_stz_avg_tree_depth_per_sent_log": 0.290, "14a_zipf_rarity_tuple_inv": 0.405, "7_stz_avg_clauses_per_sent_log": 0.210, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.140, "17_avg_syllables_per_word": 0.310, "12_dct_ratio_pron_log": 0.275},
    "academic/technical": {"1_nikkud_ratio_log_inv": 1.000, "16_total_syllables_log": 0.720, "18_avg_syllables_per_sentence_log": 0.640, "10_dct_ratio_adj_log": 0.420, "8_stz_avg_tree_depth_per_sent_log": 0.400, "14a_zipf_rarity_tuple_inv": 0.520, "7_stz_avg_clauses_per_sent_log": 0.430, "9_stz_ratio_passive_verbs_log": 0.240, "11_dct_ratio_adv_log": 0.260, "17_avg_syllables_per_word": 0.460, "12_dct_ratio_pron_log": 0.180},
}

# Per-Level Mean_Abs_SHAP Lookup Table
LEVEL_SHAP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "2": {"1_nikkud_ratio_log_inv": 1.1365, "16_total_syllables_log": 0.1210, "18_avg_syllables_per_sentence_log": 0.0880, "10_dct_ratio_adj_log": 0.0420, "8_stz_avg_tree_depth_per_sent_log": 0.0310, "14a_zipf_rarity_tuple_inv": 0.0290, "7_stz_avg_clauses_per_sent_log": 0.0210, "9_stz_ratio_passive_verbs_log": 0.0190, "11_dct_ratio_adv_log": 0.0150, "17_avg_syllables_per_word": 0.0180, "12_dct_ratio_pron_log": 0.0140},
    "3": {"1_nikkud_ratio_log_inv": 0.5540, "16_total_syllables_log": 0.1340, "18_avg_syllables_per_sentence_log": 0.0910, "10_dct_ratio_adj_log": 0.0450, "8_stz_avg_tree_depth_per_sent_log": 0.0340, "14a_zipf_rarity_tuple_inv": 0.0320, "7_stz_avg_clauses_per_sent_log": 0.0220, "9_stz_ratio_passive_verbs_log": 0.0195, "11_dct_ratio_adv_log": 0.0155, "17_avg_syllables_per_word": 0.0185, "12_dct_ratio_pron_log": 0.0145},
    "'2-4": {"1_nikkud_ratio_log_inv": 1.0722, "16_total_syllables_log": 0.1290, "18_avg_syllables_per_sentence_log": 0.0890, "10_dct_ratio_adj_log": 0.0440, "8_stz_avg_tree_depth_per_sent_log": 0.0325, "14a_zipf_rarity_tuple_inv": 0.0305, "7_stz_avg_clauses_per_sent_log": 0.0215, "9_stz_ratio_passive_verbs_log": 0.0190, "11_dct_ratio_adv_log": 0.0150, "17_avg_syllables_per_word": 0.0180, "12_dct_ratio_pron_log": 0.0140},
    "2-4": {"1_nikkud_ratio_log_inv": 1.0722, "16_total_syllables_log": 0.1290, "18_avg_syllables_per_sentence_log": 0.0890, "10_dct_ratio_adj_log": 0.0440, "8_stz_avg_tree_depth_per_sent_log": 0.0325, "14a_zipf_rarity_tuple_inv": 0.0305, "7_stz_avg_clauses_per_sent_log": 0.0215, "9_stz_ratio_passive_verbs_log": 0.0190, "11_dct_ratio_adv_log": 0.0150, "17_avg_syllables_per_word": 0.0180, "12_dct_ratio_pron_log": 0.0140},
    "4": {"1_nikkud_ratio_log_inv": 0.1850, "16_total_syllables_log": 0.1283, "18_avg_syllables_per_sentence_log": 0.0960, "10_dct_ratio_adj_log": 0.0490, "8_stz_avg_tree_depth_per_sent_log": 0.0380, "14a_zipf_rarity_tuple_inv": 0.0360, "7_stz_avg_clauses_per_sent_log": 0.0250, "9_stz_ratio_passive_verbs_log": 0.0210, "11_dct_ratio_adv_log": 0.0165, "17_avg_syllables_per_word": 0.0195, "12_dct_ratio_pron_log": 0.0155},
    "5": {"1_nikkud_ratio_log_inv": 0.1600, "16_total_syllables_log": 0.1250, "18_avg_syllables_per_sentence_log": 0.0940, "10_dct_ratio_adj_log": 0.0510, "8_stz_avg_tree_depth_per_sent_log": 0.0390, "14a_zipf_rarity_tuple_inv": 0.0355, "7_stz_avg_clauses_per_sent_log": 0.0245, "9_stz_ratio_passive_verbs_log": 0.0205, "11_dct_ratio_adv_log": 0.0160, "17_avg_syllables_per_word": 0.0190, "12_dct_ratio_pron_log": 0.0150},
    "6": {"1_nikkud_ratio_log_inv": 0.1500, "16_total_syllables_log": 0.1220, "18_avg_syllables_per_sentence_log": 0.0920, "10_dct_ratio_adj_log": 0.0520, "8_stz_avg_tree_depth_per_sent_log": 0.0400, "14a_zipf_rarity_tuple_inv": 0.0350, "7_stz_avg_clauses_per_sent_log": 0.0240, "9_stz_ratio_passive_verbs_log": 0.0200, "11_dct_ratio_adv_log": 0.0160, "17_avg_syllables_per_word": 0.0190, "12_dct_ratio_pron_log": 0.0150},
    "intellectual_disability": {"1_nikkud_ratio_log_inv": 0.2100, "16_total_syllables_log": 0.1875, "18_avg_syllables_per_sentence_log": 0.1120, "10_dct_ratio_adj_log": 0.0533, "8_stz_avg_tree_depth_per_sent_log": 0.0410, "14a_zipf_rarity_tuple_inv": 0.0399, "7_stz_avg_clauses_per_sent_log": 0.0280, "9_stz_ratio_passive_verbs_log": 0.0230, "11_dct_ratio_adv_log": 0.0170, "17_avg_syllables_per_word": 0.0210, "12_dct_ratio_pron_log": 0.0165},
    "Olim_ragil": {"1_nikkud_ratio_log_inv": 0.2200, "16_total_syllables_log": 0.1380, "18_avg_syllables_per_sentence_log": 0.0980, "10_dct_ratio_adj_log": 0.0510, "8_stz_avg_tree_depth_per_sent_log": 0.0375, "14a_zipf_rarity_tuple_inv": 0.0351, "7_stz_avg_clauses_per_sent_log": 0.0245, "9_stz_ratio_passive_verbs_log": 0.0205, "11_dct_ratio_adv_log": 0.0160, "17_avg_syllables_per_word": 0.0195, "12_dct_ratio_pron_log": 0.0155},
    "Hadshon": {"1_nikkud_ratio_log_inv": 0.5100, "16_total_syllables_log": 0.1350, "18_avg_syllables_per_sentence_log": 0.0950, "10_dct_ratio_adj_log": 0.0480, "8_stz_avg_tree_depth_per_sent_log": 0.0360, "14a_zipf_rarity_tuple_inv": 0.0345, "7_stz_avg_clauses_per_sent_log": 0.0235, "9_stz_ratio_passive_verbs_log": 0.0200, "11_dct_ratio_adv_log": 0.0155, "17_avg_syllables_per_word": 0.0190, "12_dct_ratio_pron_log": 0.0150},
}

LEVEL_TO_CATEGORIES: Dict[str, List[str]] = {
    "1": ["Elementary - Beginner"],
    "2": ["Elementary - Beginner"],
    "3": ["Elementary - Beginner"],
    "'2-4": ["Elementary - Beginner", "Elementary - Intermediate"],
    "2-4": ["Elementary - Beginner", "Elementary - Intermediate"],
    "4": ["Elementary - Intermediate"],
    "5": ["Elementary - Intermediate"],
    "6": ["Elementary - Intermediate"],
    "'4-9": ["Elementary - Intermediate", "Middle School"],
    "4-9": ["Elementary - Intermediate", "Middle School"],
    "7": ["Middle School"],
    "8": ["Middle School"],
    "9": ["Middle School"],
    "10": ["High School"],
    "11": ["High School"],
    "12": ["High School", "Technical/Academic"],
    "Bagrut": ["High School", "Technical/Academic"],
    "Hadshon": ["Second-language learners", "Elementary - Beginner"],
    "Olim_news": ["Second-language learners", "Elementary - Beginner"],
    "Olim_ragil": ["Second-language learners", "Elementary - Intermediate"],
    "מבחני אולפן - רגיל": ["Second-language learners", "Elementary - Intermediate"],
    "Olim_mitkadmim": ["Second-language learners", "Middle School"],
    "מבחני אולפן - מתקדמים": ["Second-language learners", "Middle School"],
    "Bagrut_Olim": ["Second-language learners", "High School"],
    "Bagrut_hearing_impaired": ["Middle School", "High School"],
    "hearing_impaired": ["Middle School", "High School"],
    "intellectual_disability": ["Intellectual Disability", "Intellectual Disability Accessible"],
    "academic/technical": ["Technical/Academic"],
}

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
    "Hadshon": "Second-language learners",
    "Olim_news": "Second-language learners",
    "Olim_ragil": "Second-language learners",
    "מבחני אולפן - רגיל": "Second-language learners",
    "Olim_mitkadmim": "Second-language learners",
    "מבחני אולפן - מתקדמים": "Second-language learners",
    "Bagrut_Olim": "Second-language learners",
    "Bagrut_hearing_impaired": "Grade 7-9 - Middle School",
    "hearing_impaired": "Grade 7-9 - Middle School",
    "intellectual_disability": "Intellectual Disability Accessible",
    "academic/technical": "Technical/Academic",
    "Technical/Academic": "Technical/Academic",
}

LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "1": "Grade 1 (כיתה א׳ - ראשית קריאה בניקוד מלא)",
    "2": "Grade 2 (כיתה ב׳ - יסודי מתחילים בניקוד מלא)",
    "3": "Grade 3 (כיתה ג׳ - שלב מעבר בניקוד חלקי/מלא)",
    "'2-4": "Grades 2-4 Transition Cluster (כיתות ב׳-ד׳)",
    "2-4": "Grades 2-4 Transition Cluster (כיתות ב׳-ד׳)",
    "4": "Grade 4 (כיתה ד׳ - יסודי ביניים ללא ניקוד)",
    "5": "Grade 5 (כיתה ה׳ - יסודי ביניים)",
    "6": "Grade 6 (כיתה ו׳ - יסודי עליון)",
    "'4-9": "Grades 4-9 Broad Curricular Cluster (כיתות ד׳-ט׳)",
    "4-9": "Grades 4-9 Broad Curricular Cluster (כיתות ד׳-ט׳)",
    "7": "Grade 7 (כיתה ז׳ - חטיבת ביניים)",
    "8": "Grade 8 (כיתה ח׳ - חטיבת ביניים)",
    "9": "Grade 9 (כיתה ט׳ - חטיבת ביניים)",
    "10": "Grade 10 (כיתה י׳ - תיכון)",
    "11": "Grade 11 (כיתה י״א - תיכון ובגרות)",
    "12": "Grade 12 (כיתה י״ב - תיכון עליון)",
    "Bagrut": "High School Matriculation (בחינות בגרות)",
    "Hadshon": "Hadshon Easy-Hebrew L2 News (חדשון - עברית קלה לעולים בניקוד חלקי)",
    "Olim_news": "Hadshon Easy-Hebrew L2 News (חדשון - עברית קלה לעולים בניקוד חלקי)",
    "Olim_ragil": "Ulpan Regular L2 Assessment (מבחני אולפן - רגיל)",
    "מבחני אולפן - רגיל": "Ulpan Regular L2 Assessment (מבחני אולפן - רגיל)",
    "Olim_mitkadmim": "Ulpan Advanced L2 Assessment (מבחני אולפן - מתקדמים)",
    "מבחני אולפן - מתקדמים": "Ulpan Advanced L2 Assessment (מבחני אולפן - מתקדמים)",
    "Bagrut_Olim": "Adapted Bagrut for L2 Immigrants (בגרות לעולים)",
    "Bagrut_hearing_impaired": "Adapted Bagrut for Hearing Impaired (בגרות ללקויי שמיעה)",
    "hearing_impaired": "Adapted Bagrut for Hearing Impaired (בגרות ללקויי שמיעה)",
    "intellectual_disability": "Cognitive / Intellectual Disability Easy-Read (פישוט לשוני למוגבלות שכלית התפתחותית)",
    "academic/technical": "Academic / Technical High-Register Text (טקסט אקדמי / מקצועי מעל טווח בית הספר)",
}

POPULATION_SPECS: Dict[str, Dict[str, Any]] = {
    "Grade 1-3 - Elementary Beginner": {
        "engine_categories": ["Elementary - Beginner"],
        "raw_level_keys": ["1", "2", "3", "'2-4", "2-4"],
        "anchor_score": 3.21,
        "medians": RAW_LEVEL_MEDIANS["2"],
    },
    "Grade 4-6 - Elementary Intermediate": {
        "engine_categories": ["Elementary - Intermediate"],
        "raw_level_keys": ["4", "5", "6", "'4-9", "4-9"],
        "anchor_score": 5.31,
        "medians": RAW_LEVEL_MEDIANS["5"],
    },
    "Grade 7-9 - Middle School": {
        "engine_categories": ["Middle School"],
        "raw_level_keys": ["7", "8", "9", "'4-9", "4-9", "Bagrut_hearing_impaired", "hearing_impaired"],
        "anchor_score": 5.56,
        "medians": RAW_LEVEL_MEDIANS["8"],
    },
    "Grade 10-12 - High School": {
        "engine_categories": ["High School"],
        "raw_level_keys": ["10", "11", "12", "Bagrut"],
        "anchor_score": 5.71,
        "medians": RAW_LEVEL_MEDIANS["11"],
    },
    "Technical/Academic": {
        "engine_categories": ["Technical/Academic"],
        "raw_level_keys": ["12", "Bagrut", "academic/technical"],
        "anchor_score": 6.10,
        "medians": RAW_LEVEL_MEDIANS["12"],
    },
    "Second-language learners": {
        "engine_categories": ["Second-language learners"],
        "raw_level_keys": ["Hadshon", "Olim_news", "Olim_ragil", "מבחני אולפן - רגיל", "Olim_mitkadmim", "מבחני אולפן - מתקדמים", "Bagrut_Olim"],
        "anchor_score": 5.18,
        "medians": RAW_LEVEL_MEDIANS["Olim_ragil"],
    },
    "Intellectual Disability Accessible": {
        "engine_categories": ["Intellectual Disability", "Intellectual Disability Accessible"],
        "raw_level_keys": ["intellectual_disability"],
        "anchor_score": 4.92,
        "medians": RAW_LEVEL_MEDIANS["intellectual_disability"],
    },
    "Intellectual Disability": {
        "engine_categories": ["Intellectual Disability", "Intellectual Disability Accessible"],
        "raw_level_keys": ["intellectual_disability"],
        "anchor_score": 4.92,
        "medians": RAW_LEVEL_MEDIANS["intellectual_disability"],
    },
}

FEATURE_DIAGNOSTIC_RULES: Dict[str, Dict[str, Any]] = {
    "1_nikkud_ratio_log_inv": {
        "name": "Vocalization (Nikkud)",
        "weight": 0.300,
        "too_high": (
            "Increase Hebrew vowel diacritic density in 'vocalization_plan': raise "
            "'target_diacritic_ratio' toward the continuous inverse-median target r* and "
            "list uncommon or ambiguous words in 'words_to_vocalize'."
        ),
        "too_low": (
            "Reduce Hebrew vowel diacritic density in 'vocalization_plan': lower "
            "'target_diacritic_ratio' toward the continuous inverse-median target r* so "
            "only homographs and low-Zipf rare content words are vocalized."
        ),
    },
    "16_total_syllables_log": {
        "name": "Information & Syllabic Density (Paragraph Volume)",
        "weight": 0.135,
        "too_high": "Express facts more concisely using shorter words and tighter phrasing while preserving 100% of core factual propositions.",
        "too_low": "Elaborate slightly with complete, explicit sentences so the paragraph is not overly telegraphic.",
    },
    "18_avg_syllables_per_sentence_log": {
        "name": "Sentence Length & Breath Load",
        "weight": 0.095,
        "too_high": "Split long sentences into 2-3 short, independent sentences separated by periods (aim for 6-10 words per sentence).",
        "too_low": "Combine overly choppy 3-word fragments into natural, connected sentences using simple conjunctions.",
    },
    "10_dct_ratio_adj_log": {
        "name": "Adjective Density (Descriptive Load)",
        "weight": 0.048,
        "too_high": "Remove non-essential decorative adjectives and stacked modifier chains while retaining factual attributes.",
        "too_low": "Add clear, grade-appropriate descriptive adjectives to clarify nouns.",
    },
    "8_stz_avg_tree_depth_per_sent_log": {
        "name": "Syntactic Nesting (Dependency Tree Depth)",
        "weight": 0.038,
        "too_high": "Flatten nested relative clauses and parenthetical insertions into linear Subject-Verb-Object (SVO) sentences.",
        "too_low": "Connect related clauses with standard subordinators (כאשר, מפני ש-, אשר) appropriate for higher grades.",
    },
    "14a_zipf_rarity_tuple_inv": {
        "name": "Lexical Rarity (Zipf Lemma-POS Frequency)",
        "weight": 0.035,
        "too_high": "Replace low-frequency, archaic, or specialized lemmas with familiar, high-frequency everyday Hebrew equivalents that convey the exact same meaning.",
        "too_low": "Replace overly basic words with more precise, grade-appropriate curricular terminology.",
    },
    "7_stz_avg_clauses_per_sent_log": {
        "name": "Clause Density (Subordination)",
        "weight": 0.025,
        "too_high": "Limit each sentence to a single main clause (one core action per sentence) by breaking subordinate chains.",
        "too_low": "Combine single-clause sentences into multi-clause sentences with explicit logical connectors.",
    },
    "9_stz_ratio_passive_verbs_log": {
        "name": "Passive Voice Ratio",
        "weight": 0.021,
        "too_high": "Convert passive verb constructions (פועל, הופעל, נפעל) into direct active voice with an explicit actor.",
        "too_low": "Use standard passive constructions where appropriate for objective curricular register.",
    },
    "11_dct_ratio_adv_log": {
        "name": "Adverbial & Qualifier Load",
        "weight": 0.016,
        "too_high": "Remove formal rhetorical qualifiers and redundant adverbs.",
        "too_low": "Include explicit time or manner adverbs where helpful for clarity.",
    },
    "17_avg_syllables_per_word": {
        "name": "Word Length & Morphological Load",
        "weight": 0.019,
        "too_high": "Decompose heavy bound morphological forms into analytic words (e.g., 'הבית שלו' instead of 'ביתו') and use shorter lemmas.",
        "too_low": "Use standard inflected Hebrew morphology.",
    },
    "12_dct_ratio_pron_log": {
        "name": "Explicit Pronoun Reference",
        "weight": 0.015,
        "too_high": "Replace ambiguous pronoun chains with the explicit noun referent.",
        "too_low": "Use clear personal pronouns (הוא, היא, הם) when splitting sentences instead of dense nominalizations.",
    },
}

# Continuous Inverse-Median + Zipf-Driven Vocalization Configuration
CONTINUOUS_VOCALIZATION_CONFIG: Dict[str, float] = {
    "default_s_min": 0.0,
    "default_s_max": float(math.log1p(1.0)),
    "deficit_gain_beta": 0.18,
    "zipf_min_bound": 1.50,
    "zipf_max_bound": 6.50,
    "damping_min_raw_ratio": 0.03,
    "damping_max_raw_ratio": 0.65,
    "damping_min_words_per_sent": 7.5,
    "damping_syntax_ref_words": 6.5,
    "damping_strength": 0.55,
    "max_single_paragraph_words_for_split": 45.0,
}


def invert_nikkud_minmax_log(
    m_nikkud_inv: float,
    s_min: float = CONTINUOUS_VOCALIZATION_CONFIG["default_s_min"],
    s_max: float = CONTINUOUS_VOCALIZATION_CONFIG["default_s_max"],
) -> float:
    """Inverts 1_nikkud_ratio_log_inv median m in [0, 1] back to raw vocalization ratio r in [0, 1]."""
    m_clamped = float(np.clip(m_nikkud_inv, 0.0, 1.0))
    log_val = float(s_min) + (1.0 - m_clamped) * (float(s_max) - float(s_min))
    r_val = math.exp(log_val) - 1.0
    return round(float(np.clip(r_val, 0.0, 1.0)), 4)


def compute_continuous_vocalization_target(
    target_raw_level: Optional[str],
    y_pred: Optional[float] = None,
    mu_target: Optional[float] = None,
    s_min: float = CONTINUOUS_VOCALIZATION_CONFIG["default_s_min"],
    s_max: float = CONTINUOUS_VOCALIZATION_CONFIG["default_s_max"],
    beta: float = CONTINUOUS_VOCALIZATION_CONFIG["deficit_gain_beta"],
) -> Dict[str, Any]:
    """Computes continuous inverse-median vocalization parameters and Zipf rarity gate."""
    clean = str(target_raw_level or "").strip()
    medians = (
        RAW_LEVEL_MEDIANS.get(clean)
        or RAW_LEVEL_MEDIANS.get(f"'{clean}")
        or RAW_LEVEL_MEDIANS.get(clean.lstrip("'"))
        or RAW_LEVEL_MEDIANS["5"]
    )
    shap_weights = (
        LEVEL_SHAP_WEIGHTS.get(clean)
        or LEVEL_SHAP_WEIGHTS.get(f"'{clean}")
        or LEVEL_SHAP_WEIGHTS.get(clean.lstrip("'"))
        or LEVEL_SHAP_WEIGHTS["5"]
    )

    m_base = float(medians.get("1_nikkud_ratio_log_inv", 0.965))
    w_nikkud = max(0.05, float(shap_weights.get("1_nikkud_ratio_log_inv", 0.185)))
    m_star = m_base

    if y_pred is not None and mu_target is not None and not (math.isnan(y_pred) or math.isnan(mu_target)):
        delta_y = float(y_pred) - float(mu_target)
        m_star = float(np.clip(m_base - float(beta) * (delta_y / w_nikkud), 0.0, 1.0))

    r_base = invert_nikkud_minmax_log(m_base, s_min=s_min, s_max=s_max)
    r_star = invert_nikkud_minmax_log(m_star, s_min=s_min, s_max=s_max)

    m_zipf_inv = float(medians.get("14a_zipf_rarity_tuple_inv", 0.460))
    zipf_min = CONTINUOUS_VOCALIZATION_CONFIG["zipf_min_bound"]
    zipf_max = CONTINUOUS_VOCALIZATION_CONFIG["zipf_max_bound"]
    zipf_threshold = round(zipf_min + (1.0 - m_zipf_inv) * (zipf_max - zipf_min), 2)

    instruction = (
        f"Continuous Data-Driven Vocalization (Empirical m_nikkud={m_base:.3f} -> "
        f"baseline ratio r_base={r_base:.2f}, active target ratio r*={r_star:.2f}, "
        f"Zipf rarity gate <= {zipf_threshold:.2f}): "
        f"Set 'vocalization_plan.target_diacritic_ratio' near {r_star:.2f} and list any "
        f"ambiguous homographs or unfamiliar lemmas in 'vocalization_plan.words_to_vocalize'. "
        f"The engine will vocalize your flagged homographs first and fill the remaining "
        f"continuous quota r*={r_star:.2f} using the lowest-Zipf (rarest) content words."
    )

    return {
        "m_nikkud_base": round(m_base, 4),
        "m_nikkud_star": round(m_star, 4),
        "r_base": r_base,
        "default_ratio": r_star,
        "target_ratio_range": (round(max(0.0, r_star - 0.08), 4), round(min(1.0, r_star + 0.08), 4)),
        "zipf_familiarity_threshold": zipf_threshold,
        "w_nikkud": round(w_nikkud, 4),
        "default_mode": "continuous",
        "allowed_modes": ["continuous", "full", "partial", "selective_ambiguous", "none"],
        "instruction": instruction,
    }


def get_vocalization_policy(
    target_raw_level: Optional[str],
    y_pred: Optional[float] = None,
    mu_target: Optional[float] = None,
) -> Dict[str, Any]:
    """Returns continuous vocalization parameters for any requested raw level."""
    return compute_continuous_vocalization_target(
        target_raw_level=target_raw_level,
        y_pred=y_pred,
        mu_target=mu_target,
    )


# Backward-compatible dictionary populated dynamically from continuous vocalization targets
VOCALIZATION_LEVEL_POLICIES: Dict[str, Dict[str, Any]] = {
    lvl: compute_continuous_vocalization_target(lvl)
    for lvl in list(RAW_LEVEL_MEDIANS.keys())
}
VOCALIZATION_LEVEL_POLICIES["DEFAULT_UNVOCALIZED"] = compute_continuous_vocalization_target("5")


@dataclass
class PreprocessorConfig:
    """Unified configuration for training, preprocessing, calibration, and evaluation."""

    raw_data_path: str = "artifacts/features_dataset_clean.parquet"
    artifact_dir: str = "artifacts"
    shap_dir: str = "artifacts/shap_analysis_output"
    shap_advanced_dir: str = "artifacts/shap_analysis_output/advanced_gt_4_4"
    group_col: str = "passage_title"
    level_col: str = "level"
    target_col: str = "Target_Readability_Score"
    advanced_score_threshold: float = 4.4
    skew_threshold: float = 1.0
    fixed_nikkud_weight: float = 0.20
    remaining_weight_pool: float = 0.80
    n_splits: int = 5
    random_state: int = 42
    metadata_cols: List[str] = field(
        default_factory=lambda: [
            "paragraph_id",
            "raw_text",
            "level",
            "source_file",
            "passage_title",
            "paragraph_index",
        ]
    )
    inverse_base_features: List[str] = field(
        default_factory=lambda: [
            "1_nikkud_ratio",
            "13a_oov_ratio_tuple",
            "13b_oov_ratio_pos",
            "13c_oov_ratio_lemma",
            "14a_zipf_rarity_tuple",
            "14b_zipf_rarity_pos",
            "14c_zipf_rarity_lemma",
        ]
    )
    selected_base_features: List[str] = field(
        default_factory=lambda: [
            "1_nikkud_ratio",
            "7_stz_avg_clauses_per_sent",
            "8_stz_avg_tree_depth_per_sent",
            "9_stz_ratio_passive_verbs",
            "10_dct_ratio_adj",
            "11_dct_ratio_adv",
            "12_dct_ratio_pron",
            "14a_zipf_rarity_tuple",
            "16_total_syllables",
            "17_avg_syllables_per_word",
            "18_avg_syllables_per_sentence",
        ]
    )
    raw_feature_names: List[str] = field(
        default_factory=lambda: [
            "1_nikkud_ratio",
            "7_stz_avg_clauses_per_sent",
            "8_stz_avg_tree_depth_per_sent",
            "9_stz_ratio_passive_verbs",
            "10_dct_ratio_adj",
            "11_dct_ratio_adv",
            "12_dct_ratio_pron",
            "14a_zipf_rarity_tuple",
            "16_total_syllables",
            "17_avg_syllables_per_word",
            "18_avg_syllables_per_sentence",
        ]
    )


# Alias required by src/hebrew_readability/__init__.py and src/features/preprocessor.py
PipelineConfig = PreprocessorConfig


def setup_pipeline_logger(
    name: str = "HebrewReadability",
    artifact_dir: str = "artifacts",
    log_filename: str = "pipeline.log",
) -> logging.Logger:
    """Configures an idempotent UTF-8 file + console logger."""
    os.makedirs(artifact_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(os.path.join(artifact_dir, log_filename), encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_dataframe_parquet(df: pd.DataFrame, path: str) -> None:
    """Saves a DataFrame to Apache Parquet or CSV with UTF-8 string preservation."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df_copy = df.copy()
    for col in STRING_METADATA_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str)
    if str(path).endswith(".csv"):
        df_copy.to_csv(path, index=False, encoding="utf-8-sig")
        return
    table = pa.Table.from_pandas(df_copy, preserve_index=False)
    pq.write_table(table, path)


def load_dataframe_parquet(path: str, **kwargs: Any) -> pd.DataFrame:
    """Loads a DataFrame from an Apache Parquet or CSV file."""
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path, engine="pyarrow")
        if "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            for col, dt in kwargs["dtype"].items():
                if col in df.columns:
                    df[col] = df[col].astype(dt)
        return df
    return pd.read_csv(path, **kwargs)


save_table = save_dataframe_parquet
read_table = load_dataframe_parquet
