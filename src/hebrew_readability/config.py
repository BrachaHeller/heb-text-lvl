
"""Centralized configuration, feature whitelists, population specs, and Parquet/CSV I/O helpers."""

from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Hugging Face Repositories
HF_MODEL_REPO_ID = "GolDL/hebrew-readability-classifier"
HF_DICT_REPO_ID = "GolDL/hebrew-lemma-frequency-dictionary"
HF_DATASET_REPO_ID = "GolDL/heb-readability-txt"

# Default Filenames in HF Repos
RAW_DATASET_FILENAME = "heb_paragraphs_dataset.parquet"
DICT_PARQUET_FILENAME = "frequency_dictionary_clean.parquet"
DICT_CSV_FILENAME = "frequency_dictionary_clean.csv"
DICT_ONNX_URL = "https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.int8.onnx"

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


def read_table(filepath: str, **kwargs) -> pd.DataFrame:
    """Reads a DataFrame from either .parquet or .csv automatically."""
    if str(filepath).endswith(".parquet"):
        df = pd.read_parquet(filepath, engine="pyarrow")
        if "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            for col, dt in kwargs["dtype"].items():
                if col in df.columns:
                    df[col] = df[col].astype(dt)
        return df
    return pd.read_csv(filepath, **kwargs)


def save_table(df: pd.DataFrame, filepath: str) -> None:
    """Saves a DataFrame to .parquet (enforcing clean string types for PyArrow) or .csv."""
    out_df = df.copy()
    if str(filepath).endswith(".parquet"):
        for col in STRING_METADATA_COLS:
            if col in out_df.columns:
                out_df[col] = out_df[col].fillna("").astype(str)
        out_df.to_parquet(filepath, index=False, engine="pyarrow")
    else:
        out_df.to_csv(filepath, index=False)


def setup_pipeline_logger(
    name: str, artifact_dir: str = "artifacts", log_filename: str = "pipeline_execution.log"
) -> logging.Logger:
    """Configures a logger that outputs to both stdout and a persistent UTF-8 log file."""
    os.makedirs(artifact_dir, exist_ok=True)
    log_path = os.path.join(artifact_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Exact 11 Core Engineered Features expected by the purged XGBoost regressor
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
    "Bagrut_Olim": ["High School", "Second-language - Advanced"],
    "hearing_impaired": ["High School"],
    "Bagrut_hearing_impaired": ["High School"],
    "intellectual_disability": ["Intellectual Disability"],
    "Hadshon": ["Second-language - Beginner"],
    "Olim_news": ["Second-language - Beginner"],
    "מבחני אולפן - רגיל": ["Second-language - Beginner"],
    "Olim_ragil": ["Second-language - Beginner"],
    "מבחני אולפן - מתקדמים": ["Second-language - Advanced"],
    "Olim_mitkadmim": ["Second-language - Advanced"],
}

LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "1": "Grade 1 (Elementary - Beginner)",
    "2": "Grade 2 (Elementary - Beginner)",
    "3": "Grade 3 (Elementary - Beginner)",
    "'2-4": "Grades 2-4 Cluster (Elementary - Beginner, Elementary - Intermediate)",
    "2-4": "Grades 2-4 Cluster (Elementary - Beginner, Elementary - Intermediate)",
    "4": "Grade 4 (Elementary - Intermediate)",
    "5": "Grade 5 (Elementary - Intermediate)",
    "6": "Grade 6 (Elementary - Intermediate)",
    "'4-9": "Grades 4-9 Broad Cluster (Elementary - Intermediate, Middle School)",
    "4-9": "Grades 4-9 Broad Cluster (Elementary - Intermediate, Middle School)",
    "7": "Grade 7 - Interpolated (Middle School)",
    "8": "Grade 8 (Middle School)",
    "9": "Grade 9 (Middle School)",
    "10": "Grade 10 - Interpolated (High School)",
    "11": "Grade 11 (High School)",
    "12": "Grade 12 (High School)",
    "Bagrut": "Bagrut (High School)",
    "Bagrut_Olim": "High School Bagrut (High School, Olim)",
    "hearing_impaired": "Hearing Impaired (High School)",
    "Bagrut_hearing_impaired": "Hearing Impaired Bagrut (High School)",
    "intellectual_disability": "Supported Easy-Read (Intellectual Disability)",
    "Hadshon": "Hadshon Easy News (Second-language - Beginner)",
    "Olim_news": "Hadshon Easy News (Second-language - Beginner)",
    "מבחני אולפן - רגיל": "מבחני אולפן - רגיל (Second-language - Beginner)",
    "Olim_ragil": "Ulpan Olim Regular (Second-language - Beginner)",
    "מבחני אולפן - מתקדמים": "מבחני אולפן - מתקדמים (Second-language - Advanced)",
    "Olim_mitkadmim": "Ulpan Olim Advanced (Second-language - Advanced)",
}

# Target population specifications for Deliverable 5 Agentic Workflow
POPULATION_SPECS: Dict[str, Dict[str, Any]] = {
    "Grade 1-3 - Elementary Beginner": {
        "engine_categories": ["Elementary - Beginner"],
        "raw_level_keys": ["1", "2", "3", "'2-4", "2-4"],
        "anchor_score": 3.21,
        "medians": {
            "1_nikkud_ratio_log_inv": 0.000,
            "16_total_syllables_log": 0.485,
            "18_avg_syllables_per_sentence_log": 0.468,
            "10_dct_ratio_adj_log": 0.180,
            "8_stz_avg_tree_depth_per_sent_log": 0.300,
            "14a_zipf_rarity_tuple_inv": 0.403,
            "7_stz_avg_clauses_per_sent_log": 0.220,
            "9_stz_ratio_passive_verbs_log": 0.000,
            "11_dct_ratio_adv_log": 0.150,
            "17_avg_syllables_per_word": 0.310,
            "12_dct_ratio_pron_log": 0.280,
        },
    },
    "Grade 4-6 - Elementary Intermediate": {
        "engine_categories": ["Elementary - Intermediate"],
        "raw_level_keys": ["4", "5", "6", "'4-9", "4-9"],
        "anchor_score": 5.31,
        "medians": {
            "1_nikkud_ratio_log_inv": 0.949,
            "16_total_syllables_log": 0.567,
            "18_avg_syllables_per_sentence_log": 0.512,
            "10_dct_ratio_adj_log": 0.260,
            "8_stz_avg_tree_depth_per_sent_log": 0.316,
            "14a_zipf_rarity_tuple_inv": 0.425,
            "7_stz_avg_clauses_per_sent_log": 0.290,
            "9_stz_ratio_passive_verbs_log": 0.080,
            "11_dct_ratio_adv_log": 0.200,
            "17_avg_syllables_per_word": 0.360,
            "12_dct_ratio_pron_log": 0.250,
        },
    },
    "Grade 7-9 - Middle School": {
        "engine_categories": ["Middle School"],
        "raw_level_keys": ["7", "8", "9", "'4-9", "4-9"],
        "anchor_score": 5.69,
        "medians": {
            "1_nikkud_ratio_log_inv": 0.985,
            "16_total_syllables_log": 0.621,
            "18_avg_syllables_per_sentence_log": 0.556,
            "10_dct_ratio_adj_log": 0.310,
            "8_stz_avg_tree_depth_per_sent_log": 0.351,
            "14a_zipf_rarity_tuple_inv": 0.435,
            "7_stz_avg_clauses_per_sent_log": 0.340,
            "9_stz_ratio_passive_verbs_log": 0.140,
            "11_dct_ratio_adv_log": 0.220,
            "17_avg_syllables_per_word": 0.390,
            "12_dct_ratio_pron_log": 0.230,
        },
    },
    "Grade 10-12 - High School": {
        "engine_categories": ["High School"],
        "raw_level_keys": ["10", "11", "12", "Bagrut", "hearing_impaired", "Bagrut_hearing_impaired"],
        "anchor_score": 5.95,
        "medians": {
            "1_nikkud_ratio_log_inv": 0.996,
            "16_total_syllables_log": 0.670,
            "18_avg_syllables_per_sentence_log": 0.587,
            "10_dct_ratio_adj_log": 0.350,
            "8_stz_avg_tree_depth_per_sent_log": 0.367,
            "14a_zipf_rarity_tuple_inv": 0.457,
            "7_stz_avg_clauses_per_sent_log": 0.380,
            "9_stz_ratio_passive_verbs_log": 0.180,
            "11_dct_ratio_adv_log": 0.240,
            "17_avg_syllables_per_word": 0.420,
            "12_dct_ratio_pron_log": 0.210,
        },
    },
    "Technical/Academic": {
        "engine_categories": ["Technical/Academic"],
        "raw_level_keys": ["12", "Bagrut"],
        "anchor_score": 6.80,
        "medians": {
            "1_nikkud_ratio_log_inv": 1.000,
            "16_total_syllables_log": 0.720,
            "18_avg_syllables_per_sentence_log": 0.640,
            "10_dct_ratio_adj_log": 0.420,
            "8_stz_avg_tree_depth_per_sent_log": 0.400,
            "14a_zipf_rarity_tuple_inv": 0.520,
            "7_stz_avg_clauses_per_sent_log": 0.430,
            "9_stz_ratio_passive_verbs_log": 0.240,
            "11_dct_ratio_adv_log": 0.260,
            "17_avg_syllables_per_word": 0.460,
            "12_dct_ratio_pron_log": 0.180,
        },
    },
    "Second-language learners": {
        "engine_categories": ["Second-language - Beginner", "Second-language - Advanced"],
        "raw_level_keys": [
            "Hadshon",
            "Olim_news",
            "מבחני אולפן - רגיל",
            "Olim_ragil",
            "מבחני אולפן - מתקדמים",
            "Olim_mitkadmim",
            "Bagrut_Olim",
        ],
        "anchor_score": 5.33,
        "medians": {
            "1_nikkud_ratio_log_inv": 0.900,
            "16_total_syllables_log": 0.609,
            "18_avg_syllables_per_sentence_log": 0.528,
            "10_dct_ratio_adj_log": 0.250,
            "8_stz_avg_tree_depth_per_sent_log": 0.327,
            "14a_zipf_rarity_tuple_inv": 0.474,
            "7_stz_avg_clauses_per_sent_log": 0.280,
            "9_stz_ratio_passive_verbs_log": 0.090,
            "11_dct_ratio_adv_log": 0.190,
            "17_avg_syllables_per_word": 0.350,
            "12_dct_ratio_pron_log": 0.260,
        },
    },
    "Intellectual Disability Accessible": {
        "engine_categories": ["Intellectual Disability"],
        "raw_level_keys": ["intellectual_disability"],
        "anchor_score": 4.96,
        "medians": {
            "1_nikkud_ratio_log_inv": 1.000,
            "16_total_syllables_log": 0.427,
            "18_avg_syllables_per_sentence_log": 0.464,
            "10_dct_ratio_adj_log": 0.190,
            "8_stz_avg_tree_depth_per_sent_log": 0.316,
            "14a_zipf_rarity_tuple_inv": 0.475,
            "7_stz_avg_clauses_per_sent_log": 0.210,
            "9_stz_ratio_passive_verbs_log": 0.000,
            "11_dct_ratio_adv_log": 0.160,
            "17_avg_syllables_per_word": 0.320,
            "12_dct_ratio_pron_log": 0.290,
        },
    },
}

# Diagnostic feedback rules covering all 11 core engineered features
FEATURE_DIAGNOSTIC_RULES: Dict[str, Dict[str, Any]] = {
    "1_nikkud_ratio_log_inv": {
        "name": "Vocalization (Nikkud)",
        "weight": 0.200,
        "too_high": "Add full Hebrew vowel diacritics (ניקוד) to words.",
        "too_low": "Remove Nikkud vowel marks (write unvocalized Hebrew).",
    },
    "16_total_syllables_log": {
        "name": "Paragraph Length & Volume",
        "weight": 0.274,
        "too_high": "Condense paragraph length and keep only core facts.",
        "too_low": "Expand paragraph length slightly with fuller context.",
    },
    "18_avg_syllables_per_sentence_log": {
        "name": "Sentence Length",
        "weight": 0.147,
        "too_high": "Shorten sentences significantly by splitting long sentences with periods.",
        "too_low": "Combine choppy, fragmented sentences into smoother sentences.",
    },
    "10_dct_ratio_adj_log": {
        "name": "Adjective Density",
        "weight": 0.105,
        "too_high": "Strip non-essential, decorative, or stacked adjectives.",
        "too_low": "Add precise descriptive adjectives to enrich noun phrases.",
    },
    "8_stz_avg_tree_depth_per_sent_log": {
        "name": "Syntactic Nesting",
        "weight": 0.096,
        "too_high": "Flatten nested grammatical structures into linear Subject-Verb-Object order.",
        "too_low": "Use standard connected sentence structure with natural subordination.",
    },
    "14a_zipf_rarity_tuple_inv": {
        "name": "Vocabulary Rarity",
        "weight": 0.085,
        "too_high": "Replace rare, abstract, or high-register words with everyday Hebrew words.",
        "too_low": "Use age-appropriate curricular or academic vocabulary.",
    },
    "7_stz_avg_clauses_per_sent_log": {
        "name": "Clause Density",
        "weight": 0.075,
        "too_high": "Limit each sentence to a single main clause/idea; break subordinate chains ('אשר', 'ש...').",
        "too_low": "Link related ideas with appropriate coordinating or subordinating clauses.",
    },
    "9_stz_ratio_passive_verbs_log": {
        "name": "Passive Voice Usage",
        "weight": 0.070,
        "too_high": "Convert passive verbs (פועל, הופעל, נפעל) into direct active voice with a clear subject.",
        "too_low": "Allow standard passive or impersonal verb constructions where appropriate.",
    },
    "17_avg_syllables_per_word": {
        "name": "Word Length & Morphological Load",
        "weight": 0.055,
        "too_high": "Decompose heavy bound suffixes into separate analytic words (e.g., 'הבית שלו' instead of 'ביתו').",
        "too_low": "Use standard inflected Hebrew morphological forms.",
    },
    "11_dct_ratio_adv_log": {
        "name": "Adverbial & Qualifier Load",
        "weight": 0.045,
        "too_high": "Remove unnecessary adverbs and rhetorical qualifiers.",
        "too_low": "Include appropriate qualifying adverbs for nuance.",
    },
    "12_dct_ratio_pron_log": {
        "name": "Explicit Pronoun Reference",
        "weight": 0.040,
        "too_high": "Replace ambiguous pronouns with explicit noun subjects.",
        "too_low": "Use clear personal pronouns ('הוא', 'היא', 'הם') when splitting sentences.",
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for reproducible training, calibration, and evaluation."""

    raw_data_path: str = "artifacts/features_dataset_clean.parquet"
    artifact_dir: str = "artifacts"
    shap_dir: str = "artifacts/shap_analysis_output"
    shap_advanced_dir: str = "artifacts/shap_analysis_output/advanced_gt_4_4"

    group_col: str = "passage_title"
    level_col: str = "level"
    target_col: str = "Target_Readability_Score"
    advanced_score_threshold: float = 4.4

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

    skew_threshold: float = 1.0
    fixed_nikkud_weight: float = 0.20
    remaining_weight_pool: float = 0.80
    n_splits: int = 5
    random_state: int = 42

    