"""Centralized configuration, feature whitelists, and Parquet/CSV I/O helpers."""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import logging
import os
import sys

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

        # 1. Console Handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Persistent File Handler (UTF-8 for Hebrew level names and text)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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

LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "1": "Grade 1 (Elementary - Beginner)",
    "2": "Grade 2 (Elementary - Beginner)",
    "3": "Grade 3 (Elementary - Beginner)",
    "'2-4": "Grades 2-4 Cluster (Elementary - Beginner)",
    "4": "Grade 4 (Elementary - Intermediate)",
    "5": "Grade 5 (Elementary - Intermediate)",
    "6": "Grade 6 (Elementary - Intermediate)",
    "7": "Grade 7 (Middle School - Interpolated)",
    "8": "Grade 8 (Middle School)",
    "9": "Grade 9 (Middle School)",
    "'4-9": "Grades 4-9 Broad Cluster (Elementary - Intermediate, Middle School)",
    "10": "Grade 10 (High School - Interpolated)",
    "11": "Grade 11 (High School)",
    "12": "Grade 12 (High School)",
    "Olim_ragil": "Ulpan Olim (Regular Track)",
    "Olim_mitkadmim": "Ulpan Olim (Advanced Track)",
    "Olim_news": "Hadshon Easy News (Olim)",
    "Bagrut_Olim": "High School Bagrut (High School, Olim)",
    "Bagrut_hearing_impaired": "High School Bagrut (High School, Hearing Impaired)",
    "intellectual_disability": "Supported Easy-Read (Intellectual Disability)",
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