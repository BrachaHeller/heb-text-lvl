"""Automatic artifact and dictionary resolution from Hugging Face Hub (supports Parquet & CSV)."""

import logging
import os
import urllib.request
from typing import Dict, Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from hebrew_readability.config import (
    DICT_CSV_FILENAME,
    DICT_ONNX_URL,
    DICT_PARQUET_FILENAME,
    HF_DICT_REPO_ID,
    HF_MODEL_REPO_ID,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN")


def ensure_dicta_onnx_model(cache_dir: str = ".cache") -> str:
    """Downloads the Dicta ONNX vocalization model if not already present locally."""
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, "dicta-1.0.int8.onnx")
    if not os.path.exists(onnx_path):
        logger.info(f"Downloading Dicta ONNX model from {DICT_ONNX_URL}...")
        urllib.request.urlretrieve(DICT_ONNX_URL, onnx_path)
    return onnx_path


def download_frequency_dictionary(
    repo_id: str = HF_DICT_REPO_ID,
    filename: Optional[str] = None,
    local_override: Optional[str] = None,
) -> str:
    """Pulls the cascading frequency dictionary (.parquet or .csv) from HF Hub."""
    if local_override and os.path.exists(local_override):
        logger.info(f"Using local frequency dictionary at {local_override}")
        return local_override

    token = _get_hf_token()
    candidates = [filename] if filename else [DICT_PARQUET_FILENAME, DICT_CSV_FILENAME]

    for candidate in candidates:
        try:
            logger.info(f"Fetching '{candidate}' from HF dataset repo '{repo_id}'...")
            return hf_hub_download(
                repo_id=repo_id,
                filename=candidate,
                repo_type="dataset",
                token=token,
            )
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not find {candidates} in HF dataset repository '{repo_id}'."
    )


def download_model_artifacts(
    repo_id: str = HF_MODEL_REPO_ID,
    local_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Downloads model, preprocessor/scaler, and calibration table (.parquet or .csv)."""
    if local_dir and os.path.exists(local_dir):
        calib_parquet = os.path.join(local_dir, "level_calibration_summary.parquet")
        calib_csv = os.path.join(local_dir, "level_calibration_summary.csv")
        calib_path = calib_parquet if os.path.exists(calib_parquet) else calib_csv

        req_files = {
            "preprocessor": os.path.join(local_dir, "readability_preprocessor.pkl"),
            "scaler": os.path.join(local_dir, "minmax_scaler.pkl"),
            "model": os.path.join(local_dir, "xgboost_readability_regressor.pkl"),
            "calibration_table": calib_path,
        }
        if os.path.exists(req_files["preprocessor"]) and os.path.exists(req_files["model"]) and os.path.exists(calib_path):
            logger.info(f"Using local artifacts from '{local_dir}'")
            return req_files

    token = _get_hf_token()
    logger.info(f"Downloading frozen inference artifacts from HF model repo '{repo_id}'...")

    # Try .parquet calibration table first, fallback to .csv
    try:
        calib_file = hf_hub_download(
            repo_id=repo_id, filename="level_calibration_summary.parquet", token=token
        )
    except Exception:
        calib_file = hf_hub_download(
            repo_id=repo_id, filename="level_calibration_summary.csv", token=token
        )

    artifacts = {
        "preprocessor": hf_hub_download(
            repo_id=repo_id, filename="readability_preprocessor.pkl", token=token
        ),
        "scaler": hf_hub_download(
            repo_id=repo_id, filename="minmax_scaler.pkl", token=token
        ),
        "model": hf_hub_download(
            repo_id=repo_id, filename="xgboost_readability_regressor.pkl", token=token
        ),
        "calibration_table": calib_file,
    }
    return artifacts