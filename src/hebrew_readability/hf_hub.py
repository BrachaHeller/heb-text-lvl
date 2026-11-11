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
    """Downloads dicta-1.0.int8.onnx (~293 MB) atomically via HF Hub or GitHub Releases and verifies size."""
    import shutil
    import subprocess
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, "dicta-1.0.int8.onnx")
    min_expected_bytes = 250 * 1024 * 1024  # 250 MB gate for the 307,492,637-byte model

    if os.path.exists(onnx_path) and os.path.getsize(onnx_path) >= min_expected_bytes:
        return onnx_path

    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    # Strategy 1: Download via huggingface_hub (immune to browser URL wrapping)
    hf_candidates = [
        {"repo_id": "thewh1teagle/add-diacritics-in-hebrew", "filename": "dicta-1.0.int8.onnx", "repo_type": "space"},
        {"repo_id": "thewh1teagle/dicta-onnx", "filename": "dicta-1.0.int8.onnx", "repo_type": "model"},
    ]
    for cand in hf_candidates:
        try:
            dl_path = hf_hub_download(**cand)
            if os.path.exists(dl_path) and os.path.getsize(dl_path) >= min_expected_bytes:
                shutil.copyfile(dl_path, onnx_path)
                return onnx_path
        except Exception as exc:
            logger.debug("HF candidate %s failed: %s", cand["repo_id"], exc)

    # Strategy 2: Atomic curl download from GitHub Releases using split URL string
    gh_url = (
        "https://"
        + "github.com/thewh1teagle/dicta-onnx/releases/download/"
        + "model-files-v1.0/dicta-1.0.int8.onnx"
    )
    tmp_path = onnx_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-A", "Mozilla/5.0", "-o", tmp_path, gh_url],
        check=True,
    )
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < min_expected_bytes:
        raise RuntimeError(f"Downloaded ONNX file is incomplete: {os.path.getsize(tmp_path)} bytes")
    os.replace(tmp_path, onnx_path)
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