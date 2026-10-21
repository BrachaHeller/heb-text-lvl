"""Step 5: Pushes trained .pkl and .csv/.json artifacts to GolDL/hebrew-readability-classifier."""

import argparse
import logging
import os

from huggingface_hub import HfApi

from hebrew_readability.config import (HF_MODEL_REPO_ID, PipelineConfig, read_table)

from dotenv import load_dotenv 
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_TO_UPLOAD = [
    "readability_preprocessor.pkl",
    "minmax_scaler.pkl",
    "xgboost_readability_regressor.pkl",
    "level_calibration_summary.csv",
    "calibration_stats.json",
    "filtered_eta_weighted_1d_scale.csv",
    "filtered_eta_squared_results.csv",
    "filtered_medians_vectors.csv",
    "model_evaluation_metrics.txt",
]


def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub.")
    parser.add_argument("--artifact-dir", type=str, default=PipelineConfig().artifact_dir)
    parser.add_argument("--repo-id", type=str, default=HF_MODEL_REPO_ID)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    for filename in ARTIFACTS_TO_UPLOAD:
        local_path = os.path.join(args.artifact_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"Uploading {filename} -> {args.repo_id}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=args.repo_id,
                repo_type="model",
            )
        else:
            logger.warning(f"File not found, skipping: {local_path}")

    logger.info(f"Successfully uploaded artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()