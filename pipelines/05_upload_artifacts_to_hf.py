"""Step 5: Validates and uploads trained artifacts to GolDL/hebrew-readability-classifier."""

import argparse
import logging
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

from hebrew_readability.config import HF_MODEL_REPO_ID, PipelineConfig

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
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually push files to Hugging Face. Without this flag, runs in safe DRY-RUN mode.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(f"ARTIFACT AUDIT FOR REPO: {args.repo_id}")
    print("=" * 65)

    found_files = []
    missing_files = []
    for filename in ARTIFACTS_TO_UPLOAD:
        local_path = os.path.join(args.artifact_dir, filename)
        if os.path.exists(local_path):
            size_kb = os.path.getsize(local_path) / 1024.0
            print(f"  [✓ FOUND]   {filename:<38} ({size_kb:>7.1f} KB)")
            found_files.append((filename, local_path))
        else:
            print(f"  [✗ MISSING] {filename}")
            missing_files.append(filename)

    if not args.upload:
        print("\n[SAFE DRY-RUN MODE] No files were uploaded to Hugging Face.")
        print("Once you verify the metrics in 'artifacts/model_evaluation_metrics.txt',")
        print("run with `--upload` to push to Hugging Face:")
        print("  python pipelines/05_upload_artifacts_to_hf.py --upload\n")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment or .env file. Cannot upload.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    for filename, local_path in found_files:
        logger.info(f"Uploading {filename} -> {args.repo_id}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="model",
        )

    logger.info(f"✅ Successfully uploaded {len(found_files)} artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()