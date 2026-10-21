"""Step 1: Downloads heb_paragraphs_dataset.parquet from GolDL/heb-readability-txt and extracts features."""

import argparse
import logging
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from hebrew_readability.config import (
    HF_DATASET_REPO_ID,
    RAW_DATASET_FILENAME,
    PipelineConfig,
    read_table,
)
from features.extractor import FeatureExtractor

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run batch Hebrew linguistic feature extraction.")
    parser.add_argument(
        "--input-file", type=str, default=None, help="Path to local input .parquet or .csv file."
    )
    parser.add_argument(
        "--hf-repo", type=str, default=HF_DATASET_REPO_ID, help="HF Dataset repo ID."
    )
    parser.add_argument(
        "--hf-filename", type=str, default=RAW_DATASET_FILENAME, help="Parquet file in HF repo."
    )
    parser.add_argument(
        "--output-file", type=str, default=PipelineConfig().raw_data_path, help="Output .parquet or .csv path."
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--test", action="store_true", help="Smoke test on first 3 rows.")
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        source_path = args.input_file
        logger.info(f"Loading local dataset from '{source_path}'...")
    else:
        logger.info(f"Downloading '{args.hf_filename}' from '{args.hf_repo}'...")
        source_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )

    df = read_table(source_path)
    logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

    if args.test:
        logger.info("Running smoke test on first 3 rows...")
        df = df.head(3).copy()
        # Use a separate test output path so we don't overwrite full runs
        if args.output_file == PipelineConfig().raw_data_path:
            args.output_file = "artifacts/test_smoke_features.parquet"

    extractor = FeatureExtractor()
    out_df = extractor.extract_batch_df(
        df=df,
        text_column="raw_text",
        checkpoint_path=args.output_file,
        checkpoint_every=args.checkpoint_every,
    )
    logger.info(
        f"✅ Batch extraction complete! Saved {len(out_df)} rows to '{args.output_file}'."
    )


if __name__ == "__main__":
    main()