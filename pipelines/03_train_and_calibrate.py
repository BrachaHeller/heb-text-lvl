"""Step 3: Trains XGBRegressor and exports canonical level_calibration_summary.csv (+ JSON mirror)."""

import argparse
import json
import logging
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from hebrew_readability.config import (FINAL_FEATURE_ORDER, PipelineConfig, read_table, save_table)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_and_save_calibration(
    y_train_preds: np.ndarray,
    levels_train: pd.Series,
    level_to_score: Dict[str, float],
    output_csv_path: str,
    output_json_path: str,
) -> Dict[str, Dict[str, float]]:
    """Saves canonical level_calibration_summary.csv and JSON mirror with interpolated Grades 7 & 10."""
    calib_df = pd.DataFrame({"pred": y_train_preds, "level": levels_train.astype(str).values})
    calibration_stats: Dict[str, Dict[str, float]] = {}

    for level, group in calib_df.groupby("level"):
        preds = group["pred"].values
        calibration_stats[str(level)] = {
            "target_readability_score": float(level_to_score[str(level)]),
            "mu": float(np.mean(preds)),
            "sigma": float(max(np.std(preds, ddof=1) if len(preds) > 1 else 0.0, 1e-5)),
            "P1": float(np.percentile(preds, 1)),
            "Q1": float(np.percentile(preds, 25)),
            "Q2": float(np.percentile(preds, 50)),
            "Q3": float(np.percentile(preds, 75)),
            "P99": float(np.percentile(preds, 99)),
        }

    for missing, lower, upper in [("7", "6", "8"), ("10", "9", "11")]:
        if (
            missing not in calibration_stats
            and lower in calibration_stats
            and upper in calibration_stats
        ):
            calibration_stats[missing] = {
                k: float((calibration_stats[lower][k] + calibration_stats[upper][k]) / 2.0)
                for k in calibration_stats[lower].keys()
            }
            logger.info(f"Interpolated Grade {missing} derived from {lower} and {upper}")

    summary_df = (
        pd.DataFrame.from_dict(calibration_stats, orient="index")
        .rename_axis("pedagogical_level")
        .reset_index()
        .sort_values(by="target_readability_score")
        .reset_index(drop=True)
    )
    summary_df.save_table(output_csv_path, index=False)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved canonical calibration summary to '{output_csv_path}' and '{output_json_path}'.")
    return calibration_stats


def main():
    parser = argparse.ArgumentParser(description="Train XGBRegressor and calibrate levels.")
    parser.add_argument("--artifact-dir", type=str, default=PipelineConfig().artifact_dir)
    args = parser.parse_args()

    cfg = PipelineConfig(artifact_dir=args.artifact_dir)
    train_df = read_table(os.path.join(cfg.artifact_dir, "train_dataset.csv"))
    scale_df = read_table(
        os.path.join(cfg.artifact_dir, "filtered_eta_weighted_1d_scale.csv"),
        dtype={cfg.level_col: str},
    )
    level_to_score = dict(zip(scale_df[cfg.level_col], scale_df[cfg.target_col]))

    X_train = train_df[FINAL_FEATURE_ORDER]
    y_train = train_df[cfg.target_col]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(cfg.artifact_dir, "xgboost_readability_regressor.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Trained and saved XGBRegressor to '{model_path}'.")

    y_train_preds = model.predict(X_train)
    build_and_save_calibration(
        y_train_preds=y_train_preds,
        levels_train=train_df[cfg.level_col],
        level_to_score=level_to_score,
        output_csv_path=os.path.join(cfg.artifact_dir, "level_calibration_summary.csv"),
        output_json_path=os.path.join(cfg.artifact_dir, "calibration_stats.json"),
    )


if __name__ == "__main__":
    main()