"""Step 2: Leak-proof StratifiedGroupKFold split, Stateful Preprocessing, EDA, and Hybrid 1D Scale."""

import argparse
import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hebrew_readability.config import (PipelineConfig, read_table, save_table)
from features.preprocessor import HebrewReadabilityPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def execute_stratified_group_split(
    df: pd.DataFrame, cfg: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by passage_title (zero leakage) while stratifying across pedagogical levels."""
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )
    y_labels = df[cfg.level_col].astype(str)
    groups = df[cfg.group_col].astype(str)

    train_idx, test_idx = next(sgkf.split(df, y_labels, groups))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_levels = set(train_df[cfg.level_col].astype(str).unique())
    test_levels = set(test_df[cfg.level_col].astype(str).unique())
    missing_in_train = test_levels - train_levels

    if missing_in_train:
        logger.warning(
            f"Levels {missing_in_train} have only 1 unique '{cfg.group_col}' group; "
            f"routing to Train so all levels have valid target anchors."
        )
        move_mask = test_df[cfg.level_col].astype(str).isin(missing_in_train)
        move_groups = set(test_df.loc[move_mask, cfg.group_col].astype(str).unique())
        group_move_mask = test_df[cfg.group_col].astype(str).isin(move_groups)

        train_df = pd.concat([train_df, test_df[group_move_mask]], ignore_index=True)
        test_df = test_df[~group_move_mask].reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    leakage = set(train_df[cfg.group_col]).intersection(set(test_df[cfg.group_col]))
    assert len(leakage) == 0, f"Data leakage detected across {len(leakage)} passages!"
    logger.info(
        f"Stratified Group Split Complete: Train={len(train_df)} rows, Test={len(test_df)} rows"
    )
    return train_df, test_df


def compute_eda_and_hybrid_scale(
    X_train_trans: pd.DataFrame,
    levels_train: pd.Series,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Computes Spearman matrix, per-level distributions, Eta-Squared weights, and 1D Target Scale."""
    df_calc = X_train_trans.copy()
    feature_cols = list(X_train_trans.columns)

    # 1. Spearman Correlation Matrix
    df_calc[feature_cols].corr(method="spearman").save_table(
        os.path.join(cfg.artifact_dir, "eda_spearman_correlation_matrix.csv")
    )

    # 2. Per-Level Feature Distributions
    df_calc[cfg.level_col] = levels_train.astype(str).values
    summary_wide = df_calc.groupby(cfg.level_col)[feature_cols].describe()
    try:
        summary_vert = (
            summary_wide.stack(level=0, future_stack=True)
            .reset_index()
            .rename(columns={"level_1": "Feature"})
        )
    except TypeError:
        summary_vert = (
            summary_wide.stack(level=0).reset_index().rename(columns={"level_1": "Feature"})
        )
    cols = ["Feature", cfg.level_col, "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    summary_vert[cols].save_table(
        os.path.join(cfg.artifact_dir, "feature_distributions_per_level.csv"), index=False
    )

    # 3. Eta-Squared & Hybrid Dimensional Weights (0.20 Nikkud / 0.80 Pool)
    eta_results = {}
    for feat in feature_cols:
        valid = df_calc[[cfg.level_col, feat]].dropna()
        grand_mean = valid[feat].mean()
        ss_total = ((valid[feat] - grand_mean) ** 2).sum()
        group_means = valid.groupby(cfg.level_col)[feat].transform("mean")
        ss_between = ((group_means - grand_mean) ** 2).sum()
        eta_results[feat] = round(float(ss_between / ss_total) if ss_total != 0 else 0.0, 4)

    eta_df = (
        pd.DataFrame(list(eta_results.items()), columns=["Feature", "Eta_Squared"])
        .sort_values(by="Eta_Squared", ascending=False)
        .reset_index(drop=True)
    )

    nikkud_mask = eta_df["Feature"].str.contains("nikkud", case=False)
    nikkud_df = eta_df[nikkud_mask].copy()
    rem_df = eta_df[~nikkud_mask].copy()

    nikkud_df["Normalized_Weight"] = cfg.fixed_nikkud_weight
    rem_sum = rem_df["Eta_Squared"].sum()
    rem_df["Normalized_Weight"] = (rem_df["Eta_Squared"] / rem_sum) * cfg.remaining_weight_pool

    final_eta_df = pd.concat([nikkud_df, rem_df]).sort_values(
        by="Normalized_Weight", ascending=False
    )
    final_eta_df.save_table(
        os.path.join(cfg.artifact_dir, "filtered_eta_squared_results.csv"), index=False
    )
    weights = dict(zip(final_eta_df["Feature"], final_eta_df["Normalized_Weight"]))

    # 4. Project 1D Target Scale from Medians & Save filtered_eta_weighted_1d_scale.csv
    medians_df = df_calc.groupby(cfg.level_col)[feature_cols].median().reset_index()
    weight_vec = np.array([weights[col] for col in feature_cols])
    medians_df[cfg.target_col] = (medians_df[feature_cols].values @ weight_vec) * 10.0
    medians_df.save_table(os.path.join(cfg.artifact_dir, "filtered_medians_vectors.csv"), index=False)

    scale_df = (
        medians_df[[cfg.level_col, cfg.target_col]]
        .sort_values(by=cfg.target_col)
        .reset_index(drop=True)
    )
    # Explicitly save the 1D scale CSV requested for HF upload
    scale_df.save_table(
        os.path.join(cfg.artifact_dir, "filtered_eta_weighted_1d_scale.csv"), index=False
    )

    level_to_score = dict(zip(scale_df[cfg.level_col].astype(str), scale_df[cfg.target_col]))
    return final_eta_df, level_to_score


def main():
    parser = argparse.ArgumentParser(description="Run EDA, Preprocessing, and 1D Target Scale.")
    parser.add_argument("--raw-data", type=str, default=PipelineConfig().raw_data_path)
    parser.add_argument("--artifact-dir", type=str, default=PipelineConfig().artifact_dir)
    args = parser.parse_args()

    cfg = PipelineConfig(raw_data_path=args.raw_data, artifact_dir=args.artifact_dir)
    os.makedirs(cfg.artifact_dir, exist_ok=True)

    df = read_table(cfg.raw_data_path)
    df = df.dropna(subset=[cfg.group_col, cfg.level_col] + cfg.selected_base_features).reset_index(
        drop=True
    )

    train_df_raw, test_df_raw = execute_stratified_group_split(df, cfg)

    preprocessor = HebrewReadabilityPreprocessor(
        selected_base_features=cfg.selected_base_features,
        inverse_base_features=cfg.inverse_base_features,
        skew_threshold=cfg.skew_threshold,
    )
    X_train = preprocessor.fit_transform(train_df_raw)
    X_test = preprocessor.transform(test_df_raw)

    # Save BOTH full preprocessor AND standalone minmax_scaler.pkl to artifacts/
    joblib.dump(preprocessor, os.path.join(cfg.artifact_dir, "readability_preprocessor.pkl"))
    joblib.dump(preprocessor.scaler_, os.path.join(cfg.artifact_dir, "minmax_scaler.pkl"))
    logger.info("Saved readability_preprocessor.pkl and minmax_scaler.pkl to artifacts/")

    _, level_to_score = compute_eda_and_hybrid_scale(X_train, train_df_raw[cfg.level_col], cfg)

    y_train = train_df_raw[cfg.level_col].astype(str).map(level_to_score)
    y_test = test_df_raw[cfg.level_col].astype(str).map(level_to_score)

    keep_meta = [c for c in cfg.metadata_cols if c in df.columns]
    train_export = pd.concat([train_df_raw[keep_meta], X_train], axis=1)
    train_export[cfg.target_col] = y_train.values
    train_export.save_table(os.path.join(cfg.artifact_dir, "train_dataset.csv"), index=False)

    test_export = pd.concat([test_df_raw[keep_meta], X_test], axis=1)
    test_export[cfg.target_col] = y_test.values
    test_export.save_table(os.path.join(cfg.artifact_dir, "test_dataset.csv"), index=False)
    logger.info("Saved train_dataset.csv and test_dataset.csv.")


if __name__ == "__main__":
    main()