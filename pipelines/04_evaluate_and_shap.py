"""Step 4: Full evaluation and SHAP suite (100% parity with classification_pipeline.ipynb)."""

import argparse
import logging
import os
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hebrew_readability.config import (FINAL_FEATURE_ORDER, PipelineConfig, read_table, save_table)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _run_shap_suite_on_subset(
    explainer: shap.TreeExplainer,
    X_sub: pd.DataFrame,
    levels_sub: pd.Series,
    out_dir: str,
    prefix_label: str = "Global",
):
    os.makedirs(out_dir, exist_ok=True)
    shap_values = explainer.shap_values(X_sub)
    feature_cols = list(X_sub.columns)
    level_str_series = levels_sub.astype(str)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "original_level", level_str_series.values)
    shap_df.save_table(os.path.join(out_dir, "all_computed_shap_values.csv"), index=False)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sub, show=False)
    plt.title(f"{prefix_label} SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(out_dir, "shap_summary_beeswarm.png"), bbox_inches="tight")
    plt.close("all")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_imp = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(12, 8))
    plt.title(f"{prefix_label} Feature Importance (Mean Absolute SHAP Value)")
    bars = plt.barh(global_imp.index, global_imp.values, color="steelblue")
    for bar in bars:
        w = bar.get_width()
        plt.text(
            w,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_shap_bar_plot.png"), bbox_inches="tight")
    plt.close("all")

    level_shap_summary = []
    for level in level_str_series.unique():
        mask = (level_str_series == level).values
        if not np.any(mask):
            continue
        lvl_mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        lvl_imp = pd.Series(lvl_mean_abs, index=feature_cols).sort_values(ascending=True)
        for feat, val in lvl_imp.items():
            level_shap_summary.append(
                {"Level": level, "Feature": feat, "Mean_Abs_SHAP": float(val)}
            )

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance for Level: {level} ({prefix_label})")
        bars = plt.barh(lvl_imp.index, lvl_imp.values, color="darkorange")
        for bar in bars:
            w = bar.get_width()
            plt.text(
                w,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        plt.tight_layout()
        safe_lvl = str(level).replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(out_dir, f"shap_bar_level_{safe_lvl}.png"), bbox_inches="tight")
        plt.close("all")

    pd.DataFrame(level_shap_summary).sort_values(
        by=["Level", "Mean_Abs_SHAP"], ascending=[True, False]
    ).save_table(os.path.join(out_dir, "all_shap_values_per_level.csv"), index=False)

    for feat in feature_cols:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_sub, show=False, interaction_index=None)
        plt.title(f"SHAP Decision Boundary ({prefix_label}): {feat}")
        plt.tight_layout()
        plt.gcf().savefig(
            os.path.join(out_dir, f"shap_dependence_{feat}.png"), bbox_inches="tight"
        )
        plt.close("all")


def evaluate_and_explain(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df_export: pd.DataFrame,
    calibration_stats: Dict[str, Dict[str, float]],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    levels_test = test_df_export[cfg.level_col].astype(str)

    # 1. Standard Global & Per-Level Mathematical Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    report_lines = [
        "STANDARD GLOBAL MATHEMATICAL METRICS",
        "=" * 50,
        f"Mean Absolute Error (MAE): {mae:.4f}",
        f"Root Mean Squared Error (RMSE): {rmse:.4f}",
        f"R-squared (R2): {r2:.4f}\n",
        "=" * 50,
        "EVALUATION PER PEDAGOGICAL LEVEL",
        "=" * 50,
    ]
    for lvl in sorted(levels_test.unique()):
        mask = (levels_test == lvl).values
        lvl_mae = mean_absolute_error(y_test.values[mask], y_pred[mask])
        lvl_rmse = float(np.sqrt(mean_squared_error(y_test.values[mask], y_pred[mask])))
        report_lines.append(f"Level: {lvl:<25} | MAE: {lvl_mae:>6.4f} | RMSE: {lvl_rmse:>6.4f}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(
        os.path.join(cfg.artifact_dir, "model_evaluation_metrics.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    # 2. SHAP Suite (All Levels + Advanced Subset > 4.4)
    explainer = shap.TreeExplainer(model)
    _run_shap_suite_on_subset(explainer, X_test, levels_test, cfg.shap_dir, prefix_label="All_Levels")

    adv_mask = (y_test > cfg.advanced_score_threshold).values
    if np.any(adv_mask):
        _run_shap_suite_on_subset(
            explainer,
            X_test.iloc[adv_mask].reset_index(drop=True),
            levels_test.iloc[adv_mask].reset_index(drop=True),
            cfg.shap_advanced_dir,
            prefix_label=f"Advanced_Score_gt_{cfg.advanced_score_threshold}",
        )

    # 3. Probabilistic Range & Coverage Evaluation (Core Q1-Q3 & Extended P1-P99)
    eval_out = test_df_export.copy()
    eval_out["Y_pred"] = y_pred

    in_core_hits, in_ext_hits, top_1_hits = 0, 0, 0
    boundary_penalties, directional_errors, mapped_core, mapped_ext = [], [], [], []

    for pred, true_lvl in zip(y_pred, levels_test.values):
        core_matched = [lvl for lvl, s in calibration_stats.items() if s["Q1"] <= pred <= s["Q3"]]
        ext_matched = [lvl for lvl, s in calibration_stats.items() if s["P1"] <= pred <= s["P99"]]

        mapped_core.append(", ".join(core_matched) if core_matched else "None")
        mapped_ext.append(", ".join(ext_matched) if ext_matched else "Out of Bounds")

        if true_lvl in core_matched:
            in_core_hits += 1
        if true_lvl in ext_matched:
            in_ext_hits += 1

        eval_levels = ext_matched if ext_matched else list(calibration_stats.keys())
        pdf_scores = {
            lvl: norm.pdf(pred, calibration_stats[lvl]["mu"], calibration_stats[lvl]["sigma"])
            for lvl in eval_levels
        }
        total_pdf = sum(pdf_scores.values())
        if total_pdf > 0:
            top_pred = max(pdf_scores, key=pdf_scores.get)
            if top_pred == true_lvl:
                top_1_hits += 1

        true_stats = calibration_stats.get(true_lvl)
        if true_stats:
            if pred < true_stats["P1"]:
                penalty = true_stats["P1"] - pred
            elif pred > true_stats["P99"]:
                penalty = pred - true_stats["P99"]
            else:
                penalty = 0.0
            dir_err = pred - true_stats["mu"]
        else:
            penalty, dir_err = 0.0, 0.0

        boundary_penalties.append(penalty)
        directional_errors.append(dir_err)

    eval_out["Boundary_Penalty"] = boundary_penalties
    eval_out["Directional_Error"] = directional_errors
    eval_out["Raw_Residual"] = y_pred - y_test.values
    eval_out["Core_IQR_Mapped_Levels"] = mapped_core
    eval_out["Mapped_Pedagogical_Levels"] = mapped_ext

    total_samples = len(eval_out)
    mean_bias = float(np.mean(directional_errors))
    bias_str = (
        f"OVERESTIMATION (+{mean_bias:.4f})"
        if mean_bias > 0
        else f"UNDERESTIMATION ({mean_bias:.4f})"
    )

    print("\n" + "=" * 55)
    print("STAGE 2 & 3: INFERENCE & EVALUATION (TEST SET)")
    print("=" * 55)
    print(f"Extended Coverage (P1-P99 Hit Rate):   {(in_ext_hits / total_samples) * 100:.1f}%")
    print(f"Core Coverage (IQR Hit Rate):          {(in_core_hits / total_samples) * 100:.1f}%")
    print(f"Top-1 Probability Accuracy:            {(top_1_hits / total_samples) * 100:.1f}%")
    print(f"Mean Boundary Penalty (Severity):      {np.mean(boundary_penalties):.4f} scale points")
    print(f"Systematic Model Bias:                 {bias_str}")

    # 4. Error Divergence Visualization
    level_div = (
        eval_out.groupby(cfg.level_col)["Boundary_Penalty"].mean().sort_values(ascending=False)
    )
    print("\nMean Boundary Penalty per Level:")
    for lvl, pen in level_div.items():
        print(f" - {str(lvl):<25}: {pen:.4f}")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [str(k) for k in level_div.index], level_div.values, color="coral", edgecolor="black"
    )
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.title("Error Divergence: Mean Boundary Penalty by Pedagogical Level")
    plt.xlabel("Original Pedagogical Level")
    plt.ylabel("Mean Absolute Distance from P1/P99 Extended Range")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.artifact_dir, "error_divergence_by_level.png"))
    plt.close("all")

    # 5. Error Severity & Residuals Histograms
    non_zero_pen = [p for p in boundary_penalties if p > 0]
    plt.figure(figsize=(10, 6))
    if non_zero_pen:
        plt.hist(non_zero_pen, bins=20, color="crimson", edgecolor="black", alpha=0.75)
    plt.title("Pedagogical Error Severity (Distance from P1/P99 Boundaries)")
    plt.xlabel("Absolute Distance from Extended Range (P1/P99)")
    plt.ylabel("Number of Mismatched Texts")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.artifact_dir, "pedagogical_error_severity.png"))
    plt.close("all")

    plt.figure(figsize=(10, 6))
    plt.hist(eval_out["Raw_Residual"], bins=30, color="slateblue", edgecolor="black", alpha=0.75)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Residuals Distribution (Predicted Score - Target Anchor Score)")
    plt.xlabel("Residual (Y_pred - Y_true)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.artifact_dir, "residuals_histogram.png"))
    plt.close("all")

    return eval_out


def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBRegressor and run SHAP analysis.")
    parser.add_argument("--artifact-dir", type=str, default=PipelineConfig().artifact_dir)
    args = parser.parse_args()

    cfg = PipelineConfig(artifact_dir=args.artifact_dir)
    model: xgb.XGBRegressor = joblib.load(
        os.path.join(cfg.artifact_dir, "xgboost_readability_regressor.pkl")
    )
    test_df = read_table(os.path.join(cfg.artifact_dir, "test_dataset.csv"))
    calib_df = read_table(
        os.path.join(cfg.artifact_dir, "level_calibration_summary.csv"),
        dtype={"pedagogical_level": str},
    )
    calibration_stats = calib_df.set_index("pedagogical_level").to_dict(orient="index")

    X_test = test_df[FINAL_FEATURE_ORDER]
    y_test = test_df[cfg.target_col]

    eval_df = evaluate_and_explain(model, X_test, y_test, test_df, calibration_stats, cfg)
    eval_df.save_table(
        os.path.join(cfg.artifact_dir, "test_dataset_with_predictions.csv"), index=False
    )
    logger.info("Evaluation and SHAP analysis complete.")


if __name__ == "__main__":
    main()