"""
Phase 4 (Deliverable 6): Comprehensive Quality & Proximity Analysis Script.

Evaluates each row of experimentation_results.csv against its target_level across
multiple proximity metrics (Exact IQR Hit, Cluster Hit, Broad Category Hit,
Target Probability Mass, Expected Grade Error, and Min IQR Grade Distance) and
outputs method-level and level-wise quality tables and visualizations.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hebrew_readability.config import LEVEL_TO_CATEGORIES

# Numeric grade / complexity equivalents for ordinal proximity calculation
LEVEL_NUMERIC_GRADE: Dict[str, float] = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "'2-4": 3.0,
    "2-4": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "'4-9": 6.5,
    "4-9": 6.5,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
    "10": 10.0,
    "11": 11.0,
    "12": 12.0,
    "Bagrut": 11.5,
    "Bagrut_Olim": 10.5,
    "hearing_impaired": 11.0,
    "Bagrut_hearing_impaired": 11.0,
    "מבחני אולפן - רגיל": 5.5,
    "Olim_ragil": 5.5,
    "מבחני אולפן - מתקדמים": 7.5,
    "Olim_mitkadmim": 7.5,
    "Hadshon": 3.5,
    "Olim_news": 3.5,
    "intellectual_disability": 3.5,
}

RAW_LEVEL_ALIASES: Dict[str, Set[str]] = {
    "1": {"1", "2"},
    "2": {"2"},
    "3": {"3"},
    "'2-4": {"'2-4", "2-4"},
    "2-4": {"'2-4", "2-4"},
    "4": {"4"},
    "5": {"5"},
    "6": {"6"},
    "'4-9": {"'4-9", "4-9"},
    "4-9": {"'4-9", "4-9"},
    "7": {"7"},
    "8": {"8"},
    "9": {"9"},
    "10": {"10"},
    "11": {"11"},
    "12": {"12"},
    "Bagrut": {"Bagrut", "Bagrut_Olim"},
    "Bagrut_Olim": {"Bagrut_Olim", "Bagrut"},
    "hearing_impaired": {"hearing_impaired", "Bagrut_hearing_impaired"},
    "Bagrut_hearing_impaired": {"Bagrut_hearing_impaired", "hearing_impaired"},
    "intellectual_disability": {"intellectual_disability"},
    "Hadshon": {"Hadshon", "Olim_news"},
    "Olim_news": {"Olim_news", "Hadshon"},
    "Olim_ragil": {"Olim_ragil", "מבחני אולפן - רגיל"},
    "מבחני אולפן - רגיל": {"מבחני אולפן - רגיל", "Olim_ragil"},
    "Olim_mitkadmim": {"Olim_mitkadmim", "מבחני אולפן - מתקדמים"},
    "מבחני אולפן - מתקדמים": {"מבחני אולפן - מתקדמים", "Olim_mitkadmim"},
}


def parse_simplified_level_col(s: str) -> List[Tuple[str, float]]:
    """Parses '11 (13.0%), 12 (12.2%)' into [('11', 13.0), ('12', 12.2)]."""
    if not isinstance(s, str) or s.startswith("NONE_IN_IQR"):
        return []
    parsed: List[Tuple[str, float]] = []
    for part in s.split(","):
        part = part.strip()
        m = re.match(r"^(.*?)\s*\(([\d\.]+)%\)$", part)
        if m:
            parsed.append((m.group(1).strip(), float(m.group(2))))
        elif part:
            parsed.append((part, 0.0))
    return parsed


def evaluate_row_proximity(row: pd.Series) -> Dict[str, object]:
    """Computes multi-dimensional proximity metrics for a single experiment row."""
    t_lvl = str(row["target_level"]).strip()
    t_aliases = RAW_LEVEL_ALIASES.get(t_lvl, {t_lvl, f"'{t_lvl}", t_lvl.lstrip("'")})
    t_cats = set(
        LEVEL_TO_CATEGORIES.get(t_lvl)
        or LEVEL_TO_CATEGORIES.get(t_lvl.lstrip("'"))
        or []
    )
    t_grade = LEVEL_NUMERIC_GRADE.get(
        t_lvl, LEVEL_NUMERIC_GRADE.get(t_lvl.lstrip("'"), np.nan)
    )

    parsed = parse_simplified_level_col(str(row["simplified_level"]))
    matched_lvls = [lvl for lvl, _ in parsed]
    matched_set = set(matched_lvls)
    prob_map = {lvl: prob for lvl, prob in parsed}

    # 1. Exact Raw Level IQR Hit
    exact_iqr_hit = len(t_aliases & matched_set) > 0

    # 2. Cluster-Inclusive IQR Hit ('2-4 covers 2..4, '4-9 covers 4..9)
    cluster_iqr_hit = exact_iqr_hit
    if t_lvl in {"2", "3", "4"} and ({"'2-4", "2-4"} & matched_set):
        cluster_iqr_hit = True
    if t_lvl in {"4", "5", "6", "7", "8", "9"} and ({"'4-9", "4-9"} & matched_set):
        cluster_iqr_hit = True

    # 3. Broad Pedagogical Category Hit
    matched_cats: Set[str] = set()
    for lvl in matched_lvls:
        matched_cats.update(
            LEVEL_TO_CATEGORIES.get(lvl)
            or LEVEL_TO_CATEGORIES.get(lvl.lstrip("'"))
            or []
        )
    category_iqr_hit = len(t_cats & matched_cats) > 0

    # 4. Probability Mass Assigned to Exact Target & Target Category
    target_prob_pct = sum(prob_map.get(a, 0.0) for a in t_aliases)
    cat_prob_sum_pct = sum(
        prob
        for lvl, prob in parsed
        if len(
            t_cats
            & set(
                LEVEL_TO_CATEGORIES.get(lvl)
                or LEVEL_TO_CATEGORIES.get(lvl.lstrip("'"))
                or []
            )
        )
        > 0
    )

    # 5. Ordinal Grade Proximity Metrics
    top1_lvl = matched_lvls[0] if matched_lvls else "NONE"
    valid_grades = [
        (LEVEL_NUMERIC_GRADE[lvl], prob)
        for lvl, prob in parsed
        if lvl in LEVEL_NUMERIC_GRADE
    ]
    if valid_grades and not np.isnan(t_grade):
        total_w = sum(p if p > 0 else 1.0 for _, p in valid_grades)
        expected_iqr_grade = (
            sum(g * (p if p > 0 else 1.0) for g, p in valid_grades) / total_w
        )
        weighted_grade_err = abs(expected_iqr_grade - t_grade)
        min_iqr_grade_err = (
            0.0
            if exact_iqr_hit
            else min(abs(g - t_grade) for g, _ in valid_grades)
        )
    else:
        expected_iqr_grade = np.nan
        weighted_grade_err = np.nan
        min_iqr_grade_err = np.nan

    # 6. Compression ratio (word count reduction vs source text)
    src_words = max(len(str(row["text"]).split()), 1)
    simp_words = len(str(row["simplified_text"]).split())
    word_compression_ratio = round(simp_words / src_words, 2)

    return {
        "paragraph_id": row["paragraph_id"],
        "target_level": t_lvl,
        "prompt": row["prompt"],
        "original_level": row["original_level"],
        "simplified_score": float(row["simplified_score"]),
        "exact_iqr_hit": exact_iqr_hit,
        "cluster_iqr_hit": cluster_iqr_hit,
        "category_iqr_hit": category_iqr_hit,
        "target_prob_pct": round(target_prob_pct, 2),
        "cat_prob_sum_pct": round(cat_prob_sum_pct, 2),
        "top1_matched_level": top1_lvl,
        "expected_iqr_grade": round(expected_iqr_grade, 2)
        if not np.isnan(expected_iqr_grade)
        else np.nan,
        "weighted_grade_err": round(weighted_grade_err, 2)
        if not np.isnan(weighted_grade_err)
        else np.nan,
        "min_iqr_grade_err": round(min_iqr_grade_err, 2)
        if not np.isnan(min_iqr_grade_err)
        else np.nan,
        "simplified_words": simp_words,
        "word_compression_ratio": word_compression_ratio,
    }


def run_full_analysis(
    input_csv: Path,
    output_dir: Path = Path("artifacts/analysis"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs row-by-row and method-level quality analysis and saves tables + plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    row_metrics = pd.DataFrame([evaluate_row_proximity(row) for _, row in df.iterrows()])
    row_csv_path = output_dir / "row_level_proximity_analysis.csv"
    row_metrics.to_csv(row_csv_path, index=False, encoding="utf-8-sig")

    # Method-level summary
    method_summary = (
        row_metrics.groupby("prompt", as_index=False)
        .agg(
            n_rows=("paragraph_id", "count"),
            exact_iqr_hit_pct=("exact_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            cluster_iqr_hit_pct=("cluster_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            category_iqr_hit_pct=("category_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            mean_target_prob_pct=("target_prob_pct", lambda s: round(s.mean(), 2)),
            mean_cat_prob_sum_pct=("cat_prob_sum_pct", lambda s: round(s.mean(), 2)),
            mean_weighted_grade_err=("weighted_grade_err", lambda s: round(s.mean(), 2)),
            mean_min_iqr_grade_err=("min_iqr_grade_err", lambda s: round(s.mean(), 2)),
            mean_simplified_score=("simplified_score", lambda s: round(s.mean(), 4)),
            std_simplified_score=("simplified_score", lambda s: round(s.std(), 4)),
            mean_words=("simplified_words", lambda s: round(s.mean(), 1)),
            mean_compression_ratio=("word_compression_ratio", lambda s: round(s.mean(), 2)),
        )
        .sort_values(
            ["exact_iqr_hit_pct", "mean_cat_prob_sum_pct", "mean_weighted_grade_err"],
            ascending=[False, False, True],
        )
    )
    summary_csv_path = output_dir / "method_quality_summary.csv"
    method_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print("1. ROW-LEVEL PROXIMITY TO TARGET LEVEL")
    print("=" * 95)
    print(
        row_metrics[
            [
                "paragraph_id",
                "target_level",
                "prompt",
                "simplified_score",
                "exact_iqr_hit",
                "cluster_iqr_hit",
                "category_iqr_hit",
                "cat_prob_sum_pct",
                "expected_iqr_grade",
                "weighted_grade_err",
                "min_iqr_grade_err",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 95)
    print("2. AGGREGATED SIMPLIFICATION METHOD QUALITY RANKING")
    print("=" * 95)
    print(method_summary.to_string(index=False))
    print("=" * 95 + "\n")

    # Generate comparison plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(
        data=row_metrics,
        x="target_level",
        y="simplified_score",
        hue="prompt",
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Simplified Continuous Score by Target Level", fontweight="bold")
    axes[0].set_xlabel("Target Raw Level")
    axes[0].set_ylabel("Simplified Score (Y_pred)")

    sns.barplot(
        data=row_metrics,
        x="target_level",
        y="weighted_grade_err",
        hue="prompt",
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_title("Weighted IQR Grade Error (Lower = Better)", fontweight="bold")
    axes[1].set_xlabel("Target Raw Level")
    axes[1].set_ylabel("|Expected IQR Grade - Target Grade|")

    sns.barplot(
        data=row_metrics,
        x="target_level",
        y="cat_prob_sum_pct",
        hue="prompt",
        ax=axes[2],
        palette="viridis",
    )
    axes[2].set_title("Target Category Probability Mass (%)", fontweight="bold")
    axes[2].set_xlabel("Target Raw Level")
    axes[2].set_ylabel("Cumulative Target Category Probability (%)")

    plt.tight_layout()
    plot_path = output_dir / "experimentation_quality_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved row analysis -> {row_csv_path}")
    print(f"[✓] Saved method summary -> {summary_csv_path}")
    print(f"[✓] Saved comparison plot -> {plot_path}")

    return row_metrics, method_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experimentation_results.csv")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/experimentation_results.csv"),
        help="Path to experimentation_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis"),
        help="Directory to save analysis CSVs and charts",
    )
    args, _ = parser.parse_known_args()
    run_full_analysis(input_csv=args.input_csv, output_dir=args.output_dir)
