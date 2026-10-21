"""
ChurnSense - Exploratory Data Analysis (EDA) Module
Generates churn-focused visualizations with mandatory data-grounded insights
underneath every chart.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.chart_insights import (
    get_categorical_churn_insight,
    get_churn_dist_insight,
    get_correlation_insight,
    get_missing_values_insight,
    render_chart_with_insight,
)


# Set visual styling for dark theme
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#161b22",
    "figure.facecolor": "#0d1117",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#30363d"
})


def run_full_eda(
    df: pd.DataFrame,
    target_col: str,
    positive_class: Any,
    excluded_cols: Optional[List[str]] = None
) -> None:
    """
    Renders the complete EDA dashboard in Streamlit with dynamic,
    data-grounded interpretations beneath every single chart.
    """
    if excluded_cols is None:
        excluded_cols = []

    st.markdown("### 🔍 Exploratory Data Analysis & Churn Diagnostic")

    total_rows = len(df)
    y_binary = (df[target_col] == positive_class).astype(int)
    churn_count = int(y_binary.sum())
    non_churn_count = total_rows - churn_count
    churn_rate = float(y_binary.mean())

    # --------------------------------------------------------------------------
    # 1. Target Class Distribution
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Non-Churn", "Churn (Positive)"],
        [non_churn_count, churn_count],
        color=["#238636", "#da3633"],
        width=0.5
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + total_rows * 0.015,
            f"{yval:,} ({(yval/total_rows)*100:.1f}%)",
            ha='center', va='bottom', color='#f0f6fc', fontweight='bold'
        )
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.set_ylim(0, max(non_churn_count, churn_count) * 1.18)

    insight = get_churn_dist_insight(total_rows, churn_count, non_churn_count, churn_rate)
    render_chart_with_insight(fig, f"Target Distribution: '{target_col}'", insight)

    # --------------------------------------------------------------------------
    # 2. Missing Values Analysis
    # --------------------------------------------------------------------------
    missing_counts = df.isnull().sum()
    missing_dict = {str(col): int(cnt) for col, cnt in missing_counts.items() if cnt > 0}

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if missing_dict:
        cols = list(missing_dict.keys())
        counts = list(missing_dict.values())
        ax.barh(cols, counts, color="#f0883e")
        ax.set_xlabel("Missing Count")
    else:
        # Show all-zero bar or status
        ax.text(0.5, 0.5, "No Missing Values Found in Dataset", ha='center', va='center',
                fontsize=14, color="#3fb950", fontweight='bold')
        ax.set_axis_off()

    missing_insight = get_missing_values_insight(missing_dict, total_rows)
    render_chart_with_insight(fig, "Missing Values Profile", missing_insight)

    # --------------------------------------------------------------------------
    # 3. Correlation Heatmap (Numeric Features)
    # --------------------------------------------------------------------------
    numeric_df = df.select_dtypes(include=['number']).copy()
    # Add binarized target for correlation
    numeric_df["Churn_Target"] = y_binary.values

    # Remove identifier columns
    for col in excluded_cols:
        if col in numeric_df.columns:
            numeric_df.drop(columns=[col], inplace=True)

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar=True,
            ax=ax,
            linewidths=0.5
        )
        corr_insight = get_correlation_insight(corr)
        render_chart_with_insight(fig, "Correlation Heatmap (Numeric Features)", corr_insight)

    # --------------------------------------------------------------------------
    # 4. Categorical Churn Rates
    # --------------------------------------------------------------------------
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                if c != target_col and c not in excluded_cols and df[c].nunique() <= 10]

    # Select up to top 3 categorical features for display
    for col in cat_cols[:3]:
        grouped = y_binary.groupby(df[col], observed=False).agg(
            total='count',
            churners='sum',
            rate='mean'
        ).reset_index().sort_values('rate', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(grouped[col].astype(str), grouped['rate'] * 100, color="#58a6ff", width=0.5)
        ax.axhline(churn_rate * 100, color="#f85149", linestyle="--", label=f"Baseline Churn ({churn_rate*100:.1f}%)")

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.2,
                f"{yval:.1f}%",
                ha='center', va='bottom', color='#f0f6fc', fontsize=10, fontweight='bold'
            )

        ax.set_ylabel("Observed Churn Rate (%)", fontsize=11)
        ax.set_ylim(0, max(grouped['rate'] * 100) * 1.25)
        ax.legend(loc="upper right")
        plt.xticks(rotation=15, ha='right')

        high_grp = str(grouped.iloc[0][col])
        high_rate = float(grouped.iloc[0]['rate'])
        low_grp = str(grouped.iloc[-1][col])
        low_rate = float(grouped.iloc[-1]['rate'])

        cat_insight = get_categorical_churn_insight(
            col, high_grp, high_rate, low_grp, low_rate, churn_rate
        )
        render_chart_with_insight(fig, f"Observed Churn Rate by '{col}'", cat_insight)

    # --------------------------------------------------------------------------
    # 5. Key Numerical Distributions (e.g. tenure, MonthlyCharges)
    # --------------------------------------------------------------------------
    num_cols_to_plot = [c for c in numeric_df.columns if c != "Churn_Target"][:2]
    for col in num_cols_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=col,
            hue=y_binary.map({0: "Non-Churn", 1: "Churn"}),
            kde=True,
            palette={"Non-Churn": "#238636", "Churn": "#da3633"},
            ax=ax,
            bins=25
        )
        ax.set_ylabel("Customer Count")

        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        skew_raw: Any = df[col].skew()
        skew_val = float(skew_raw) if skew_raw is not None else 0.0

        skew_desc = "positively skewed (right tail)" if skew_val > 0.5 else ("negatively skewed (left tail)" if skew_val < -0.5 else "approximately symmetric")
        num_insight = {
            "tells_us": (
                f"Feature '{col}' exhibits mean = {mean_val:.2f}, std = {std_val:.2f}, and skewness = {skew_val:.2f} ({skew_desc}). "
                "The overlay shows how churners compare to non-churners across values."
            ),
            "why_it_matters": (
                f"Distribution spread reveals key risk thresholds (e.g., tenure concentration in early months). "
                "Skewed numeric variables require robust scaling (standardization/median imputation) to prevent outlier distortion."
            ),
            "limitation": (
                "Bivariate histograms show density overlaps, but interactions with other features are not visible in 1D projections."
            )
        }
        render_chart_with_insight(fig, f"Distribution of '{col}' by Churn Status", num_insight)
