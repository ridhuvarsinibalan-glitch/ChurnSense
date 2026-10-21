"""
ChurnSense - Mandatory Chart Insights Component
Renders every visualization with an automatically generated, data-grounded
interpretation directly beneath the chart.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def render_chart_with_insight(
    fig: plt.Figure,
    title: str,
    insight: Dict[str, str],
    key: Optional[str] = None
) -> None:
    """
    Renders a Matplotlib figure followed immediately by a structured,
    data-grounded insight container containing:
    1. What this graph tells us
    2. Why it matters
    3. Caution / Limitation
    """
    st.markdown(f"#### 📊 {title}")
    st.pyplot(fig, clear_figure=True)

    # Structured insight container directly underneath the chart
    st.markdown(
        f"""
        <div style="background-color: #161b22; border-left: 4px solid #00ffcc; border-radius: 6px; padding: 14px 18px; margin-top: -10px; margin-bottom: 24px;">
            <p style="margin: 0 0 8px 0; font-size: 0.95rem; color: #f0f6fc;">
                <strong style="color: #00ffcc;">📌 What this graph tells us:</strong> {insight.get('tells_us', '')}
            </p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem; color: #f0f6fc;">
                <strong style="color: #58a6ff;">💡 Why it matters:</strong> {insight.get('why_it_matters', '')}
            </p>
            <p style="margin: 0; font-size: 0.90rem; color: #8b949e;">
                <strong style="color: #f0883e;">⚠️ Caution / Limitation:</strong> {insight.get('limitation', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close(fig)


# ==============================================================================
# DYNAMIC INSIGHT GENERATORS GROUNDED IN COMPUTED VALUES
# ==============================================================================

def get_churn_dist_insight(
    total: int,
    churn_cnt: int,
    non_churn_cnt: int,
    churn_rate: float
) -> Dict[str, str]:
    is_imbalanced = churn_rate < 0.35 or churn_rate > 0.65
    imbalance_text = "exhibits notable class imbalance" if is_imbalanced else "is relatively balanced"

    return {
        "tells_us": (
            f"Of the {total:,} total customers, {churn_cnt:,} ({churn_rate*100:.1f}%) have churned (positive class), "
            f"while {non_churn_cnt:,} ({(1-churn_rate)*100:.1f}%) remain active. The dataset {imbalance_text}."
        ),
        "why_it_matters": (
            f"Because churn represents only {churn_rate*100:.1f}% of the sample, a naive classifier predicting 'No Churn' for every customer "
            f"would achieve an unearned accuracy of {(1-churn_rate)*100:.1f}%. Models must be optimized and evaluated using Recall, F1, and PR-AUC instead of Accuracy."
        ),
        "limitation": (
            "Historical class prevalence in this cohort may shift over time if broader market conditions, pricing changes, or competitive actions occur."
        )
    }


def get_missing_values_insight(missing_summary: Dict[str, int], total_rows: int) -> Dict[str, str]:
    if not missing_summary or sum(missing_summary.values()) == 0:
        return {
            "tells_us": f"Zero missing values detected across all {total_rows:,} records.",
            "why_it_matters": "The dataset is complete across all analyzed attributes, eliminating the risk of imputation bias during model training.",
            "limitation": "A zero-null count does not preclude silent missing representations, such as whitespace strings, placeholder codes ('999'), or unknown sentinel values."
        }

    top_missing_col, top_missing_cnt = max(missing_summary.items(), key=lambda x: x[1])
    pct = (top_missing_cnt / total_rows) * 100
    return {
        "tells_us": f"Missing values detected. The highest missing count occurs in '{top_missing_col}' with {top_missing_cnt:,} missing entries ({pct:.1f}% of records).",
        "why_it_matters": "Features with missing values require systematic imputation (median for numeric, mode for categorical) to prevent runtime exceptions and bias.",
        "limitation": "Simple imputation assumes data is Missing At Random (MAR); if non-response carries informative intent, imputation can introduce systematic error."
    }


def get_correlation_insight(corr_matrix: pd.DataFrame) -> Dict[str, str]:
    # Extract highest off-diagonal correlation safely
    cols = list(corr_matrix.columns)
    top_pair = None
    max_abs_corr = -1.0
    arr = corr_matrix.to_numpy(dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                val = float(arr[i, j])
                if np.isnan(val):
                    continue
                if abs(val) > max_abs_corr:
                    max_abs_corr = abs(val)
                    top_pair = (str(cols[i]), str(cols[j]), val)
            except Exception:
                continue

    if top_pair is not None:
        c1, c2, top_val = top_pair
        pair_desc = f"The strongest observed linear correlation is between '{c1}' and '{c2}' (r = {top_val:.2f})."
    else:
        pair_desc = "Linear correlations among numerical features are weak to moderate."

    return {
        "tells_us": f"{pair_desc} Collinear pairs reflect overlapping information.",
        "why_it_matters": "High multicollinearity can destabilize coefficient estimates in linear models, although tree-based ensembles are generally robust to collinearity.",
        "limitation": "Pearson correlation only captures linear relationships and does not detect non-linear dependencies. High correlation does not imply causality."
    }


def get_categorical_churn_insight(
    feature: str,
    high_group: str,
    high_rate: float,
    low_group: str,
    low_rate: float,
    overall_rate: float
) -> Dict[str, str]:
    diff = (high_rate - overall_rate) * 100
    return {
        "tells_us": (
            f"In '{feature}', the highest observed churn rate is {high_rate*100:.1f}% in the '{high_group}' cohort "
            f"({diff:+.1f}% points vs. company baseline of {overall_rate*100:.1f}%). "
            f"The lowest observed churn rate is {low_rate*100:.1f}% in the '{low_group}' cohort."
        ),
        "why_it_matters": (
            f"Segment disparities indicate that '{feature}' provides strong discriminatory signal for classification models "
            "and represents a primary target for tailored operational retention playbooks."
        ),
        "limitation": (
            "These differences are observational associations in historical data. Altering a customer's category does not guarantee an immediate reduction in churn risk."
        )
    }


def get_confusion_matrix_insight(
    tn: int, fp: int, fn: int, tp: int,
    precision: float, recall: float, f1: float,
    threshold: float
) -> Dict[str, str]:
    total = tn + fp + fn + tp
    return {
        "tells_us": (
            f"At threshold {threshold:.2f}, the model correctly identified {tp:,} churners (TP) and {tn:,} non-churners (TN). "
            f"It missed {fn:,} actual churners (FN) and produced {fp:,} false alarms (FP). "
            f"This yields Precision = {precision:.3f}, Recall = {recall:.3f}, and F1 = {f1:.3f}."
        ),
        "why_it_matters": (
            f"False Negatives ({fn:,} missed churners) represent direct revenue attrition where customers leave undetected. "
            f"False Positives ({fp:,} false alarms) represent unnecessary operational outreach or discounted retention costs."
        ),
        "limitation": (
            "The confusion matrix depends directly on the chosen decision threshold. Adjusting the threshold shifts the balance between FP and FN."
        )
    }


def get_roc_curve_insight(roc_auc: Optional[float], model_name: str) -> Dict[str, str]:
    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    return {
        "tells_us": (
            f"The ROC curve plots the True Positive Rate against the False Positive Rate across all thresholds for {model_name}. "
            f"Achieved Area Under the ROC Curve (ROC-AUC) = {auc_str}."
        ),
        "why_it_matters": (
            f"An AUC of {auc_str} measures the model's ability to rank a randomly chosen churner higher than a non-churner. "
            "It confirms the model's overall discriminatory power independent of any specific threshold."
        ),
        "limitation": (
            "ROC-AUC can be overly optimistic on imbalanced datasets because the False Positive Rate denominator (TN + FP) "
            "is dominated by the large majority class. It should always be viewed alongside the PR-AUC curve."
        )
    }


def get_pr_curve_insight(pr_auc: Optional[float], baseline_rate: float, model_name: str) -> Dict[str, str]:
    pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
    return {
        "tells_us": (
            f"The Precision-Recall curve tracks Precision across all Recall levels for {model_name}. "
            f"Achieved PR-AUC (Average Precision) = {pr_str}, compared to a random guessing baseline of {baseline_rate:.3f}."
        ),
        "why_it_matters": (
            "PR-AUC focuses strictly on the positive churn class, providing the most reliable assessment of practical utility "
            "under class imbalance."
        ),
        "limitation": (
            "PR curves do not incorporate True Negatives; a model with high PR-AUC may still produce a high absolute count of False Positives if overall cohort size is large."
        )
    }


def get_model_comparison_insight(
    winner: str,
    metric_name: str,
    winner_score: float,
    runner_up: Optional[str],
    runner_up_score: Optional[float]
) -> Dict[str, str]:
    runner_text = f"followed closely by {runner_up} ({runner_up_score:.4f})" if runner_up else "with no runner-up"
    diff_text = f" (Δ = {winner_score - runner_up_score:.4f})" if runner_up_score is not None else ""
    return {
        "tells_us": (
            f"Under the primary metric {metric_name.upper()}, {winner} ranks highest with a score of {winner_score:.4f}, "
            f"{runner_text}{diff_text}."
        ),
        "why_it_matters": (
            f"Model selection aligns with the primary business objective ({metric_name.upper()}). "
            "Comparing multiple architectures ensures we do not overfit to a single model family."
        ),
        "limitation": (
            f"{winner} is the best performer specifically under this dataset, split, and metric. "
            "It is not universally superior; other models may offer faster inference speed, better interpretability, or superior calibration."
        )
    }


def get_threshold_tradeoff_insight(
    threshold: float,
    prec: float,
    rec: float,
    f1: float,
    flagged_cnt: int,
    flagged_pct: float
) -> Dict[str, str]:
    return {
        "tells_us": (
            f"At threshold {threshold:.2f}, the model flags {flagged_cnt:,} customers ({flagged_pct:.1f}% of cohort), "
            f"yielding Precision = {prec:.3f}, Recall = {rec:.3f}, and F1-score = {f1:.3f}."
        ),
        "why_it_matters": (
            f"Lowering the threshold captures more churners (higher Recall) at the expense of more false alarms (lower Precision). "
            f"Raising it flags fewer customers (higher Precision) but misses more churners (lower Recall)."
        ),
        "limitation": (
            "Threshold selection must reflect business capacity (how many customers can retention agents contact?) "
            "rather than relying purely on the arbitrary default 0.50 threshold."
        )
    }


def get_feature_importance_insight(top_features: List[str], method: str) -> Dict[str, str]:
    feat_str = ", ".join(f"'{f}'" for f in top_features[:4])
    return {
        "tells_us": (
            f"The most influential features identified by {method} are: {feat_str}."
        ),
        "why_it_matters": (
            "These features drive the model's split decisions and probability outputs, indicating where customer disengagement signals are concentrated."
        ),
        "limitation": (
            "Permutation and tree importances reflect statistical reliance in predictive modeling, not real-world causation. "
            "Interventions must be validated via pilot experiments."
        )
    }


def get_risk_distribution_insight(
    high_cnt: int, med_cnt: int, low_cnt: int,
    high_thresh: float, med_thresh: float, total: int
) -> Dict[str, str]:
    high_pct = (high_cnt / total) * 100 if total > 0 else 0
    med_pct = (med_cnt / total) * 100 if total > 0 else 0
    low_pct = (low_cnt / total) * 100 if total > 0 else 0
    return {
        "tells_us": (
            f"Risk distribution: {high_cnt:,} High Risk ({high_pct:.1f}% at prob ≥ {high_thresh:.2f}), "
            f"{med_cnt:,} Medium Risk ({med_pct:.1f}% at prob ≥ {med_thresh:.2f}), and "
            f"{low_cnt:,} Low Risk ({low_pct:.1f}%)."
        ),
        "why_it_matters": (
            f"Enables tiered operational resource allocation: high-risk accounts receive immediate high-touch intervention, "
            f"medium-risk receive automated digital nurture, and low-risk receive standard service."
        ),
        "limitation": (
            "Tier counts depend on the user's operational threshold settings. Predicted risk reflects historical patterns, not guaranteed future events."
        )
    }
