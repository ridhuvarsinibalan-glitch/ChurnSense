"""
ChurnSense - Interactive Churn Threshold Explorer
Provides probability threshold tuning, curve analysis, business cost optimization,
and test-set evaluation without data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluates classification outcomes for a specific decision threshold T in [0.0, 1.0].
    """
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    n_total = len(y_true)
    flagged_count = int(tp + fp)
    flagged_pct = float((flagged_count / n_total) * 100) if n_total > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "flagged_count": flagged_count,
        "flagged_pct": flagged_pct,
        "total_samples": n_total
    }


def compute_threshold_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 100
) -> pd.DataFrame:
    """
    Evaluates metrics across a uniform threshold grid [0.01, 0.99] to support
    instant interactive charting without retraining.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    rows = []
    for t in thresholds:
        metrics = evaluate_threshold(y_true, y_prob, t)
        rows.append(metrics)

    return pd.DataFrame(rows)


def recommend_threshold_validation(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    strategy: str = "max_f1",
    min_recall: float = 0.80,
    min_precision: float = 0.70,
    cost_fn: float = 200.0,
    cost_fp: float = 25.0
) -> Dict[str, Any]:
    """
    Recommends an optimal decision threshold strictly using the VALIDATION set
    to prevent test-set data leakage.
    """
    curves_df = compute_threshold_curves(y_val_true, y_val_prob, steps=99)

    best_thresh = 0.50
    rationale = ""
    tie_breaker = "Nearest to 0.50 default"

    if strategy == "max_f1":
        max_f1 = curves_df["f1"].max()
        candidates = curves_df[curves_df["f1"] >= max_f1 - 1e-4]
        # Tie-breaker: closest to 0.50
        candidates = candidates.copy()
        candidates["dist"] = (candidates["threshold"] - 0.50).abs()
        best_row = candidates.sort_values("dist").iloc[0]
        best_thresh = float(best_row["threshold"])
        rationale = f"Maximizes validation F1-score ({best_row['f1']:.4f}). Balances missed churners against unnecessary contacts."

    elif strategy == "min_recall":
        eligible = curves_df[curves_df["recall"] >= min_recall]
        if len(eligible) > 0:
            # Among eligible, maximize precision
            best_row = eligible.sort_values(["precision", "f1"], ascending=False).iloc[0]
            best_thresh = float(best_row["threshold"])
            rationale = f"Ensures at least {min_recall*100:.0f}% Recall (achieved: {best_row['recall']:.4f}) while maximizing Precision ({best_row['precision']:.4f})."
        else:
            best_row = curves_df.sort_values("recall", ascending=False).iloc[0]
            best_thresh = float(best_row["threshold"])
            rationale = f"Requested {min_recall*100:.0f}% Recall could not be achieved; selected threshold yielding highest available Recall ({best_row['recall']:.4f})."

    elif strategy == "min_precision":
        eligible = curves_df[curves_df["precision"] >= min_precision]
        if len(eligible) > 0:
            best_row = eligible.sort_values(["recall", "f1"], ascending=False).iloc[0]
            best_thresh = float(best_row["threshold"])
            rationale = f"Ensures at least {min_precision*100:.0f}% Precision (achieved: {best_row['precision']:.4f}) while maximizing Recall ({best_row['recall']:.4f})."
        else:
            best_row = curves_df.sort_values("precision", ascending=False).iloc[0]
            best_thresh = float(best_row["threshold"])
            rationale = f"Requested {min_precision*100:.0f}% Precision could not be achieved; selected threshold yielding highest available Precision ({best_row['precision']:.4f})."

    elif strategy == "min_cost":
        costs = curves_df["fn"] * cost_fn + curves_df["fp"] * cost_fp
        curves_df["total_cost"] = costs
        min_cost = curves_df["total_cost"].min()
        candidates = curves_df[curves_df["total_cost"] <= min_cost + 1e-4]
        candidates = candidates.copy()
        candidates["dist"] = (candidates["threshold"] - 0.50).abs()
        best_row = candidates.sort_values("dist").iloc[0]
        best_thresh = float(best_row["threshold"])
        rationale = (
            f"Minimizes total assumed business cost (${best_row['total_cost']:,.0f}) "
            f"given assumed False Negative cost (${cost_fn:,.0f}) and False Positive cost (${cost_fp:,.0f})."
        )

    val_eval = evaluate_threshold(y_val_true, y_val_prob, best_thresh)

    return {
        "recommended_threshold": best_thresh,
        "strategy": strategy,
        "rationale": rationale,
        "tie_breaker": tie_breaker,
        "validation_metrics": val_eval,
        "assumptions": {
            "min_recall": min_recall,
            "min_precision": min_precision,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp
        }
    }


def evaluate_locked_threshold_on_test(
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    locked_threshold: float
) -> Dict[str, Any]:
    """
    Evaluates the final locked threshold on the unseen TEST set.
    Provides completely unbiased performance assessment.
    """
    test_eval = evaluate_threshold(y_test_true, y_test_prob, locked_threshold)
    default_eval = evaluate_threshold(y_test_true, y_test_prob, 0.50)

    delta_f1 = test_eval["f1"] - default_eval["f1"]
    delta_recall = test_eval["recall"] - default_eval["recall"]
    delta_precision = test_eval["precision"] - default_eval["precision"]

    return {
        "locked_threshold": locked_threshold,
        "test_metrics": test_eval,
        "default_metrics": default_eval,
        "delta": {
            "f1": delta_f1,
            "recall": delta_recall,
            "precision": delta_precision
        }
    }
