"""
ChurnSense - Machine Learning Engine
Implements the full 5-model classification suite, rigorous business metrics,
and multi-model comparison.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """Returns factory dictionary for the 5 required models."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ),
            "description": "Linear baseline using sigmoid function with balanced class weights for churn imbalance."
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced"
            ),
            "description": "Interpretable tree-based partitioning model with depth regularization."
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ),
            "description": "Ensemble of decorrelated decision trees with bootstrap aggregation and feature subsampling."
        },
        "Bagging Classifier": {
            "estimator": BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            ),
            "description": "Bootstrap Aggregation over unconstrained base decision trees to minimize model variance."
        },
        "Boosting Classifier": {
            "estimator": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            "description": "Sequential boosting model iteratively fitting shallow trees on pseudo-residuals."
        }
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Computes business-focused classification metrics safely.
    Handles zero divisions and edge cases without fabricating scores.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback for single-class edge case
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = None

    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception:
        report = {}

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_score": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report
    }


def normalize_metric_key(metric_name: str) -> str:
    """Normalizes UI metric labels like 'F1-score', 'ROC-AUC' to internal metric dictionary keys."""
    raw = metric_name.lower().replace("-", "_").strip()
    if "f1" in raw:
        return "f1"
    elif "roc" in raw:
        return "roc_auc"
    elif "pr" in raw or ("precision" in raw and "average" in raw):
        return "pr_auc"
    elif "accuracy" in raw:
        return "accuracy"
    elif "precision" in raw:
        return "precision"
    elif "recall" in raw:
        return "recall"
    elif "specificity" in raw:
        return "specificity"
    return raw



def train_single_model(model_name: str, split_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains an individual model on X_train_proc, computes probabilities and metrics
    on both validation and test splits, and records training duration.
    """
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' is not in registry.")

    model_info = registry[model_name]
    model = model_info["estimator"]

    X_train = split_data["X_train_proc"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val_proc"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test_proc"]
    y_test = split_data["y_test"]

    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    # Probabilities & default 0.50 predictions
    val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    val_preds = (val_probs >= 0.50).astype(int) if val_probs is not None else model.predict(X_val)

    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    test_preds = (test_probs >= 0.50).astype(int) if test_probs is not None else model.predict(X_test)

    val_metrics = compute_metrics(y_val, val_preds, val_probs)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)

    return {
        "name": model_name,
        "model": model,
        "training_time": fit_duration,
        "description": model_info["description"],
        "parameters": model.get_params(),
        "val_probs": val_probs,
        "val_preds": val_preds,
        "val_metrics": val_metrics,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_metrics": test_metrics,
    }


def train_all_models(split_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Trains all 5 models sequentially on the identical train split."""
    results = {}
    for name in get_model_registry().keys():
        results[name] = train_single_model(name, split_data)
    return results


def select_best_model(
    model_results: Dict[str, Dict[str, Any]],
    primary_metric: str = "f1",
    dataset_split: str = "val"
) -> Dict[str, Any]:
    """
    Automatically identifies the best model using the selected primary metric.
    Compares winner and runner-up, analyzes trade-offs, and details limitations.
    """
    if not model_results:
        return {"winner": None, "message": "No trained models available."}

    metric_raw = primary_metric.lower().replace("-", "_")
    if "f1" in metric_raw:
        metric_key = "f1"
    elif "roc" in metric_raw:
        metric_key = "roc_auc"
    elif "pr" in metric_raw or "precision" in metric_raw and "average" in metric_raw:
        metric_key = "pr_auc"
    else:
        metric_key = metric_raw

    ranked = []
    for name, res in model_results.items():
        metrics = res["val_metrics"] if dataset_split == "val" else res["test_metrics"]
        score = metrics.get(metric_key)
        if score is not None:
            ranked.append((name, score, metrics))

    if not ranked:
        return {"winner": None, "message": f"Metric '{primary_metric}' could not be evaluated."}

    # Sort descending by primary score
    ranked.sort(key=lambda x: x[1], reverse=True)

    winner_name, winner_score, winner_metrics = ranked[0]
    runner_up_name = ranked[1][0] if len(ranked) > 1 else None
    runner_up_score = ranked[1][1] if len(ranked) > 1 else None
    diff = (winner_score - runner_up_score) if runner_up_score is not None else 0.0

    # Trade-off and explanation analysis
    rationale = f"Selected because it achieved the highest {primary_metric.upper()} ({winner_score:.4f}) on the {dataset_split.upper()} set."
    if abs(diff) < 0.01 and runner_up_name:
        rationale += f" Note: The performance difference compared to runner-up ({runner_up_name}: {runner_up_score:.4f}) is marginal (Δ = {diff:.4f})."

    limitations = (
        f"{winner_name} was chosen under the {dataset_split} split optimizing {primary_metric.upper()}. "
        "It is not universally superior; for instance, high Recall may sacrifice Precision by increasing False Positives, "
        "while high Precision may miss viable churners (higher False Negatives)."
    )

    return {
        "winner": winner_name,
        "winner_score": winner_score,
        "winner_metrics": winner_metrics,
        "runner_up": runner_up_name,
        "runner_up_score": runner_up_score,
        "difference": diff,
        "primary_metric": primary_metric,
        "dataset_split": dataset_split,
        "rationale": rationale,
        "limitations": limitations
    }


def get_metric_explanations() -> Dict[str, Dict[str, str]]:
    """Provides plain-English explanations and business caveats for each metric."""
    return {
        "Recall": {
            "plain": "Of the customers who actually churned, how many did the model successfully identify?",
            "business": "Crucial when missing a churner (False Negative) has high revenue consequences. High recall minimizes surprise cancellations.",
            "limitation": "High recall can be trivially achieved by flagging almost all customers, causing excessive false alarms."
        },
        "Precision": {
            "plain": "Of the customers flagged as likely to churn, how many actually churned?",
            "business": "Vital when customer retention interventions (e.g. expensive loyalty discounts, high-touch account management) are costly.",
            "limitation": "A model can boast high precision by only flagging the few absolute certain churners, leaving most churners undetected."
        },
        "F1-score": {
            "plain": "How well does the model balance Precision and Recall via their harmonic mean?",
            "business": "The gold standard single metric for imbalanced churn datasets; penalizes extreme trade-offs between precision and recall.",
            "limitation": "Treats precision and recall with equal weighting, which may not align if false positives and false negatives carry vastly different business costs."
        },
        "ROC-AUC": {
            "plain": "How effectively does the model rank churners higher than non-churners across all possible decision thresholds?",
            "business": "Evaluates overall discriminatory power independent of any specific operational probability threshold.",
            "limitation": "Can present an overly optimistic score on severely imbalanced datasets because False Positive Rate is diluted by a large negative class."
        },
        "PR-AUC / Average Precision": {
            "plain": "How effectively does the model identify the positive churn class specifically across varying thresholds?",
            "business": "The most informative curve metric for imbalanced churn domains because it focuses solely on the minority churn class.",
            "limitation": "Less familiar to non-technical stakeholders compared to ROC-AUC; baseline PR-AUC equals the raw churn prevalence rate."
        },
        "Specificity": {
            "plain": "How well does the model correctly identify customers who will genuinely remain with the service?",
            "business": "Measures avoidance of unnecessary interventions on loyal, satisfied customers.",
            "limitation": "Does not reflect how well actual churners are being found."
        }
    }


def save_full_pipeline(
    filepath: str,
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    metadata: Dict[str, Any]
) -> None:
    """Saves an end-to-end deployable inference bundle."""
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metadata": metadata
    }
    joblib.dump(bundle, filepath)
