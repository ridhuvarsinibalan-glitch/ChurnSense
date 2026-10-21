"""
Automated Unit and Acceptance Test Suite for ChurnSense ML Intelligence Engine
Tests data cleaning, leakage prevention, 5-model training, metrics,
threshold explorer, AI insights, and chart insight generators.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.business_impact import (
    calculate_scenario_financials,
    get_retention_action_planner,
    segment_customers_by_risk,
)
from src.chart_insights import (
    get_churn_dist_insight,
    get_confusion_matrix_insight,
    get_feature_importance_insight,
    get_model_comparison_insight,
    get_pr_curve_insight,
    get_risk_distribution_insight,
    get_roc_curve_insight,
    get_threshold_tradeoff_insight,
)
from src.insights import (
    compute_model_feature_importance,
    explain_single_customer,
    generate_dataset_insights,
    generate_retention_recommendations,
)
from src.models import (
    compute_metrics,
    get_model_registry,
    select_best_model,
    train_all_models,
    train_single_model,
)
from src.preprocessing import clean_dataset, prepare_data_splits
from src.threshold_explorer import (
    compute_threshold_curves,
    evaluate_locked_threshold_on_test,
    evaluate_threshold,
    recommend_threshold_validation,
)


@pytest.fixture
def sample_churn_df():
    """Generates a controlled synthetic churn dataset for unit tests."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "customerID": [f"CUST_{i:04d}" for i in range(n)],
        "tenure": np.random.randint(1, 72, size=n),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
        "TotalCharges": [f"{v:.2f}" if i % 10 != 0 else " " for i, v in enumerate(np.random.uniform(20.0, 5000.0, size=n))],
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2]),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n),
        "Churn": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    })
    return df


def test_clean_dataset(sample_churn_df):
    clean_df, meta = clean_dataset(sample_churn_df)
    assert len(clean_df) == len(sample_churn_df)
    assert meta["detected_target"] == "Churn"
    assert meta["detected_positive"] == "Yes"
    assert "customerID" in meta["id_columns"]
    # TotalCharges whitespace should be coerced to numeric float
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].isna().sum() == 0


def test_prepare_data_splits_no_leakage(sample_churn_df):
    clean_df, meta = clean_dataset(sample_churn_df)
    splits = prepare_data_splits(
        clean_df,
        target_col="Churn",
        positive_class="Yes",
        excluded_cols=["customerID"],
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=42
    )

    # Verify split sizes
    assert splits["train_size"] == 120
    assert splits["val_size"] == 40
    assert splits["test_size"] == 40

    # customerID must be completely excluded from features
    assert "customerID" not in splits["X_train_raw"].columns
    assert "Churn" not in splits["X_train_raw"].columns

    # Verify transformed array shapes
    n_feats = len(splits["feature_names"])
    assert splits["X_train_proc"].shape == (120, n_feats)
    assert splits["X_val_proc"].shape == (40, n_feats)
    assert splits["X_test_proc"].shape == (40, n_feats)


def test_train_all_models(sample_churn_df):
    clean_df, _ = clean_dataset(sample_churn_df)
    splits = prepare_data_splits(clean_df, "Churn", "Yes", ["customerID"], random_state=42)

    results = train_all_models(splits)

    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Bagging Classifier",
        "Boosting Classifier"
    ]

    for model_name in expected_models:
        assert model_name in results, f"Missing model: {model_name}"
        res = results[model_name]
        assert res["val_probs"] is not None
        assert res["test_probs"] is not None

        # Verify validation metrics
        vm = res["val_metrics"]
        assert 0.0 <= vm["accuracy"] <= 1.0
        assert 0.0 <= vm["precision"] <= 1.0
        assert 0.0 <= vm["recall"] <= 1.0
        assert 0.0 <= vm["f1"] <= 1.0
        assert 0.0 <= vm["specificity"] <= 1.0
        assert vm["roc_auc"] is not None and 0.0 <= vm["roc_auc"] <= 1.0
        assert vm["pr_auc"] is not None and 0.0 <= vm["pr_auc"] <= 1.0
        assert "tn" in vm["confusion_matrix"]


def test_select_best_model(sample_churn_df):
    clean_df, _ = clean_dataset(sample_churn_df)
    splits = prepare_data_splits(clean_df, "Churn", "Yes", ["customerID"], random_state=42)
    results = train_all_models(splits)

    selection = select_best_model(results, primary_metric="F1-score", dataset_split="val")
    assert selection["winner"] in results
    assert selection["winner_score"] >= 0.0
    assert "limitations" in selection
    assert "rationale" in selection


def test_threshold_explorer(sample_churn_df):
    clean_df, _ = clean_dataset(sample_churn_df)
    splits = prepare_data_splits(clean_df, "Churn", "Yes", ["customerID"], random_state=42)
    res = train_single_model("Random Forest", splits)

    y_val = splits["y_val"]
    val_probs = res["val_probs"]

    # Test single threshold evaluation
    t_eval = evaluate_threshold(y_val, val_probs, 0.50)
    assert t_eval["tp"] + t_eval["fp"] + t_eval["tn"] + t_eval["fn"] == len(y_val)
    assert 0.0 <= t_eval["f1"] <= 1.0

    # Test curves computation
    curves_df = compute_threshold_curves(y_val, val_probs, steps=10)
    assert len(curves_df) == 10
    assert "precision" in curves_df.columns
    assert "recall" in curves_df.columns

    # Test validation-based recommendation without test leakage
    rec = recommend_threshold_validation(y_val, val_probs, strategy="max_f1")
    assert 0.01 <= rec["recommended_threshold"] <= 0.99

    # Test locking threshold on test set
    test_eval = evaluate_locked_threshold_on_test(splits["y_test"], res["test_probs"], rec["recommended_threshold"])
    assert "locked_threshold" in test_eval
    assert "delta" in test_eval


def test_ai_insights_and_recommendations(sample_churn_df):
    clean_df, _ = clean_dataset(sample_churn_df)
    splits = prepare_data_splits(clean_df, "Churn", "Yes", ["customerID"], random_state=42)
    res = train_single_model("Random Forest", splits)

    dataset_insights = generate_dataset_insights(clean_df, "Churn", "Yes", ["customerID"])
    assert isinstance(dataset_insights, list)

    feat_imp = compute_model_feature_importance(
        res["model"],
        splits["X_val_proc"],
        splits["y_val"],
        splits["feature_names"],
        n_repeats=2
    )
    assert len(feat_imp["aggregated_features"]) > 0

    recs = generate_retention_recommendations(dataset_insights, feat_imp)
    assert len(recs) > 0
    first_rec = recs[0]
    assert "finding" in first_rec
    assert "evidence" in first_rec
    assert "suggested_action" in first_rec
    assert "expected_objective" in first_rec
    assert "validation_plan" in first_rec
    assert "limitations" in first_rec

    # Single customer explanation
    single_cust = splits["X_train_raw"].iloc[0]
    cust_exp = explain_single_customer(
        single_cust,
        splits["preprocessor"],
        res["model"],
        splits["feature_names"],
        splits["X_train_raw"],
        threshold=0.50
    )
    assert 0.0 <= cust_exp["churn_probability"] <= 1.0
    assert cust_exp["risk_tier"] in ["High", "Medium", "Low"]
    assert "explanation" in cust_exp


def test_business_impact_and_scenario():
    probs = np.array([0.85, 0.72, 0.65, 0.45, 0.20, 0.10])
    summary = segment_customers_by_risk(probs, high_threshold=0.70, med_threshold=0.40)
    assert summary["high_count"] == 2
    assert summary["med_count"] == 2
    assert summary["low_count"] == 2

    scenario = calculate_scenario_financials(
        target_count=100,
        cost_per_outreach=20.0,
        assumed_success_rate=0.20,
        value_per_retained_customer=500.0,
        expected_churn_rate_in_target=0.80
    )
    assert scenario["campaign_cost"] == 2000.0
    assert scenario["estimated_saved_churners"] == 16.0
    assert scenario["gross_value_preserved"] == 8000.0
    assert scenario["net_financial_benefit"] == 6000.0
    assert scenario["roi_pct"] == 300.0


def test_chart_insight_generators():
    # Verify all insight generators return required 3 keys: tells_us, why_it_matters, limitation
    dist_ins = get_churn_dist_insight(1000, 260, 740, 0.26)
    assert "tells_us" in dist_ins and "why_it_matters" in dist_ins and "limitation" in dist_ins

    cm_ins = get_confusion_matrix_insight(700, 40, 50, 210, 0.84, 0.808, 0.824, 0.50)
    assert "tells_us" in cm_ins and "why_it_matters" in cm_ins and "limitation" in cm_ins

    roc_ins = get_roc_curve_insight(0.85, "Random Forest")
    assert "tells_us" in roc_ins and "why_it_matters" in roc_ins and "limitation" in roc_ins

    pr_ins = get_pr_curve_insight(0.68, 0.26, "Random Forest")
    assert "tells_us" in pr_ins and "why_it_matters" in pr_ins and "limitation" in pr_ins

    comp_ins = get_model_comparison_insight("Random Forest", "F1", 0.65, "Boosting", 0.64)
    assert "tells_us" in comp_ins and "why_it_matters" in comp_ins and "limitation" in comp_ins

    trade_ins = get_threshold_tradeoff_insight(0.50, 0.65, 0.70, 0.67, 300, 30.0)
    assert "tells_us" in trade_ins and "why_it_matters" in trade_ins and "limitation" in trade_ins

    risk_ins = get_risk_distribution_insight(200, 300, 500, 0.70, 0.40, 1000)
    assert "tells_us" in risk_ins and "why_it_matters" in risk_ins and "limitation" in risk_ins
