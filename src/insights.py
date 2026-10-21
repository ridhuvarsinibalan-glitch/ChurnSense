"""
ChurnSense - Evidence-Based AI Insights & Retention Intelligence
Extracts local, statistical patterns, permutation importance, individual customer
explanations, and grounded retention recommendations without paid external APIs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def generate_dataset_insights(
    df: pd.DataFrame,
    target_col: str,
    positive_class: Any,
    excluded_cols: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Dynamically discovers features in the dataset, analyzes group-level churn rates,
    and returns evidence-backed findings with sample size caveats and non-causal language.
    """
    if excluded_cols is None:
        excluded_cols = []

    y_binary = (df[target_col] == positive_class).astype(int)
    overall_churn_rate = float(y_binary.mean())
    total_customers = len(df)

    insights = []
    cols_to_analyze = [c for c in df.columns if c != target_col and c not in excluded_cols]

    for col in cols_to_analyze:
        # Categorical columns or binned numeric
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric and series.nunique() > 10:
            # Create quantile bins for numeric feature (e.g. tenure, monthly charges)
            try:
                binned = pd.qcut(series, q=4, duplicates='drop')
                grouped = y_binary.groupby(binned, observed=False)
            except Exception:
                continue
        else:
            if series.nunique() > 20:
                continue  # Skip high-cardinality non-categoricals
            grouped = y_binary.groupby(series, observed=False)

        summary = grouped.agg(
            count='count',
            churners='sum',
            rate='mean'
        ).reset_index()

        if len(summary) < 2:
            continue

        # Sort by churn rate
        summary_sorted = summary.sort_values('rate', ascending=False)
        highest = summary_sorted.iloc[0]
        lowest = summary_sorted.iloc[-1]

        high_rate = float(highest['rate'])
        low_rate = float(lowest['rate'])
        high_group = str(highest.iloc[0])
        low_group = str(lowest.iloc[0])
        high_count = int(highest['count'])
        high_churners = int(highest['churners'])

        diff_vs_baseline = high_rate - overall_churn_rate

        # Only report if meaningful disparity exists
        if abs(diff_vs_baseline) >= 0.05 and high_count >= 10:
            caveat = f"Small sample size ({high_count} customers); observe with caution." if high_count < 50 else ""

            interpretation = (
                f"Customers in segment '{high_group}' exhibit an observed churn rate of {high_rate*100:.1f}%, "
                f"which is {diff_vs_baseline*100:+.1f}% points compared to the baseline ({overall_churn_rate*100:.1f}%). "
                f"Conversely, segment '{low_group}' exhibited a churn rate of {low_rate*100:.1f}%. "
                "This indicates a measurable statistical association with churn in this dataset, "
                "but does not alone establish that this feature causes customer departures."
            )

            insights.append({
                "feature": col,
                "highest_group": high_group,
                "highest_rate": high_rate,
                "highest_count": high_count,
                "highest_churners": high_churners,
                "lowest_group": low_group,
                "lowest_rate": low_rate,
                "baseline_rate": overall_churn_rate,
                "diff_vs_baseline": diff_vs_baseline,
                "caveat": caveat,
                "interpretation": interpretation,
                "table": summary.to_dict(orient='records')
            })

    # Sort insights by magnitude of deviation from baseline
    insights.sort(key=lambda x: abs(x['diff_vs_baseline']), reverse=True)
    return insights


def compute_model_feature_importance(
    model: Any,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes model-agnostic Permutation Importance on held-out validation data.
    Aggregates one-hot encoded categories back to their parent features.
    """
    perm = permutation_importance(
        model, X_val_proc, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='f1',
        n_jobs=-1
    )

    importances = perm.importances_mean

    # Raw transformed feature ranking
    raw_ranking = []
    for name, imp, std in zip(feature_names, importances, perm.importances_std):
        raw_ranking.append({
            "feature": name,
            "importance": float(imp),
            "std": float(std)
        })
    raw_ranking.sort(key=lambda x: x["importance"], reverse=True)

    # Aggregate to parent features (e.g. Contract_One year -> Contract)
    parent_map: Dict[str, float] = {}
    for item in raw_ranking:
        feat = item["feature"]
        parent = feat.split("_")[0] if "_" in feat else feat
        parent_map[parent] = parent_map.get(parent, 0.0) + max(0.0, item["importance"])

    aggregated_ranking = [
        {"parent_feature": k, "aggregated_importance": float(v)}
        for k, v in parent_map.items()
    ]
    aggregated_ranking.sort(key=lambda x: x["aggregated_importance"], reverse=True)

    method_explanation = (
        "Permutation Importance (Model-Agnostic, evaluated on held-out validation data). "
        "Measures the decrease in validation F1-score when a feature's values are randomly shuffled. "
        "A higher value indicates that the model relies heavily on that feature for discriminatory power."
    )

    limitations = (
        "Permutation importance measures model dependence, not direct real-world causality. "
        "Correlated features may split importance among each other, artificially lowering individual scores."
    )

    return {
        "raw_features": raw_ranking[:15],
        "aggregated_features": aggregated_ranking[:10],
        "method": method_explanation,
        "limitations": limitations
    }


def explain_single_customer(
    customer_raw: pd.Series,
    preprocessor: Any,
    model: Any,
    feature_names: List[str],
    X_train_raw: pd.DataFrame,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Generates a localized, evidence-based feature attribution breakdown for an individual customer.
    Compares the customer's attributes against training population medians/modes.
    """
    cust_df = pd.DataFrame([customer_raw])
    cust_proc = preprocessor.transform(cust_df)

    prob = float(model.predict_proba(cust_proc)[0, 1]) if hasattr(model, "predict_proba") else 0.50
    flagged = prob >= threshold
    risk_tier = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")

    # Local perturbation contribution vs training baseline
    # We measure how replacing customer's features one-by-one with baseline affects probability
    contributions = []
    base_row = cust_df.copy()

    for col in cust_df.columns:
        if col not in X_train_raw.columns:
            continue
        orig_val = cust_df[col].iloc[0]

        # Use training median/mode as counterfactual baseline
        if pd.api.types.is_numeric_dtype(X_train_raw[col]):
            baseline_val = X_train_raw[col].median()
        else:
            baseline_val = X_train_raw[col].mode().iloc[0]

        # Perturbed copy
        perturbed_df = cust_df.copy()
        perturbed_df[col] = baseline_val
        try:
            perturbed_proc = preprocessor.transform(perturbed_df)
            prob_perturbed = float(model.predict_proba(perturbed_proc)[0, 1])
            impact = prob - prob_perturbed  # positive impact means customer's value increased risk
            contributions.append({
                "feature": col,
                "customer_value": str(orig_val),
                "baseline_value": str(baseline_val),
                "risk_contribution": float(impact),
                "direction": "Risk Driver (+)" if impact > 0.005 else ("Protective Factor (-)" if impact < -0.005 else "Neutral")
            })
        except Exception:
            continue

    contributions.sort(key=lambda x: abs(x["risk_contribution"]), reverse=True)
    top_factors = contributions[:5]

    plain_explanation = (
        f"The model estimated a churn probability of {prob*100:.1f}% ({risk_tier} Risk). "
        f"At decision threshold {threshold:.2f}, this customer is {'flagged as a churn risk' if flagged else 'not flagged'}. "
    )
    if top_factors:
        lead = top_factors[0]
        plain_explanation += (
            f"The primary contributing factor is '{lead['feature']}' ({lead['customer_value']}), "
            f"which altered risk by {lead['risk_contribution']*100:+.1f}% relative to standard baseline."
        )

    return {
        "churn_probability": prob,
        "threshold": threshold,
        "flagged": flagged,
        "risk_tier": risk_tier,
        "top_factors": top_factors,
        "explanation": plain_explanation,
        "warning": "This output is a statistical estimation derived from training correlations, not a definitive guarantee of customer churn."
    }


def generate_retention_recommendations(
    dataset_insights: List[Dict[str, Any]],
    feature_importance: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Constructs actionable, structured retention recommendations grounded in verified findings.
    Each recommendation satisfies the required 6-part framework.
    """
    recommendations = []

    for ins in dataset_insights[:4]:
        feat = ins["feature"].lower()
        high_grp = ins["highest_group"]
        rate = ins["highest_rate"]
        baseline = ins["baseline_rate"]

        if "contract" in feat and "month" in high_grp.lower():
            recommendations.append({
                "finding": f"Customers on '{ins['highest_group']}' exhibit an elevated churn rate of {rate*100:.1f}% (vs. baseline {baseline*100:.1f}%).",
                "evidence": f"{ins['highest_churners']:,} churners observed out of {ins['highest_count']:,} customers in this cohort.",
                "suggested_action": "Design and pilot an annual contract migration incentive program (e.g., offer 2 months discounted or streaming add-on for 12-month commitment).",
                "expected_objective": "Increase contract tenure stability and reduce month-to-month volatility.",
                "validation_plan": "Run a randomized controlled A/B test with 500 month-to-month customers receiving the incentive vs. 500 receiving standard communications; monitor 90-day churn.",
                "limitations": "Observational correlation does not prove that forcing contract lock-in will satisfy dissatisfied customers."
            })

        elif "tenure" in feat:
            recommendations.append({
                "finding": f"Customers in the newest tenure bracket ({ins['highest_group']}) show the highest attrition ({rate*100:.1f}%).",
                "evidence": f"{ins['highest_churners']:,} churners out of {ins['highest_count']:,} new subscribers.",
                "suggested_action": "Revamp the first 90-day onboarding journey with proactive check-ins, automated milestone guides, and dedicated early-life technical assistance.",
                "expected_objective": "Improve early customer product adoption and address setup friction before disengagement occurs.",
                "validation_plan": "Implement revised onboarding for all new signups in a single territory; compare 6-month retention against historic regional benchmarks.",
                "limitations": "Early churn may stem from product misfit or unrealistic marketing expectations that onboarding tweaks cannot remedy."
            })

        elif "payment" in feat and ("electronic" in high_grp.lower() or "check" in high_grp.lower()):
            recommendations.append({
                "finding": f"Subscribers utilizing '{ins['highest_group']}' experience {rate*100:.1f}% churn, notably above the {baseline*100:.1f}% company baseline.",
                "evidence": f"Accounts using this method represent {ins['highest_count']:,} customers and {ins['highest_churners']:,} churn events.",
                "suggested_action": "Encourage migration to Automated Credit Card or Direct Debit billing with a one-time bill credit incentive ($10 on next statement).",
                "expected_objective": "Reduce involuntary churn caused by payment processing failures, expired checks, and manual billing friction.",
                "validation_plan": "Deploy targeted email campaign offering the billing incentive; measure payment failure rates and retention at 60 and 120 days.",
                "limitations": "Customers selecting manual payments may already be less committed to the service."
            })

        elif "internet" in feat or "techsupport" in feat or "security" in feat:
            recommendations.append({
                "finding": f"Subscribers in '{ins['feature']} = {ins['highest_group']}' show distinct churn vulnerability ({rate*100:.1f}% vs. {baseline*100:.1f}%).",
                "evidence": f"Observed in {ins['highest_count']:,} accounts with {ins['highest_churners']:,} churn cases.",
                "suggested_action": "Bundle complimentary security monitoring or priority technical support for vulnerable service tiers.",
                "expected_objective": "Enhance perceived service reliability and remove frustration during connection outages.",
                "validation_plan": "Offer 30-day free trial of priority support to high-risk subscribers; measure support ticket satisfaction and renewal rate.",
                "limitations": "Additional services may incur cost without improving customer sentiment if core service quality remains deficient."
            })

    # Fallback generic grounded recommendation if specific rules don't match
    if not recommendations and dataset_insights:
        top_ins = dataset_insights[0]
        recommendations.append({
            "finding": f"Feature '{top_ins['feature']}' group '{top_ins['highest_group']}' has {top_ins['highest_rate']*100:.1f}% churn rate.",
            "evidence": f"{top_ins['highest_churners']:,} of {top_ins['highest_count']:,} accounts churned.",
            "suggested_action": f"Conduct focused qualitative customer interviews with customers in {top_ins['highest_group']} to pinpoint friction points.",
            "expected_objective": "Identify actionable operational improvements.",
            "validation_plan": "Gather sentiment from 50 interviews and pilot targeted solution.",
            "limitations": "Observational findings require controlled experimentation."
        })

    return recommendations
