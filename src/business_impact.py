"""
ChurnSense - Business Impact & Retention Planning Engine
Translates probability distributions into risk segments, operational retention actions,
and transparent financial scenario calculations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def segment_customers_by_risk(
    y_prob: np.ndarray,
    df_raw: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    y_true: Optional[np.ndarray] = None,
    high_threshold: float = 0.70,
    med_threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Categorizes predicted customers into non-overlapping risk segments:
    - High Risk: prob >= high_threshold
    - Medium Risk: med_threshold <= prob < high_threshold
    - Low Risk: prob < med_threshold
    """
    if med_threshold >= high_threshold:
        raise ValueError("Medium-risk threshold must be strictly less than high-risk threshold.")

    n_total = len(y_prob)

    high_mask = y_prob >= high_threshold
    med_mask = (y_prob >= med_threshold) & (y_prob < high_threshold)
    low_mask = y_prob < med_threshold

    high_count = int(high_mask.sum())
    med_count = int(med_mask.sum())
    low_count = int(low_mask.sum())

    categories = np.empty(n_total, dtype=object)
    categories[high_mask] = "High"
    categories[med_mask] = "Medium"
    categories[low_mask] = "Low"

    # Build risk DataFrame for inspection & CSV export
    risk_df = pd.DataFrame({
        "churn_probability": np.round(y_prob, 4),
        "risk_tier": categories
    })

    if df_raw is not None and id_col is not None and id_col in df_raw.columns:
        risk_df.insert(0, "customer_id", list(df_raw[id_col].iloc[:n_total]))
    else:
        risk_df.insert(0, "customer_index", list(range(n_total)))

    observed_churn = {}
    if y_true is not None and len(y_true) == n_total:
        risk_df["actual_churn"] = y_true
        for tier, mask in [("High", high_mask), ("Medium", med_mask), ("Low", low_mask)]:
            tier_total = int(mask.sum())
            tier_churners = int(y_true[mask].sum()) if tier_total > 0 else 0
            tier_rate = float(tier_churners / tier_total) if tier_total > 0 else 0.0
            observed_churn[tier] = {
                "total": tier_total,
                "churners": tier_churners,
                "rate": tier_rate
            }

    summary = {
        "total_scored": n_total,
        "high_threshold": high_threshold,
        "med_threshold": med_threshold,
        "high_count": high_count,
        "high_pct": float(high_count / n_total * 100) if n_total > 0 else 0.0,
        "med_count": med_count,
        "med_pct": float(med_count / n_total * 100) if n_total > 0 else 0.0,
        "low_count": low_count,
        "low_pct": float(low_count / n_total * 100) if n_total > 0 else 0.0,
        "observed_churn": observed_churn,
        "risk_table": risk_df
    }

    return summary


def get_retention_action_planner() -> Dict[str, Dict[str, Any]]:
    """Returns recommended operational playbooks for each risk tier."""
    return {
        "High": {
            "tier": "High Risk (Probability >= High Threshold)",
            "priority": "P0 — Immediate Intervention",
            "recommended_workflow": "Direct human outreach via dedicated retention / account specialists.",
            "suggested_actions": [
                "Schedule a proactive satisfaction check-in phone call or 1-on-1 account review.",
                "Conduct a billing and plan audit to resolve pricing dissatisfaction or unexpected overages.",
                "Offer customized contract migration incentives or loyalty benefits.",
                "Expedite resolution of any open technical support tickets."
            ],
            "caution": "Avoid blanket discounts across all high-risk accounts; target only customers with high lifetime value."
        },
        "Medium": {
            "tier": "Medium Risk (Med Threshold <= Probability < High Threshold)",
            "priority": "P1 — Proactive Engagement & Monitoring",
            "recommended_workflow": "Automated digital re-engagement campaigns and customer satisfaction monitoring.",
            "suggested_actions": [
                "Send personalized educational content highlighting underutilized service features.",
                "Issue a brief satisfaction survey to identify emerging friction before escalation.",
                "Provide promotional add-ons (e.g. temporary speed boost, complimentary streaming trial).",
                "Monitor product usage telemetry for steep drops in engagement."
            ],
            "caution": "Keep communication subtle so satisfied customers are not reminded to reconsider their plan."
        },
        "Low": {
            "tier": "Low Risk (Probability < Med Threshold)",
            "priority": "P2 — Standard Service Excellence",
            "recommended_workflow": "Routine service delivery and loyalty reinforcement.",
            "suggested_actions": [
                "Maintain reliable core service delivery without intrusive retention interventions.",
                "Include in standard periodic brand newsletters and seasonal promotions.",
                "Invite to participate in customer advocacy, referral, or beta-feature programs."
            ],
            "caution": "Do not allocate costly manual retention budgets to low-risk customers."
        }
    }


def calculate_scenario_financials(
    target_count: int,
    cost_per_outreach: float = 20.0,
    assumed_success_rate: float = 0.15,
    value_per_retained_customer: float = 800.0,
    expected_churn_rate_in_target: float = 0.70
) -> Dict[str, Any]:
    """
    Calculates a hypothetical business scenario model.
    Explicitly separates assumed inputs from measured facts.
    """
    campaign_cost = float(target_count * cost_per_outreach)
    estimated_potential_churners = float(target_count * expected_churn_rate_in_target)
    estimated_saved_churners = float(estimated_potential_churners * assumed_success_rate)
    gross_value_preserved = float(estimated_saved_churners * value_per_retained_customer)
    net_financial_benefit = float(gross_value_preserved - campaign_cost)
    roi_pct = float((net_financial_benefit / campaign_cost) * 100) if campaign_cost > 0 else 0.0

    formula_explanation = (
        "1. Campaign Cost = Target Customers × Cost per Outreach\n"
        "2. Potential Churners = Target Customers × Assumed Churn Rate in Cohort\n"
        "3. Churners Saved = Potential Churners × Assumed Retention Success Rate\n"
        "4. Gross Value Preserved = Churners Saved × Customer Value\n"
        "5. Net Benefit = Gross Value Preserved - Campaign Cost\n"
        "6. ROI = (Net Benefit / Campaign Cost) × 100%"
    )

    academic_disclaimer = (
        "IMPORTANT ACADEMIC & BUSINESS DISCLAIMER: "
        "These financial figures represent a hypothetical business scenario model based exclusively on user-configured assumptions. "
        "High classification accuracy does not guarantee that targeted retention outreach will causally prevent customer departures. "
        "Real-world retention ROI must be validated via randomized controlled A/B test pilots."
    )

    return {
        "inputs": {
            "target_count": target_count,
            "cost_per_outreach": cost_per_outreach,
            "assumed_success_rate": assumed_success_rate,
            "value_per_retained_customer": value_per_retained_customer,
            "expected_churn_rate_in_target": expected_churn_rate_in_target
        },
        "campaign_cost": campaign_cost,
        "estimated_potential_churners": estimated_potential_churners,
        "estimated_saved_churners": estimated_saved_churners,
        "gross_value_preserved": gross_value_preserved,
        "net_financial_benefit": net_financial_benefit,
        "roi_pct": roi_pct,
        "formula": formula_explanation,
        "disclaimer": academic_disclaimer
    }
