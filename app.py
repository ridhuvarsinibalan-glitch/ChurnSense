"""
ChurnSense - Advanced ML Intelligence & Customer Retention Dashboard
Executive-Ready Web Application for Academic Evaluation & Business Intelligence
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import streamlit as st

from src.business_impact import (
    calculate_scenario_financials,
    get_retention_action_planner,
    segment_customers_by_risk,
)
from src.chart_insights import (
    get_confusion_matrix_insight,
    get_feature_importance_insight,
    get_model_comparison_insight,
    get_pr_curve_insight,
    get_risk_distribution_insight,
    get_roc_curve_insight,
    get_threshold_tradeoff_insight,
    render_chart_with_insight,
)
from src.eda import run_full_eda
from src.insights import (
    compute_model_feature_importance,
    generate_dataset_insights,
    generate_retention_recommendations,
)
from src.models import (
    get_metric_explanations,
    get_model_registry,
    normalize_metric_key,
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
from src.utils import (
    detect_id_columns,
    detect_positive_class,
    detect_target,
    format_metric,
    get_dataset_fingerprint,
)


# ==============================================================================
# APP CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="ChurnSense — Customer Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Glassmorphism Dark Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1322 0%, #080c14 90%);
        color: #e2e8f0;
    }
    
    /* Header branding */
    .hero-container {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.9) 0%, rgba(13, 19, 34, 0.8) 100%);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ffcc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(0, 255, 204, 0.12);
        border: 1px solid #00ffcc;
        color: #00ffcc;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    /* Executive KPI cards */
    .kpi-card {
        background: rgba(19, 27, 46, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
        backdrop-filter: blur(6px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 255, 204, 0.35);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 18px;
        color: #94a3b8;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 255, 204, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%) !important;
        color: #00ffcc !important;
        font-weight: 600 !important;
        border: 1px solid rgba(0, 255, 204, 0.4) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7, #06b6d4);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.25rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee);
        box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
    }

    /* Glass card containers */
    .glass-panel {
        background: rgba(19, 27, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state():
    defaults = {
        "raw_df": None,
        "dataset_name": None,
        "cleaned_df": None,
        "df_fingerprint": None,
        "split_data": None,
        "model_results": {},
        "selected_model": "Random Forest",
        "primary_metric": "F1-score",
        "threshold": 0.50,
        "locked_threshold": 0.50,
        "threshold_eval_split": "val",
        "dataset_insights": None,
        "feature_importance": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚡ ChurnSense")
    st.caption("Machine Learning Retention System")
    st.markdown("---")

    default_data_path = "data/telco_churn.csv"
    has_demo_data = os.path.exists(default_data_path)

    if st.session_state.raw_df is not None:
        st.success(f"Active Data: **{st.session_state.dataset_name}** ({len(st.session_state.raw_df):,} rows)")
        if st.button("🔄 Change / Upload Another Dataset"):
            st.session_state.raw_df = None
            st.session_state.dataset_name = None
            st.session_state.cleaned_df = None
            st.session_state.split_data = None
            st.session_state.model_results = {}
            st.session_state.dataset_insights = None
            st.session_state.feature_importance = None
            st.rerun()

        st.markdown("---")
        st.subheader("⚙️ Pipeline Configuration")

        clean_df = st.session_state.cleaned_df
        target_options = list(clean_df.columns)
        detected_tgt = detect_target(clean_df)
        tgt_idx = target_options.index(detected_tgt) if detected_tgt in target_options else len(target_options) - 1
        target_col = st.selectbox("Target Variable (Churn)", options=target_options, index=tgt_idx)

        unique_targets = clean_df[target_col].dropna().unique().tolist()
        detected_pos = detect_positive_class(clean_df, target_col)
        pos_idx = unique_targets.index(detected_pos) if detected_pos in unique_targets else 0
        positive_class = st.selectbox("Positive Churn Label", options=unique_targets, index=pos_idx)

        all_features = [c for c in clean_df.columns if c != target_col]
        detected_ids = detect_id_columns(clean_df)
        default_ids = [c for c in detected_ids if c in all_features]
        excluded_cols = st.multiselect(
            "Exclude Identifiers (No Leakage)",
            options=all_features,
            default=default_ids,
            help="Removes customer IDs and unique identifiers from predictive features."
        )

        if st.button("🚀 Re-split & Fit Pipeline"):
            with st.spinner("Partitioning data into stratified 60/20/20 splits..."):
                try:
                    split_data = prepare_data_splits(
                        clean_df,
                        target_col=target_col,
                        positive_class=positive_class,
                        excluded_cols=excluded_cols,
                        train_ratio=0.60,
                        val_ratio=0.20,
                        test_ratio=0.20,
                        random_state=42
                    )
                    st.session_state.split_data = split_data
                    st.session_state.model_results = {}
                    st.session_state.dataset_insights = None
                    st.session_state.feature_importance = None
                    st.success("Pipeline re-split ready! Transformers fitted strictly on Train.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Partitioning error: {e}")
    else:
        st.info("Upload your dataset or explore with the demo data on the main screen.")


# ==============================================================================
# STATE A: NO DATASET LOADED YET (WELCOME & INGESTION LANDING PAGE)
# ==============================================================================
if st.session_state.raw_df is None:
    st.markdown("""
    <div class="hero-container">
        <div class="badge-pill">● MACHINE LEARNING CAT II EVALUATION</div>
        <h1 class="hero-title">ChurnSense — Customer Churn Intelligence System</h1>
        <p class="hero-subtitle">
            An end-to-end Machine Learning retention platform with leakage-free preprocessing, multi-model benchmarking,
            interactive decision-threshold tuning, and automated data-grounded insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architectural highlights
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🛡️</div>
            <div style="font-weight: 700; color: #00ffcc; margin-bottom: 4px;">Zero Data Leakage</div>
            <div style="font-size: 0.88rem; color: #94a3b8;">
                Strict 60/20/20 stratified split. Imputers, scalers, and encoders are fitted strictly on training data.
                Customer IDs are automatically isolated.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🤖</div>
            <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">5-Model Benchmark Lab</div>
            <div style="font-size: 0.88rem; color: #94a3b8;">
                Logistic Regression, Decision Tree, Random Forest, Bagging & Boosting evaluated on identical splits
                with Recall, F1, PR-AUC and ROC-AUC.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🎛️</div>
            <div style="font-weight: 700; color: #a78bfa; margin-bottom: 4px;">Threshold & Business Impact</div>
            <div style="font-size: 0.88rem; color: #94a3b8;">
                Interactive probability threshold tuning, validation locking, customer risk tiers, and transparent
                financial ROI scenario modeling.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ingestion Box
    st.markdown("""
    <div style="background: rgba(19, 27, 46, 0.7); border: 1px dashed rgba(0, 255, 204, 0.35); border-radius: 12px; padding: 28px; text-align: center;">
        <h3 style="color: #00ffcc; margin: 0 0 8px 0;">📂 Step 1: Supply Your Churn Dataset</h3>
        <p style="color: #94a3b8; font-size: 0.95rem; margin: 0 0 18px 0;">
            Upload your customer records CSV file to initialize the executive dashboard and ML pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    upload_col, demo_col = st.columns([1.5, 1])
    with upload_col:
        uploaded_csv = st.file_uploader("Upload CSV Dataset", type=['csv'], label_visibility="collapsed")
        if uploaded_csv is not None:
            try:
                raw_df = pd.read_csv(uploaded_csv)
                cleaned, meta = clean_dataset(raw_df)
                st.session_state.raw_df = raw_df
                st.session_state.cleaned_df = cleaned
                st.session_state.dataset_name = uploaded_csv.name
                st.session_state.df_fingerprint = get_dataset_fingerprint(raw_df)
                st.success(f"Uploaded successfully ({len(raw_df):,} rows)! Initializing dashboard...")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with demo_col:
        st.write("")
        if has_demo_data:
            if st.button("📁 Or Load Demo Benchmark Dataset (Telco Churn: 7,043 Customers)", use_container_width=True):
                raw_df = pd.read_csv(default_data_path)
                cleaned, meta = clean_dataset(raw_df)
                st.session_state.raw_df = raw_df
                st.session_state.cleaned_df = cleaned
                st.session_state.dataset_name = "IBM Telco Churn (Benchmark)"
                st.session_state.df_fingerprint = get_dataset_fingerprint(raw_df)
                st.success("Loaded Telco benchmark dataset! Initializing dashboard...")
                st.rerun()

    st.stop()


# ==============================================================================
# STATE B: DATASET IS LOADED — EXECUTIVE DASHBOARD & PIPELINE
# ==============================================================================
clean_df = st.session_state.cleaned_df

# Ensure splits exist
if st.session_state.split_data is None:
    tgt = detect_target(clean_df)
    pos = detect_positive_class(clean_df, tgt)
    ids = detect_id_columns(clean_df)
    try:
        st.session_state.split_data = prepare_data_splits(
            clean_df,
            target_col=tgt,
            positive_class=pos,
            excluded_cols=ids,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            random_state=42
        )
    except Exception as e:
        st.error(f"Data partitioning error: {e}")
        st.stop()

split_data = st.session_state.split_data
target_col = split_data["target_col"]
pos_class = split_data["positive_class"]
y_binary = (clean_df[target_col] == pos_class).astype(int)

# Hero Header Bar
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div class="badge-pill">● DATASET ACTIVE: {st.session_state.dataset_name.upper()}</div>
            <h1 class="hero-title" style="margin: 4px 0 6px 0;">ChurnSense Executive Intelligence Dashboard</h1>
            <p class="hero-subtitle">
                Target: <strong style="color: #00ffcc;">{target_col}</strong> (Positive: <code>{pos_class}</code>) | 
                Identifiers Isolated: <code>{', '.join(split_data['excluded_cols']) if split_data['excluded_cols'] else 'None'}</code>
            </p>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-size: 0.85rem; font-weight: 600; padding: 6px 14px; border-radius: 8px;">
                Split: 60% Train ({split_data['train_size']:,}) | 20% Val ({split_data['val_size']:,}) | 20% Test ({split_data['test_size']:,})
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_exec, tab_eda, tab_models, tab_thresh, tab_ai, tab_biz = st.tabs([
    "📌 Executive Overview",
    "📊 Deep-Dive EDA",
    "🤖 Model Benchmark Lab",
    "🎛️ Threshold Explorer",
    "🧠 AI Retention Intelligence",
    "💼 Business Impact & Segments"
])


# ==============================================================================
# TAB 0: EXECUTIVE OVERVIEW DASHBOARD
# ==============================================================================
with tab_exec:
    st.markdown("### 📌 Executive Churn Summary at a Glance")
    st.caption("High-level business health KPIs, attrition prevalence, and immediate strategic priorities.")

    total_cust = len(clean_df)
    churn_count = int(y_binary.sum())
    non_churn_count = total_cust - churn_count
    churn_rate = float(y_binary.mean())

    # Calculate estimated revenue at risk if MonthlyCharges exists
    revenue_at_risk = 0.0
    if "MonthlyCharges" in clean_df.columns:
        revenue_at_risk = float(clean_df[clean_df[target_col] == pos_class]["MonthlyCharges"].sum())

    # Executive KPI Cards Strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Customer Base</div>
            <div class="kpi-value">{total_cust:,}</div>
            <div class="kpi-sub" style="color: #38bdf8;">Active Scored Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Observed Churn Rate</div>
            <div class="kpi-value" style="color: {'#f87171' if churn_rate > 0.20 else '#34d399'};">{churn_rate*100:.1f}%</div>
            <div class="kpi-sub" style="color: #f87171;">{churn_count:,} Total Departures</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Monthly Revenue at Risk</div>
            <div class="kpi-value" style="color: #fbbf24;">${revenue_at_risk:,.0f}</div>
            <div class="kpi-sub" style="color: #94a3b8;">Annualized: ${revenue_at_risk*12:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Class Balance Status</div>
            <div class="kpi-value" style="font-size: 1.5rem; color: #a78bfa;">{'Imbalanced' if churn_rate < 0.35 else 'Balanced'}</div>
            <div class="kpi-sub" style="color: #a78bfa;">Optimizing for Recall/F1</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-Column Visual Overview
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # Donut Chart for Executive Breakdown
        fig_donut, ax_donut = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax_donut.pie(
            [non_churn_count, churn_count],
            labels=["Retained", "Churned"],
            autopct='%1.1f%%',
            startangle=90,
            colors=["#238636", "#da3633"],
            wedgeprops=dict(width=0.4, edgecolor='#0d1117', linewidth=2)
        )
        for t in texts:
            t.set_color('#f8fafc')
        for at in autotexts:
            at.set_color('#ffffff')
            at.set_fontweight('bold')
        ax_donut.set_title(f"Customer Retention vs. Attrition Ratio", color='#00ffcc', fontsize=12)

        donut_insight = {
            "tells_us": f"{non_churn_count:,} accounts ({(1-churn_rate)*100:.1f}%) remain retained while {churn_count:,} accounts ({churn_rate*100:.1f}%) have churned.",
            "why_it_matters": "Visualizes baseline customer retention health. Demonstrates class imbalance requiring Precision-Recall optimization over raw accuracy.",
            "limitation": "Overall churn prevalence does not disclose which customer segments are driving the attrition."
        }
        render_chart_with_insight(fig_donut, "Customer Base Retention Ratio", donut_insight)

    with col_chart2:
        # Top Risk Drivers Quick Overview
        st.markdown("""
        <div class="glass-panel" style="height: 95%;">
            <h4 style="color: #00ffcc; margin: 0 0 10px 0;">⚡ Immediate Strategic Priorities</h4>
            <ul style="color: #cbd5e1; line-height: 1.8; font-size: 0.93rem;">
                <li><strong>Class Imbalance Mitigation:</strong> Standard accuracy is deceptive; models must be tuned using <strong>Recall and PR-AUC</strong>.</li>
                <li><strong>Contract Migration:</strong> Month-to-month subscribers represent the highest concentration of voluntary churn.</li>
                <li><strong>Onboarding Support:</strong> Newer accounts (tenure &lt; 12 months) experience steepest attrition rates.</li>
                <li><strong>Billing Optimization:</strong> Electronic check users show elevated payment friction and turnover.</li>
            </ul>
            <div style="margin-top: 16px; padding: 12px; background: rgba(0, 255, 204, 0.08); border-left: 3px solid #00ffcc; border-radius: 6px;">
                <span style="font-size: 0.85rem; color: #00ffcc; font-weight: 600;">Next Recommended Action:</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;"> Navigate to <strong>Model Benchmark Lab</strong> to train and compare the 5 classification algorithms.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 1: DEEP-DIVE EDA
# ==============================================================================
with tab_eda:
    st.markdown("### 📊 Deep-Dive Exploratory Data Analysis")
    st.caption("Every graph below features an automatically generated, data-grounded interpretation directly beneath it.")

    run_full_eda(
        clean_df,
        target_col=target_col,
        positive_class=pos_class,
        excluded_cols=split_data["excluded_cols"]
    )


# ==============================================================================
# TAB 2: MODEL BENCHMARK LAB
# ==============================================================================
with tab_models:
    st.markdown("### 🤖 Multi-Model Laboratory & Benchmarking")
    st.caption("Train models individually or benchmark all 5 algorithms on identical stratified splits.")

    reg = get_model_registry()
    model_names = list(reg.keys())

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        selected_to_train = st.selectbox("Select Individual Model to Train", options=model_names)
    with ctrl_col2:
        st.write("")
        st.write("")
        if st.button("Train Selected Model"):
            with st.spinner(f"Training {selected_to_train}..."):
                res = train_single_model(selected_to_train, split_data)
                st.session_state.model_results[selected_to_train] = res
                st.session_state.selected_model = selected_to_train
                st.success(f"{selected_to_train} trained in {res['training_time']:.2f}s!")
                st.rerun()
    with ctrl_col3:
        st.write("")
        st.write("")
        if st.button("🔥 Train All 5 Models"):
            with st.spinner("Training all 5 models sequentially on identical splits..."):
                all_res = train_all_models(split_data)
                st.session_state.model_results = all_res
                st.success("All 5 models trained successfully!")
                st.rerun()

    if not st.session_state.model_results:
        st.info("Click 'Train All 5 Models' or 'Train Selected Model' above to begin evaluation.")
    else:
        results = st.session_state.model_results

        st.markdown("#### 📋 Comprehensive Model Comparison Table")
        eval_split_choice = st.radio(
            "Evaluation Dataset Split:",
            options=["Validation Split (Recommended for model selection)", "Test Split (Final unbiased benchmark)"],
            horizontal=True
        )
        split_key = "val_metrics" if "Validation" in eval_split_choice else "test_metrics"

        table_rows = []
        for name, data in results.items():
            m = data[split_key]
            table_rows.append({
                "Model": name,
                "Accuracy": format_metric(m.get("accuracy")),
                "Precision": format_metric(m.get("precision")),
                "Recall": format_metric(m.get("recall")),
                "F1-Score": format_metric(m.get("f1")),
                "Specificity": format_metric(m.get("specificity")),
                "ROC-AUC": format_metric(m.get("roc_auc")),
                "PR-AUC": format_metric(m.get("pr_auc")),
                "Train Time (s)": f"{data['training_time']:.2f}"
            })

        comp_df = pd.DataFrame(table_rows)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("#### 🏆 Business-Oriented Model Selection")
        metric_col1, metric_col2 = st.columns([1, 2])
        with metric_col1:
            primary_metric = st.selectbox(
                "Primary Business Selection Metric",
                options=["F1-score", "Recall", "Precision", "ROC-AUC", "PR-AUC"],
                index=0
            )
            st.session_state.primary_metric = primary_metric

        with metric_col2:
            st.caption(
                "**Business Objective Alignment:** "
                "Use **Recall** if missing churners carries massive revenue risk. "
                "Use **Precision** if retention outreach is expensive. "
                "Use **F1-score** for balanced performance on imbalanced data. "
                "Use **PR-AUC** to measure ranking quality focused specifically on churners."
            )

        best_info = select_best_model(
            results,
            primary_metric=primary_metric,
            dataset_split="val" if "Validation" in eval_split_choice else "test"
        )

        if best_info["winner"]:
            winner_m = best_info["winner_metrics"]
            st.markdown(
                f"""
                <div style="background-color: #161b22; border: 2px solid #238636; border-radius: 8px; padding: 18px; margin: 15px 0;">
                    <h3 style="color: #3fb950; margin: 0 0 8px 0;">🥇 Winning Model: {best_info['winner']}</h3>
                    <p style="margin: 0 0 8px 0; font-size: 1.05rem;">
                        <strong>Score ({primary_metric}):</strong> <span style="color: #00ffcc; font-size: 1.2rem; font-weight: bold;">{best_info['winner_score']:.4f}</span> |
                        <strong>Accuracy:</strong> {winner_m.get('accuracy', 0):.4f} |
                        <strong>Precision:</strong> {winner_m.get('precision', 0):.4f} |
                        <strong>Recall:</strong> {winner_m.get('recall', 0):.4f} |
                        <strong>ROC-AUC:</strong> {format_metric(winner_m.get('roc_auc'))} |
                        <strong>PR-AUC:</strong> {format_metric(winner_m.get('pr_auc'))}
                    </p>
                    <p style="margin: 0 0 6px 0; color: #e6edf3;"><strong>Why Selected:</strong> {best_info['rationale']}</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #8b949e;"><strong>Important Limitations:</strong> {best_info['limitations']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            model_plot_names = list(results.keys())
            plot_metric_key = normalize_metric_key(primary_metric)
            model_scores = [results[n][split_key].get(plot_metric_key, 0.0) or 0.0 for n in model_plot_names]

            bars = ax.bar(model_plot_names, model_scores, color="#00ffcc", width=0.5)
            ax.set_ylabel(f"{primary_metric} Score")
            ax.set_ylim(0, 1.15)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha='center', va='bottom', color='#f0f6fc', fontweight='bold')
            plt.xticks(rotation=15, ha='right')

            comp_insight = get_model_comparison_insight(
                best_info["winner"],
                primary_metric,
                best_info["winner_score"],
                best_info["runner_up"],
                best_info["runner_up_score"]
            )
            render_chart_with_insight(fig, f"Model Benchmark Comparison ({primary_metric})", comp_insight)

        # ROC & PR Curves
        st.markdown("#### 📈 Diagnostic Curves for Winning Architecture")
        curve_col1, curve_col2 = st.columns(2)
        winner_data = results[best_info["winner"]]

        with curve_col1:
            y_val_true = split_data["y_val"]
            y_val_prob = winner_data["val_probs"]
            if y_val_prob is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_val_true, y_val_prob)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
                ax_roc.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"ROC (AUC = {winner_data['val_metrics'].get('roc_auc', 0):.3f})")
                ax_roc.plot([0, 1], [0, 1], color="#8b949e", linestyle="--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate (Recall)")
                ax_roc.legend(loc="lower right")

                roc_insight = get_roc_curve_insight(winner_data['val_metrics'].get('roc_auc'), best_info["winner"])
                render_chart_with_insight(fig_roc, f"ROC Curve — {best_info['winner']}", roc_insight)

        with curve_col2:
            if y_val_prob is not None:
                from sklearn.metrics import precision_recall_curve
                prec_curve, rec_curve, _ = precision_recall_curve(y_val_true, y_val_prob)
                baseline_rate = float(np.mean(y_val_true))
                fig_pr, ax_pr = plt.subplots(figsize=(6, 4.5))
                ax_pr.plot(rec_curve, prec_curve, color="#3fb950", lw=2, label=f"PR (AUC = {winner_data['val_metrics'].get('pr_auc', 0):.3f})")
                ax_pr.axhline(baseline_rate, color="#f85149", linestyle="--", label=f"Baseline ({baseline_rate:.3f})")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.legend(loc="upper right")

                pr_insight = get_pr_curve_insight(winner_data['val_metrics'].get('pr_auc'), baseline_rate, best_info["winner"])
                render_chart_with_insight(fig_pr, f"Precision-Recall Curve — {best_info['winner']}", pr_insight)

        with st.expander("📘 Metric Reference Guide in Plain English"):
            explanations = get_metric_explanations()
            for m_name, exp in explanations.items():
                st.markdown(f"**{m_name}:** {exp['plain']}")
                st.markdown(f"- *Business Context:* {exp['business']}")
                st.markdown(f"- *Limitation:* {exp['limitation']}")
                st.markdown("---")


# ==============================================================================
# TAB 3: THRESHOLD EXPLORER
# ==============================================================================
with tab_thresh:
    st.markdown("### 🎛️ Interactive Churn Probability Threshold Explorer")
    st.caption("Adjust decision thresholds interactively to balance False Negatives vs. False Positives without retraining.")

    if not st.session_state.model_results:
        st.warning("Please train at least one model in the Model Benchmark Lab tab first.")
    else:
        avail_models = list(st.session_state.model_results.keys())
        default_model = st.session_state.selected_model if st.session_state.selected_model in avail_models else avail_models[0]

        top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
        with top_col1:
            active_model = st.selectbox("Active Model for Threshold Tuning", options=avail_models, index=avail_models.index(default_model))
            st.session_state.selected_model = active_model
        with top_col2:
            eval_split = st.selectbox(
                "Evaluation Split",
                options=["Validation Set (Recommend threshold here)", "Test Set (Final locked evaluation)"],
                index=0
            )
        with top_col3:
            st.write("")
            st.write("")
            if st.button("↺ Reset Threshold to 0.50"):
                st.session_state.threshold = 0.50
                st.rerun()

        m_data = st.session_state.model_results[active_model]
        is_val = "Validation" in eval_split
        y_true = split_data["y_val"] if is_val else split_data["y_test"]
        y_prob = m_data["val_probs"] if is_val else m_data["test_probs"]

        if y_prob is None:
            st.error(f"Model {active_model} does not output probability estimates.")
        else:
            threshold = st.slider(
                "Select Decision Threshold (P ≥ T flags customer as Churn Risk)",
                min_value=0.01,
                max_value=0.99,
                value=float(st.session_state.threshold),
                step=0.01
            )
            st.session_state.threshold = threshold

            cur_metrics = evaluate_threshold(y_true, y_prob, threshold)

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Flagged Accounts</div><div class="kpi-value">{cur_metrics['flagged_count']:,}</div><div class="kpi-sub" style="color: #38bdf8;">{cur_metrics['flagged_pct']:.1f}% of cohort</div></div>""", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Recall (TP)</div><div class="kpi-value" style="color: #34d399;">{cur_metrics['recall']:.3f}</div><div class="kpi-sub" style="color: #34d399;">TP: {cur_metrics['tp']:,}</div></div>""", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Precision (FP)</div><div class="kpi-value" style="color: #38bdf8;">{cur_metrics['precision']:.3f}</div><div class="kpi-sub" style="color: #38bdf8;">FP: {cur_metrics['fp']:,}</div></div>""", unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">F1-Score</div><div class="kpi-value" style="color: #00ffcc;">{cur_metrics['f1']:.3f}</div><div class="kpi-sub" style="color: #00ffcc;">Harmonic Mean</div></div>""", unsafe_allow_html=True)
            with kpi5:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Missed Churn (FN)</div><div class="kpi-value" style="color: #f87171;">{cur_metrics['fn']:,}</div><div class="kpi-sub" style="color: #f87171;">Undetected</div></div>""", unsafe_allow_html=True)

            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
                cm_matrix = np.array([[cur_metrics['tn'], cur_metrics['fp']], [cur_metrics['fn'], cur_metrics['tp']]])
                sns.heatmap(
                    cm_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax_cm,
                    xticklabels=["Pred Non-Churn", "Pred Churn"],
                    yticklabels=["Actual Non-Churn", "Actual Churn"]
                )
                ax_cm.set_ylabel("Ground Truth")
                ax_cm.set_xlabel("Prediction")

                cm_insight = get_confusion_matrix_insight(
                    cur_metrics['tn'], cur_metrics['fp'], cur_metrics['fn'], cur_metrics['tp'],
                    cur_metrics['precision'], cur_metrics['recall'], cur_metrics['f1'],
                    threshold
                )
                render_chart_with_insight(fig_cm, f"Confusion Matrix at Threshold {threshold:.2f}", cm_insight)

            with viz_col2:
                curves_df = compute_threshold_curves(y_true, y_prob, steps=99)
                fig_crv, ax_crv = plt.subplots(figsize=(6.5, 4.5))
                ax_crv.plot(curves_df['threshold'], curves_df['precision'], label='Precision', color='#58a6ff', lw=2)
                ax_crv.plot(curves_df['threshold'], curves_df['recall'], label='Recall', color='#3fb950', lw=2)
                ax_crv.plot(curves_df['threshold'], curves_df['f1'], label='F1-Score', color='#00ffcc', lw=2.5)
                ax_crv.axvline(threshold, color='#f85149', linestyle='--', label=f'Current T={threshold:.2f}')
                ax_crv.set_xlabel("Threshold")
                ax_crv.set_ylabel("Metric Value")
                ax_crv.set_ylim(-0.05, 1.05)
                ax_crv.legend(loc="lower left")

                tradeoff_insight = get_threshold_tradeoff_insight(
                    threshold, cur_metrics['precision'], cur_metrics['recall'], cur_metrics['f1'],
                    cur_metrics['flagged_count'], cur_metrics['flagged_pct']
                )
                render_chart_with_insight(fig_crv, "Threshold Trade-off Curves", tradeoff_insight)

            st.markdown("---")
            st.markdown("#### 🎯 Business Strategy Threshold Recommendation & Locking")
            st.caption("Recommend an optimal threshold strictly using Validation data to prevent test-set data leakage, then evaluate on Test.")

            strat_col1, strat_col2, strat_col3 = st.columns([1.5, 1, 1])
            with strat_col1:
                strategy = st.selectbox(
                    "Recommendation Objective",
                    options=[
                        ("max_f1", "Maximize Validation F1-Score"),
                        ("min_recall", "Meet Minimum Target Recall"),
                        ("min_precision", "Meet Minimum Target Precision"),
                        ("min_cost", "Minimize Business Cost (Configurable FN & FP costs)")
                    ],
                    format_func=lambda x: x[1]
                )[0]

            with strat_col2:
                cost_fn = 200.0
                cost_fp = 25.0
                target_rec = 0.80
                target_prec = 0.70

                if strategy == "min_cost":
                    cost_fn = st.number_input("Cost of Missed Churner (FN $)", value=200.0, step=25.0)
                    cost_fp = st.number_input("Cost of Unnecessary Outreach (FP $)", value=25.0, step=5.0)
                elif strategy == "min_recall":
                    target_rec = st.slider("Minimum Target Recall", 0.50, 0.95, 0.80, 0.05)
                elif strategy == "min_precision":
                    target_prec = st.slider("Minimum Target Precision", 0.50, 0.95, 0.70, 0.05)

            with strat_col3:
                st.write("")
                st.write("")
                recommend_btn = st.button("🔒 Recommend & Lock Threshold")

            if recommend_btn:
                rec_res = recommend_threshold_validation(
                    split_data["y_val"],
                    m_data["val_probs"],
                    strategy=strategy,
                    min_recall=target_rec,
                    min_precision=target_prec,
                    cost_fn=cost_fn,
                    cost_fp=cost_fp
                )
                st.session_state.locked_threshold = rec_res["recommended_threshold"]
                st.session_state.threshold = rec_res["recommended_threshold"]

                test_eval = evaluate_locked_threshold_on_test(
                    split_data["y_test"],
                    m_data["test_probs"],
                    rec_res["recommended_threshold"]
                )

                st.markdown(
                    f"""
                    <div style="background-color: #161b22; border-left: 4px solid #3fb950; border-radius: 6px; padding: 16px; margin-top: 10px;">
                        <h4 style="color: #3fb950; margin: 0 0 6px 0;">✅ Recommended Threshold Locked: {rec_res['recommended_threshold']:.2f}</h4>
                        <p style="margin: 0 0 6px 0;"><strong>Optimization Strategy:</strong> {rec_res['rationale']}</p>
                        <hr style="border-color: #30363d; margin: 8px 0;" />
                        <h5 style="color: #00ffcc; margin: 0 0 6px 0;">Unbiased Final Test Set Evaluation (Locked T = {rec_res['recommended_threshold']:.2f}):</h5>
                        <p style="margin: 0 0 4px 0;">
                            <strong>Test Recall:</strong> {test_eval['test_metrics']['recall']:.3f} (Δ vs default 0.50: {test_eval['delta']['recall']:+.3f}) |
                            <strong>Test Precision:</strong> {test_eval['test_metrics']['precision']:.3f} (Δ: {test_eval['delta']['precision']:+.3f}) |
                            <strong>Test F1:</strong> {test_eval['test_metrics']['f1']:.3f} (Δ: {test_eval['delta']['f1']:+.3f})
                        </p>
                        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8;">
                            Flagged on Test Set: {test_eval['test_metrics']['flagged_count']:,} of {test_eval['test_metrics']['total_samples']:,} customers ({test_eval['test_metrics']['flagged_pct']:.1f}%).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==============================================================================
# TAB 4: AI RETENTION INTELLIGENCE
# ==============================================================================
with tab_ai:
    st.markdown("### 🧠 Evidence-Based AI Insights & Retention Intelligence")
    st.caption("Derived locally from statistical patterns, permutation importance, and non-causal behavioral analysis without external paid APIs.")

    st.markdown("#### 1. Dataset-Level Churn Disparities")
    if st.session_state.dataset_insights is None:
        with st.spinner("Analyzing group-level churn disparities across dataset features..."):
            st.session_state.dataset_insights = generate_dataset_insights(
                clean_df,
                target_col=split_data["target_col"],
                positive_class=split_data["positive_class"],
                excluded_cols=split_data["excluded_cols"]
            )

    insights = st.session_state.dataset_insights
    if insights:
        for ins in insights[:3]:
            st.markdown(
                f"""
                <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                    <h5 style="color: #58a6ff; margin: 0 0 4px 0;">Feature: {ins['feature']}</h5>
                    <p style="margin: 0 0 6px 0; font-size: 0.95rem;">{ins['interpretation']}</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #8b949e;">
                        Cohort Evidence: {ins['highest_churners']:,} churners / {ins['highest_count']:,} customers. {ins['caveat']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No significant group-level churn disparities detected in this dataset.")

    st.markdown("#### 2. Model-Level Feature Importance (Permutation Importance)")
    if not st.session_state.model_results:
        st.info("Train models in Model Benchmark Lab to view feature importances.")
    else:
        active_m = st.session_state.selected_model
        model_obj = st.session_state.model_results[active_m]["model"]

        if st.session_state.feature_importance is None:
            with st.spinner(f"Computing model-agnostic permutation importance on validation set for {active_m}..."):
                st.session_state.feature_importance = compute_model_feature_importance(
                    model_obj,
                    split_data["X_val_proc"],
                    split_data["y_val"],
                    split_data["feature_names"]
                )

        feat_imp = st.session_state.feature_importance
        agg_feats = feat_imp["aggregated_features"]

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4.5))
        p_names = [f["parent_feature"] for f in agg_feats[:8]]
        p_vals = [f["aggregated_importance"] for f in agg_feats[:8]]
        ax_imp.barh(p_names[::-1], p_vals[::-1], color="#00ffcc")
        ax_imp.set_xlabel("Mean Decrease in Validation F1-Score")

        imp_insight = get_feature_importance_insight(p_names, "Permutation Importance on held-out validation data")
        render_chart_with_insight(fig_imp, f"Top Predictive Features — {active_m}", imp_insight)

    st.markdown("#### 3. Evidence-Grounded Retention Recommendations")
    st.caption("Practical operational playbooks tied directly to measured empirical findings.")

    if st.session_state.dataset_insights and st.session_state.feature_importance:
        recs = generate_retention_recommendations(st.session_state.dataset_insights, st.session_state.feature_importance)
        for i, rec in enumerate(recs, 1):
            with st.expander(f"Recommendation #{i}: {rec['suggested_action'][:65]}...", expanded=(i==1)):
                st.markdown(f"**📌 Finding:** {rec['finding']}")
                st.markdown(f"**📊 Evidence:** {rec['evidence']}")
                st.markdown(f"**💡 Suggested Action:** {rec['suggested_action']}")
                st.markdown(f"**🎯 Expected Objective:** {rec['expected_objective']}")
                st.markdown(f"**🔬 Validation Plan:** {rec['validation_plan']}")
                st.markdown(f"**⚠️ Limitations:** {rec['limitations']}")


# ==============================================================================
# TAB 5: BUSINESS IMPACT & RETENTION PLANNING
# ==============================================================================
with tab_biz:
    st.markdown("### 💼 Business Impact & Retention Planning")
    st.caption("Translate probabilistic outputs into operational risk segments and financial scenarios.")

    if not st.session_state.model_results:
        st.warning("Please train at least one model in Model Benchmark Lab first.")
    else:
        active_m = st.session_state.selected_model
        m_data = st.session_state.model_results[active_m]

        y_test_prob = m_data["test_probs"]
        y_test_true = split_data["y_test"]

        st.markdown("#### 1. Configurable Risk Segmentation Tiers")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            high_thresh = st.slider("High Risk Threshold (Probability ≥ T)", 0.50, 0.95, 0.70, 0.05)
        with t_col2:
            med_thresh = st.slider("Medium Risk Threshold (T ≤ Probability < High)", 0.20, high_thresh - 0.05, 0.40, 0.05)

        risk_summary = segment_customers_by_risk(
            y_test_prob,
            df_raw=clean_df,
            id_col=split_data["excluded_cols"][0] if split_data["excluded_cols"] else None,
            y_true=y_test_true,
            high_threshold=high_thresh,
            med_threshold=med_thresh
        )

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Scored Cohort</div><div class="kpi-value">{risk_summary['total_scored']:,}</div><div class="kpi-sub" style="color: #38bdf8;">Test Records</div></div>""", unsafe_allow_html=True)
        with bcol2:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value" style="color: #f87171;">{risk_summary['high_count']:,}</div><div class="kpi-sub" style="color: #f87171;">{risk_summary['high_pct']:.1f}% of cohort</div></div>""", unsafe_allow_html=True)
        with bcol3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Medium Risk</div><div class="kpi-value" style="color: #fbbf24;">{risk_summary['med_count']:,}</div><div class="kpi-sub" style="color: #fbbf24;">{risk_summary['med_pct']:.1f}% of cohort</div></div>""", unsafe_allow_html=True)
        with bcol4:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Low Risk</div><div class="kpi-value" style="color: #34d399;">{risk_summary['low_count']:,}</div><div class="kpi-sub" style="color: #34d399;">{risk_summary['low_pct']:.1f}% of cohort</div></div>""", unsafe_allow_html=True)

        fig_risk, ax_risk = plt.subplots(figsize=(7, 3.5))
        r_bars = ax_risk.bar(
            ["High Risk", "Medium Risk", "Low Risk"],
            [risk_summary['high_count'], risk_summary['med_count'], risk_summary['low_count']],
            color=["#da3633", "#f0883e", "#238636"],
            width=0.45
        )
        for b in r_bars:
            h = b.get_height()
            ax_risk.text(b.get_x() + b.get_width()/2.0, h + 10, f"{h:,}", ha='center', va='bottom', color='#f0f6fc', fontweight='bold')
        ax_risk.set_ylabel("Customers")
        ax_risk.set_ylim(0, max(risk_summary['high_count'], risk_summary['med_count'], risk_summary['low_count']) * 1.2)

        risk_insight = get_risk_distribution_insight(
            risk_summary['high_count'], risk_summary['med_count'], risk_summary['low_count'],
            high_thresh, med_thresh, risk_summary['total_scored']
        )
        render_chart_with_insight(fig_risk, "Predicted Customer Churn Risk Tiers", risk_insight)

        st.markdown("#### 2. Customer Risk Table")
        risk_table = risk_summary["risk_table"]
        st.dataframe(risk_table.head(10), use_container_width=True)
        csv_data = risk_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Scored Risk Table (CSV)",
            data=csv_data,
            file_name="churn_risk_scores.csv",
            mime="text/csv"
        )

        st.markdown("#### 3. Operational Retention Action Planner")
        planner = get_retention_action_planner()
        for tier_key, plan in planner.items():
            with st.expander(f"{plan['tier']} — {plan['priority']}"):
                st.markdown(f"**Recommended Workflow:** {plan['recommended_workflow']}")
                st.markdown("**Suggested Interventions:**")
                for act in plan["suggested_actions"]:
                    st.markdown(f"- {act}")
                st.markdown(f"⚠️ *Operational Caution:* {plan['caution']}")

        st.markdown("---")
        st.markdown("#### 4. Hypothetical Scenario Financial Estimator")
        st.caption("Simulate potential campaign ROI based on explicit business assumptions. Explicitly separated from empirical metrics.")

        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            target_outreach = st.number_input("Targeted Customer Volume", min_value=10, max_value=100000, value=min(500, risk_summary['high_count']), step=50)
            cost_outreach = st.number_input("Cost per Retention Outreach ($)", min_value=1.0, max_value=500.0, value=25.0, step=5.0)
        with sc_col2:
            retention_success = st.slider("Assumed Retention Success Rate (%)", 1, 50, 15, 1) / 100.0
            churn_in_target = st.slider("Expected Churn Rate in Cohort (%)", 20, 100, 75, 5) / 100.0
        with sc_col3:
            value_retention = st.number_input("Avg Retained Customer Value ($)", min_value=50.0, max_value=5000.0, value=800.0, step=50.0)

        scenario = calculate_scenario_financials(
            target_count=target_outreach,
            cost_per_outreach=cost_outreach,
            assumed_success_rate=retention_success,
            value_per_retained_customer=value_retention,
            expected_churn_rate_in_target=churn_in_target
        )

        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px; margin-top: 10px;">
                <h4 style="color: #00ffcc; margin: 0 0 10px 0;">Estimated Scenario Financial Outputs</h4>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 12px;">
                    <div><strong>Campaign Cost:</strong> ${scenario['campaign_cost']:,.0f}</div>
                    <div><strong>Est. Churners Saved:</strong> {scenario['estimated_saved_churners']:.1f} customers</div>
                    <div><strong>Gross Value Preserved:</strong> ${scenario['gross_value_preserved']:,.0f}</div>
                    <div><strong>Net Financial Benefit:</strong> <span style="color: {'#3fb950' if scenario['net_financial_benefit'] >= 0 else '#da3633'}; font-weight: bold;">${scenario['net_financial_benefit']:,.0f}</span></div>
                    <div><strong>Estimated ROI:</strong> <span style="color: {'#3fb950' if scenario['roi_pct'] >= 0 else '#da3633'}; font-weight: bold;">{scenario['roi_pct']:+.1f}%</span></div>
                </div>
                <details>
                    <summary style="color: #58a6ff; cursor: pointer;">View Calculation Formulas</summary>
                    <pre style="background-color: #0d1117; padding: 10px; border-radius: 4px; font-size: 0.85rem; margin-top: 6px;">{scenario['formula']}</pre>
                </details>
                <p style="margin: 8px 0 0 0; font-size: 0.82rem; color: #8b949e;">{scenario['disclaimer']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
