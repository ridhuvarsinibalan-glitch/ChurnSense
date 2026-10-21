"""
ChurnSense - Utility Helpers
Provides target detection, schema fingerprinting, and formatting utilities.
"""

import hashlib
from typing import Any, List, Optional
import pandas as pd


def detect_target(df: pd.DataFrame) -> str:
    """Intelligently detects target column matching churn keywords, or returns last column."""
    candidates = ['churn', 'churned', 'churn_label', 'attrition', 'target', 'eligible', 'status']
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return df.columns[-1]


def detect_positive_class(df: pd.DataFrame, target_col: str) -> Any:
    """Detects positive churn class ('Yes', '1', True, etc.) from target column."""
    if target_col not in df.columns:
        return None
    unique_vals = df[target_col].dropna().unique().tolist()
    for candidate in ['Yes', 'yes', '1', 1, 'True', 'true', 'Churn', 'churn', 'Churned', 'churned']:
        if candidate in unique_vals:
            return candidate
    return unique_vals[-1] if unique_vals else None


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Detects candidate identifier columns to exclude from training."""
    id_cols = []
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['customerid', 'customer_id', 'id', 'cust_id', 'name', 'client_id']:
            id_cols.append(col)
        elif df[col].dtype == 'object' and df[col].nunique() == len(df) and len(df) > 50:
            id_cols.append(col)
    return id_cols


def get_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Generates an MD5 fingerprint of dataset shape and column names to detect data switches."""
    sig = f"{df.shape}_{list(df.columns)}"
    return hashlib.md5(sig.encode()).hexdigest()[:10]


def format_metric(val: Optional[float], is_percent: bool = False) -> str:
    """Formats float metric cleanly."""
    if val is None or pd.isna(val):
        return "N/A"
    if is_percent:
        return f"{val * 100:.1f}%"
    return f"{val:.4f}"
