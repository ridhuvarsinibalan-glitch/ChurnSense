"""
ChurnSense - Data Preprocessing & Pipeline Architecture
Handles robust dataset ingestion, schema validation, identifier isolation,
leakage-free 3-way stratified splitting, and transformation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw churn data: handles whitespace strings, coerces numeric types,
    and identifies candidate target and identifier columns.
    """
    df_clean = df.copy()

    # Drop exact duplicates if any
    df_clean = df_clean.drop_duplicates()

    # Detect known identifier columns
    id_candidates = []
    for col in df_clean.columns:
        col_lower = col.lower().strip()
        if col_lower in ['customerid', 'customer_id', 'id', 'cust_id', 'name', 'client_id']:
            id_candidates.append(col)
        elif df_clean[col].dtype == 'object' and df_clean[col].nunique() == len(df_clean) and len(df_clean) > 50:
            id_candidates.append(col)

    # Detect candidate target column
    target_candidates = []
    for col in df_clean.columns:
        col_lower = col.lower().strip()
        if col_lower in ['churn', 'churned', 'churn_label', 'attrition', 'target', 'eligible', 'status']:
            target_candidates.append(col)

    detected_target = target_candidates[0] if target_candidates else df_clean.columns[-1]

    # Specific cleanups for standard churn datasets (e.g. Telco Customer Churn)
    for col in df_clean.columns:
        if col == detected_target or col in id_candidates:
            continue
        # Check if column is object or string with numeric representations (like TotalCharges)
        if pd.api.types.is_string_dtype(df_clean[col]) or df_clean[col].dtype == 'object':
            # Try converting to numeric
            converted = pd.to_numeric(df_clean[col].astype(str).str.strip(), errors='coerce')
            if converted.notna().sum() > 0.8 * len(df_clean):
                df_clean[col] = converted
                # Impute missing values with median
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    # Detect default positive class
    detected_pos = None
    if detected_target in df_clean.columns:
        unique_vals = df_clean[detected_target].dropna().unique().tolist()
        for val in ['Yes', 'yes', '1', 1, 'True', 'true', 'Churn', 'churn', 'Churned', 'churned']:
            if val in unique_vals:
                detected_pos = val
                break
        if detected_pos is None and len(unique_vals) == 2:
            detected_pos = unique_vals[1]

    metadata = {
        'id_columns': id_candidates,
        'detected_target': detected_target,
        'detected_positive': detected_pos,
        'total_rows': len(df_clean),
        'total_cols': len(df_clean.columns),
        'columns': list(df_clean.columns)
    }

    return df_clean, metadata


def build_preprocessor(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """
    Builds a scikit-learn ColumnTransformer fitted exclusively on training data.
    Separates numeric and categorical features cleanly.
    """
    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    preprocessor.fit(X_train)

    # Extract transformed feature names
    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_cols_encoded = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_cols_encoded)

    return preprocessor, feature_names


def prepare_data_splits(
    df: pd.DataFrame,
    target_col: str,
    positive_class: Any,
    excluded_cols: Optional[List[str]] = None,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs a strict 3-way stratified train/val/test split (e.g. 60/20/20)
    and fits the preprocessing pipeline strictly on X_train to prevent data leakage.
    """
    if excluded_cols is None:
        excluded_cols = []

    # Features X: exclude target and identifiers
    cols_to_drop = [target_col] + [c for c in excluded_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Binary positive churn target: 1 for churn, 0 for non-churn
    y_raw = df[target_col]
    y = (y_raw == positive_class).astype(int)

    # Verify class balance
    class_counts = y.value_counts().to_dict()
    if len(class_counts) < 2 or min(class_counts.values()) < 5:
        raise ValueError("Target variable must contain at least 5 examples of both positive and negative classes.")

    # 1st split: Train+Val (1 - test_ratio) vs Test (test_ratio)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )

    # 2nd split: Train vs Val from Train+Val
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_rel_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    # Fit preprocessor strictly on X_train
    preprocessor, feature_names = build_preprocessor(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    return {
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'X_test_raw': X_test,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'X_train_proc': X_train_proc,
        'X_val_proc': X_val_proc,
        'X_test_proc': X_test_proc,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'target_col': target_col,
        'positive_class': positive_class,
        'excluded_cols': cols_to_drop,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
        'churn_rate_train': float(y_train.mean()),
        'churn_rate_val': float(y_val.mean()),
        'churn_rate_test': float(y_test.mean())
    }
