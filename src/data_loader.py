from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML config as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw Telco churn CSV.

    Raises a clear error if file is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Strip whitespace from string columns
    - Convert TotalCharges to numeric and drop rows where it is NaN
    """
    df = df.copy()

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Fix TotalCharges type issue: in Kaggle/IBM data it is 'object' and has blanks [web:74][web:71]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Ensure numeric columns are numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def encode_target(
    df: pd.DataFrame, target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Encode Churn Yes/No as 1/0 and drop id column.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in DataFrame")

    df = df.copy()

    # Map target: Yes -> 1, No -> 0 [web:77]
    target_mapping = {"No": 0, "Yes": 1}
    if not set(df[target_col].unique()).issubset(set(target_mapping.keys())):
        raise ValueError(f"Unexpected values in target column '{target_col}'")

    y = df[target_col].map(target_mapping)

    # Drop target and customerID from features
    drop_cols = [target_col]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols)

    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split so churn ratio is preserved [web:65].
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    config_path: str | Path = "configs/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    High-level helper used by trainer.py:
    - Load config
    - Load raw CSV
    - Clean data
    - Encode target
    - Train/test split
    """
    config = load_config(config_path)
    data_cfg = config["data"]

    raw_path = data_cfg["raw_path"]
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)

    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    X, y = encode_target(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick manual sanity check
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Churn rate train:", y_train.mean())
    print("Churn rate test :", y_test.mean())
