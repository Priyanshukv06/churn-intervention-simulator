from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import load_preprocessor, transform_features


def load_artifacts(
    model_path: str | Path = "models/xgb_model.pkl",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
):
    """
    Load the trained XGBoost model and fitted preprocessor from disk.
    """
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    return model, preprocessor


def _dict_to_dataframe(record: Dict) -> pd.DataFrame:
    """
    Helper: convert a single customer record dict into a one-row DataFrame.

    This expects keys that match the feature column names used during training,
    e.g. tenure, MonthlyCharges, TotalCharges, gender, Contract, etc.
    """
    return pd.DataFrame([record])


def predict_proba_single(
    record: Dict,
    model,
    preprocessor,
) -> float:
    """
    Predict churn probability for a single customer record.

    Parameters
    ----------
    record : dict
        Raw feature values, e.g.:
        {
          "gender": "Male",
          "SeniorCitizen": 0,
          "Partner": "Yes",
          ...
          "MonthlyCharges": 70.35,
          "TotalCharges": 1397.5,
        }

    Returns
    -------
    float
        Probability of churn (between 0 and 1).
    """
    df = _dict_to_dataframe(record)
    X = transform_features(preprocessor, df)
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def predict_proba_batch(
    df: pd.DataFrame,
    model,
    preprocessor,
) -> np.ndarray:
    """
    Predict churn probabilities for a batch of customers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the same feature columns used in training.

    Returns
    -------
    np.ndarray
        1D array of churn probabilities.
    """
    X = transform_features(preprocessor, df)
    proba = model.predict_proba(X)[:, 1]
    return proba


if __name__ == "__main__":
    # Quick manual smoke test
    from src.data_loader import load_and_prepare_data

    # Load artifacts
    model, preprocessor = load_artifacts()

    # Grab a small sample from the test set
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    sample = X_test.iloc[0].to_dict()

    proba = predict_proba_single(sample, model, preprocessor)
    print("Sample record churn probability:", proba)
