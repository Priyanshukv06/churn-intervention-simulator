from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import load_preprocessor, transform_features

# Suppress XGBoost device mismatch warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def load_artifacts(
    model_path: str | Path = "models/xgb_model.pkl",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
):
    """
    Load the trained XGBoost model and fitted preprocessor from disk.
    Forces model to CPU for inference to eliminate device mismatch warning.
    """
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = load_preprocessor(preprocessor_path)

    # Model was trained on GPU (device=cuda) but numpy inputs are CPU arrays.
    # Switching to CPU for inference eliminates the mismatch warning entirely.
    model.set_params(device="cpu")

    return model, preprocessor


def _dict_to_dataframe(record: Dict) -> pd.DataFrame:
    """Convert a single customer record dict into a one-row DataFrame."""
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
        Raw feature values matching training column names, e.g.:
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "tenure": 12,
            "MonthlyCharges": 70.35,
            "TotalCharges": 1397.5,
            "Contract": "Month-to-month",
            ...
        }

    Returns
    -------
    float
        Churn probability between 0 and 1.
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
        1D array of churn probabilities, one per row.
    """
    X = transform_features(preprocessor, df)
    return model.predict_proba(X)[:, 1]


def predict_labels(
    df: pd.DataFrame,
    model,
    preprocessor,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict binary churn labels (0 or 1) using a custom threshold.

    Parameters
    ----------
    threshold : float
        Probability cutoff above which a customer is predicted to churn.
    """
    probas = predict_proba_batch(df, model, preprocessor)
    return (probas >= threshold).astype(int)


if __name__ == "__main__":
    from src.data_loader import load_and_prepare_data

    model, preprocessor = load_artifacts()
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Single prediction test
    sample = X_test.iloc[0].to_dict()
    proba = predict_proba_single(sample, model, preprocessor)
    print(f"Single record churn probability : {proba:.4f}")

    # Batch prediction test
    batch_probas = predict_proba_batch(X_test.head(5), model, preprocessor)
    print(f"Batch probabilities (first 5)   : {batch_probas.round(4)}")

    # Label prediction test
    labels = predict_labels(X_test.head(5), model, preprocessor)
    print(f"Predicted labels (first 5)      : {labels}")
