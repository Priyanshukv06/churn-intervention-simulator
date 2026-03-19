from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_groups() -> dict:
    """
    Define groups of features based on the IBM Telco churn schema [web:75][web:72].
    """
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    binary_features = [
        "SeniorCitizen",         # 0/1
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]

    # These are multi-class categorical with multiple options [web:75]
    multiclass_features = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    return {
        "numeric": numeric_features,
        "binary": binary_features,
        "multiclass": multiclass_features,
    }


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes + scales numeric features
    - Ordinal-encodes Yes/No-like binary features
    - One-hot encodes multi-class categorical features
    """
    groups = get_feature_groups()
    numeric_features = groups["numeric"]
    binary_features = groups["binary"]
    multiclass_features = groups["multiclass"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Binary Yes/No columns -> 0/1
    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    multiclass_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("bin", binary_pipeline, binary_features),
            ("cat", multiclass_pipeline, multiclass_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer, X_train: pd.DataFrame
) -> ColumnTransformer:
    """Fit the preprocessor on training features."""
    preprocessor.fit(X_train)
    return preprocessor


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    save_path: str | Path,
) -> ColumnTransformer:
    """
    Convenience function:
    - build preprocessor
    - fit on X_train
    - save to disk
    """
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)

    return preprocessor


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    """Load a fitted preprocessor from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {path}")
    preprocessor = joblib.load(path)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> np.ndarray:
    """Transform raw features into model-ready numeric array."""
    return preprocessor.transform(X)


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get human-readable feature names after transformation.
    Useful for SHAP visualizations [web:61].
    """
    names = preprocessor.get_feature_names_out()
    return names.tolist()


if __name__ == "__main__":
    # Quick manual sanity check when run directly
    from src.data_loader import load_and_prepare_data

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    preprocessor = fit_and_save_preprocessor(
        X_train, "models/preprocessor.pkl"
    )
    X_train_trans = transform_features(preprocessor, X_train)
    print("Transformed X_train shape:", X_train_trans.shape)
    print("First 5 feature names:", get_feature_names_out(preprocessor)[:5])
