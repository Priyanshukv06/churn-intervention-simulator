from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.predictor import load_artifacts
from src.preprocessor import get_feature_names_out


def build_explainer(
    model,
    preprocessor,
    background_samples: int = 100,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer with a background dataset for stable explanations.
    """
    # Create background dataset from a sample of training data
    from src.data_loader import load_and_prepare_data

    X_train, _, _, _ = load_and_prepare_data()
    X_train_trans = preprocessor.transform(X_train)

    # Sample background for faster computation
    background_idx = np.random.choice(
        X_train_trans.shape[0], size=min(background_samples, X_train_trans.shape[0]), replace=False
    )
    background = X_train_trans[background_idx]

    explainer = shap.TreeExplainer(model, background)
    return explainer


def load_explainer(
    model_path: str | Path = "models/xgb_model.pkl",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
    background_samples: int = 100,
) -> shap.TreeExplainer:
    """
    Convenience function: load model + preprocessor + build explainer.
    """
    model, preprocessor = load_artifacts(model_path, preprocessor_path)
    return build_explainer(model, preprocessor, background_samples)


def get_shap_explanation(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """
    Compute SHAP values for a transformed input.
    Returns a shap.Explanation object that can be passed to st_shap().
    """
    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)

    # FIXED: TreeExplainer returns 2D (n_samples, n_features) for binary XGBoost [web:107][web:109]
    shap_values = explainer.shap_values(X_transformed)

    # Binary case: shap_values is 2D (samples, features) for positive class
    if isinstance(shap_values, list):
        # Multi-class case (not ours)
        raise ValueError("Multi-class SHAP values not yet supported.")
    else:
        # Binary case: 2D array for positive class
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_transformed,
            feature_names=feature_names,
        )
    return explanation


def get_global_feature_importance(
    explainer: shap.TreeExplainer,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature for global importance plot.
    """
    shap_values = explainer.shap_values(X_test_transformed)
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_shap_value": importance,
        },
    )
    importance_df = importance_df.sort_values("mean_shap_value", ascending=True)
    return importance_df


if __name__ == "__main__":
    # Quick smoke test
    explainer = load_explainer()
    model, preprocessor = load_artifacts()

    from src.data_loader import load_and_prepare_data

    _, X_test, _, _ = load_and_prepare_data()
    feature_names = get_feature_names_out(preprocessor)

    # Single sample explanation
    sample = X_test.iloc[0:1]  # one row
    X_sample_trans = preprocessor.transform(sample)
    explanation = get_shap_explanation(explainer, X_sample_trans, feature_names)
    print("SHAP explanation computed successfully!")
    print("Explanation shape:", explanation.values.shape)
