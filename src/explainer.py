from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import shap

from src.predictor import load_artifacts
from src.preprocessor import get_feature_names_out


def build_explainer(model, preprocessor, background_samples: int = 100) -> shap.TreeExplainer:
    from src.data_loader import load_and_prepare_data
    X_train, _, _, _ = load_and_prepare_data()
    X_train_trans = preprocessor.transform(X_train)
    background_idx = np.random.choice(
        X_train_trans.shape[0],
        size=min(background_samples, X_train_trans.shape[0]),
        replace=False,
    )
    background = X_train_trans[background_idx]
    explainer = shap.TreeExplainer(model, background)
    return explainer


def load_explainer(
    model_path: str | Path = "models/xgb_model.pkl",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
    background_samples: int = 100,
) -> shap.TreeExplainer:
    model, preprocessor = load_artifacts(model_path, preprocessor_path)
    return build_explainer(model, preprocessor, background_samples)


def get_shap_values(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
) -> np.ndarray:
    """Return raw 2D SHAP values array (samples x features)."""
    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)
    return explainer.shap_values(X_transformed)


def get_shap_force_plot(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: Optional[List[str]] = None,
):
    """
    Return a SHAP force plot object — this has .html() and works with st_shap().
    """
    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)
    shap_values = get_shap_values(explainer, X_transformed)
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=X_transformed[0],
        feature_names=feature_names,
        matplotlib=False,
    )
    return force_plot


def get_shap_waterfall_fig(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: Optional[List[str]] = None,
):
    """
    Return a matplotlib figure of the SHAP waterfall plot.
    Use with st.pyplot() — NOT st_shap().
    """
    import matplotlib.pyplot as plt

    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)

    shap_values = get_shap_values(explainer, X_transformed)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig


def get_shap_explanation(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """Legacy helper kept for compatibility."""
    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)
    shap_values = get_shap_values(explainer, X_transformed)
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_transformed,
        feature_names=feature_names,
    )


def get_global_feature_importance(
    explainer: shap.TreeExplainer,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    shap_values = get_shap_values(explainer, X_test_transformed)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "mean_shap_value": importance,
    }).sort_values("mean_shap_value", ascending=True)
