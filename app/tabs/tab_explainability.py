import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

from src.data_loader import load_and_prepare_data
from src.explainer import get_shap_values, get_shap_waterfall_fig
from src.preprocessor import get_feature_names_out


@st.cache_data
def get_global_shap_data(_preprocessor, _explainer, n_samples: int = 300):
    """
    Compute SHAP values for a sample of test data.
    Cached so it only runs once per session.
    """
    _, X_test, _, _ = load_and_prepare_data()
    X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
    X_trans = _preprocessor.transform(X_sample)
    shap_values = _explainer.shap_values(X_trans)
    feature_names = get_feature_names_out(_preprocessor)
    return shap_values, X_trans, feature_names, X_sample


def plot_global_importance(shap_values, feature_names, top_n: int = 15):
    """Bar chart of top-N features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Global Feature Importances")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_beeswarm(shap_values, X_trans, feature_names, top_n: int = 15):
    """SHAP beeswarm (summary) plot — shows direction + magnitude per feature."""
    explanation = shap.Explanation(
        values=shap_values,
        data=X_trans,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    plt.tight_layout()
    return fig


def render_explainability_tab(
    model,
    preprocessor,
    explainer,
    feature_names,
    current_customer,
    **kwargs,
):
    st.header("🔍 SHAP Explainability")
    st.markdown(
        """
        Understand **why** the model makes each prediction.
        Global plots show patterns across all customers;
        local plots explain a single selected customer.
        """
    )

    shap_values, X_trans, feat_names, X_sample = get_global_shap_data(
        preprocessor, explainer
    )

    # ── Global section ─────────────────────────────────────────────────────
    st.subheader("🌍 Global Feature Importance")
    st.caption(f"Based on {len(X_sample)} test customers")

    top_n = st.slider("Number of top features to show", min_value=5, max_value=30, value=15)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean |SHAP| Bar Chart**")
        bar_fig = plot_global_importance(shap_values, feat_names, top_n=top_n)
        st.pyplot(bar_fig, clear_figure=True)

    with col2:
        st.markdown("**Beeswarm Plot** (direction + magnitude)")
        beeswarm_fig = plot_beeswarm(shap_values, X_trans, feat_names, top_n=top_n)
        st.pyplot(beeswarm_fig, clear_figure=True)

    st.info(
        "📌 **Reading the beeswarm**: Each dot = one customer. "
        "Color = feature value (red=high, blue=low). "
        "X-axis = SHAP value (positive → pushes toward churn)."
    )

    st.divider()

    # ── Local section ──────────────────────────────────────────────────────
    st.subheader("👤 Local Explanation — Selected Customer")

    if not current_customer:
        st.warning("Select a customer from the sidebar to see their individual explanation.")
        return

    X_cust = preprocessor.transform(pd.DataFrame([current_customer]))
    waterfall_fig = get_shap_waterfall_fig(explainer, X_cust, feature_names)
    st.pyplot(waterfall_fig, clear_figure=True)

    st.info(
        "📌 **Reading the waterfall**: "
        "Red bars push churn probability **higher**; "
        "blue bars push it **lower**. "
        "Starting from the base rate (E[f(x)]), bars stack to reach final prediction."
    )
