import streamlit as st
from pathlib import Path
import pandas as pd


def render_performance_tab():
    st.header("📊 Model Performance Dashboard")
    st.markdown("Evaluation results from the trained XGBoost model on the held-out test set.")

    # Key metrics summary
    st.subheader("📈 Key Metrics")
    metrics = {
        "ROC-AUC": "0.8228",
        "PR-AUC": "0.6272",
        "Accuracy": "75.12%",
        "Precision": "52.34%",
        "Recall": "71.66%",
        "F1 Score": "60.50%",
    }
    cols = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        cols[i % 3].metric(label=metric, value=value)

    st.divider()

    # Evaluation plots
    st.subheader("📉 Evaluation Plots")
    artifacts = Path("models/eval_artifacts")

    col1, col2, col3 = st.columns(3)
    with col1:
        if (artifacts / "roc_curve.png").exists():
            st.image(
                str(artifacts / "roc_curve.png"),
                caption="ROC Curve (AUC = 0.82)",
                use_container_width=True,
            )
    with col2:
        if (artifacts / "pr_curve.png").exists():
            st.image(
                str(artifacts / "pr_curve.png"),
                caption="Precision-Recall Curve",
                use_container_width=True,
            )
    with col3:
        if (artifacts / "confusion_matrix.png").exists():
            st.image(
                str(artifacts / "confusion_matrix.png"),
                caption="Confusion Matrix (threshold=0.5)",
                use_container_width=True,
            )

    st.divider()

    # Model config summary
    st.subheader("⚙️ Model Configuration")
    config_data = {
        "Parameter": [
            "Algorithm", "n_estimators", "max_depth",
            "learning_rate", "subsample", "colsample_bytree",
            "scale_pos_weight", "eval_metric", "Training device",
        ],
        "Value": [
            "XGBoost Classifier", "300", "6",
            "0.05", "0.8", "0.8",
            "2.763 (neg/pos ratio)", "logloss", "CUDA (RTX 3060)",
        ],
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
