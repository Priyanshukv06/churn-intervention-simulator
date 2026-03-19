import streamlit as st
import pandas as pd
from src.data_loader import load_and_prepare_data


@st.cache_data
def get_sample_customers() -> pd.DataFrame:
    _, X_test, _, _ = load_and_prepare_data()
    return X_test.sample(n=50, random_state=42).reset_index(drop=True)


def render_sidebar() -> dict:
    st.sidebar.header("👤 Customer Selector")

    sample_df = get_sample_customers()

    customer_idx = st.sidebar.selectbox(
        "Select Customer:",
        options=range(len(sample_df)),
        format_func=lambda i: f"Customer #{i+1}",
        key="customer_select",
    )
    current_customer = sample_df.iloc[customer_idx].to_dict()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selected Customer Info**")
    st.sidebar.caption(f"Tenure: {int(current_customer.get('tenure', 0))} months")
    st.sidebar.caption(f"Monthly Charges: ${current_customer.get('MonthlyCharges', 0):.2f}")
    st.sidebar.caption(f"Contract: {current_customer.get('Contract', 'N/A')}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with XGBoost + SHAP 🚀")

    return current_customer
