import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit_shap import st_shap

from src.predictor import predict_proba_single
from src.explainer import get_shap_force_plot, get_shap_waterfall_fig


def render_probability_gauge(original_prob: float, new_prob: float):
    delta = new_prob - original_prob
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Churn Risk", f"{original_prob:.1%}")
    with col2:
        st.metric(
            "Simulated Churn Risk",
            f"{new_prob:.1%}",
            delta=f"{delta:+.1%}",
            delta_color="inverse",
        )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=new_prob * 100,
        title={"text": "Simulated Churn Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "salmon"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=220)
    st.plotly_chart(fig, use_container_width=True)


def render_simulator_tab(model, preprocessor, explainer, feature_names, current_customer):
    st.header("🎛️ What-If Intervention Simulator")

    if not current_customer:
        st.warning("Please select a customer from the sidebar.")
        return

    original_prob = predict_proba_single(current_customer, model, preprocessor)

    st.subheader("📋 Original Customer Profile")
    st.dataframe(pd.DataFrame([current_customer]), use_container_width=True)

    st.subheader("🔧 Simulate Business Interventions")
    st.caption("Adjust parameters below — churn risk and SHAP explanation update instantly.")

    with st.expander("Intervention Controls", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            contract_options = ["Month-to-month", "One year", "Two year"]
            orig_contract = current_customer.get("Contract", "Month-to-month")
            contract_idx = contract_options.index(orig_contract) if orig_contract in contract_options else 0
            contract = st.selectbox("📄 Contract Type", options=contract_options, index=contract_idx)

            tenure = st.slider(
                "📅 Tenure (months)",
                min_value=0, max_value=72,
                value=int(current_customer.get("tenure", 12)),
            )

        with col2:
            monthly_charges = st.slider(
                "💵 Monthly Charges ($)",
                min_value=18.0, max_value=120.0,
                value=float(current_customer.get("MonthlyCharges", 70.0)),
                step=0.5,
            )

            internet_options = ["DSL", "Fiber optic", "No"]
            orig_internet = current_customer.get("InternetService", "DSL")
            internet_idx = internet_options.index(orig_internet) if orig_internet in internet_options else 0
            internet_service = st.selectbox("🌐 Internet Service", options=internet_options, index=internet_idx)

            payment_options = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            orig_payment = current_customer.get("PaymentMethod", "Electronic check")
            payment_idx = payment_options.index(orig_payment) if orig_payment in payment_options else 0
            payment_method = st.selectbox("💳 Payment Method", options=payment_options, index=payment_idx)

    modified_customer = current_customer.copy()
    modified_customer.update({
        "Contract": contract,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
    })

    new_prob = predict_proba_single(modified_customer, model, preprocessor)

    st.subheader("📊 Churn Risk Comparison")
    render_probability_gauge(original_prob, new_prob)

    st.subheader("🧠 SHAP Explanation (Before vs After Intervention)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Original Customer — Force Plot**")
        orig_trans = preprocessor.transform(pd.DataFrame([current_customer]))
        orig_force = get_shap_force_plot(explainer, orig_trans, feature_names)
        st_shap(orig_force, height=200)

        st.markdown("**Original — Waterfall Detail**")
        orig_fig = get_shap_waterfall_fig(explainer, orig_trans, feature_names)
        st.pyplot(orig_fig, clear_figure=True)

    with col_b:
        st.markdown("**Post-Intervention — Force Plot**")
        mod_trans = preprocessor.transform(pd.DataFrame([modified_customer]))
        mod_force = get_shap_force_plot(explainer, mod_trans, feature_names)
        st_shap(mod_force, height=200)

        st.markdown("**Post-Intervention — Waterfall Detail**")
        mod_fig = get_shap_waterfall_fig(explainer, mod_trans, feature_names)
        st.pyplot(mod_fig, clear_figure=True)

    st.divider()
    st.subheader("📝 Log This Simulation")

    notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g. Offered 2-year contract discount to reduce churn",
    )

    if st.button("💾 Save to Session History", type="primary"):
        from src.database import log_simulation, init_db
        init_db()
        customer_idx = st.session_state.get("customer_select", 0)
        row_id = log_simulation(
            customer_index=int(customer_idx),
            original_customer=current_customer,
            modified_customer=modified_customer,
            original_prob=original_prob,
            simulated_prob=new_prob,
            notes=notes,
        )
        delta = new_prob - original_prob
        if delta < 0:
            st.success(
                f"✅ Session #{row_id} saved! "
                f"Risk reduced: {original_prob:.1%} → {new_prob:.1%} ({delta:+.1%} 🎉)"
            )
        else:
            st.warning(
                f"⚠️ Session #{row_id} saved. "
                f"Risk unchanged: {original_prob:.1%} → {new_prob:.1%} ({delta:+.1%})"
            )
