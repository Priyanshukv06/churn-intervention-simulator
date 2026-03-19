from src.database import init_db
init_db()  # Ensure DB and table exist on every cold start


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import streamlit as st
from src.explainer import build_explainer, load_explainer
from src.predictor import load_artifacts
from src.preprocessor import get_feature_names_out

st.set_page_config(
    page_title="Churn Intervention Simulator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_all_artifacts():
    model, preprocessor = load_artifacts()
    explainer = build_explainer(model, preprocessor)
    feature_names = get_feature_names_out(preprocessor)
    return model, preprocessor, explainer, feature_names


def main():
    st.title("📱 Customer Churn Intervention Simulator")
    st.markdown(
        """
        **Predict and explain churn risk, then simulate business interventions.**

        Select a customer → adjust interventions → see churn probability + SHAP explanation update **live**.
        """
    )

    model, preprocessor, explainer, feature_names = load_all_artifacts()

    # ── Sidebar ──────────────────────────────────────────
    from app.components.sidebar import render_sidebar
    current_customer = render_sidebar()

    # ── Tabs ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ What-If Simulator",
        "🔍 SHAP Explainability",
        "📊 Model Performance",
        "🗃️ Session History",
    ])

    with tab1:
        from app.tabs.tab_simulator import render_simulator_tab
        render_simulator_tab(
            model=model,
            preprocessor=preprocessor,
            explainer=explainer,
            feature_names=feature_names,
            current_customer=current_customer,
        )

    with tab2:
        from app.tabs.tab_explainability import render_explainability_tab
        render_explainability_tab(
            model=model,
            preprocessor=preprocessor,
            explainer=explainer,
            feature_names=feature_names,
            current_customer=current_customer,
        )

    with tab3:
        from app.tabs.tab_performance import render_performance_tab
        render_performance_tab()

    with tab4:
        from app.tabs.tab_history import render_history_tab
        render_history_tab()


main()
