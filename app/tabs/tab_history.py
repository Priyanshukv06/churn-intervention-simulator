import streamlit as st
import pandas as pd
import plotly.express as px

from src.database import (
    fetch_all_sessions,
    delete_session,
    clear_all_sessions,
    get_session_stats,
    init_db,
)


def render_history_tab():
    init_db()

    st.header("🗃️ Session History")
    st.markdown("Every simulation you log is stored here with full before/after details.")

    # ── Stats cards ───────────────────────────────────────────────────────
    stats = get_session_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", stats["total_sessions"])
    col2.metric(
        "Avg Original Risk",
        f"{float(stats['avg_original_risk']):.1%}" if stats["avg_original_risk"] else "—",
    )
    col3.metric(
        "Avg Simulated Risk",
        f"{float(stats['avg_simulated_risk']):.1%}" if stats["avg_simulated_risk"] else "—",
    )
    col4.metric(
        "Interventions That Helped",
        stats["interventions_that_helped"],
        help="Sessions where simulated risk < original risk",
    )

    st.divider()

    # ── Session table ─────────────────────────────────────────────────────
    df = fetch_all_sessions()

    if df.empty:
        st.info("No sessions logged yet. Use the **What-If Simulator** tab and click '📝 Log This Simulation'.")
        return

    st.subheader(f"📋 Logged Sessions ({len(df)} total)")

    # Filters
    with st.expander("🔍 Filters", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            min_delta = st.slider(
                "Min probability delta",
                min_value=-1.0, max_value=1.0,
                value=-1.0, step=0.01,
            )
        with col_b:
            show_helped = st.checkbox("Only show interventions that reduced risk", value=False)

    filtered_df = df.copy()
    filtered_df["prob_delta"] = filtered_df["prob_delta"].astype(float)
    filtered_df = filtered_df[filtered_df["prob_delta"] >= min_delta]
    if show_helped:
        filtered_df = filtered_df[filtered_df["prob_delta"] < 0]

    # Display columns
    display_cols = [
        "id", "timestamp", "customer_index",
        "original_prob", "simulated_prob", "prob_delta",
        "contract_original", "contract_simulated",
        "tenure_original", "tenure_simulated",
    ]
    display_df = filtered_df[display_cols].copy()
    display_df["original_prob"] = display_df["original_prob"].apply(lambda x: f"{x:.1%}")
    display_df["simulated_prob"] = display_df["simulated_prob"].apply(lambda x: f"{x:.1%}")
    display_df["prob_delta"] = display_df["prob_delta"].apply(
        lambda x: f"{'↓' if x < 0 else '↑'} {abs(x):.1%}"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Trend chart ───────────────────────────────────────────────────────
    if len(df) >= 2:
        st.subheader("📈 Risk Reduction Over Sessions")
        chart_df = df[["id", "original_prob", "simulated_prob"]].copy()
        chart_df = chart_df.melt(
            id_vars="id",
            value_vars=["original_prob", "simulated_prob"],
            var_name="Type",
            value_name="Churn Probability",
        )
        chart_df["Type"] = chart_df["Type"].map({
            "original_prob": "Original Risk",
            "simulated_prob": "Simulated Risk",
        })
        fig = px.line(
            chart_df,
            x="id", y="Churn Probability",
            color="Type",
            markers=True,
            title="Original vs Simulated Churn Risk Per Session",
            color_discrete_map={
                "Original Risk": "salmon",
                "Simulated Risk": "steelblue",
            },
        )
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Session ID")
        st.plotly_chart(fig, use_container_width=True)

    # ── Danger zone ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        col_del, col_clear = st.columns(2)
        with col_del:
            del_id = st.number_input("Delete session by ID", min_value=1, step=1)
            if st.button("🗑️ Delete Session", type="secondary"):
                delete_session(int(del_id))
                st.success(f"Session {del_id} deleted.")
                st.rerun()
        with col_clear:
            st.warning("This will wipe ALL sessions permanently.")
            if st.button("💣 Clear All Sessions", type="secondary"):
                clear_all_sessions()
                st.success("All sessions cleared.")
                st.rerun()
