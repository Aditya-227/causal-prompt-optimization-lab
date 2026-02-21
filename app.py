import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(
    page_title="Causal Prompt Optimization Lab",
    layout="wide"
)

st.title("Causal Prompt Optimization Lab")

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
df = pd.read_csv("causal_prompt_results.csv")

# -----------------------------------------------------
# Top KPI Section
# -----------------------------------------------------
st.subheader("Overall Performance Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Mean Accuracy", round(df["correct"].mean(), 3))
col2.metric("Mean Tokens", round(df["total_tokens"].mean(), 1))
col3.metric("Mean Latency (s)", round(df["latency"].mean(), 2))

st.divider()

# -----------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------
st.sidebar.header("Controls")

metric_choice = st.sidebar.selectbox(
    "Select Metric",
    ["correct", "total_tokens", "latency"]
)

show_labels = st.sidebar.checkbox("Show Frontier Labels")

st.sidebar.markdown("---")

st.sidebar.subheader("Configuration Explorer")

cot_filter = st.sidebar.selectbox("CoT", [0,1])
fewshot_filter = st.sidebar.selectbox("Few-shot", [0,1])
role_filter = st.sidebar.selectbox("Role", [0,1])
constraint_filter = st.sidebar.selectbox("Constraint", [0,1])

filtered = df[
    (df["cot"] == cot_filter) &
    (df["fewshot"] == fewshot_filter) &
    (df["role"] == role_filter) &
    (df["constraint"] == constraint_filter)
]
st.subheader("Selected Configuration Performance")

if len(filtered) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Accuracy", round(filtered["correct"].mean(),3))
    col2.metric("Mean Tokens", round(filtered["total_tokens"].mean(),1))
    col3.metric("Mean Latency", round(filtered["latency"].mean(),2))
else:
    st.warning("No data for selected configuration.")
# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Effects", "Optimization", "Statistical Models", "Documentation"]
)

# =====================================================
# TAB 1 — Effects
# =====================================================
with tab1:

    st.subheader("Main Effect (Chain-of-Thought)")
    fig_bar = px.bar(
        df.groupby("cot")["correct"].mean().reset_index(),
        x="cot",
        y="correct",
        labels={"cot":"CoT", "correct":"Accuracy"},
        title="CoT Effect on Accuracy"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Interaction Heatmap (CoT × Few-shot)")
    heat_data = df.groupby(["cot","fewshot"])[metric_choice].mean().reset_index()
    fig_heat = px.density_heatmap(
        heat_data,
        x="cot",
        y="fewshot",
        z=metric_choice,
        text_auto=True,
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# TAB 2 — Optimization
# =====================================================
with tab2:

    agg = df.groupby(["cot","fewshot","role","constraint"]).agg({
        "correct":"mean",
        "total_tokens":"mean",
        "latency":"mean"
    }).reset_index()

    st.subheader("Token–Accuracy Frontier")

    fig_frontier = px.scatter(
        agg,
        x="total_tokens",
        y="correct",
        hover_data=["cot","fewshot","role","constraint"],
        labels={"total_tokens":"Mean Tokens", "correct":"Mean Accuracy"},
        title="Token–Accuracy Tradeoff"
    )

    st.plotly_chart(fig_frontier, use_container_width=True)

    st.subheader("Latency–Accuracy Frontier")

    fig_latency = px.scatter(
        agg,
        x="latency",
        y="correct",
        hover_data=["cot","fewshot","role","constraint"],
        labels={"latency":"Mean Latency", "correct":"Mean Accuracy"},
        title="Latency–Accuracy Tradeoff"
    )

    st.plotly_chart(fig_latency, use_container_width=True)

    st.subheader("Efficiency Ranking")

    df["accuracy_per_token"] = df["correct"] / df["total_tokens"]
    df["accuracy_per_second"] = df["correct"] / df["latency"]

    eff = df.groupby(["cot","fewshot","role","constraint"]).agg({
        "accuracy_per_token":"mean",
        "accuracy_per_second":"mean",
        "correct":"mean",
        "total_tokens":"mean",
        "latency":"mean"
    }).reset_index()

    st.dataframe(
        eff.sort_values("accuracy_per_token", ascending=False),
        use_container_width=True
    )

    st.download_button(
        "Download Efficiency Table",
        eff.to_csv(index=False),
        file_name="efficiency_results.csv"
    )

# =====================================================
# TAB 3 — Statistical Models
# =====================================================
with tab3:

    st.subheader("Fixed Effects Regression (Accuracy)")

    model = smf.ols(
        "correct ~ cot + fewshot + role + constraint + C(question_id)",
        data=df
    ).fit()

    st.text(model.summary())

    st.subheader("Average Treatment Effects (ATE)")

    ate_cot = df[df["cot"]==1]["correct"].mean() - df[df["cot"]==0]["correct"].mean()
    ate_few = df[df["fewshot"]==1]["correct"].mean() - df[df["fewshot"]==0]["correct"].mean()

    st.write(f"ATE (CoT): {round(ate_cot,3)}")
    st.write(f"ATE (Few-shot): {round(ate_few,3)}")

# =====================================================
# TAB 4 — Documentation
# =====================================================
with tab4:

    st.markdown("""
    ## Project Overview

    This dashboard evaluates the causal impact of prompt components on LLM performance.

    ### Experimental Design
    - Role Instruction (0/1)
    - Chain-of-Thought (0/1)
    - Few-shot Examples (0/1)
    - Output Constraint (0/1)

    Factorial within-subject design using GSM8K benchmark.

    ### Key Findings
    - CoT significantly improves accuracy.
    - Role instruction reduces performance.
    - Output constraint reduces token and latency cost.
    - No strong interaction detected.

    Optimal Policy: Chain-of-Thought + Output Constraint.
    """)
