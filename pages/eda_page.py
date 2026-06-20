"""
pages/eda_page.py
=================
Data Exploration page — distributions, correlation heatmap,
class balance, EMI distribution, feature analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data(show_spinner="Loading data…")
def load_data():
    dfs = {}
    paths = {
        "train":      "data/train_fe.csv",
        "train_smote":"data/train_fe_smote.csv",
        "val":        "data/val_fe.csv",
        "test":       "data/test_fe.csv",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path, low_memory=False)
    return dfs


LABEL_MAP  = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
CLF_COLORS = {"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}


def show():
    st.title("📊 Data Exploration")
    st.markdown("Interactive exploration of the training data after feature engineering.")
    st.divider()

    dfs = load_data()

    if not dfs:
        st.warning("No data files found. Run the preprocessing and feature engineering steps first.")
        return

    df = dfs.get("train", next(iter(dfs.values())))

    # ── DATASET OVERVIEW ──────────────────────────────────────
    st.subheader("📋 Dataset Overview")

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Rows",    f"{len(df):,}")
    ov2.metric("Total Columns", len(df.columns))
    ov3.metric("Numeric Cols",  len(df.select_dtypes(include="number").columns))
    ov4.metric("Missing Values",f"{df.isnull().sum().sum():,}")

    with st.expander("🔍 Show raw data sample"):
        st.dataframe(df.head(100), use_container_width=True)

    st.divider()

    # ── CLASS BALANCE ─────────────────────────────────────────
    st.subheader("⚖️ Class Balance")

    bc1, bc2 = st.columns(2)

    with bc1:
        if "emi_eligibility_enc" in df.columns:
            dist_orig = (
                df["emi_eligibility_enc"]
                .map(LABEL_MAP)
                .value_counts()
                .reset_index()
            )
            dist_orig.columns = ["Class", "Count"]
            fig = px.pie(
                dist_orig, names="Class", values="Count",
                color="Class", color_discrete_map=CLF_COLORS,
                title="Original Train — Class Distribution",
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with bc2:
        if "train_smote" in dfs and "emi_eligibility_enc" in dfs["train_smote"].columns:
            dist_smote = (
                dfs["train_smote"]["emi_eligibility_enc"]
                .map(LABEL_MAP)
                .value_counts()
                .reset_index()
            )
            dist_smote.columns = ["Class", "Count"]
            fig2 = px.pie(
                dist_smote, names="Class", values="Count",
                color="Class", color_discrete_map=CLF_COLORS,
                title="SMOTE Train — Class Distribution",
            )
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── EMI DISTRIBUTION ─────────────────────────────────────
    st.subheader("💳 Max Monthly EMI Distribution")

    if "max_monthly_emi" in df.columns:
        emi_col = df["max_monthly_emi"].dropna()

        ec1, ec2 = st.columns(2)
        with ec1:
            fig = px.histogram(
                emi_col, nbins=60,
                title="EMI Distribution — Original Train",
                labels={"value": "Max Monthly EMI (₹)"},
                color_discrete_sequence=["#42a5f5"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with ec2:
            # Per-class box plot
            if "emi_eligibility_enc" in df.columns:
                box_df = df[["max_monthly_emi", "emi_eligibility_enc"]].copy()
                box_df["Class"] = box_df["emi_eligibility_enc"].map(LABEL_MAP)
                fig2 = px.box(
                    box_df, x="Class", y="max_monthly_emi",
                    color="Class", color_discrete_map=CLF_COLORS,
                    title="EMI by Eligibility Class",
                    labels={"max_monthly_emi": "Max Monthly EMI (₹)"},
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── NUMERIC FEATURE DISTRIBUTIONS ────────────────────────
    st.subheader("📈 Feature Distributions")

    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ["emi_eligibility_enc", "max_monthly_emi"]
    ]

    selected_feat = st.selectbox("Select a feature to explore:", num_cols)

    if selected_feat:
        fd1, fd2 = st.columns(2)
        series = df[selected_feat].dropna()

        with fd1:
            fig = px.histogram(
                series, nbins=50,
                title=f"Distribution of {selected_feat}",
                color_discrete_sequence=["#ab47bc"],
                labels={"value": selected_feat},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with fd2:
            if "emi_eligibility_enc" in df.columns:
                scatter_df = df[[selected_feat, "max_monthly_emi", "emi_eligibility_enc"]].copy()
                scatter_df["Class"] = scatter_df["emi_eligibility_enc"].map(LABEL_MAP)
                fig2 = px.scatter(
                    scatter_df.sample(min(2000, len(scatter_df)), random_state=42),
                    x=selected_feat, y="max_monthly_emi",
                    color="Class", color_discrete_map=CLF_COLORS,
                    opacity=0.5,
                    title=f"{selected_feat} vs Max EMI",
                    labels={"max_monthly_emi": "Max Monthly EMI (₹)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        with st.expander(f"📐 Summary statistics for {selected_feat}"):
            st.dataframe(series.describe().rename("Value").to_frame(), use_container_width=True)

    st.divider()

    # ── CORRELATION HEATMAP ───────────────────────────────────
    st.subheader("🔥 Correlation Heatmap")

    top_n = st.slider("Number of top numeric features to include:", 5, min(30, len(num_cols)), 15)

    # Select top features by correlation to max_monthly_emi
    if "max_monthly_emi" in df.columns and len(num_cols) > 0:
        corr_target = (
            df[num_cols + ["max_monthly_emi"]]
            .corr()["max_monthly_emi"]
            .abs()
            .drop("max_monthly_emi")
            .nlargest(top_n)
            .index.tolist()
        )
        heat_cols = corr_target + ["max_monthly_emi"]
        corr_matrix = df[heat_cols].corr().round(2)

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Correlation Matrix — Top {top_n} Features",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── MISSING VALUES ────────────────────────────────────────
    st.subheader("🕳️ Missing Values Check")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("✅ No missing values found in the training set.")
    else:
        miss_df = missing.reset_index()
        miss_df.columns = ["Column", "Missing Count"]
        miss_df["Missing %"] = (miss_df["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
