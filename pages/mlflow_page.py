"""
pages/mlflow_page.py
====================
Model Performance page — MLflow experiment results, metric comparison
tables and bar charts for all 4 classifiers and 4 regressors.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data(ttl=30, show_spinner="Fetching MLflow runs…")
def fetch_runs(experiment_name: str) -> pd.DataFrame:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            filter_string="status = 'FINISHED'",   # only completed runs
        )
        return runs
    except Exception:
        return pd.DataFrame()


CLF_METRICS = ["test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_roc_auc"]
REG_METRICS = ["test_rmse", "test_mae", "test_r2", "test_mape"]

CLF_DISPLAY = {
    "test_accuracy" : "Accuracy",
    "test_precision": "Precision",
    "test_recall"   : "Recall",
    "test_f1_score" : "F1-Score",
    "test_roc_auc"  : "ROC-AUC",
}
REG_DISPLAY = {
    "test_rmse": "RMSE (₹)",
    "test_mae" : "MAE (₹)",
    "test_r2"  : "R²",
    "test_mape": "MAPE (%)",
}


def _build_clf_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["tags.mlflow.runName", "start_time"] + [f"metrics.{m}" for m in CLF_METRICS]
    avail = [c for c in cols if c in runs.columns]
    df = runs[avail].copy().dropna(subset=["tags.mlflow.runName"])
    df = df.rename(columns={
        "tags.mlflow.runName": "Model",
        **{f"metrics.{k}": v for k, v in CLF_DISPLAY.items()},
    })
    # Keep only the LATEST run per model name
    if "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False)
        df = df.drop_duplicates(subset=["Model"]).drop(columns=["start_time"])
    else:
        df = df.drop_duplicates(subset=["Model"])
    return df.reset_index(drop=True)


def _build_reg_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["tags.mlflow.runName", "start_time"] + [f"metrics.{m}" for m in REG_METRICS]
    avail = [c for c in cols if c in runs.columns]
    df = runs[avail].copy().dropna(subset=["tags.mlflow.runName"])
    df = df.rename(columns={
        "tags.mlflow.runName": "Model",
        **{f"metrics.{k}": v for k, v in REG_DISPLAY.items()},
    })
    # Keep only the LATEST run per model name
    if "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False)
        df = df.drop_duplicates(subset=["Model"]).drop(columns=["start_time"])
    else:
        df = df.drop_duplicates(subset=["Model"])
    return df.reset_index(drop=True)


def _highlight_best_clf(df: pd.DataFrame):
    styled = df.style
    for col in CLF_DISPLAY.values():
        if col in df.columns:
            best_idx = df[col].idxmax()
            styled = styled.highlight_max(subset=[col], color="#2ecc7133")
    return styled


def _highlight_best_reg(df: pd.DataFrame):
    styled = df.style
    for col in ["R²"]:
        if col in df.columns:
            styled = styled.highlight_max(subset=[col], color="#2ecc7133")
    for col in ["RMSE (₹)", "MAE (₹)", "MAPE (%)"]:
        if col in df.columns:
            styled = styled.highlight_min(subset=[col], color="#2ecc7133")
    return styled


def _get_production_model(registry_name: str) -> str:
    """Return the Production model version info, or empty string."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{registry_name}'")
        for v in versions:
            if v.current_stage == "Production":
                return f"v{v.version} — {v.description or v.run_id[:8]}"
    except Exception:
        pass
    return ""


def show():
    st.title("📈 Model Performance")
    st.markdown("Compare all trained models using test-set metrics tracked in **MLflow**.")

    # ── Production model badges ───────────────────────────────
    clf_prod = _get_production_model("EMI_Best_Classifier")
    reg_prod = _get_production_model("EMI_Best_Regressor")
    if clf_prod or reg_prod:
        c1, c2 = st.columns(2)
        if clf_prod:
            c1.success(f"🏭 **Production Classifier:** {clf_prod}")
        if reg_prod:
            c2.success(f"🏭 **Production Regressor:** {reg_prod}")

    st.divider()

    import os
    if not os.path.exists("mlflow.db"):
        st.warning(
            "MLflow database not found (`mlflow.db`). "
            "Run `python step4_train_models.py` first, then reload this page."
        )
        return

    # ── CLASSIFICATION ────────────────────────────────────────
    st.subheader("🔵 Classification Results")
    st.caption("4 Models trained on SMOTE-balanced data | Evaluated on held-out test set")

    clf_runs = fetch_runs("EMI_Classification")

    if clf_runs.empty:
        st.info("No classification runs found in MLflow yet.")
    else:
        clf_df = _build_clf_table(clf_runs)

        if not clf_df.empty:
            st.dataframe(
                _highlight_best_clf(clf_df.set_index("Model")),
                use_container_width=True,
            )

            # Bar chart — pick metric
            st.markdown("")
            clf_metric_choice = st.selectbox(
                "Plot metric:", list(CLF_DISPLAY.values()), key="clf_metric"
            )

            if clf_metric_choice in clf_df.columns:
                sorted_df = clf_df.sort_values(clf_metric_choice, ascending=False)
                colors = px.colors.qualitative.Set2[:len(sorted_df)]
                fig = go.Figure(go.Bar(
                    x=sorted_df["Model"],
                    y=sorted_df[clf_metric_choice],
                    marker_color=colors,
                    text=sorted_df[clf_metric_choice].map(lambda v: f"{v:.4f}"),
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Classification — {clf_metric_choice} Comparison",
                    yaxis_title=clf_metric_choice,
                    yaxis=dict(range=[0, 1.05]),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Best model callout
            if "Accuracy" in clf_df.columns:
                best_row = clf_df.loc[clf_df["Accuracy"].idxmax()]
                st.success(
                    f"🏆 **Best Classifier:** {best_row['Model']}  "
                    f"| Accuracy={best_row.get('Accuracy', 0):.4f}  "
                    f"| F1={best_row.get('F1-Score', 0):.4f}  "
                    f"| ROC-AUC={best_row.get('ROC-AUC', 0):.4f}"
                )

    st.divider()

    # ── REGRESSION ────────────────────────────────────────────
    st.subheader("🟠 Regression Results")
    st.caption("4 Models trained on original-balance data | Evaluated on held-out test set")

    reg_runs = fetch_runs("EMI_Regression")

    if reg_runs.empty:
        st.info("No regression runs found in MLflow yet.")
    else:
        reg_df = _build_reg_table(reg_runs)

        if not reg_df.empty:
            st.dataframe(
                _highlight_best_reg(reg_df.set_index("Model")),
                use_container_width=True,
            )

            st.markdown("")
            reg_metric_choice = st.selectbox(
                "Plot metric:", list(REG_DISPLAY.values()), key="reg_metric"
            )

            if reg_metric_choice in reg_df.columns:
                ascending = reg_metric_choice != "R²"
                sorted_df = reg_df.sort_values(reg_metric_choice, ascending=ascending)
                colors = px.colors.qualitative.Pastel[:len(sorted_df)]
                fig = go.Figure(go.Bar(
                    x=sorted_df["Model"],
                    y=sorted_df[reg_metric_choice],
                    marker_color=colors,
                    text=sorted_df[reg_metric_choice].map(
                        lambda v: f"₹{v:,.0f}" if "₹" in reg_metric_choice else f"{v:.4f}"
                    ),
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Regression — {reg_metric_choice} Comparison",
                    yaxis_title=reg_metric_choice,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Best model callout
            if "RMSE (₹)" in reg_df.columns:
                best_row = reg_df.loc[reg_df["RMSE (₹)"].idxmin()]
                st.success(
                    f"🏆 **Best Regressor:** {best_row['Model']}  "
                    f"| RMSE=₹{best_row.get('RMSE (₹)', 0):,.0f}  "
                    f"| MAE=₹{best_row.get('MAE (₹)', 0):,.0f}  "
                    f"| R²={best_row.get('R²', 0):.4f}"
                )

    st.divider()

    # ── MLFLOW UI LINK ────────────────────────────────────────
    st.subheader("🔗 MLflow UI")
    st.markdown(
        "Launch the full MLflow dashboard to inspect individual runs, "
        "artifacts, and confusion matrices:"
    )
    st.code(
        "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000",
        language="bash",
    )
    st.caption("Then open http://localhost:5000 in your browser.")

    if st.button("🔄 Refresh Results"):
        fetch_runs.clear()
        st.rerun()
