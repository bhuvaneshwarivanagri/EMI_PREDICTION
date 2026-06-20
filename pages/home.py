"""pages/home.py — Home Page (v3)"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DATA_PATH  = "data/train_fe.csv"
PALETTE    = {"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
LABEL_MAP  = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}

FINAL_FEATURES = [
    "age", "gender", "marital_status", "education",
    "monthly_salary", "employment_type", "years_of_employment", "company_type",
    "house_type", "existing_loans", "current_emi_amount",
    "credit_score", "bank_balance",
    "requested_amount", "requested_tenure",
    "total_monthly_expenses",
]


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, low_memory=False)
    return None


def show():
    # ── HERO HEADER ──────────────────────────────────────────
    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#1a237e 0%,#0d47a1 50%,#01579b 100%);
                    border-radius:16px;padding:36px 32px;text-align:center;margin-bottom:8px'>
            <div style='font-size:3rem;margin-bottom:6px'>💰</div>
            <h1 style='color:white;margin:0;font-size:2.4rem;letter-spacing:1px'>
                EMIPredict AI
            </h1>
            <p style='color:#90caf9;margin:10px 0 0;font-size:1.1rem'>
                Intelligent Financial Risk Assessment Platform &nbsp;|&nbsp; v3
            </p>
            <div style='margin-top:18px;display:flex;justify-content:center;gap:32px;flex-wrap:wrap'>
                <span style='color:#a5d6a7;font-size:0.9rem'>✅ 4 Classification Models</span>
                <span style='color:#fff176;font-size:0.9rem'>📉 4 Regression Models</span>
                <span style='color:#ef9a9a;font-size:0.9rem'>⚖️ SMOTE Class Balancing</span>
                <span style='color:#80deea;font-size:0.9rem'>🔬 MLflow Experiment Tracking</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── ABOUT + TECH STACK ───────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
### 📌 About This Project
**EMIPredict AI** provides real-time EMI eligibility assessment using ML:
- 🤖 **4 Classification models** — predicts Eligible / High_Risk / Not_Eligible
- 📉 **4 Regression models** — predicts maximum safe monthly EMI (₹)
- ⚖️ **SMOTE balancing** — synthetic oversampling for fair class distribution
- 🔬 **MLflow tracking** — all 8 experiments logged with params, metrics, artifacts
- ⚡ **Real-time predictions** via interactive Streamlit form
- 🗄️ **CRUD management** for loan applications and customer profiles
        """)
    with col2:
        st.markdown("""
### 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| ML Models | Scikit-learn, XGBoost |
| Balancing | imbalanced-learn (SMOTE) |
| Tracking  | MLflow + SQLite |
| Frontend  | Streamlit |
| Data      | Pandas, NumPy |
| Charts    | Plotly |
        """)

    st.divider()

    # ── MODEL ARCHITECTURE CARDS ─────────────────────────────
    st.markdown("### 🤖 Model Architecture")
    clf_models = ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree"]
    reg_models = ["Linear Regression",   "Random Forest", "XGBoost", "Decision Tree"]

    st.markdown("**Classification (Eligibility)**")
    clf_cols = st.columns(4)
    icons = ["📐", "🌲", "⚡", "🌿"]
    for col, name, icon in zip(clf_cols, clf_models, icons):
        with col:
            st.markdown(
                f"""<div style='background:#1a237e22;border:1px solid #1a237e;
                    border-radius:10px;padding:10px;text-align:center;height:80px'>
                    <div style='font-size:1.4rem'>{icon}</div>
                    <div style='font-size:0.72rem;margin-top:4px;color:#90caf9'>{name}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>**Regression (Max EMI)**", unsafe_allow_html=True)
    reg_cols = st.columns(4)
    reg_icons = ["📏", "🌳", "⚡", "🌿"]
    for col, name, icon in zip(reg_cols, reg_models, reg_icons):
        with col:
            st.markdown(
                f"""<div style='background:#01579b22;border:1px solid #01579b;
                    border-radius:10px;padding:10px;text-align:center;height:80px'>
                    <div style='font-size:1.4rem'>{icon}</div>
                    <div style='font-size:0.72rem;margin-top:4px;color:#80deea'>{name}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── DATASET OVERVIEW ─────────────────────────────────────
    st.markdown("### 📦 Dataset Overview")

    df = load_data()

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",     "404,800")
    c2.metric("Raw Features",      "27 → 16")
    c3.metric("SMOTE Train Rows",  "657,021")
    c4.metric("Val / Test Split",  "15% / 15%")
    c5.metric("Target Classes",    "3 + 1 (reg)")

    if df is not None:
        # Map label column
        if "emi_eligibility_enc" in df.columns:
            df["Eligibility"] = df["emi_eligibility_enc"].map(LABEL_MAP)
        elif "emi_eligibility" in df.columns:
            df["Eligibility"] = df["emi_eligibility"]

        # ── ROW 1: Donut + Expense histogram ─────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            if "Eligibility" in df.columns:
                elig = df["Eligibility"].value_counts().reset_index()
                elig.columns = ["Class", "Count"]
                fig = px.pie(
                    elig, values="Count", names="Class",
                    title="📊 EMI Eligibility Distribution",
                    color="Class", color_discrete_map=PALETTE,
                    hole=0.4,
                )
                fig.update_traces(textinfo="percent+label", pull=[0.04, 0.04, 0.04])
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "total_monthly_expenses" in df.columns and "Eligibility" in df.columns:
                sample = df.sample(min(5000, len(df)), random_state=42)
                fig2 = px.histogram(
                    sample, x="total_monthly_expenses",
                    color="Eligibility",
                    color_discrete_map=PALETTE,
                    barmode="overlay", nbins=50, opacity=0.75,
                    title="💸 Monthly Expenses Distribution by Eligibility",
                    labels={"total_monthly_expenses": "Total Monthly Expenses (₹)"},
                )
                fig2.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig2, use_container_width=True)

        # ── ROW 2: Salary box + Credit score violin ───────────
        col_c, col_d = st.columns(2)

        with col_c:
            if "monthly_salary" in df.columns and "Eligibility" in df.columns:
                sample = df.sample(min(5000, len(df)), random_state=1)
                fig3 = px.box(
                    sample, x="Eligibility", y="monthly_salary",
                    color="Eligibility", color_discrete_map=PALETTE,
                    title="💼 Monthly Salary by Eligibility Class",
                    labels={"monthly_salary": "Monthly Salary (₹)"},
                    points=False,
                )
                fig3.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            if "credit_score" in df.columns and "Eligibility" in df.columns:
                sample = df.sample(min(5000, len(df)), random_state=2)
                fig4 = px.violin(
                    sample, x="Eligibility", y="credit_score",
                    color="Eligibility", color_discrete_map=PALETTE,
                    title="🏦 Credit Score Distribution by Eligibility",
                    labels={"credit_score": "Credit Score"},
                    box=True, points=False,
                )
                fig4.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

        # ── ROW 3: Requested amount vs Max EMI scatter ────────
        if "requested_amount" in df.columns and "max_monthly_emi" in df.columns and "Eligibility" in df.columns:
            sample = df.sample(min(3000, len(df)), random_state=3)
            fig5 = px.scatter(
                sample,
                x="requested_amount", y="max_monthly_emi",
                color="Eligibility", color_discrete_map=PALETTE,
                opacity=0.5, size_max=6,
                title="🏠 Requested Loan Amount vs Maximum Safe EMI",
                labels={
                    "requested_amount": "Requested Loan Amount (₹)",
                    "max_monthly_emi":  "Max Monthly EMI (₹)",
                },
            )
            fig5.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig5, use_container_width=True)

        # ── Expandable sections ───────────────────────────────
        with st.expander("📋 Sample Data (first 5 rows — processed features)"):
            st.dataframe(df.head(5), use_container_width=True)

    else:
        st.warning(
            "Processed data not found at `data/train_fe.csv`. "
            "Run the pipeline first to see charts."
        )

    with st.expander("📋 Final Feature List (16 input features)"):
        for feat in FINAL_FEATURES:
            st.markdown(f"✅ `{feat}`")

    st.divider()

    # ── SETUP INSTRUCTIONS ────────────────────────────────────
    with st.expander("⚙️ Setup & Run Instructions"):
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocessing — cleaning, outlier clipping, train/val/test split
python step1_data_preprocessing.py

# 3. Feature engineering — ratios, SMOTE, scaling
python step3_feature_engineering.py

# 4. Train all 10 models + MLflow logging
python step4_train_models.py

# 5. View MLflow dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 6. Launch Streamlit app
streamlit run app.py
        """, language="bash")

    st.caption("💰 EMIPredict AI v3 | GUVI × HCL Final Project | FinTech & Banking")
