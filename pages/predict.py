"""
pages/predict.py — Real-time EMI Prediction (v3)
=================================================
Simplified input form:
  • Single total_monthly_expenses field (no sub-categories)
  • Gauge chart for max EMI + horizontal probability bars
  • Live expense ratio indicator
  • Detailed AI recommendation per eligibility class
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

CLF_LABEL_INV = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}

CLF_COLORS = {
    "Eligible"    : "#2ecc71",
    "High_Risk"   : "#f39c12",
    "Not_Eligible": "#e74c3c",
}
CLF_ICONS = {
    "Eligible"    : "✅",
    "High_Risk"   : "⚠️",
    "Not_Eligible": "❌",
}


@st.cache_resource
def load_models():
    try:
        clf       = joblib.load("models/best_classifier.pkl")
        reg       = joblib.load("models/best_regressor.pkl")
        scaler    = joblib.load("models/scaler.pkl")
        encoders  = joblib.load("models/encoders.pkl")
        feat_cols = joblib.load("models/feature_cols.pkl")
        log_path  = "models/log_transformed_cols.pkl"
        log_cols  = joblib.load(log_path) if os.path.exists(log_path) else []
        return clf, reg, scaler, encoders, feat_cols, log_cols, True
    except FileNotFoundError:
        return None, None, None, None, None, [], False


def preprocess_input(raw: dict, scaler, encoders, feat_cols, log_cols) -> np.ndarray:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from step3_feature_engineering import (
        add_financial_ratios, add_risk_features, add_interaction_features
    )

    df = pd.DataFrame([raw])

    # Feature engineering
    add_financial_ratios(df)
    add_risk_features(df)
    add_interaction_features(df)

    # Log1p transform (same columns used during training)
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            val = val if val in le.classes_ else le.classes_[0]
            df[col] = le.transform([val])[0]

    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail].astype(float).fillna(0).values
    return scaler.transform(X)


def gauge_chart(value: float, max_val: float = 50000) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Maximum Safe Monthly EMI"},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar" : {"color": "#2ecc71"},
            "steps": [
                {"range": [0,              max_val * 0.33], "color": "#fee2e2"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#fef9c3"},
                {"range": [max_val * 0.66, max_val],        "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
        number={"prefix": "₹", "valueformat": ",.0f"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def probability_chart(proba: list) -> go.Figure:
    labels = ["Eligible", "High_Risk", "Not_Eligible"]
    colors = [CLF_COLORS[l] for l in labels]
    fig = go.Figure(go.Bar(
        x=[f"{p:.1%}" for p in proba],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in proba],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Confidence",
        xaxis=dict(title="Probability", range=[0, 1.1]),
        height=280,
        template="plotly_dark",
        margin=dict(t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
def show():
    st.title("🔮 EMI Eligibility Prediction")
    st.caption(
        "Enter your financial details to get instant eligibility assessment "
        "and maximum safe EMI amount."
    )

    clf, reg, scaler, encoders, feat_cols, log_cols, models_ready = load_models()

    if not models_ready:
        st.error("⚠️ Models not found. Run the training pipeline first.")
        st.code(
            "python step1_data_preprocessing.py\n"
            "python step3_feature_engineering.py\n"
            "python step4_train_models.py",
            language="bash",
        )
        return

    # ── Input Form ────────────────────────────
    st.markdown("### 👤 Personal Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age            = st.number_input("Age", 18, 70, 35)
        gender         = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col2:
        education           = st.selectbox("Education",
                                ["Graduate", "Post_Graduate", "Under_Graduate", "PhD"])
        employment_type     = st.selectbox("Employment Type",
                                ["Salaried", "Self_Employed", "Freelancer"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0, step=0.5)
    with col3:
        company_type = st.selectbox("Company Type",
                            ["Private", "Government", "MNC", "Startup"])
        house_type   = st.selectbox("House Type", ["Owned", "Rented", "Parental"])

    st.divider()
    st.markdown("### 💰 Income & Financial Status")
    col4, col5, col6 = st.columns(3)
    with col4:
        monthly_salary = st.number_input(
            "Monthly Salary (₹)", 5000, 10_000_000, 50000, step=1000,
            help="Your gross monthly income before deductions",
        )
    with col5:
        credit_score = st.number_input(
            "Credit Score", 300, 900, 700,
            help="CIBIL / credit score (300–900)",
        )
    with col6:
        bank_balance = st.number_input(
            "Bank Balance (₹)", 0, 50_000_000, 100000, step=5000,
            help="Current savings / account balance",
        )

    st.divider()
    st.markdown("### 💸 Monthly Expenses")
    st.info(
        "Enter your **total monthly expenses** — this includes rent, groceries, "
        "school fees, travel, utilities, and all other regular costs. "
        "**Do not include your current EMI** — that is entered separately below."
    )

    col7, col8 = st.columns(2)
    with col7:
        total_monthly_expenses = st.number_input(
            "Total Monthly Expenses (₹)",
            min_value=0, max_value=5_000_000,
            value=20000, step=500,
            help=(
                "Sum of: rent + groceries + utilities + school/college fees "
                "+ travel + all other recurring monthly costs. "
                "Do NOT include your existing EMI here."
            ),
        )
    with col8:
        if monthly_salary > 0:
            ratio = total_monthly_expenses / monthly_salary * 100
            color = "🟢" if ratio < 40 else ("🟡" if ratio < 60 else "🔴")
            st.metric(
                "Expense Ratio",
                f"{ratio:.1f}% of salary",
                delta=f"{color} {'Low' if ratio < 40 else ('Moderate' if ratio < 60 else 'High')}",
                delta_color="normal" if ratio < 60 else "inverse",
            )

    st.divider()
    st.markdown("### 🏦 Existing Loans & Loan Request")
    col9, col10, col11, col12, col13 = st.columns(5)
    with col9:
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
    with col10:
        current_emi_amount = st.number_input(
            "Current EMI (₹)", 0, 500_000, 0, step=500,
            help="Total EMI you are already paying each month for existing loans.",
        )
    with col11:
        requested_amount = st.number_input(
            "Loan Amount (₹)", 10000, 100_000_000, 300000, step=5000,
        )
    with col12:
        requested_tenure = st.number_input("Tenure (months)", 6, 360, 24)
    with col13:
        if requested_amount > 0 and requested_tenure > 0:
            est_emi = requested_amount / requested_tenure * 1.08
            st.metric("Est. New EMI", f"₹{est_emi:,.0f}/mo",
                      help="Rough estimate assuming ~8% interest")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict EMI Eligibility", type="primary",
                             use_container_width=True)

    # ── RUN PREDICTION ────────────────────────
    if predict_btn:
        total_all_expenses = total_monthly_expenses + current_emi_amount

        raw_input = {
            "age"                   : float(age),
            "gender"                : gender,
            "marital_status"        : marital_status,
            "education"             : education,
            "monthly_salary"        : float(monthly_salary),
            "employment_type"       : employment_type,
            "years_of_employment"   : float(years_of_employment),
            "company_type"          : company_type,
            "house_type"            : house_type,
            "existing_loans"        : 1 if existing_loans == "Yes" else 0,
            "current_emi_amount"    : float(current_emi_amount),
            "credit_score"          : float(credit_score),
            "bank_balance"          : float(bank_balance),
            "requested_amount"      : float(requested_amount),
            "requested_tenure"      : int(requested_tenure),
            "total_monthly_expenses": float(total_all_expenses),
        }

        with st.spinner("🤖 Running AI models..."):
            X         = preprocess_input(raw_input, scaler, encoders, feat_cols, log_cols)
            clf_pred  = int(clf.predict(X)[0])
            clf_proba = (clf.predict_proba(X)[0].tolist()
                         if hasattr(clf, "predict_proba") else [0.0, 0.0, 0.0])
            reg_pred  = float(max(reg.predict(X)[0], 0))
            eligibility = CLF_LABEL_INV[clf_pred]

        # ── RESULTS ───────────────────────────
        st.divider()
        st.markdown("## 🎯 Prediction Results")

        color = CLF_COLORS[eligibility]
        icon  = CLF_ICONS[eligibility]
        st.markdown(
            f"<div style='background:{color}22;border:2px solid {color};"
            f"border-radius:12px;padding:20px;text-align:center'>"
            f"<h2>{icon} {eligibility.replace('_', ' ')}</h2>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gauge_chart(reg_pred), use_container_width=True)
        with c2:
            st.plotly_chart(probability_chart(clf_proba), use_container_width=True)

        # ── Key metrics ───────────────────────
        st.markdown("### 📊 Your Financial Snapshot")
        disposable = monthly_salary - total_all_expenses
        dti        = total_all_expenses / max(monthly_salary, 1) * 100
        emi_ratio  = current_emi_amount / max(monthly_salary, 1) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Monthly Salary",    f"₹{monthly_salary:,.0f}")
        m2.metric("Total Expenses",    f"₹{total_all_expenses:,.0f}")
        m3.metric("Disposable Income", f"₹{disposable:,.0f}",
                  delta="Positive ✅" if disposable > 0 else "Negative ❌",
                  delta_color="normal" if disposable > 0 else "inverse")
        m4.metric("Expense Ratio",     f"{dti:.1f}%",
                  delta="OK ✅" if dti < 50 else "High ⚠️",
                  delta_color="normal" if dti < 50 else "inverse")
        m5.metric("Max Safe EMI",      f"₹{reg_pred:,.0f}")

        # ── Recommendation ────────────────────
        st.markdown("### 💡 AI Recommendation")
        if eligibility == "Eligible":
            st.success(f"""
**✅ Loan Recommended**
- Maximum safe monthly EMI you can afford: **₹{reg_pred:,.0f}**
- Your expense ratio ({dti:.1f}%) is within healthy limits.
- Credit score {credit_score} indicates good creditworthiness.
- Requested loan amount: ₹{requested_amount:,.0f} over {requested_tenure} months.
            """)
        elif eligibility == "High_Risk":
            st.warning(f"""
**⚠️ Loan Possible with Conditions**
- Maximum safe monthly EMI: **₹{reg_pred:,.0f}**
- Consider a higher interest rate to offset risk.
- Suggestions:
  - Reduce requested amount to lower monthly EMI
  - Extend tenure to reduce monthly burden
  - Provide collateral or a co-applicant
- Expense ratio ({dti:.1f}%) needs improvement — aim for < 50%.
            """)
        else:
            st.error(f"""
**❌ Loan Not Recommended at This Time**
- Maximum safe monthly EMI: **₹{reg_pred:,.0f}**
- Your current expenses ({dti:.1f}% of salary) exceed safe thresholds.
- Action steps:
  - Reduce monthly expenses (target below 50% of salary)
  - Clear existing loans before applying (current EMI: ₹{current_emi_amount:,.0f})
  - Improve credit score (currently {credit_score} — target 700+)
  - Build bank balance to strengthen financial profile
  - Consider a smaller loan amount of ₹{requested_amount * 0.5:,.0f} or less
            """)
