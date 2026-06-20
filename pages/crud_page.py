"""
pages/crud_page.py
==================
Data Management page — Create, Read, Update, Delete loan applications.
Records are stored in data/loan_applications.csv.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime

APPS_CSV = "data/loan_applications.csv"

COLUMNS = [
    "id", "submitted_at",
    "age", "gender", "marital_status", "education",
    "monthly_salary", "employment_type", "years_of_employment", "company_type",
    "house_type", "existing_loans", "current_emi_amount",
    "credit_score", "bank_balance",
    "requested_amount", "requested_tenure", "total_monthly_expenses",
    "predicted_eligibility", "predicted_max_emi", "notes",
]


# ── FILE HELPERS ──────────────────────────────────────────────
def _load() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(APPS_CSV):
        return pd.read_csv(APPS_CSV)
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(APPS_CSV, index=False)


def _next_id(df: pd.DataFrame) -> int:
    return int(df["id"].max()) + 1 if not df.empty else 1


# ══════════════════════════════════════════════════════════════
def show():
    st.title("🛠️ Data Management")
    st.markdown("Manage loan applications — add, view, edit, and delete records.")
    st.divider()

    df = _load()

    # ── TABS ─────────────────────────────────────────────────
    tab_view, tab_add, tab_edit, tab_delete, tab_export = st.tabs([
        "📋 View All", "➕ Add New", "✏️ Edit", "🗑️ Delete", "📥 Export"
    ])

    # ════════════════════════════════════════════════════════
    # VIEW
    # ════════════════════════════════════════════════════════
    with tab_view:
        st.subheader("All Loan Applications")

        if df.empty:
            st.info("No applications yet. Use **Add New** to create one.")
        else:
            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                elig_filter = st.multiselect(
                    "Filter by Eligibility",
                    options=["Eligible", "High_Risk", "Not_Eligible", "—"],
                    default=[],
                )
            with fc2:
                gender_filter = st.multiselect("Filter by Gender", options=["Male", "Female"], default=[])
            with fc3:
                salary_range = st.slider(
                    "Monthly Salary (₹)",
                    int(df["monthly_salary"].min() if "monthly_salary" in df.columns and not df.empty else 0),
                    int(df["monthly_salary"].max() if "monthly_salary" in df.columns and not df.empty else 1_000_000),
                    (
                        int(df["monthly_salary"].min() if "monthly_salary" in df.columns and not df.empty else 0),
                        int(df["monthly_salary"].max() if "monthly_salary" in df.columns and not df.empty else 1_000_000),
                    ),
                )

            view_df = df.copy()
            if elig_filter:
                view_df = view_df[view_df["predicted_eligibility"].isin(elig_filter)]
            if gender_filter:
                view_df = view_df[view_df["gender"].isin(gender_filter)]
            if "monthly_salary" in view_df.columns:
                view_df = view_df[
                    (view_df["monthly_salary"] >= salary_range[0]) &
                    (view_df["monthly_salary"] <= salary_range[1])
                ]

            st.caption(f"Showing {len(view_df):,} of {len(df):,} records")
            st.dataframe(view_df, use_container_width=True, hide_index=True)

            # Quick stats
            if not view_df.empty and "predicted_eligibility" in view_df.columns:
                st.divider()
                st.markdown("**Quick Stats**")
                s1, s2, s3, s4 = st.columns(4)
                vc = view_df["predicted_eligibility"].value_counts()
                s1.metric("Eligible",     int(vc.get("Eligible", 0)))
                s2.metric("High Risk",    int(vc.get("High_Risk", 0)))
                s3.metric("Not Eligible", int(vc.get("Not_Eligible", 0)))
                s4.metric("Avg Salary",
                          f"₹{view_df['monthly_salary'].mean():,.0f}"
                          if "monthly_salary" in view_df.columns else "—")

    # ════════════════════════════════════════════════════════
    # ADD
    # ════════════════════════════════════════════════════════
    with tab_add:
        st.subheader("Add New Loan Application")

        with st.form("add_form", clear_on_submit=True):
            st.markdown("**Personal**")
            a1, a2, a3 = st.columns(3)
            with a1: age            = st.number_input("Age", 18, 70, 30)
            with a2: gender         = st.selectbox("Gender", ["Male", "Female"])
            with a3: marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

            a4, a5 = st.columns(2)
            with a4: education  = st.selectbox("Education", ["Graduate", "Post_Graduate", "Under_Graduate", "PhD"])
            with a5: house_type = st.selectbox("House Type", ["Owned", "Rented", "Parental"])

            st.markdown("**Employment**")
            b1, b2, b3 = st.columns(3)
            with b1: employment_type      = st.selectbox("Employment Type", ["Salaried", "Self_Employed", "Freelancer"])
            with b2: company_type         = st.selectbox("Company Type", ["Private", "Government", "MNC", "Startup"])
            with b3: years_of_employment  = st.number_input("Years Employed", 0.0, 40.0, 3.0, 0.5)

            st.markdown("**Financials**")
            c1, c2 = st.columns(2)
            with c1: monthly_salary          = st.number_input("Monthly Salary (₹)", 5000, 10_000_000, 50_000, 1000)
            with c2: bank_balance             = st.number_input("Bank Balance (₹)", 0, 50_000_000, 100_000, 5000)

            c3, c4 = st.columns(2)
            with c3: credit_score             = st.number_input("Credit Score", 300, 900, 700)
            with c4: total_monthly_expenses   = st.number_input("Monthly Expenses (₹)", 0, 5_000_000, 20_000, 500)

            c5, c6 = st.columns(2)
            with c5: existing_loans           = st.selectbox("Existing Loans?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with c6: current_emi_amount       = st.number_input("Current EMI (₹)", 0, 500_000, 0, 500)

            st.markdown("**Loan Request**")
            d1, d2 = st.columns(2)
            with d1: requested_amount  = st.number_input("Loan Amount (₹)", 10_000, 100_000_000, 500_000, 10_000)
            with d2: requested_tenure  = st.number_input("Tenure (months)", 6, 360, 60, 6)

            st.markdown("**Prediction (optional)**")
            e1, e2 = st.columns(2)
            with e1: pred_elig    = st.selectbox("Predicted Eligibility", ["—", "Eligible", "High_Risk", "Not_Eligible"])
            with e2: pred_max_emi = st.number_input("Predicted Max EMI (₹)", 0, 10_000_000, 0, 500)

            notes = st.text_area("Notes", placeholder="Any remarks…", height=80)

            if st.form_submit_button("✅ Save Application", use_container_width=True):
                new_row = {
                    "id": _next_id(df),
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "age": age, "gender": gender, "marital_status": marital_status,
                    "education": education, "monthly_salary": monthly_salary,
                    "employment_type": employment_type, "years_of_employment": years_of_employment,
                    "company_type": company_type, "house_type": house_type,
                    "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
                    "credit_score": credit_score, "bank_balance": bank_balance,
                    "requested_amount": requested_amount, "requested_tenure": requested_tenure,
                    "total_monthly_expenses": total_monthly_expenses,
                    "predicted_eligibility": pred_elig if pred_elig != "—" else "",
                    "predicted_max_emi": pred_max_emi if pred_max_emi > 0 else "",
                    "notes": notes,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                _save(df)
                st.success(f"✅ Application #{new_row['id']} saved successfully!")
                st.rerun()

    # ════════════════════════════════════════════════════════
    # EDIT
    # ════════════════════════════════════════════════════════
    with tab_edit:
        st.subheader("Edit an Application")

        if df.empty:
            st.info("No applications to edit.")
        else:
            edit_id = st.selectbox("Select Application ID to edit:", df["id"].tolist())
            row = df[df["id"] == edit_id].iloc[0]

            with st.form("edit_form"):
                st.markdown(f"**Editing Application #{edit_id}** — submitted {row.get('submitted_at', '—')}")

                notes_edit = st.text_area("Notes", value=str(row.get("notes", "")), height=100)
                pred_elig_edit = st.selectbox(
                    "Predicted Eligibility",
                    ["—", "Eligible", "High_Risk", "Not_Eligible"],
                    index=["—", "Eligible", "High_Risk", "Not_Eligible"].index(
                        row.get("predicted_eligibility", "—")
                        if row.get("predicted_eligibility") in ["Eligible", "High_Risk", "Not_Eligible"]
                        else "—"
                    ),
                )
                pred_emi_edit = st.number_input(
                    "Predicted Max EMI (₹)", 0, 10_000_000,
                    int(float(row["predicted_max_emi"])) if str(row.get("predicted_max_emi", "")).strip() not in ["", "nan"] else 0,
                    500,
                )
                monthly_salary_edit = st.number_input(
                    "Monthly Salary (₹)", 5000, 10_000_000,
                    int(row.get("monthly_salary", 50_000)), 1000,
                )
                credit_score_edit = st.number_input(
                    "Credit Score", 300, 900, int(row.get("credit_score", 700))
                )

                if st.form_submit_button("💾 Save Changes", use_container_width=True):
                    df.loc[df["id"] == edit_id, "notes"]                = notes_edit
                    df.loc[df["id"] == edit_id, "predicted_eligibility"] = pred_elig_edit if pred_elig_edit != "—" else ""
                    df.loc[df["id"] == edit_id, "predicted_max_emi"]    = pred_emi_edit if pred_emi_edit > 0 else ""
                    df.loc[df["id"] == edit_id, "monthly_salary"]       = monthly_salary_edit
                    df.loc[df["id"] == edit_id, "credit_score"]         = credit_score_edit
                    _save(df)
                    st.success(f"✅ Application #{edit_id} updated.")
                    st.rerun()

    # ════════════════════════════════════════════════════════
    # DELETE
    # ════════════════════════════════════════════════════════
    with tab_delete:
        st.subheader("Delete an Application")

        if df.empty:
            st.info("No applications to delete.")
        else:
            del_id = st.selectbox("Select Application ID to delete:", df["id"].tolist(), key="del_id")
            row_del = df[df["id"] == del_id].iloc[0]

            st.warning(
                f"You are about to delete **Application #{del_id}** "
                f"(submitted {row_del.get('submitted_at', '—')}, "
                f"{row_del.get('gender', '')} age {row_del.get('age', '')}, "
                f"salary ₹{int(float(row_del.get('monthly_salary', 0))):,})."
            )

            confirm = st.checkbox("Yes, I want to permanently delete this record.")
            if st.button("🗑️ Delete", disabled=not confirm, use_container_width=True):
                df = df[df["id"] != del_id].reset_index(drop=True)
                _save(df)
                st.success(f"✅ Application #{del_id} deleted.")
                st.rerun()

    # ════════════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════════════
    with tab_export:
        st.subheader("Export Applications")

        if df.empty:
            st.info("No applications to export.")
        else:
            st.metric("Total Records", len(df))
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download as CSV",
                data=csv_bytes,
                file_name="loan_applications_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()
            st.markdown("**Preview (first 20 rows)**")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
