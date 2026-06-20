"""
utils/predict_utils.py
======================
Shared prediction utilities for EMIPredict AI v3.

Handles:
  • Loading model artefacts (classifier, regressor, scaler, encoders,
    feature_cols, log_transformed_cols)
  • Building feature vectors from raw user input
    (FE → log transform → encode → scale)
  • Running dual predictions (classification + regression)
  • Computing key financial ratios for display
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# ── Label mappings ────────────────────────────
CLF_LABEL_MAP = {"Eligible": 0, "High_Risk": 1, "Not_Eligible": 2}
CLF_LABEL_INV = {v: k for k, v in CLF_LABEL_MAP.items()}

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

MODELS_DIR = "models"


# ══════════════════════════════════════════════
# 1. LOAD ARTEFACTS
# ══════════════════════════════════════════════
def load_artifacts(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """
    Load all saved model artefacts.
    Raises FileNotFoundError if any required artefact is missing.

    Returns dict with keys:
        clf, reg, scaler, encoders, feat_cols, log_cols
    """
    required = {
        "clf"      : "best_classifier.pkl",
        "reg"      : "best_regressor.pkl",
        "scaler"   : "scaler.pkl",
        "encoders" : "encoders.pkl",
        "feat_cols": "feature_cols.pkl",
    }

    missing = [
        fname for fname in required.values()
        if not os.path.exists(os.path.join(models_dir, fname))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefacts: {missing}\n"
            f"Run the training pipeline:\n"
            f"  python step1_data_preprocessing.py\n"
            f"  python step3_feature_engineering.py\n"
            f"  python step4_train_models.py"
        )

    artifacts = {
        key: joblib.load(os.path.join(models_dir, fname))
        for key, fname in required.items()
    }

    # log_transformed_cols is optional (v3+ artefact)
    log_path = os.path.join(models_dir, "log_transformed_cols.pkl")
    artifacts["log_cols"] = (
        joblib.load(log_path) if os.path.exists(log_path) else []
    )

    return artifacts


def models_ready(models_dir: str = MODELS_DIR) -> bool:
    """Return True if all required artefacts exist on disk."""
    fnames = ["best_classifier.pkl", "best_regressor.pkl",
              "scaler.pkl", "encoders.pkl", "feature_cols.pkl"]
    return all(os.path.exists(os.path.join(models_dir, f)) for f in fnames)


# ══════════════════════════════════════════════
# 2. BUILD FEATURE VECTOR
# ══════════════════════════════════════════════
def build_feature_vector(raw: Dict[str, Any],
                          artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Convert raw user input dict → scaled feature vector ready for prediction.

    raw must contain these 16 keys:
        age, gender, marital_status, education,
        monthly_salary, employment_type, years_of_employment, company_type,
        house_type, existing_loans (int 0/1), current_emi_amount,
        credit_score, bank_balance,
        requested_amount, requested_tenure,
        total_monthly_expenses

    Pipeline applied:
        1. Feature engineering (financial ratios, risk feats, interactions)
        2. Log1p transform on the same columns identified during training
        3. Label-encode categoricals with saved LabelEncoders
        4. Select features in training order
        5. StandardScaler transform
    """
    from step3_feature_engineering import (
        add_financial_ratios,
        add_risk_features,
        add_interaction_features,
    )

    df = pd.DataFrame([raw])

    # ── Step 1: Feature engineering ──────────
    add_financial_ratios(df)
    add_risk_features(df)
    add_interaction_features(df)

    # ── Step 2: Log1p transform ───────────────
    log_cols = artifacts.get("log_cols", [])
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── Step 3: Encode categoricals ──────────
    for col, le in artifacts["encoders"].items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            val = val if val in le.classes_ else le.classes_[0]
            df[col] = le.transform([val])[0]

    # ── Step 4: Select features ───────────────
    feat_cols = artifacts["feat_cols"]
    avail     = [c for c in feat_cols if c in df.columns]
    X         = df[avail].astype(float).fillna(0).values

    # ── Step 5: Scale ────────────────────────
    X_sc = artifacts["scaler"].transform(X)
    return X_sc


# ══════════════════════════════════════════════
# 3. PREDICT
# ══════════════════════════════════════════════
def predict_emi(raw: Dict[str, Any],
                artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run both classification (eligibility) and regression (max EMI).

    Returns:
        {
          "eligibility"     : str,   # "Eligible" | "High_Risk" | "Not_Eligible"
          "class_id"        : int,   # 0 / 1 / 2
          "probabilities"   : list,  # [P(Eligible), P(High_Risk), P(Not_Eligible)]
          "max_monthly_emi" : float, # predicted max safe EMI in ₹
          "color"           : str,   # hex color for the eligibility class
          "icon"            : str,   # emoji icon
        }
    """
    X = build_feature_vector(raw, artifacts)

    clf = artifacts["clf"]
    reg = artifacts["reg"]

    class_id    = int(clf.predict(X)[0])
    proba       = (clf.predict_proba(X)[0].tolist()
                   if hasattr(clf, "predict_proba") else [0.0, 0.0, 0.0])
    max_emi     = float(max(reg.predict(X)[0], 0))
    eligibility = CLF_LABEL_INV[class_id]

    return {
        "eligibility"    : eligibility,
        "class_id"       : class_id,
        "probabilities"  : proba,
        "max_monthly_emi": max_emi,
        "color"          : CLF_COLORS[eligibility],
        "icon"           : CLF_ICONS[eligibility],
    }


# ══════════════════════════════════════════════
# 4. FINANCIAL RATIO HELPERS
# ══════════════════════════════════════════════
def compute_ratios(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute key financial display metrics from raw user input.
    Does NOT need the ML pipeline — uses simple arithmetic.
    """
    salary   = float(raw.get("monthly_salary", 1)) or 1.0
    tot_exp  = float(raw.get("total_monthly_expenses", 0))
    curr_emi = float(raw.get("current_emi_amount", 0))
    bank_bal = float(raw.get("bank_balance", 0))
    req_amt  = float(raw.get("requested_amount", 0))
    req_ten  = float(raw.get("requested_tenure", 1)) or 1.0

    total_all  = tot_exp + curr_emi
    disposable = salary - total_all
    exp_ratio  = total_all / salary
    emi_burden = curr_emi / salary
    loan_to_bal = req_amt / max(bank_bal, 1)
    est_new_emi = req_amt / req_ten * 1.08  # rough 8% annual interest proxy

    return {
        "total_all_expenses"   : total_all,
        "disposable_income"    : disposable,
        "expense_to_income_pct": exp_ratio * 100,
        "emi_burden_pct"       : emi_burden * 100,
        "loan_to_balance"      : loan_to_bal,
        "estimated_new_emi"    : est_new_emi,
        "savings_proxy"        : max(disposable, 0),
    }


def risk_summary(ratios: Dict[str, float]) -> Tuple[str, str]:
    """
    Return a (risk_level, advice) tuple based on computed ratios.
    Used for instant feedback before running the ML model.
    """
    exp_pct = ratios["expense_to_income_pct"]
    disp    = ratios["disposable_income"]

    if exp_pct < 40 and disp > 10000:
        return "Low Risk 🟢", (
            "Your expense ratio is healthy. "
            "Good chance of loan approval."
        )
    elif exp_pct < 60:
        return "Moderate Risk 🟡", (
            "Expense ratio is manageable but tight. "
            "Consider reducing loan amount or extending tenure."
        )
    else:
        return "High Risk 🔴", (
            "Expenses consume a large portion of income. "
            "Loan approval is unlikely without reducing obligations."
        )
