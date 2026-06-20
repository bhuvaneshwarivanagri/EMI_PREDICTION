"""
=============================================================
step3_feature_engineering.py
EMIPredict AI — Step 3: Feature Engineering (v3)
=============================================================
Pipeline inside this step:
  [1] Financial ratios  (7 features)
  [2] Risk features     (8 features)
  [3] Interaction feats (4 features)
  [3b] Correlation check + redundancy removal (threshold = 0.90)
  [4]  Skewness check  + log1p transform
  [5]  Encode categoricals (LabelEncoder fit on train)
  [6]  Encode targets
  [7]  SMOTE on training set (class balance for classification)
  [8]  Scale (StandardScaler fit on original train)
  [9]  Save:
         data/train_fe.csv        ← scaled, original balance (for regression)
         data/train_fe_smote.csv  ← scaled, SMOTE-balanced  (for classification)
         data/val_fe.csv
         data/test_fe.csv
         models/scaler.pkl | encoders.pkl | feature_cols.pkl
         models/dropped_features.pkl | log_transformed_cols.pkl

Input features (16):
  age, gender, marital_status, education,
  monthly_salary, employment_type, years_of_employment, company_type,
  house_type, existing_loans, current_emi_amount,
  credit_score, bank_balance,
  requested_amount, requested_tenure,
  total_monthly_expenses

After FE → ~35 features before redundancy removal.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore")

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

TRAIN_PATH = "data/train.csv"
VAL_PATH   = "data/val.csv"
TEST_PATH  = "data/test.csv"

CLF_TARGET  = "emi_eligibility"
REG_TARGET  = "max_monthly_emi"

CLF_LABEL_MAP = {"Eligible": 0, "High_Risk": 1, "Not_Eligible": 2}
CLF_LABEL_INV = {v: k for k, v in CLF_LABEL_MAP.items()}

RANDOM_STATE         = 42
CORR_THRESHOLD       = 0.90    # drop feature if |corr| with another > this
SKEW_THRESHOLD       = 1.0     # apply log1p if |skewness| > this

# Categorical columns to label-encode
CAT_COLS = [
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
    "credit_score_band", "age_band",
]


# ═══════════════════════════════════════════════════════════
# [1] FINANCIAL RATIOS
# ═══════════════════════════════════════════════════════════
def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    sal = df["monthly_salary"]

    df["expense_to_income"]   = df["total_monthly_expenses"] / (sal + eps)
    df["disposable_income"]   = sal - df["total_monthly_expenses"]
    df["emi_burden_ratio"]    = df["current_emi_amount"] / (sal + eps)
    df["savings_proxy"]       = (sal - df["total_monthly_expenses"]).clip(lower=0)
    df["loan_to_balance"]     = df["requested_amount"] / (df["bank_balance"] + eps)
    df["affordability_ratio"] = df["disposable_income"] / (
        df["requested_amount"] / (df["requested_tenure"] + eps) + eps
    )
    df["net_worth_proxy"]     = df["bank_balance"] + df["savings_proxy"]

    # Cap extremes
    for col in ["expense_to_income", "emi_burden_ratio",
                "loan_to_balance", "affordability_ratio"]:
        df[col] = df[col].clip(-20, 20)

    return df


# ═══════════════════════════════════════════════════════════
# [2] RISK FEATURES
# ═══════════════════════════════════════════════════════════
def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df["credit_score_band"] = pd.cut(
        df["credit_score"],
        bins=[0, 500, 580, 670, 740, 800, 900],
        labels=["Very Poor", "Poor", "Fair", "Good", "Very Good", "Excellent"]
    ).astype(str)

    df["credit_risk_score"]  = ((df["credit_score"] - 300) / 600 * 100).clip(0, 100)

    emp_map = {"Government": 3, "Private": 2, "Self-employed": 1}
    df["emp_type_score"]       = df["employment_type"].map(emp_map).fillna(1)
    df["employment_stability"] = (
        df["emp_type_score"] * 0.5 +
        df["years_of_employment"].clip(0, 20) / 20 * 50
    )

    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 100],
        labels=["Young", "Mid", "Senior", "Retiring"]
    ).astype(str)

    df["high_expense_flag"]  = (df["expense_to_income"] > 0.6).astype(int)
    df["existing_loan_flag"] = df["existing_loans"].astype(int)
    df["negative_disp_flag"] = (df["disposable_income"] < 0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════
# [3] INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    df["salary_x_credit"]      = df["monthly_salary"] * df["credit_score"] / 1e6
    df["balance_x_disposable"] = df["bank_balance"] * df["disposable_income"].clip(0) / 1e8
    df["tenure_x_amount"]      = df["requested_tenure"] * df["requested_amount"] / 1e6
    df["emi_to_disposable"]    = (
        df["current_emi_amount"] / (df["disposable_income"].abs() + eps)
    ).clip(-20, 20)
    return df


# Public helper: apply all FE to a single DataFrame (used by predict page)
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_financial_ratios(df)
    df = add_risk_features(df)
    df = add_interaction_features(df)
    return df


# ═══════════════════════════════════════════════════════════
# [3b] CORRELATION CHECK + REDUNDANCY REMOVAL
# ═══════════════════════════════════════════════════════════
def check_and_remove_redundant(
    train_df: pd.DataFrame,
    feat_cols: List[str],
    y_train: np.ndarray,
    threshold: float = CORR_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Identify pairs of features with |Pearson correlation| > threshold.
    For each redundant pair, drop the feature that has lower absolute
    correlation with the target (y_train).

    Note: correlation is computed on numeric columns only (categorical
    string columns are skipped at this stage — they are encoded in [5]).

    Returns:
        retained_cols : list of feature column names to keep
        dropped_cols  : list of column names that were removed
    """
    print(f"\n[3b] Correlation check (threshold = {threshold})...")

    # Only numeric columns can be correlated — categoricals are still strings here
    numeric_feat_cols = [
        c for c in feat_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]
    skipped_cats = [c for c in feat_cols if c not in numeric_feat_cols]
    if skipped_cats:
        print(f"  Skipping {len(skipped_cats)} categorical cols (encoded in step 5): "
              f"{skipped_cats}")

    X    = train_df[numeric_feat_cols].astype(float)
    corr = X.corr().abs()

    # Upper triangle only (avoid double-counting)
    upper = corr.where(
        np.triu(np.ones(corr.shape, dtype=bool), k=1)
    )

    to_drop = set()
    pairs_found = []

    for col in upper.columns:
        high_corr_peers = upper[col][upper[col] > threshold].index.tolist()
        for peer in high_corr_peers:
            if col in to_drop or peer in to_drop:
                continue
            # Keep the feature with higher |corr| to target
            try:
                col_target_corr  = abs(np.corrcoef(X[col].values, y_train)[0, 1])
                peer_target_corr = abs(np.corrcoef(X[peer].values, y_train)[0, 1])
            except Exception:
                col_target_corr = peer_target_corr = 0.0

            victim = peer if col_target_corr >= peer_target_corr else col
            to_drop.add(victim)
            pairs_found.append((col, peer, corr.loc[col, peer], victim))

    # Retain all feat_cols except dropped numeric ones
    # (categorical string cols are untouched)
    retained = [c for c in feat_cols if c not in to_drop]

    if pairs_found:
        print(f"  {'Feature A':<30} {'Feature B':<30} {'|Corr|':>7}  {'Dropped'}")
        print(f"  {'-'*30} {'-'*30} {'-'*7}  {'-'*25}")
        for a, b, r, v in pairs_found:
            print(f"  {a:<30} {b:<30} {r:>7.4f}  {v}")
        print(f"\n  Removed {len(to_drop)} redundant feature(s): {sorted(to_drop)}")
        print(f"  Retained {len(retained)} feature(s)")
    else:
        print(f"  ✅ No feature pairs exceed threshold — all {len(retained)} retained")

    return retained, sorted(list(to_drop))


# ═══════════════════════════════════════════════════════════
# [4] SKEWNESS CHECK + LOG1P TRANSFORM
# ═══════════════════════════════════════════════════════════
def check_and_fix_skewness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
    threshold: float = SKEW_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute skewness of each numeric feature on training data.
    Apply np.log1p to features with |skewness| > threshold
    that also have min >= 0 (log1p requires non-negative inputs).

    Returns updated DataFrames and the list of log-transformed columns.
    """
    print(f"\n[4] Skewness check (threshold = |{threshold}|)...")

    numeric_feats = [c for c in feat_cols
                     if pd.api.types.is_numeric_dtype(train_df[c])]

    skew_report = []
    log_cols    = []

    for col in numeric_feats:
        sk = stats.skew(train_df[col].dropna())
        skew_report.append((col, sk))

        if abs(sk) > threshold and train_df[col].min() >= 0:
            for df_ in [train_df, val_df, test_df]:
                df_[col] = np.log1p(df_[col])
            log_cols.append(col)

    # Sort for display
    skew_report.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  {'Feature':<35} {'Skewness':>10}  {'Action'}")
    print(f"  {'-'*35} {'-'*10}  {'-'*20}")
    for col, sk in skew_report:
        action = "→ log1p applied" if col in log_cols else (
            "→ skewed but has negatives (skip)" if abs(sk) > threshold else "—"
        )
        print(f"  {col:<35} {sk:>10.4f}  {action}")

    print(f"\n  log1p applied to {len(log_cols)} feature(s): {log_cols}")
    return train_df, val_df, test_df, log_cols


# ═══════════════════════════════════════════════════════════
# [5] ENCODE CATEGORICALS
# ═══════════════════════════════════════════════════════════
def encode_categorical(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:

    print("\n[5] Encoding categorical columns...")
    encoders: Dict[str, LabelEncoder] = {}

    for col in CAT_COLS:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        val[col]   = le.transform(
            val[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
        )
        test[col]  = le.transform(
            test[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
        )
        encoders[col] = le

    print(f"  ✅ Label-encoded {len(encoders)} columns: {list(encoders.keys())}")
    return train, val, test, encoders


# ═══════════════════════════════════════════════════════════
# [6] ENCODE TARGETS
# ═══════════════════════════════════════════════════════════
def encode_targets(train, val, test):
    for df in [train, val, test]:
        df["emi_eligibility_enc"] = df[CLF_TARGET].map(CLF_LABEL_MAP)
    print("\n[6] Target encoded: Eligible=0, High_Risk=1, Not_Eligible=2")
    return train, val, test


# ═══════════════════════════════════════════════════════════
# [7] SMOTE — class balancing on training set
# ═══════════════════════════════════════════════════════════
def apply_smote(
    X_train: pd.DataFrame,
    y_clf_train: np.ndarray,
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the encoded (but pre-scale) training features
    for classification target only.

    Returns:
        X_sm       : np.ndarray  — balanced feature matrix
        y_clf_sm   : np.ndarray  — balanced class labels
    """
    print("\n[7] SMOTE — balancing training classes...")

    before = pd.Series(y_clf_train).value_counts().sort_index()
    print("  Before SMOTE:")
    label_names = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
    for cls_id, cnt in before.items():
        print(f"    {label_names.get(cls_id, cls_id):<15}: {cnt:,}")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_sm, y_clf_sm = smote.fit_resample(
        X_train[feat_cols].astype(float).values,
        y_clf_train,
    )

    after = pd.Series(y_clf_sm).value_counts().sort_index()
    print("  After SMOTE:")
    for cls_id, cnt in after.items():
        print(f"    {label_names.get(cls_id, cls_id):<15}: {cnt:,}")
    print(f"  Synthetic samples added: {len(X_sm) - len(X_train):,}")
    print(f"  Total train size (SMOTE): {len(X_sm):,}")

    return X_sm, y_clf_sm


# ═══════════════════════════════════════════════════════════
# [8] SCALE
# ═══════════════════════════════════════════════════════════
def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: List[str],
    X_sm: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on original (non-SMOTE) training features.
    Transform:
      • original train → train_sc  (used for regression)
      • val, test      → val_sc, test_sc
      • X_sm (SMOTE)   → X_sm_sc   (used for classification)
    """
    print(f"\n[8] Scaling features ({len(feat_cols)} features)...")

    scaler = StandardScaler()

    X_tr_raw = train[feat_cols].astype(float)
    scaler.fit(X_tr_raw)

    train_sc = pd.DataFrame(scaler.transform(X_tr_raw),  columns=feat_cols)
    val_sc   = pd.DataFrame(
        scaler.transform(val[feat_cols].astype(float)),   columns=feat_cols
    )
    test_sc  = pd.DataFrame(
        scaler.transform(test[feat_cols].astype(float)),  columns=feat_cols
    )
    X_sm_sc  = scaler.transform(X_sm)

    print(f"  ✅ StandardScaler fit on {len(X_tr_raw):,} original training rows")
    return train_sc, val_sc, test_sc, X_sm_sc, scaler


# ═══════════════════════════════════════════════════════════
# BASE FEATURE COLS (before redundancy removal)
# ═══════════════════════════════════════════════════════════
ALL_FEATURE_COLS = [
    # Original numeric
    "age", "monthly_salary", "years_of_employment",
    "current_emi_amount", "credit_score", "bank_balance",
    "requested_amount", "requested_tenure",
    "total_monthly_expenses", "existing_loans",
    # Encoded categoricals
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
    "credit_score_band", "age_band",
    # Financial ratios
    "expense_to_income", "disposable_income", "emi_burden_ratio",
    "savings_proxy", "loan_to_balance", "affordability_ratio",
    "net_worth_proxy",
    # Risk features
    "credit_risk_score", "employment_stability", "emp_type_score",
    "high_expense_flag", "existing_loan_flag", "negative_disp_flag",
    # Interaction
    "salary_x_credit", "balance_x_disposable",
    "tenure_x_amount", "emi_to_disposable",
]


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("🏦 EMIPredict AI — Step 3: Feature Engineering (v3)")
    print("=" * 65)
    print("Pipeline: FE → Correlation/Redundancy → Skewness/LogTransform →")
    print("          Encode → Targets → SMOTE → Scale → Save\n")

    # ── Load splits ──────────────────────────────
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    val   = pd.read_csv(VAL_PATH,   low_memory=False)
    test  = pd.read_csv(TEST_PATH,  low_memory=False)
    print(f"Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")

    # ── [1–3] Apply FE to all splits ─────────────
    print("\n[1] Financial ratios...")
    print("[2] Risk features...")
    print("[3] Interaction features...")
    for df in [train, val, test]:
        add_financial_ratios(df)
        add_risk_features(df)
        add_interaction_features(df)
    print("  ✅ FE applied to all splits")

    # Select only columns that actually exist
    feat_cols = [c for c in ALL_FEATURE_COLS if c in train.columns]
    print(f"\n  Features after FE: {len(feat_cols)}")

    # ── [3b] Correlation + redundancy removal ────
    y_clf_train_raw = train[CLF_TARGET].map(CLF_LABEL_MAP).values
    feat_cols, dropped_features = check_and_remove_redundant(
        train, feat_cols, y_clf_train_raw
    )

    # ── [4] Skewness check + log1p ───────────────
    train, val, test, log_cols = check_and_fix_skewness(
        train, val, test, feat_cols
    )

    # ── [5] Encode categoricals ──────────────────
    train, val, test, encoders = encode_categorical(train, val, test)

    # ── [6] Encode targets ───────────────────────
    train, val, test = encode_targets(train, val, test)

    y_clf_train = train["emi_eligibility_enc"].astype(int).values
    y_reg_train = train[REG_TARGET].astype(float).values
    y_clf_val   = val["emi_eligibility_enc"].astype(int).values
    y_reg_val   = val[REG_TARGET].astype(float).values
    y_clf_test  = test["emi_eligibility_enc"].astype(int).values
    y_reg_test  = test[REG_TARGET].astype(float).values

    # ── [7] SMOTE (classification only) ──────────
    X_sm, y_clf_sm = apply_smote(train, y_clf_train, feat_cols)

    # ── [8] Scale ─────────────────────────────────
    train_sc, val_sc, test_sc, X_sm_sc, scaler = scale_features(
        train, val, test, feat_cols, X_sm
    )

    # ── [9] Save ──────────────────────────────────
    print("\n[9] Saving all artefacts...")

    # Attach targets to scaled splits (original balance)
    train_sc[CLF_TARGET]            = train[CLF_TARGET].values
    train_sc["emi_eligibility_enc"] = y_clf_train
    train_sc[REG_TARGET]            = y_reg_train

    val_sc[CLF_TARGET]              = val[CLF_TARGET].values
    val_sc["emi_eligibility_enc"]   = y_clf_val
    val_sc[REG_TARGET]              = y_reg_val

    test_sc[CLF_TARGET]             = test[CLF_TARGET].values
    test_sc["emi_eligibility_enc"]  = y_clf_test
    test_sc[REG_TARGET]             = y_reg_test

    # SMOTE-balanced training data (classification only)
    smote_df = pd.DataFrame(X_sm_sc, columns=feat_cols)
    smote_df["emi_eligibility_enc"] = y_clf_sm

    # Save CSVs
    train_sc.to_csv("data/train_fe.csv",       index=False)
    val_sc.to_csv("data/val_fe.csv",           index=False)
    test_sc.to_csv("data/test_fe.csv",         index=False)
    smote_df.to_csv("data/train_fe_smote.csv", index=False)

    # Save model artefacts
    joblib.dump(scaler,           "models/scaler.pkl")
    joblib.dump(encoders,         "models/encoders.pkl")
    joblib.dump(feat_cols,        "models/feature_cols.pkl")
    joblib.dump(dropped_features, "models/dropped_features.pkl")
    joblib.dump(log_cols,         "models/log_transformed_cols.pkl")

    print(f"\n  💾 data/train_fe.csv          ({len(train_sc):,} rows)")
    print(f"  💾 data/train_fe_smote.csv    ({len(smote_df):,} rows — SMOTE balanced)")
    print(f"  💾 data/val_fe.csv            ({len(val_sc):,} rows)")
    print(f"  💾 data/test_fe.csv           ({len(test_sc):,} rows)")
    print(f"  💾 models/scaler.pkl | encoders.pkl | feature_cols.pkl")
    print(f"  💾 models/dropped_features.pkl | log_transformed_cols.pkl")
    print(f"\n  Final feature count : {len(feat_cols)}")
    print(f"  Dropped (redundant) : {dropped_features}")
    print(f"  Log-transformed     : {log_cols}")
    print(f"  Feature list        : {feat_cols}")

    print("\n✅ Step 3 complete! Run step4_train_models.py next.\n")
    return train_sc, val_sc, test_sc, scaler, encoders, feat_cols


if __name__ == "__main__":
    main()