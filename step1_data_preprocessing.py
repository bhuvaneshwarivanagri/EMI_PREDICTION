"""
=============================================================
step1_data_preprocessing.py
EMIPredict AI — Step 1: Data Loading & Preprocessing (v3)
=============================================================
Pipeline:
  1. Load
  2. Clean
       ├─ dtype correction
       ├─ null check (%)
       ├─ fill (median / mode)
       ├─ post-fill verify
       ├─ duplicate removal
       └─ outlier clip (IQR)
  3. Combine monthly expenses → total_monthly_expenses
  4. Drop unwanted columns
  5. Business-rule validation + data_quality_flag
  6. Split  70 / 15 / 15  (stratified on emi_eligibility)
  7. Save

MODIFIED COLUMN STRUCTURE:
  DROPPED  : emi_scenario, emergency_fund, family_size, dependents,
             school_fees, college_fees, travel_expenses,
             groceries_utilities, other_monthly_expenses, monthly_rent
  ADDED    : total_monthly_expenses (all monthly costs combined,
             including current_emi_amount)

Final input features (16):
  age, gender, marital_status, education,
  monthly_salary, employment_type, years_of_employment, company_type,
  house_type, existing_loans, current_emi_amount,
  credit_score, bank_balance,
  requested_amount, requested_tenure,
  total_monthly_expenses

Targets (2):
  emi_eligibility    (Classification: Eligible / High_Risk / Not_Eligible)
  max_monthly_emi    (Regression: max safe EMI in INR)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RAW_DATA_PATH   = "emi_prediction_dataset.csv"
CLEAN_DATA_PATH = "data/emi_cleaned.csv"
TRAIN_PATH      = "data/train.csv"
VAL_PATH        = "data/val.csv"
TEST_PATH       = "data/test.csv"
RANDOM_STATE    = 42

# Columns combined into total_monthly_expenses
EXPENSE_SUB_COLS = [
    "monthly_rent",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",      # included so total = all monthly cash out
]

# Columns dropped entirely (not combined into anything)
DROP_COLS = [
    "emi_scenario",
    "emergency_fund",
    "family_size",
    "dependents",
]

# Final feature set used for modelling
KEEP_FEATURES = [
    "age", "gender", "marital_status", "education",
    "monthly_salary", "employment_type", "years_of_employment", "company_type",
    "house_type", "existing_loans", "current_emi_amount",
    "credit_score", "bank_balance",
    "requested_amount", "requested_tenure",
    "total_monthly_expenses",
]

TARGET_CLF = "emi_eligibility"
TARGET_REG = "max_monthly_emi"

# Numeric columns expected in the dataset
NUMERIC_COLS = [
    "age", "monthly_salary", "years_of_employment",
    "current_emi_amount", "credit_score", "bank_balance",
    "requested_amount", "requested_tenure",
    "max_monthly_emi",
]

# IQR clip limits (upper tail only; lower is floored at 0 for monetary cols)
IQR_MULTIPLIER = 3.0   # generous — preserves legitimate high earners

os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 2A — DTYPE CORRECTION & QUALITY REPORT
# ═══════════════════════════════════════════════════════════
def dtype_report(df: pd.DataFrame, label: str = "raw") -> None:
    """Print column dtypes, null %, and unique counts."""
    print(f"\n{'='*65}")
    print(f"DATA QUALITY REPORT ({label})")
    print(f"{'='*65}")
    report = pd.DataFrame({
        "dtype"     : df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "null_pct"  : (df.isnull().sum() / len(df) * 100).round(2),
        "unique"    : df.nunique(),
        "sample"    : [df[c].dropna().iloc[0] if df[c].notna().any() else "—"
                       for c in df.columns],
    })
    print(report.to_string())
    print(f"\n  Duplicate rows : {df.duplicated().sum():,}")
    print(f"  Total cells    : {df.size:,}")
    print(f"  Missing cells  : {df.isnull().sum().sum():,}"
          f" ({df.isnull().sum().sum() / df.size * 100:.2f}%)")


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct mixed-type / string-encoded numeric columns.
    Handles patterns like '38' vs '38.0' in object columns.
    """
    print("\n[2A] Fixing dtypes...")

    def _coerce_int_str(series: pd.Series) -> pd.Series:
        """Strip trailing .0 from stringified floats before numeric cast."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
            errors="coerce",
        )

    # Age often stored as '38' or '38.0'
    df["age"]                = _coerce_int_str(df["age"]).round(0)

    # Pure numeric — cast directly
    for col in ["monthly_salary", "bank_balance", "credit_score",
                "current_emi_amount", "requested_amount",
                "max_monthly_emi", "years_of_employment",
                "requested_tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary flag: Yes/No → 1/0
    if df["existing_loans"].dtype == object:
        df["existing_loans"] = df["existing_loans"].map({"Yes": 1, "No": 0})
        print("  existing_loans : Yes/No → 1/0")

    # Ensure expense sub-columns are numeric (filled later)
    for col in EXPENSE_SUB_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Report any remaining object columns that should be numeric
    still_obj = df[NUMERIC_COLS].select_dtypes(include="object").columns.tolist()
    if still_obj:
        print(f"  ⚠️  Still object after cast: {still_obj}")
    else:
        print("  ✅ All numeric columns correctly typed")

    return df


# ═══════════════════════════════════════════════════════════
# STEP 2B — NULL CHECK (%)
# ═══════════════════════════════════════════════════════════
def null_check(df: pd.DataFrame) -> None:
    """Print columns with missing values and their percentages."""
    null_series = df.isnull().sum()
    null_cols   = null_series[null_series > 0]
    if null_cols.empty:
        print("\n[2B] Null check: ✅ No missing values found")
        return

    print("\n[2B] Null check — columns with missing values:")
    print(f"  {'Column':<35} {'Count':>8}  {'Pct':>7}")
    print(f"  {'-'*35} {'-'*8}  {'-'*7}")
    for col, cnt in null_cols.sort_values(ascending=False).items():
        pct = cnt / len(df) * 100
        tag = "  ← HIGH" if pct > 5 else ""
        print(f"  {col:<35} {cnt:>8,}  {pct:>6.2f}%{tag}")
    print(f"\n  Total null cells : {null_cols.sum():,}")


# ═══════════════════════════════════════════════════════════
# STEP 2C — FILL MISSING VALUES (median / mode)
# ═══════════════════════════════════════════════════════════
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute:
      Numeric   → median (robust to outliers)
      Categorical → mode  (most frequent)
    """
    print("\n[2C] Filling missing values...")
    filled_log = {}

    # Numeric → median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            filled_log[col] = f"median={med:.2f}  ({n_null:,} cells)"

    # Categorical → mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            filled_log[col] = f"mode='{mode_val}'  ({n_null:,} cells)"

    if filled_log:
        for col, info in filled_log.items():
            print(f"  {col:<35} → {info}")
    else:
        print("  No nulls to fill")

    return df


# ═══════════════════════════════════════════════════════════
# STEP 2D — POST-FILL VERIFICATION
# ═══════════════════════════════════════════════════════════
def post_fill_verify(df: pd.DataFrame) -> None:
    """Assert no nulls remain; warn if any do."""
    remaining = df.isnull().sum().sum()
    if remaining == 0:
        print("\n[2D] Post-fill verify : ✅ Zero nulls remaining")
    else:
        print(f"\n[2D] Post-fill verify : ⚠️  {remaining:,} nulls still present")
        print(df.isnull().sum()[df.isnull().sum() > 0].to_string())


# ═══════════════════════════════════════════════════════════
# STEP 2E — DUPLICATE REMOVAL
# ═══════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"\n[2E] Duplicates removed : {removed:,}  → {len(df):,} rows remain")
    else:
        print(f"\n[2E] Duplicates : ✅ None found  ({len(df):,} rows)")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 2F — OUTLIER CLIPPING (IQR-based)
# ═══════════════════════════════════════════════════════════
def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip numeric features at IQR_MULTIPLIER × IQR from Q1/Q3.
    Monetary columns are floored at 0.
    Skip target columns and binary flags.
    """
    print(f"\n[2F] Outlier clipping (IQR × {IQR_MULTIPLIER})...")

    skip_cols = {TARGET_CLF, TARGET_REG, "existing_loans", "data_quality_flag"}
    num_cols  = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in skip_cols]

    clip_log = []
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - IQR_MULTIPLIER * IQR
        hi  = Q3 + IQR_MULTIPLIER * IQR

        # Floor monetary / count columns at 0
        if col in {"monthly_salary", "bank_balance", "current_emi_amount",
                   "requested_amount", "max_monthly_emi",
                   "total_monthly_expenses", "years_of_employment",
                   "requested_tenure", "age"}:
            lo = max(lo, 0)

        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()

        if n_lo + n_hi > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
            clip_log.append((col, lo, hi, n_lo, n_hi))

    if clip_log:
        print(f"  {'Column':<35} {'Lower':>12} {'Upper':>12} {'Lo#':>7} {'Hi#':>7}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*7} {'-'*7}")
        for col, lo, hi, n_lo, n_hi in clip_log:
            print(f"  {col:<35} {lo:>12,.1f} {hi:>12,.1f} {n_lo:>7,} {n_hi:>7,}")
    else:
        print("  ✅ No extreme outliers found")

    return df


# ═══════════════════════════════════════════════════════════
# STEP 3 — COMBINE MONTHLY EXPENSES
# ═══════════════════════════════════════════════════════════
def combine_monthly_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total_monthly_expenses = sum of all monthly cost sub-columns.
    Sub-cols filled with 0 before summing (NaN → 0 already done in fill_missing).
    """
    print("\n[3] Creating total_monthly_expenses...")

    existing_sub = [c for c in EXPENSE_SUB_COLS if c in df.columns]
    df["total_monthly_expenses"] = df[existing_sub].sum(axis=1)

    print(f"  Combined {len(existing_sub)} sub-columns")
    print(f"  Range  : ₹{df['total_monthly_expenses'].min():,.0f} – "
          f"₹{df['total_monthly_expenses'].max():,.0f}")
    print(f"  Mean   : ₹{df['total_monthly_expenses'].mean():,.0f}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 4 — DROP UNWANTED COLUMNS
# ═══════════════════════════════════════════════════════════
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop:
      • emi_scenario, emergency_fund, family_size, dependents
      • All expense sub-columns except current_emi_amount
        (current_emi_amount is kept as its own feature)
    """
    sub_without_emi = [c for c in EXPENSE_SUB_COLS
                       if c != "current_emi_amount" and c in df.columns]
    cols_to_drop = list(set(DROP_COLS + sub_without_emi))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    print(f"\n[4] Dropping {len(cols_to_drop)} columns...")
    for c in sorted(cols_to_drop):
        print(f"  – {c}")
    df = df.drop(columns=cols_to_drop)
    print(f"  Remaining: {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 5 — BUSINESS-RULE VALIDATION
# ═══════════════════════════════════════════════════════════
def validate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Business-rule validation...")

    flags  = pd.Series(False, index=df.index)
    checks = {
        "age out of [18-75]"           : ~df["age"].between(18, 75),
        "credit_score out of [300-900]": ~df["credit_score"].between(300, 900),
        "monthly_salary <= 0"          : df["monthly_salary"] <= 0,
        "total_monthly_expenses < 0"   : df["total_monthly_expenses"] < 0,
        "requested_amount <= 0"        : df["requested_amount"] <= 0,
        "requested_tenure <= 0"        : df["requested_tenure"] <= 0,
    }

    for label, mask in checks.items():
        n = mask.sum()
        flags |= mask
        status = "✅" if n == 0 else "⚠️ "
        print(f"  {status} {label:<40} : {n:,} rows")

    df["data_quality_flag"] = flags.astype(int)
    print(f"\n  Total flagged rows : {flags.sum():,}  ({flags.mean()*100:.2f}%)")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 6 — TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ═══════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    print("\n[6] Splitting dataset (70/15/15) stratified on emi_eligibility...")

    train, temp = train_test_split(
        df, test_size=0.30,
        stratify=df[TARGET_CLF],
        random_state=RANDOM_STATE,
    )
    val, test = train_test_split(
        temp, test_size=0.50,
        stratify=temp[TARGET_CLF],
        random_state=RANDOM_STATE,
    )

    print(f"\n  {'Split':<8} {'Rows':>8}   {'Pct':>6}")
    print(f"  {'-'*8} {'-'*8}   {'-'*6}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pct = len(split) / len(df) * 100
        print(f"  {name:<8} {len(split):>8,}   {pct:>5.1f}%")

    print("\n  Train class distribution:")
    dist = train[TARGET_CLF].value_counts(normalize=True).map(lambda x: f"{x*100:.1f}%")
    for cls, pct in dist.items():
        print(f"    {cls:<15} : {pct}")

    return train, val, test


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("🏦 EMIPredict AI — Step 1: Data Preprocessing (v3)")
    print("=" * 65)
    print("Pipeline: Load → DType Fix → Null Check → Fill → Verify →")
    print("          Dedup → Outlier Clip → Combine Expenses →")
    print("          Drop Cols → Validate → Split → Save\n")

    # ── 1. Load ──────────────────────────────────
    df = load_data(RAW_DATA_PATH)

    # ── 2A. Dtype correction + quality report ────
    dtype_report(df, label="raw")
    df = fix_dtypes(df)

    # ── 2B. Null check ───────────────────────────
    null_check(df)

    # ── 2C. Fill missing ─────────────────────────
    df = fill_missing(df)

    # ── 2D. Post-fill verification ───────────────
    post_fill_verify(df)

    # ── 2E. Duplicate removal ────────────────────
    df = remove_duplicates(df)

    # ── 2F. Outlier clipping ─────────────────────
    df = clip_outliers(df)

    # ── 3. Combine monthly expenses ──────────────
    df = combine_monthly_expenses(df)

    # ── 4. Drop unwanted columns ─────────────────
    df = drop_columns(df)

    # ── 2F (post-drop). Clip on total_monthly_expenses ──
    # Re-clip after combining to handle any sum-level extremes
    if "total_monthly_expenses" in df.columns:
        Q1  = df["total_monthly_expenses"].quantile(0.25)
        Q3  = df["total_monthly_expenses"].quantile(0.75)
        hi  = Q3 + IQR_MULTIPLIER * (Q3 - Q1)
        df["total_monthly_expenses"] = df["total_monthly_expenses"].clip(lower=0, upper=hi)

    # ── 5. Business-rule validation ──────────────
    df = validate(df)

    # Final dtype report
    df["age"] = df["age"].astype(float)
    print("\n[Final] Column summary after preprocessing:")
    dtype_report(df, label="clean")

    # ── Save clean dataset ────────────────────────
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n💾 Clean data → {CLEAN_DATA_PATH}  ({len(df):,} rows, {df.shape[1]} cols)")
    print(f"   Columns: {list(df.columns)}")

    # ── 6. Split ──────────────────────────────────
    train, val, test = split_data(df)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH,     index=False)
    test.to_csv(TEST_PATH,   index=False)
    print(f"\n💾 Splits → data/train.csv | val.csv | test.csv")
    print("\n✅ Step 1 complete! Run step2_eda.py next.\n")

    return df, train, val, test


if __name__ == "__main__":
    main()
