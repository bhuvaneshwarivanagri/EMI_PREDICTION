"""
=============================================================
step2_eda.py
EMIPredict AI — Step 2: Exploratory Data Analysis (v2)
=============================================================
Adapted for reduced feature set with total_monthly_expenses.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("eda_plots", exist_ok=True)

CLEAN_DATA_PATH = "data/emi_cleaned.csv"
PALETTE = {
    "Eligible"    : "#2ecc71",
    "High_Risk"   : "#f39c12",
    "Not_Eligible": "#e74c3c",
}

NUMERIC_FEATURES = [
    "age", "monthly_salary", "years_of_employment",
    "current_emi_amount", "credit_score", "bank_balance",
    "requested_amount", "requested_tenure",
    "total_monthly_expenses", "max_monthly_emi",
]

CAT_FEATURES = [
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
]


def load(path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows for EDA | Columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
def basic_stats(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BASIC STATS")
    print("=" * 60)
    avail = [c for c in NUMERIC_FEATURES if c in df.columns]
    print(df[avail].describe().round(2).to_string())

    print("\n📊 emi_eligibility distribution:")
    vc = df["emi_eligibility"].value_counts()
    for cls, cnt in vc.items():
        print(f"  {cls:<15}: {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print("\n📊 max_monthly_emi (regression target):")
    print(df["max_monthly_emi"].describe().round(2).to_string())


# ──────────────────────────────────────────────
def plot_eligibility_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pie
    vc = df["emi_eligibility"].value_counts()
    axes[0].pie(vc, labels=vc.index, autopct="%1.1f%%",
                colors=[PALETTE.get(k, "#999") for k in vc.index],
                startangle=90)
    axes[0].set_title("Overall EMI Eligibility")

    # Credit score vs class
    for cls, color in PALETTE.items():
        sub = df[df["emi_eligibility"] == cls]["credit_score"]
        axes[1].hist(sub, bins=40, alpha=0.6, label=cls, color=color)
    axes[1].set_title("Credit Score by Eligibility")
    axes[1].set_xlabel("Credit Score")
    axes[1].legend()

    # Max monthly EMI distribution by class
    for cls, color in PALETTE.items():
        sub = df[df["emi_eligibility"] == cls]["max_monthly_emi"]
        axes[2].hist(sub, bins=40, alpha=0.6, label=cls, color=color)
    axes[2].set_title("Max Monthly EMI by Class")
    axes[2].set_xlabel("Max EMI (₹)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("eda_plots/01_eligibility_distribution.png", dpi=150)
    plt.close()
    print("📊 Saved → eda_plots/01_eligibility_distribution.png")


# ──────────────────────────────────────────────
def plot_financial_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    pairs = [
        ("monthly_salary",         axes[0, 0]),
        ("total_monthly_expenses", axes[0, 1]),
        ("bank_balance",           axes[0, 2]),
        ("current_emi_amount",     axes[1, 0]),
        ("requested_amount",       axes[1, 1]),
        ("requested_tenure",       axes[1, 2]),
    ]
    for col, ax in pairs:
        if col not in df.columns:
            continue
        df.boxplot(column=col, by="emi_eligibility", ax=ax, patch_artist=True)
        ax.set_title(f"{col.replace('_',' ').title()} by Eligibility")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.suptitle("Financial Variables vs EMI Eligibility", fontsize=14)
    plt.tight_layout()
    plt.savefig("eda_plots/02_financial_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved → eda_plots/02_financial_analysis.png")


# ──────────────────────────────────────────────
def plot_expense_analysis(df: pd.DataFrame):
    """Deep-dive on the new total_monthly_expenses column."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Expense-to-income ratio distribution
    if "monthly_salary" in df.columns:
        df["_eti"] = df["total_monthly_expenses"] / (df["monthly_salary"] + 1)
        for cls, color in PALETTE.items():
            sub = df[df["emi_eligibility"] == cls]["_eti"].clip(0, 3)
            axes[0].hist(sub, bins=40, alpha=0.6, label=cls, color=color)
        axes[0].set_title("Expense-to-Income Ratio by Eligibility")
        axes[0].set_xlabel("Total Expenses / Salary")
        axes[0].legend()
        df.drop(columns=["_eti"], inplace=True)

    # Salary vs total expenses scatter
    sample = df.sample(min(5000, len(df)), random_state=42)
    for cls, color in PALETTE.items():
        sub = sample[sample["emi_eligibility"] == cls]
        axes[1].scatter(sub["monthly_salary"], sub["total_monthly_expenses"],
                        alpha=0.4, s=8, color=color, label=cls)
    axes[1].set_title("Monthly Salary vs Total Expenses")
    axes[1].set_xlabel("Monthly Salary (₹)")
    axes[1].set_ylabel("Total Monthly Expenses (₹)")
    axes[1].legend()

    # Disposable income
    df["_disp"] = df["monthly_salary"] - df["total_monthly_expenses"]
    for cls, color in PALETTE.items():
        sub = df[df["emi_eligibility"] == cls]["_disp"].clip(-50000, 150000)
        axes[2].hist(sub, bins=40, alpha=0.6, label=cls, color=color)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1, label="Break-even")
    axes[2].set_title("Disposable Income by Eligibility")
    axes[2].set_xlabel("Salary − Total Expenses (₹)")
    axes[2].legend()
    df.drop(columns=["_disp"], inplace=True)

    plt.tight_layout()
    plt.savefig("eda_plots/03_expense_analysis.png", dpi=150)
    plt.close()
    print("📊 Saved → eda_plots/03_expense_analysis.png")


# ──────────────────────────────────────────────
def plot_demographic_analysis(df: pd.DataFrame):
    avail_cats = [c for c in CAT_FEATURES if c in df.columns]
    n = len(avail_cats)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(avail_cats):
        ct = (pd.crosstab(df[col], df["emi_eligibility"], normalize="index") * 100)
        for cls in ["Eligible", "High_Risk", "Not_Eligible"]:
            if cls in ct.columns:
                ct[[c for c in ["Eligible", "High_Risk", "Not_Eligible"]
                    if c in ct.columns]].plot(
                    kind="bar", ax=axes[i],
                    color=[PALETTE[c] for c in ["Eligible", "High_Risk", "Not_Eligible"]
                           if c in ct.columns]
                )
                break
        axes[i].set_title(f"Eligibility % by {col.replace('_',' ').title()}")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right", fontsize=8)
        axes[i].set_ylabel("%")
        axes[i].legend(fontsize=7)

    for j in range(len(avail_cats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Demographic Patterns vs Eligibility", fontsize=14)
    plt.tight_layout()
    plt.savefig("eda_plots/04_demographic_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved → eda_plots/04_demographic_analysis.png")


# ──────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[num_cols].corr()

    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                annot_kws={"size": 8})
    plt.title("Feature Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig("eda_plots/05_correlation_heatmap.png", dpi=150)
    plt.close()
    print("📊 Saved → eda_plots/05_correlation_heatmap.png")

    print("\n📈 Top correlations with max_monthly_emi:")
    print(corr["max_monthly_emi"].sort_values(ascending=False).head(10).to_string())


# ──────────────────────────────────────────────
def business_insights(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 KEY BUSINESS INSIGHTS (v2 column set)")
    print("=" * 60)
    total = len(df)
    for cls, color in PALETTE.items():
        cnt = (df["emi_eligibility"] == cls).sum()
        avg_max_emi = df[df["emi_eligibility"] == cls]["max_monthly_emi"].mean()
        avg_salary  = df[df["emi_eligibility"] == cls]["monthly_salary"].mean()
        avg_expense = df[df["emi_eligibility"] == cls]["total_monthly_expenses"].mean()
        avg_credit  = df[df["emi_eligibility"] == cls]["credit_score"].mean()
        print(f"\n  {cls} ({cnt/total*100:.1f}%):")
        print(f"    Avg Max EMI          : ₹{avg_max_emi:>10,.0f}")
        print(f"    Avg Monthly Salary   : ₹{avg_salary:>10,.0f}")
        print(f"    Avg Monthly Expenses : ₹{avg_expense:>10,.0f}")
        print(f"    Avg Credit Score     : {avg_credit:>8.0f}")

    # Existing loans impact
    print("\n  Existing Loans → Not_Eligible rate:")
    loan_rej = df.groupby("existing_loans")["emi_eligibility"].apply(
        lambda x: (x == "Not_Eligible").mean() * 100
    ).round(1)
    print(loan_rej.to_string())


# ──────────────────────────────────────────────
def main():
    print("🏦 EMIPredict AI — Step 2: EDA (v2)\n")
    df = load()
    basic_stats(df)
    plot_eligibility_distribution(df)
    plot_financial_analysis(df)
    plot_expense_analysis(df)
    plot_demographic_analysis(df)
    plot_correlation(df)
    business_insights(df)
    print("\n✅ Step 2 complete! Plots in eda_plots/")


if __name__ == "__main__":
    main()
