"""
=============================================================
step4_train_models.py
EMIPredict AI — Step 4: ML Training + MLflow (v3)
=============================================================
Models:
  Classification (4):
    1. Logistic Regression          — baseline interpretable results
    2. Random Forest Classifier     — feature importance analysis
    3. XGBoost Classifier           — high-performance gradient boosting
    4. Decision Tree Classifier     — visual interpretability

  Regression (4):
    1. Linear Regression            — baseline performance
    2. Random Forest Regressor      — ensemble-based predictions
    3. XGBoost Regressor            — advanced gradient boosting
    4. Decision Tree Regressor      — rule-based predictions
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

os.makedirs("models",    exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

EXPERIMENT_CLF = "EMI_Classification"
EXPERIMENT_REG = "EMI_Regression"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

CLF_TARGET   = "emi_eligibility_enc"
REG_TARGET   = "max_monthly_emi"
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════
def load_splits() -> Tuple:
    """
    Classifiers use SMOTE-balanced training data.
    Regressors  use original-balance training data.
    Val and test are shared.
    """
    feat_cols = joblib.load("models/feature_cols.pkl")

    # ── SMOTE training (classification) ──────────
    smote_train = pd.read_csv("data/train_fe_smote.csv", low_memory=False)
    X_clf_train = smote_train[feat_cols].astype(float).values
    y_clf_train = smote_train[CLF_TARGET].astype(int).values

    # ── Original training (regression) ───────────
    orig_train  = pd.read_csv("data/train_fe.csv", low_memory=False)
    X_reg_train = orig_train[feat_cols].astype(float).values
    y_reg_train = orig_train[REG_TARGET].astype(float).values

    # ── Shared val / test ─────────────────────────
    val  = pd.read_csv("data/val_fe.csv",  low_memory=False)
    test = pd.read_csv("data/test_fe.csv", low_memory=False)

    X_val      = val[feat_cols].astype(float).values
    y_clf_val  = val[CLF_TARGET].astype(int).values
    y_reg_val  = val[REG_TARGET].astype(float).values

    X_test     = test[feat_cols].astype(float).values
    y_clf_test = test[CLF_TARGET].astype(int).values
    y_reg_test = test[REG_TARGET].astype(float).values

    print(f"✅ CLF  Train (SMOTE) : {X_clf_train.shape}")
    print(f"   REG  Train (orig)  : {X_reg_train.shape}")
    print(f"   Val                : {X_val.shape}")
    print(f"   Test               : {X_test.shape}")
    print(f"   Features ({len(feat_cols)}): {feat_cols}")

    unique, counts = np.unique(y_clf_train, return_counts=True)
    label_map = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
    print("\n  SMOTE train class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {label_map.get(cls, cls):<15}: {cnt:,}")

    return (X_clf_train, y_clf_train,
            X_reg_train, y_reg_train,
            X_val, y_clf_val, y_reg_val,
            X_test, y_clf_test, y_reg_test,
            feat_cols)


# ═══════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════
def clf_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    m = {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred,
                                           average="weighted", zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred,
                                        average="weighted", zero_division=0)),
        "f1_score" : float(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            ))
        except Exception:
            m["roc_auc"] = 0.0
    return m


def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "r2"  : float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(
            np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
        ),
    }


# ═══════════════════════════════════════════════════════════
# ARTIFACT HELPERS
# ═══════════════════════════════════════════════════════════
def save_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    labels = ["Eligible", "High_Risk", "Not_Eligible"]
    cm     = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()


def save_feature_importance(model, feat_cols: List[str],
                             model_name: str, save_path: str, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feat_cols).nlargest(top_n)
    plt.figure(figsize=(10, 6))
    imp.sort_values().plot(kind="barh", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()


# ═══════════════════════════════════════════════════════════
# TRAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════
def train_classifier(name, model, params,
                     X_tr, y_tr,
                     X_va, y_va,
                     X_te, y_te,
                     feat_cols):
    mlflow.set_experiment(EXPERIMENT_CLF)
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        print(f"\n  📌 {name}  ({run_id[:8]}...)")
        mlflow.log_params({**params, "model_name": name,
                           "n_features": len(feat_cols),
                           "smote_balanced": True})

        # ── Train on full training set ────────────────────
        model.fit(X_tr, y_tr)

        y_pv  = model.predict(X_va);  y_pt = model.predict(X_te)
        y_ppv = (model.predict_proba(X_va)
                 if hasattr(model, "predict_proba") else None)
        y_ppt = (model.predict_proba(X_te)
                 if hasattr(model, "predict_proba") else None)

        vm = clf_metrics(y_va, y_pv, y_ppv)
        tm = clf_metrics(y_te, y_pt, y_ppt)
        mlflow.log_metrics({f"val_{k}":  v for k, v in vm.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})

        print(f"    Val   Acc={vm['accuracy']:.4f}  Prec={vm['precision']:.4f}  "
              f"Rec={vm['recall']:.4f}  F1={vm['f1_score']:.4f}  "
              f"AUC={vm.get('roc_auc', 0):.4f}")
        print(f"    Test  Acc={tm['accuracy']:.4f}  Prec={tm['precision']:.4f}  "
              f"Rec={tm['recall']:.4f}  F1={tm['f1_score']:.4f}  "
              f"AUC={tm.get('roc_auc', 0):.4f}")

        sn   = name.replace(" ", "_")
        cm_p = f"artifacts/cm_{sn}.png"
        save_confusion_matrix(y_te, y_pt, name, cm_p)
        mlflow.log_artifact(cm_p)

        fi_p = f"artifacts/fi_{sn}.png"
        save_feature_importance(model, feat_cols, name, fi_p)
        if os.path.exists(fi_p):
            mlflow.log_artifact(fi_p)

        rp = f"artifacts/report_{sn}.txt"
        with open(rp, "w") as f:
            f.write(classification_report(
                y_te, y_pt,
                target_names=["Eligible", "High_Risk", "Not_Eligible"]
            ))
        mlflow.log_artifact(rp)
        mlflow.sklearn.log_model(model, name="model")

    return {"name": name, "run_id": run_id, "model": model,
            "val_metrics": vm, "test_metrics": tm}


# ═══════════════════════════════════════════════════════════
# TRAIN REGRESSOR
# ═══════════════════════════════════════════════════════════
def train_regressor(name, model, params,
                    X_tr, y_tr,
                    X_va, y_va,
                    X_te, y_te,
                    feat_cols):
    mlflow.set_experiment(EXPERIMENT_REG)
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        print(f"\n  📌 {name}  ({run_id[:8]}...)")
        mlflow.log_params({**params, "model_name": name,
                           "n_features": len(feat_cols)})

        # ── Train on full training set ────────────────────
        model.fit(X_tr, y_tr)

        y_pv = model.predict(X_va); y_pt = model.predict(X_te)
        vm   = reg_metrics(y_va, y_pv)
        tm   = reg_metrics(y_te, y_pt)
        mlflow.log_metrics({f"val_{k}":  v for k, v in vm.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})

        print(f"    Val   RMSE=₹{vm['rmse']:>8,.2f}  "
              f"MAE=₹{vm['mae']:>8,.2f}  R²={vm['r2']:.4f}  MAPE={vm['mape']:.2f}%")
        print(f"    Test  RMSE=₹{tm['rmse']:>8,.2f}  "
              f"MAE=₹{tm['mae']:>8,.2f}  R²={tm['r2']:.4f}  MAPE={tm['mape']:.2f}%")

        sn        = name.replace(" ", "_")
        residuals = y_te - y_pt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(y_pt, residuals, alpha=0.3, s=4, color="steelblue")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title(f"Residuals — {name}")
        axes[0].set_xlabel("Predicted EMI")
        axes[1].hist(residuals, bins=50, color="orange", edgecolor="white")
        axes[1].set_title(f"Residual Distribution — {name}")
        plt.tight_layout()
        res_p = f"artifacts/residuals_{sn}.png"
        plt.savefig(res_p, dpi=120); plt.close()
        mlflow.log_artifact(res_p)

        fi_p = f"artifacts/fi_reg_{sn}.png"
        save_feature_importance(model, feat_cols, name, fi_p)
        if os.path.exists(fi_p):
            mlflow.log_artifact(fi_p)

        mlflow.sklearn.log_model(model, name="model")

    return {"name": name, "run_id": run_id, "model": model,
            "val_metrics": vm, "test_metrics": tm}


# ═══════════════════════════════════════════════════════════
# REGISTER BEST MODEL
# ═══════════════════════════════════════════════════════════
def register_best_model(results, experiment,
                         metric_key, higher_is_better,
                         registry_name):
    best = (max if higher_is_better else min)(
        results, key=lambda x: x["test_metrics"][metric_key]
    )
    print(f"\n🏆 Best {experiment}: {best['name']}"
          f"  ({metric_key}={best['test_metrics'][metric_key]:.4f})")
    try:
        mv = mlflow.register_model(
            f"runs:/{best['run_id']}/model", registry_name
        )
        print(f"   Registered as '{registry_name}' v{mv.version}")
    except Exception as e:
        print(f"   Registry note (non-fatal): {e}")
    return best


# ═══════════════════════════════════════════════════════════
# MODEL EVALUATION & SELECTION
# ═══════════════════════════════════════════════════════════
def model_evaluation_and_selection(clf_results, reg_results):
    # ── BUILD DATAFRAMES ─────────────────────────────────────
    clf_df = pd.DataFrame([
        {
            "Model"    : r["name"],
            "Accuracy" : r["test_metrics"]["accuracy"],
            "Precision": r["test_metrics"]["precision"],
            "Recall"   : r["test_metrics"]["recall"],
            "F1-Score" : r["test_metrics"]["f1_score"],
            "ROC-AUC"  : r["test_metrics"].get("roc_auc", 0.0),
        }
        for r in clf_results
    ]).set_index("Model")

    reg_df = pd.DataFrame([
        {
            "Model": r["name"],
            "RMSE" : r["test_metrics"]["rmse"],
            "MAE"  : r["test_metrics"]["mae"],
            "R²"   : r["test_metrics"]["r2"],
            "MAPE" : r["test_metrics"]["mape"],
        }
        for r in reg_results
    ]).set_index("Model")

    # ── CONSOLE: CLASSIFICATION TABLE ────────────────────────
    print("\n" + "=" * 80)
    print("MODEL EVALUATION & SELECTION")
    print("=" * 80)

    print("\n── CLASSIFICATION  (test set, SMOTE-balanced train) ──")
    print(f"\n  {'Model':<38} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1-Score':>9} {'ROC-AUC':>9}")
    print(f"  {'-'*38} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*9}")

    best_clf_name = clf_df["F1-Score"].idxmax()
    for model_name, row in clf_df.iterrows():
        mark = "  ← BEST" if model_name == best_clf_name else ""
        print(f"  {model_name:<38} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f} "
              f"{row['Recall']:>8.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f}{mark}")

    clf_df["Rank_F1"]  = clf_df["F1-Score"].rank(ascending=False).astype(int)
    clf_df["Rank_AUC"] = clf_df["ROC-AUC"].rank(ascending=False).astype(int)
    clf_df["Rank_Acc"] = clf_df["Accuracy"].rank(ascending=False).astype(int)
    clf_df["Avg_Rank"] = (clf_df["Rank_F1"] + clf_df["Rank_AUC"] + clf_df["Rank_Acc"]) / 3

    print(f"\n  Ranking (lower = better)")
    print(f"  {'Model':<38} {'F1 Rank':>8} {'AUC Rank':>9} {'Acc Rank':>9} {'Avg Rank':>9}")
    print(f"  {'-'*38} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")
    for model_name, row in clf_df.sort_values("Avg_Rank").iterrows():
        mark = "  ✅ SELECTED" if model_name == best_clf_name else ""
        print(f"  {model_name:<38} {row['Rank_F1']:>8} {row['Rank_AUC']:>9} "
              f"{row['Rank_Acc']:>9} {row['Avg_Rank']:>9.2f}{mark}")

    # ── CONSOLE: REGRESSION TABLE ─────────────────────────────
    print("\n\n── REGRESSION  (test set, original-balance train) ──")
    print(f"\n  {'Model':<38} {'RMSE (₹)':>12} {'MAE (₹)':>10} {'R²':>8} {'MAPE (%)':>10}")
    print(f"  {'-'*38} {'-'*12} {'-'*10} {'-'*8} {'-'*10}")

    best_reg_name = reg_df["R²"].idxmax()
    for model_name, row in reg_df.iterrows():
        mark = "  ← BEST" if model_name == best_reg_name else ""
        print(f"  {model_name:<38} {row['RMSE']:>12,.2f} {row['MAE']:>10,.2f} "
              f"{row['R²']:>8.4f} {row['MAPE']:>10.2f}{mark}")

    reg_df["Rank_R2"]   = reg_df["R²"].rank(ascending=False).astype(int)
    reg_df["Rank_RMSE"] = reg_df["RMSE"].rank(ascending=True).astype(int)
    reg_df["Rank_MAE"]  = reg_df["MAE"].rank(ascending=True).astype(int)
    reg_df["Rank_MAPE"] = reg_df["MAPE"].rank(ascending=True).astype(int)
    reg_df["Avg_Rank"]  = (
        reg_df["Rank_R2"] + reg_df["Rank_RMSE"] +
        reg_df["Rank_MAE"] + reg_df["Rank_MAPE"]
    ) / 4

    print(f"\n  Ranking (lower = better)")
    print(f"  {'Model':<38} {'R² Rank':>8} {'RMSE Rank':>10} {'MAE Rank':>9} "
          f"{'MAPE Rank':>10} {'Avg Rank':>9}")
    print(f"  {'-'*38} {'-'*8} {'-'*10} {'-'*9} {'-'*10} {'-'*9}")
    for model_name, row in reg_df.sort_values("Avg_Rank").iterrows():
        mark = "  ✅ SELECTED" if model_name == best_reg_name else ""
        print(f"  {model_name:<38} {row['Rank_R2']:>8} {row['Rank_RMSE']:>10} "
              f"{row['Rank_MAE']:>9} {row['Rank_MAPE']:>10} {row['Avg_Rank']:>9.2f}{mark}")

    # ── SELECTION SUMMARY ─────────────────────────────────────
    best_clf_row = clf_df.loc[best_clf_name]
    best_reg_row = reg_df.loc[best_reg_name]

    print("\n" + "─" * 80)
    print("FINAL MODEL SELECTION")
    print("─" * 80)
    print(f"\n  🏆 Classification  →  {best_clf_name}")
    print(f"       Accuracy  : {best_clf_row['Accuracy']:.4f}")
    print(f"       Precision : {best_clf_row['Precision']:.4f}")
    print(f"       Recall    : {best_clf_row['Recall']:.4f}")
    print(f"       F1-Score  : {best_clf_row['F1-Score']:.4f}   ← selection criterion")
    print(f"       ROC-AUC   : {best_clf_row['ROC-AUC']:.4f}")
    print(f"\n  🏆 Regression     →  {best_reg_name}")
    print(f"       RMSE      : ₹{best_reg_row['RMSE']:,.2f}")
    print(f"       MAE       : ₹{best_reg_row['MAE']:,.2f}")
    print(f"       R²        : {best_reg_row['R²']:.4f}   ← selection criterion")
    print(f"       MAPE      : {best_reg_row['MAPE']:.2f}%")
    print("─" * 80)

    _plot_clf_comparison(clf_df.drop(columns=["Rank_F1","Rank_AUC","Rank_Acc","Avg_Rank"]))
    _plot_reg_comparison(reg_df.drop(columns=["Rank_R2","Rank_RMSE","Rank_MAE","Rank_MAPE","Avg_Rank"]))
    _save_evaluation_report(clf_df, reg_df, best_clf_name, best_reg_name)

    return best_clf_name, best_reg_name


def _plot_clf_comparison(clf_df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x       = np.arange(len(clf_df))
    n       = len(metrics)
    w       = 0.14
    colors  = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + (i - n/2 + 0.5) * w,
                      clf_df[metric], w, label=metric,
                      color=color, edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(clf_df.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Classification — All Metrics Comparison (Test Set)", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(clf_df) - 0.5, 0.905, "0.90 threshold", fontsize=7, color="gray")
    plt.tight_layout()
    p = "artifacts/clf_metrics_comparison.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  📊 Classification chart → {p}")


def _plot_reg_comparison(reg_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(reg_df)); w = 0.3

    b1 = ax1.bar(x - w/2, reg_df["RMSE"], w, label="RMSE (₹)",
                 color="#e74c3c", edgecolor="white", alpha=0.85)
    b2 = ax1.bar(x + w/2, reg_df["MAE"],  w, label="MAE (₹)",
                 color="#3498db", edgecolor="white", alpha=0.85)
    ax1.bar_label(b1, fmt="%.0f", fontsize=7, padding=2)
    ax1.bar_label(b2, fmt="%.0f", fontsize=7, padding=2)
    ax1.set_ylabel("Error (₹)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(reg_df.index, rotation=15, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, reg_df["R²"],        "g^-",  label="R²",       markersize=10, linewidth=2)
    ax2.plot(x, reg_df["MAPE"] / 100,"m*--", label="MAPE/100", markersize=10, linewidth=1.5)
    for i, (r2, mape) in enumerate(zip(reg_df["R²"], reg_df["MAPE"])):
        ax2.annotate(f"{r2:.3f}",   (i, r2),      textcoords="offset points",
                     xytext=(0, 8),  ha="center", fontsize=7.5, color="green")
        ax2.annotate(f"{mape:.1f}%",(i, mape/100),textcoords="offset points",
                     xytext=(0,-14), ha="center", fontsize=7.5, color="purple")
    ax2.set_ylabel("R²  /  MAPE÷100", fontsize=10)
    ax2.set_ylim(0, 1.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("Regression — All Metrics Comparison (Test Set)", fontsize=13, pad=12)
    plt.tight_layout()
    p = "artifacts/reg_metrics_comparison.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  📊 Regression chart    → {p}")


def _save_evaluation_report(clf_df, reg_df, best_clf_name, best_reg_name):
    lines = ["=" * 80,
             "EMIPredict AI — Model Evaluation & Selection Report",
             "=" * 80,
             "\n[CLASSIFICATION — Test Set Metrics]",
             f"  {'Model':<38} {'Accuracy':>9} {'Precision':>10} "
             f"{'Recall':>8} {'F1-Score':>9} {'ROC-AUC':>9}",
             f"  {'-'*86}"]
    for model_name, row in clf_df.iterrows():
        mark = "  ← SELECTED" if model_name == best_clf_name else ""
        lines.append(
            f"  {model_name:<38} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f} "
            f"{row['Recall']:>8.4f} {row['F1-Score']:>9.4f} {row['ROC-AUC']:>9.4f}{mark}"
        )

    lines += ["\n[REGRESSION — Test Set Metrics]",
              f"  {'Model':<38} {'RMSE':>12} {'MAE':>10} {'R²':>8} {'MAPE (%)':>10}",
              f"  {'-'*80}"]
    for model_name, row in reg_df.iterrows():
        mark = "  ← SELECTED" if model_name == best_reg_name else ""
        lines.append(
            f"  {model_name:<38} {row['RMSE']:>12,.2f} {row['MAE']:>10,.2f} "
            f"{row['R²']:>8.4f} {row['MAPE']:>10.2f}{mark}"
        )

    bcr = clf_df.loc[best_clf_name]
    brr = reg_df.loc[best_reg_name]
    lines += ["\n[SELECTED MODELS]",
              f"\n  Classification  →  {best_clf_name}",
              f"    Accuracy={bcr['Accuracy']:.4f}  Precision={bcr['Precision']:.4f}  "
              f"Recall={bcr['Recall']:.4f}  F1={bcr['F1-Score']:.4f}  "
              f"ROC-AUC={bcr['ROC-AUC']:.4f}",
              f"\n  Regression      →  {best_reg_name}",
              f"    RMSE=₹{brr['RMSE']:,.2f}  MAE=₹{brr['MAE']:,.2f}  "
              f"R²={brr['R²']:.4f}  MAPE={brr['MAPE']:.2f}%",
              "\n" + "=" * 80]

    report_path = "artifacts/evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  📄 Evaluation report   → {report_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("🏦 EMIPredict AI — Step 4: Model Training + MLflow (v3)")
    print("=" * 65)
    print("  Classifiers : Logistic Regression, Random Forest,")
    print("                XGBoost, Decision Tree")
    print("  Regressors  : Linear Regression, Random Forest,")
    print("                XGBoost, Decision Tree\n")

    (X_clf_tr, y_clf_tr,
     X_reg_tr, y_reg_tr,
     X_va, y_clf_va, y_reg_va,
     X_te, y_clf_te, y_reg_te,
     feat_cols) = load_splits()

    # ── CLASSIFICATION ────────────────────────────────────
    print("\n" + "=" * 65)
    print("CLASSIFICATION MODELS  (train on SMOTE-balanced data)")
    print("=" * 65)

    clf_configs = [
        (
            "Logistic Regression",
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               multi_class="auto", random_state=RANDOM_STATE),
            {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        ),
        (
            "Random Forest Classifier",
            RandomForestClassifier(n_estimators=300, max_depth=20,
                                   min_samples_split=4, min_samples_leaf=2,
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 300, "max_depth": 20},
        ),
        (
            "XGBoost Classifier",
            XGBClassifier(n_estimators=500, max_depth=9, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="mlogloss", verbosity=0,
                          random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 500, "max_depth": 9, "lr": 0.05},
        ),
        (
            "Decision Tree Classifier",
            DecisionTreeClassifier(max_depth=15, min_samples_split=10,
                                   min_samples_leaf=5,
                                   random_state=RANDOM_STATE),
            {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5},
        ),
    ]

    clf_results = []
    for name, model, params in clf_configs:
        clf_results.append(train_classifier(
            name, model, params,
            X_clf_tr, y_clf_tr,
            X_va, y_clf_va,
            X_te, y_clf_te,
            feat_cols,
        ))

    best_clf = register_best_model(
        clf_results, EXPERIMENT_CLF, "f1_score", True, "EMI_Best_Classifier"
    )

    # ── REGRESSION ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("REGRESSION MODELS  (train on original-balance data)")
    print("=" * 65)

    reg_configs = [
        (
            "Linear Regression",
            LinearRegression(n_jobs=-1),
            {"fit_intercept": True},
        ),
        (
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=300, max_depth=20,
                                  min_samples_split=4, min_samples_leaf=2,
                                  random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 300, "max_depth": 20},
        ),
        (
            "XGBoost Regressor",
            XGBRegressor(n_estimators=500, max_depth=9, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         verbosity=0, random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 500, "max_depth": 9, "lr": 0.05},
        ),
        (
            "Decision Tree Regressor",
            DecisionTreeRegressor(max_depth=15, min_samples_split=10,
                                  min_samples_leaf=5,
                                  random_state=RANDOM_STATE),
            {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5},
        ),
    ]

    reg_results = []
    for name, model, params in reg_configs:
        reg_results.append(train_regressor(
            name, model, params,
            X_reg_tr, y_reg_tr,
            X_va, y_reg_va,
            X_te, y_reg_te,
            feat_cols,
        ))

    best_reg = register_best_model(
        reg_results, EXPERIMENT_REG, "r2", True, "EMI_Best_Regressor"
    )

    # ── Save best models ──────────────────────────────────
    joblib.dump(best_clf["model"], "models/best_classifier.pkl")
    joblib.dump(best_reg["model"], "models/best_regressor.pkl")
    print("\n💾 best_classifier.pkl and best_regressor.pkl → models/")

    # ── Model Evaluation & Selection ──────────────────────
    best_clf_name, best_reg_name = model_evaluation_and_selection(
        clf_results, reg_results
    )

    # ── Performance targets check ─────────────────────────
    print("\n── PERFORMANCE TARGETS ──")
    best_clf_acc  = best_clf["test_metrics"]["accuracy"]
    best_reg_rmse = best_reg["test_metrics"]["rmse"]
    clf_ok = "✅" if best_clf_acc  > 0.90   else "❌"
    reg_ok = "✅" if best_reg_rmse < 2000.0 else "❌"
    print(f"  {clf_ok} Classification accuracy > 90%  : {best_clf_acc*100:.2f}%"
          f"  [{best_clf['name']}]")
    print(f"  {reg_ok} Regression RMSE < ₹2,000       : ₹{best_reg_rmse:,.2f}"
          f"  [{best_reg['name']}]")

    print("\n✅ Step 4 complete!")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("   streamlit run app.py")

    return clf_results, reg_results, best_clf, best_reg


if __name__ == "__main__":
    main()
