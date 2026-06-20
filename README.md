# 💰 EMIPredict AI — Intelligent Financial Risk Assessment Platform

> **GUVI × HCL Final Project | FinTech & Banking Domain | v3**

EMIPredict AI is an end-to-end machine learning system that predicts **loan EMI eligibility** and **maximum safe monthly EMI** for loan applicants using advanced ML models, SMOTE class balancing, and a real-time Streamlit dashboard.

---

## 🎯 Project Objectives

- Predict whether a loan applicant is **Eligible**, **High Risk**, or **Not Eligible**
- Predict the **maximum safe monthly EMI** (₹) the applicant can afford
- Track all experiments using **MLflow**
- Provide an interactive **multi-page Streamlit dashboard** for real-time inference

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📁 Project Structure

```
EMIPredict-AI/
│
├── app.py                          # Streamlit main entry point
├── predict_utils.py                # Shared prediction utilities
├── step1_data_preprocessing.py     # Data cleaning & splitting pipeline
├── step3_feature_engineering.py    # Feature engineering, SMOTE, scaling
├── step4_train_models.py           # Model training + MLflow logging
├── requirements.txt                # Python dependencies
│
├── pages/
│   ├── home.py                     # 🏠 Home — overview & dataset stats
│   ├── predict.py                  # 🔮 EMI Prediction — real-time form
│   ├── eda_page.py                 # 📊 Data Exploration — EDA charts
│   ├── mlflow_page.py              # 📈 Model Performance — MLflow results
│   └── crud_page.py                # 🛠️ Data Management — loan CRUD
│
├── models/                         # Saved model artefacts (post-training)
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_cols.pkl
│   └── log_transformed_cols.pkl
│
├── data/                           # Processed data splits (post-pipeline)
│   ├── train_fe.csv
│   ├── train_fe_smote.csv
│   ├── val_fe.csv
│   └── test_fe.csv
│
└── artifacts/                      # Plots, reports (auto-generated)
    ├── evaluation_report.txt
    ├── cm_*.png
    └── fi_*.png
```

---

## 🧠 ML Pipeline

```
Raw Dataset (404,800 rows)
        │
        ▼
┌─────────────────────────┐
│  Step 1: Preprocessing  │  Dtype fix · Null fill · Duplicate removal
│                         │  IQR outlier clipping · Train/Val/Test split
└───────────┬─────────────┘  (70 / 15 / 15)
            │
            ▼
┌─────────────────────────┐
│  Step 3: Feature Engg.  │  Financial ratios · Risk features
│                         │  Correlation/redundancy removal
│                         │  Skewness fix (log1p) · Encoding
│                         │  SMOTE balancing · StandardScaler
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Step 4: Model Training │  5 Classifiers + 5 Regressors
│                         │  MLflow experiment tracking
│                         │  Model evaluation & selection
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Streamlit Dashboard    │  Real-time prediction · EDA · CRUD
└─────────────────────────┘
```

---

## 🤖 Models Trained

### Classification (Eligibility Prediction)
| # | Model | Description |
|---|-------|-------------|
| 1 | Logistic Regression | Baseline linear model |
| 2 | Random Forest | 300 trees, depth 20 |
| 3 | XGBoost | 500 estimators, learning rate 0.05 |
| 4 | HistGradientBoosting | Histogram-based gradient boosting |
| 5 | Voting Ensemble | Soft voting — RF + XGB + HGB |

### Regression (Max EMI Prediction)
| # | Model | Description |
|---|-------|-------------|
| 1 | Linear Regression | Baseline model |
| 2 | Random Forest | 300 trees, depth 20 |
| 3 | XGBoost | 500 estimators, learning rate 0.05 |
| 4 | HistGradientBoosting | Histogram-based gradient boosting |
| 5 | Voting Ensemble | Mean averaging — RF + XGB + HGB |

---

## 📊 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy · Precision · Recall · F1-Score · ROC-AUC |
| Regression | RMSE · MAE · R² · MAPE |

**Performance Targets:**
- ✅ Classification Accuracy > **90%**
- ✅ Regression RMSE < **₹2,000**

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Total Records | 404,800 |
| Raw Features | 27 |
| Final Features | 16 |
| Target (Classification) | `emi_eligibility` (3 classes) |
| Target (Regression) | `max_monthly_emi` (₹) |
| Class Balance | SMOTE applied → 219,007 per class |

### Input Features
| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Applicant age (18–70) |
| `gender` | Categorical | Male / Female |
| `marital_status` | Categorical | Single / Married / Divorced |
| `education` | Categorical | Graduate / Post_Graduate / Under_Graduate / PhD |
| `monthly_salary` | Numeric | Gross monthly income (₹) |
| `employment_type` | Categorical | Salaried / Self_Employed / Freelancer |
| `years_of_employment` | Numeric | Work experience (years) |
| `company_type` | Categorical | Private / Government / MNC / Startup |
| `house_type` | Categorical | Owned / Rented / Parental |
| `existing_loans` | Binary | 0 = No, 1 = Yes |
| `current_emi_amount` | Numeric | Existing monthly EMI (₹) |
| `credit_score` | Numeric | CIBIL score (300–900) |
| `bank_balance` | Numeric | Current savings (₹) |
| `requested_amount` | Numeric | Loan requested (₹) |
| `requested_tenure` | Numeric | Loan tenure (months) |
| `total_monthly_expenses` | Numeric | All monthly expenses excluding EMI (₹) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML Models | Scikit-learn, XGBoost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow + SQLite |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy, SciPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Model Persistence | Joblib |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/emipredict-ai.git
cd emipredict-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML pipeline

```bash
# Step 1 — Data preprocessing
python step1_data_preprocessing.py

# Step 3 — Feature engineering + SMOTE
python step3_feature_engineering.py

# Step 4 — Train all 10 models
python step4_train_models.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

### 5. View MLflow dashboard (optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open http://localhost:5000

---

## 🌐 Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, dataset stats, eligibility distribution charts |
| 🔮 **EMI Prediction** | Real-time eligibility + max EMI form with gauge chart |
| 📊 **Data Exploration** | Interactive EDA — distributions, correlation heatmap, scatter plots |
| 📈 **Model Performance** | MLflow experiment comparison tables and bar charts |
| 🛠️ **Data Management** | Create, view, edit, delete loan application records |

---

## 📈 Key Features

- **SMOTE Balancing** — Synthetic Minority Over-sampling equalises all 3 classes (219,007 each) to eliminate model bias
- **Redundancy Removal** — Drops highly correlated features (threshold > 0.90) to reduce overfitting
- **Skewness Correction** — Applies `log1p` transform to features with |skew| > 1.0
- **Voting Ensemble** — Combines RF + XGB + HistGB for maximum accuracy
- **Full MLflow Tracking** — Every run logs params, val/test metrics, confusion matrices, feature importance plots
- **Inference Consistency** — predict_utils applies the exact same FE + log1p + scaling pipeline used during training

---

## 🗂️ Output Files

After running the pipeline:

```
models/
  best_classifier.pkl     ← Best classification model
  best_regressor.pkl      ← Best regression model
  scaler.pkl              ← Fitted StandardScaler
  encoders.pkl            ← Fitted LabelEncoders
  feature_cols.pkl        ← Ordered feature column list
  log_transformed_cols.pkl← Columns that had log1p applied
  dropped_features.pkl    ← Features removed by redundancy check

artifacts/
  evaluation_report.txt   ← Full model comparison table
  cm_*.png                ← Confusion matrices
  fi_*.png                ← Feature importance plots
  residuals_*.png         ← Regression residual plots
```

---

## 👩‍💻 Author

**Bhuvaneshwari**
GUVI × HCL Final Project — FinTech & Banking Domain

---