"""
=============================================================
app_hf.py — EMIPredict AI  Hugging Face Entry Point (v3)
=============================================================
Same as app.py but skips Google Drive download —
models are bundled directly inside the Hugging Face Space.

Deploy on Hugging Face Spaces:
  - SDK       : Streamlit
  - app_file  : app_hf.py
"""

import os
import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EMIPREDICT-AI-MODELS",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none; }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
            }
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.3rem !important; }
            h3 { font-size: 1.1rem !important; }
        }

        .stApp { background-color: #0a1628 !important; }

        [data-testid="metric-container"] {
            background: #112240;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 12px 16px;
        }

        [data-testid="stSidebar"] { background: #071020 !important; }

        [data-testid="stExpander"] {
            background: #112240;
            border: 1px solid #1e3a5f;
        }

        .stButton > button { border-radius: 8px; font-weight: 600; }

        hr { border-color: #2d3250; }
    </style>
""", unsafe_allow_html=True)


# ── Download models from HF Hub if not present ────────────────
MODEL_REPO = "bhuvana86jaishu/EMIPREDICT-AI-MODELS"
REQUIRED_FILES = [
    "models/best_classifier.pkl",
    "models/best_regressor.pkl",
    "models/scaler.pkl",
    "models/encoders.pkl",
    "models/feature_cols.pkl",
    "models/log_transformed_cols.pkl",
    "models/dropped_features.pkl",
]

os.makedirs("models", exist_ok=True)
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing:
    try:
        from huggingface_hub import hf_hub_download
        with st.spinner("⬇️ Loading models… (first load only, ~30 sec)"):
            for filepath in missing:
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filepath,
                    repo_type="model",
                    local_dir=".",
                )
        st.success("✅ Models loaded!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:12px 0'>
            <div style='font-size:2rem'>💰</div>
            <h3 style='margin:4px 0;color:#90caf9'>EMIPredict AI</h3>
            <small style='color:#aaa'>Financial Risk Assessment</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "🔮 EMI Prediction",
            "📊 Data Exploration",
            "📈 Model Performance",
            "🛠️ Data Management",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.success("✅ Models ready")
    st.divider()
    st.caption("GUVI × HCL Final Project")
    st.caption("Domain: FinTech & Banking")
    st.caption("v3 — Hugging Face Deploy")


# ── Route to pages ────────────────────────────────────────────
try:
    if page == "🏠 Home":
        from pages.home import show
        show()

    elif page == "🔮 EMI Prediction":
        from pages.predict import show
        show()

    elif page == "📊 Data Exploration":
        from pages.eda_page import show
        show()

    elif page == "📈 Model Performance":
        from pages.mlflow_page import show
        show()

    elif page == "🛠️ Data Management":
        from pages.crud_page import show
        show()

except Exception as e:
    st.error("❌ An unexpected error occurred on this page.")
    with st.expander("🔍 Error details (for debugging)"):
        st.exception(e)
    st.info("💡 Try navigating to another page and coming back, or refresh the browser.")
