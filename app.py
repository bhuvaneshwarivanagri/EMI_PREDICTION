"""
=============================================================
app.py — EMIPredict AI  Main Entry Point (v3)
=============================================================
Multi-page Streamlit application.

Pages:
  🏠 Home            — Project overview & dataset stats
  🔮 EMI Prediction  — Real-time eligibility + max EMI form
  📊 Data Exploration— Interactive EDA charts
  📈 Model Performance— MLflow experiment comparison
  🛠️ Data Management — Loan application CRUD

Run:
    streamlit run app.py
"""

import os
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Responsive CSS + Hide default multipage menu ──────────────
st.markdown("""
    <style>
        /* Hide default Streamlit multipage nav */
        [data-testid="stSidebarNav"] { display: none; }

        /* Responsive: tighten padding on small screens */
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

        /* Navy blue background override */
        .stApp {
            background-color: #0a1628 !important;
        }

        /* Metric card styling */
        [data-testid="metric-container"] {
            background: #112240;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 12px 16px;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: #071020 !important;
        }

        /* Expander and containers */
        [data-testid="stExpander"] {
            background: #112240;
            border: 1px solid #1e3a5f;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }

        /* Divider colour */
        hr { border-color: #2d3250; }
    </style>
""", unsafe_allow_html=True)


# ── Download models from Google Drive on first run ────────────
@st.cache_resource(show_spinner=False)
def initialise():
    """Download model artefacts once per session if not present."""
    required = [
        "models/best_classifier.pkl",
        "models/best_regressor.pkl",
        "models/scaler.pkl",
        "models/encoders.pkl",
        "models/feature_cols.pkl",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        try:
            from download_models import download_models
            with st.spinner("⬇️ Downloading models for first-time setup… (1–2 min)"):
                download_models()
            return True, None
        except Exception as e:
            return False, str(e)
    return True, None


models_ok, init_error = initialise()

if not models_ok:
    st.error(
        f"❌ Failed to download models: {init_error}\n\n"
        "Please refresh the page or run the training pipeline locally first."
    )
    st.stop()


# ── Page imports with error handling ─────────────────────────
def _safe_import(module_path: str, label: str):
    try:
        import importlib
        return importlib.import_module(module_path)
    except Exception as e:
        st.error(f"❌ Could not load **{label}** page: `{e}`")
        return None


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

    # Model status indicator
    required_files = [
        "best_classifier.pkl", "best_regressor.pkl",
        "scaler.pkl", "encoders.pkl", "feature_cols.pkl",
    ]
    models_exist = all(os.path.exists(f"models/{f}") for f in required_files)

    if models_exist:
        st.success("✅ Models ready")
    else:
        st.warning("⚠️ Models not found")
        st.caption("Refresh page to retry download")

    st.divider()
    st.caption("GUVI × HCL Final Project")
    st.caption("Domain: FinTech & Banking")
    st.caption("v3 — Production Build")


# ── Route to pages ────────────────────────────────────────────
try:
    if page == "🏠 Home":
        from pages.home import show
        show()

    elif page == "🔮 EMI Prediction":
        if not models_exist:
            st.error("⚠️ Models not loaded yet. Please refresh the page.")
            st.info("If this persists, check that all model files are available.")
        else:
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
    st.error(f"❌ An unexpected error occurred on this page.")
    with st.expander("🔍 Error details (for debugging)"):
        st.exception(e)
    st.info("💡 Try navigating to another page and coming back, or refresh the browser.")
