"""
download_models.py
==================
Downloads model artefacts from Google Drive on first run.
Called automatically by app.py before loading models.
"""

import os
import gdown

os.makedirs("models", exist_ok=True)

# ── Add your Google Drive file IDs here ──────────────────────
# Get the ID from the share link:
# https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing

MODEL_FILES = {
    "models/best_classifier.pkl"     : "1b86SVZiN-0vX6IKmibhsuYR5CbZ9rylK",
    "models/best_regressor.pkl"      : "1GGTXNaqaJ_VOG5CS1IRcXs9z4gA-WGl4",
    "models/scaler.pkl"              : "1xQ-VLMcoBJqH6bld7IIu8zSq6d2xODhA",
    "models/encoders.pkl"            : "1aM8PR8RiLakyX7WZ7l-TmYyHfa-qWNO5",
    "models/feature_cols.pkl"        : "1vyUZv0p5JX5AcOQIiL1kt3q21Ro2UdpP",
    "models/log_transformed_cols.pkl": "1d43KcDZgWuSm-K3Zlm_Zdafwi5wdeHZv",
}


def download_models():
    for path, file_id in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)
            print(f"  ✅ Saved to {path}")
        else:
            print(f"  ✓ {path} already exists, skipping.")


if __name__ == "__main__":
    download_models()
