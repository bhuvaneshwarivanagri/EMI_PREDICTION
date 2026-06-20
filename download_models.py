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
    "models/best_classifier.pkl"     : "1fTOTTGdwA-9KGGLmPM1RQQDywKkb4fZ_",
    "models/best_regressor.pkl"      : "1vgwD6weYeusYh5GI76mx-1E4K2vKMsz2",
    "models/scaler.pkl"              : "1zC7woSZOoXJjS180vtZ6ulnM1Efv7GUi",
    "models/encoders.pkl"            : "1r27Xba6aMQwMk8ouIJnPiDp1JQ7PxrNR",
    "models/feature_cols.pkl"        : "1V2zH-IHmRdA9H7PfJHRCNd_4BLmRBBfl",
    "models/log_transformed_cols.pkl": "152oJEH0MvNEtq9ZmUC0lntVF4xE8im10",
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
