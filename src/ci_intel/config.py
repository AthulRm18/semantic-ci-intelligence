"""
Central config. Every script/module reads from here instead of hardcoding
values â€” change the repo, run count, or paths in exactly one place.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- GitHub / data collection ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"

TARGET_REPO = "tiangolo/fastapi"
MAX_RUNS_PER_COLLECTION = 500
PER_PAGE = 100

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = DATA_RAW_DIR / "fastapi_ci_data.csv"
CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "fastapi_ci_cleaned.csv"

# --- Labels considered valid signal (drop everything else before training) ---
VALID_JOB_CONCLUSIONS = {"success", "failure"}
