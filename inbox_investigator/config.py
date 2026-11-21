"""Paths and settings. Everything lives on the local machine."""
import os
from pathlib import Path

# INBOX_INVESTIGATOR_HOME lets the tests point the app at a temporary folder.
ROOT = Path(os.environ.get("INBOX_INVESTIGATOR_HOME",
                           Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DB_PATH = DATA_DIR / "inbox_investigator.db"
CURRENT_MODEL_FILE = MODELS_DIR / "current.json"

RANDOM_SEED = 42
DEFAULT_UNCERTAIN_BELOW = 0.75   # below this confidence the verdict is "not sure"
BALANCE_RATIO = 1.5              # balance the training set only above this class ratio
BASELINE_VERSION = "keyword-baseline"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_EXTENSIONS = {".eml", ".txt"}
MAX_CHARS = 20000                # longer emails are truncated before scoring


def ensure_dirs():
    for d in (DATA_DIR, RAW_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
