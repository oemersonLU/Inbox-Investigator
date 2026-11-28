"""Loading the Kaggle CSV(s) from data/raw and writing the 80/10/10 splits."""
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

TEXT_COLUMNS = ["email text", "email_text", "text", "body", "email", "content", "message", "text_combined"]
LABEL_COLUMNS = ["email type", "email_type", "label", "class", "type", "category", "target"]


class DatasetError(Exception):
    pass


def _find_column(columns, candidates):
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _to_binary(value):
    """Map a label to 1 (phishing) or 0 (legitimate); None if unknown."""
    s = str(value).strip().lower()
    if s in ("1", "1.0", "true") or "phish" in s or "spam" in s or "fraud" in s:
        return 1
    if s in ("0", "0.0", "false") or "safe" in s or "legit" in s or "ham" in s or "not" in s:
        return 0
    return None


def load_raw_dataset(raw_dir=None):
    """Read every CSV in data/raw. Returns (DataFrame[text, label], list of file names)."""
    raw_dir = raw_dir or config.RAW_DIR
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise DatasetError(
            f"No CSV files found in {raw_dir}. Download the Kaggle Phishing Email Dataset and "
            "put the CSV file(s) there (README step 4).")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding_errors="replace")
        text_col = _find_column(df.columns, TEXT_COLUMNS)
        label_col = _find_column(df.columns, LABEL_COLUMNS)
        if text_col is None or label_col is None:
            raise DatasetError(
                f"{f.name}: could not find the text and label columns. Columns present: "
                f"{list(df.columns)}. Rename them to 'text' and 'label' or edit "
                "TEXT_COLUMNS / LABEL_COLUMNS in inbox_investigator/data.py.")
        part = pd.DataFrame({"text": df[text_col], "label": df[label_col].map(_to_binary)})
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.slice(0, config.MAX_CHARS)
    data = data[data["text"].str.strip().str.len() > 0]
    data = data.drop_duplicates(subset="text").reset_index(drop=True)   # before splitting: no leakage
    data["label"] = data["label"].astype(int)
    return data, [f.name for f in files], max(f.stat().st_mtime for f in files)


def split_dataset(data):
    """Stratified 80/10/10 split. Done first so the test split is never touched until the end."""
    train, rest = train_test_split(data, test_size=0.2, stratify=data["label"],
                                   random_state=config.RANDOM_SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"],
                                 random_state=config.RANDOM_SEED)
    return train, val, test


def save_splits(train, val, test):
    config.ensure_dirs()
    for name, df in (("train", train), ("validation", val), ("test", test)):
        df.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)


def load_split(name):
    path = config.SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise DatasetError(f"{path} not found. Run scripts/train.py first.")
    return pd.read_csv(path)


def balance_train(train):
    """Downsample the bigger class, but only when the ratio is above config.BALANCE_RATIO."""
    counts = train["label"].value_counts()
    big, small = counts.idxmax(), counts.idxmin()
    if counts[big] / counts[small] <= config.BALANCE_RATIO:
        return train, False
    keep = train[train["label"] == big].sample(n=int(counts[small] * config.BALANCE_RATIO),
                                               random_state=config.RANDOM_SEED)
    balanced = pd.concat([keep, train[train["label"] == small]]).sample(
        frac=1, random_state=config.RANDOM_SEED)
    return balanced, True


def mtime_iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")
