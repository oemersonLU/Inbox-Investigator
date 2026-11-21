"""Train the model (offline).  Usage:  python scripts/train.py [--uncertain-below 0.75]

Steps: load the CSV(s) in data/raw -> split 80/10/10 -> balance the training set only if the
classes are more than 1.5 to 1 -> fit -> score the validation split -> save the model and
register it in the database.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402

from inbox_investigator import config, data, db, model, service  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertain-below", type=float, default=config.DEFAULT_UNCERTAIN_BELOW,
                    help="confidence below this is shown as 'not sure' (0.5 to 1)")
    args = ap.parse_args()
    if not 0.5 <= args.uncertain_below <= 1:
        sys.exit("--uncertain-below must be between 0.5 and 1")

    config.ensure_dirs()
    conn = db.connect()
    db.init_db(conn)
    try:
        df, files, newest = data.load_raw_dataset()
    except data.DatasetError as exc:
        sys.exit(f"ERROR: {exc}")

    n_phish, n_legit = int((df.label == 1).sum()), int((df.label == 0).sum())
    print(f"Loaded {len(df)} unique emails from {', '.join(files)} "
          f"({n_phish} phishing, {n_legit} legitimate).")
    if min(n_phish, n_legit) < 20:
        sys.exit("ERROR: need at least 20 emails of each class to train.")
    if any("demo" in f.lower() for f in files):
        print("\n*** WARNING: this is the synthetic DEMO data. Its metrics are not valid for "
              "your report. Use the Kaggle dataset. ***\n")

    train, val, test = data.split_dataset(df)
    data.save_splits(train, val, test)
    train_fit, balanced = data.balance_train(train)
    print(f"Split: train {len(train)}, validation {len(val)}, test {len(test)}. "
          f"Training set balanced: {'yes (' + str(len(train_fit)) + ' rows)' if balanced else 'no'}.")

    pipe = model.build_pipeline()
    pipe.fit(train_fit["text"], train_fit["label"])

    probs = pipe.predict_proba(val["text"])[:, 1]
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(val["label"], pred),
        "precision": precision_score(val["label"], pred, zero_division=0),
        "recall": recall_score(val["label"], pred, zero_division=0),
        "f1": f1_score(val["label"], pred, zero_division=0),
    }
    print("Validation: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    version = "logreg-tfidf-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    path = model.save_model(pipe, version, args.uncertain_below)

    now = db.now_iso()
    name = ", ".join(files)
    source = ("Synthetic demo data (scripts/make_demo_data.py)" if "demo" in name.lower()
              else "Kaggle: charanrajasekhar/phishing-email-dataset")
    collected = data.mtime_iso(newest)
    row = conn.execute("SELECT dataset_id FROM dataset WHERE dataset_name = ? AND collected_at = ?",
                       (name, collected)).fetchone()
    if row:
        dataset_id = row[0]
        conn.execute("UPDATE dataset SET total_samples=?, phishing_samples=?, legitimate_samples=? "
                     "WHERE dataset_id=?", (len(df), n_phish, n_legit, dataset_id))
    else:
        dataset_id = conn.execute(
            "INSERT INTO dataset (dataset_name, source, total_samples, phishing_samples, "
            "legitimate_samples, collected_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, source, len(df), n_phish, n_legit, collected)).lastrowid

    mv_id = conn.execute(
        "INSERT INTO model_version (version_name, algorithm, artifact_path, dataset_id, "
        "uncertain_below, trained_at, is_active) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (version, model.ALGORITHM, f"models/{path.name}", dataset_id,
         args.uncertain_below, now)).lastrowid
    service.set_active(conn, mv_id)
    for k, v in metrics.items():
        conn.execute("INSERT OR REPLACE INTO model_metric (model_version_id, split_name, "
                     "metric_name, metric_value, recorded_at) VALUES (?, 'validation', ?, ?, ?)",
                     (mv_id, k, float(v), now))
    db.audit(conn, "model_trained", "model_version", mv_id,
             {"version": version, "dataset": name, "balanced": balanced, **metrics})
    conn.commit()
    print(f"Saved models/{path.name} and set it as the active model.")


if __name__ == "__main__":
    main()
