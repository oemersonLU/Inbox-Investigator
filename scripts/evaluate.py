"""Score the model AND the keyword baseline on the held-out 10% test split.

Usage:  python scripts/evaluate.py
Writes results/results_table.md (the table for Milestone 3, section 4.3), results/results.json,
results/fpr_by_group.csv, and records the numbers in the model_metric table.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score)

from inbox_investigator import baseline, config, data, db, model, service  # noqa: E402
from inbox_investigator.model import decide  # noqa: E402

MIN_GROUP_LEGIT = 20     # smallest number of legitimate emails for a group's FPR to count


def style_groups(text):
    """Writing-style groups an email belongs to (an email can be in several)."""
    letters = [c for c in text if c.isalpha()]
    n_words = len(text.split())
    groups = []
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.25:
        groups.append("ALL-CAPS heavy")
    if n_words >= 20 and not any(c.isupper() for c in text):
        groups.append("no capital letters (informal)")
    if sum(ord(c) > 127 for c in text) / max(len(text), 1) > 0.01:
        groups.append("non-English characters")
    if text.count("!") + text.count("?") >= 3:
        groups.append("heavy punctuation")
    if re.search(r"<(html|div|table|body|p|br)\b", text, re.I):
        groups.append("HTML formatted")
    groups.append("short (<50 words)" if n_words < 50 else
                  "long (300+ words)" if n_words >= 300 else "medium length")
    return groups


def metrics_for(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fpr_by_group(texts, y_true, y_pred):
    rows = []
    groups = [style_groups(t) for t in texts]
    for g in sorted({x for gs in groups for x in gs}):
        mask = np.array([g in gs for gs in groups])
        legit = mask & (y_true == 0)
        n_legit = int(legit.sum())
        fp = int((legit & (y_pred == 1)).sum())
        rows.append({"group": g, "n_legitimate": n_legit, "false_positives": fp,
                     "fpr": fp / n_legit if n_legit else None,
                     "counted": n_legit >= MIN_GROUP_LEGIT})
    return pd.DataFrame(rows)


def fmt_pct(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


def main():
    bundle, reason = model.load_current()
    if bundle is None:
        sys.exit(f"ERROR: {reason} Run scripts/train.py first.")
    try:
        test = data.load_split("test")
    except data.DatasetError as exc:
        sys.exit(f"ERROR: {exc}")
    texts, y = test["text"].tolist(), test["label"].to_numpy()
    print(f"Evaluating {bundle.version_name} on {len(test)} test emails ...")

    probs = bundle.predict_proba_many(texts)
    pred_model = (probs >= 0.5).astype(int)
    pred_base = np.array([1 if baseline.classify(t)[0] == "phishing" else 0 for t in texts])
    m_model, m_base = metrics_for(y, pred_model), metrics_for(y, pred_base)
    uncertain_rate = float(np.mean([decide(p, bundle.uncertain_below)[0] == "uncertain"
                                    for p in probs]))

    # Time to classify ONE email at a time (what a user waits for), on up to 300 test emails.
    sample = texts[:300]
    times = []
    for t in sample:
        start = time.perf_counter()
        decide(bundle.predict_proba(t), bundle.uncertain_below)
        times.append(time.perf_counter() - start)
    t_mean, t_max = float(np.mean(times)), float(np.max(times))

    fpr_df = fpr_by_group(texts, y, pred_model)
    counted = fpr_df[fpr_df["counted"] & fpr_df["fpr"].notna()]
    gap = float(counted["fpr"].max() - counted["fpr"].min()) if len(counted) >= 2 else None

    # ---- save everything ----
    config.ensure_dirs()
    fpr_df.to_csv(config.RESULTS_DIR / "fpr_by_group.csv", index=False)
    results = {"model_version": bundle.version_name, "test_size": len(test),
               "model": m_model, "baseline": m_base, "uncertain_rate": uncertain_rate,
               "latency_seconds_mean": t_mean, "latency_seconds_max": t_max,
               "fpr_gap": gap}
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2),
                                                     encoding="utf-8")
    target_f1 = "above 95%"
    md = [
        f"# Results - {bundle.version_name}",
        "",
        f"Held-out test split: {len(test)} emails (10% of the dataset, never used for training "
        "or tuning). Phishing is the positive class. Hard labels use the 50% line; the "
        "\"not sure\" band is reported separately below.",
        "",
        "| Metric | Baseline | Your system | Target from M2 |",
        "|---|---|---|---|",
        f"| Accuracy | {fmt_pct(m_base['accuracy'])} | {fmt_pct(m_model['accuracy'])} | at least 95% |",
        f"| Precision | {fmt_pct(m_base['precision'])} | {fmt_pct(m_model['precision'])} | reported (no numeric target) |",
        f"| Recall | {fmt_pct(m_base['recall'])} | {fmt_pct(m_model['recall'])} | reported (no numeric target) |",
        f"| F1 score | {fmt_pct(m_base['f1'])} | {fmt_pct(m_model['f1'])} | {target_f1} |",
        f"| Time to classify one email (mean / max) | n/a | {t_mean:.3f} s / {t_max:.3f} s | under 60 seconds |",
        f"| False-positive rate gap across writing-style groups | n/a | "
        f"{'n/a (too few emails per group)' if gap is None else f'{gap * 100:.2f} percentage points'} "
        "| checked and reported |",
        "",
        f"Share of test emails the model marks \"not sure\" (confidence below "
        f"{bundle.uncertain_below:.0%}): {uncertain_rate:.1%}.",
        "",
        "## Confusion matrices (rows: actual, columns: predicted)",
        "",
        "| | Model: legitimate | Model: phishing | Baseline: legitimate | Baseline: phishing |",
        "|---|---|---|---|---|",
        f"| Actual legitimate | {m_model['tn']} | {m_model['fp']} | {m_base['tn']} | {m_base['fp']} |",
        f"| Actual phishing | {m_model['fn']} | {m_model['tp']} | {m_base['fn']} | {m_base['tp']} |",
        "",
        "## False-positive rate by writing style (model)",
        "",
        f"Groups need at least {MIN_GROUP_LEGIT} legitimate test emails to count toward the gap. "
        "An email can belong to more than one group. Sender region is not in the dataset, so it "
        "is not checked.",
        "",
        "| Group | Legitimate emails | False positives | FPR | Counted |",
        "|---|---|---|---|---|",
    ]
    for _, r in fpr_df.iterrows():
        md.append(f"| {r['group']} | {r['n_legitimate']} | {r['false_positives']} | "
                  f"{fmt_pct(r['fpr'])} | {'yes' if r['counted'] else 'no (too few)'} |")
    (config.RESULTS_DIR / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    conn = db.connect()
    db.init_db(conn)
    mv_model = service.register_model_version(conn, bundle)
    mv_base = service.ensure_baseline_version(conn)
    service.set_active(conn, mv_model)
    now = db.now_iso()

    def put(mv, name, value):
        conn.execute("INSERT OR REPLACE INTO model_metric (model_version_id, split_name, "
                     "metric_name, metric_value, recorded_at) VALUES (?, 'test', ?, ?, ?)",
                     (mv, name, float(value), now))

    for k in ("accuracy", "precision", "recall", "f1"):
        put(mv_model, k, m_model[k])
        put(mv_base, k, m_base[k])
    put(mv_model, "uncertain_rate", uncertain_rate)
    put(mv_model, "latency_mean_s", t_mean)
    put(mv_model, "latency_max_s", t_max)
    if gap is not None:
        put(mv_model, "fpr_gap", gap)
    for _, r in fpr_df.iterrows():
        if r["fpr"] is not None and not pd.isna(r["fpr"]):
            put(mv_model, f"fpr: {r['group']}", r["fpr"])
    db.audit(conn, "evaluated", "model_version", mv_model,
             {"test_size": len(test), "f1": m_model["f1"], "baseline_f1": m_base["f1"]})
    conn.commit()

    print("\n".join(md[:14]))
    print("\nSaved results/results_table.md, results.json, fpr_by_group.csv")


if __name__ == "__main__":
    main()
