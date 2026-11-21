"""Classify an email and keep the database in step. Used by the web app."""
import hashlib
import time

from . import baseline, config
from .db import audit, now_iso
from .model import ALGORITHM, decide


def ensure_baseline_version(conn):
    row = conn.execute("SELECT model_version_id FROM model_version WHERE version_name = ?",
                       (config.BASELINE_VERSION,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO model_version (version_name, algorithm, artifact_path, dataset_id, "
        "uncertain_below, trained_at, is_active) VALUES (?, ?, NULL, NULL, NULL, ?, 0)",
        (config.BASELINE_VERSION, "keyword rule (2+ suspicious keywords)", now_iso()))
    return cur.lastrowid


def register_model_version(conn, bundle, dataset_id=None):
    """Make sure the loaded model has a model_version row and mark it as the active one."""
    row = conn.execute("SELECT model_version_id FROM model_version WHERE version_name = ?",
                       (bundle.version_name,)).fetchone()
    if row:
        mv_id = row[0]
    else:
        cur = conn.execute(
            "INSERT INTO model_version (version_name, algorithm, artifact_path, dataset_id, "
            "uncertain_below, trained_at, is_active) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (bundle.version_name, ALGORITHM, bundle.artifact_path, dataset_id,
             bundle.uncertain_below, now_iso()))
        mv_id = cur.lastrowid
    set_active(conn, mv_id)
    return mv_id


def set_active(conn, model_version_id):
    conn.execute("UPDATE model_version SET is_active = 0 WHERE model_version_id != ?",
                 (model_version_id,))
    conn.execute("UPDATE model_version SET is_active = 1 WHERE model_version_id = ?",
                 (model_version_id,))


def classify_and_store(conn, parsed, bundle):
    """Return (result_id, already_checked). Uses the model, or the keyword rule if bundle is None."""
    content_hash = hashlib.sha256(parsed.raw_content.encode("utf-8")).hexdigest()
    row = conn.execute("SELECT email_id FROM email WHERE content_hash = ?",
                       (content_hash,)).fetchone()
    if row:
        email_id = row[0]
    else:
        cur = conn.execute(
            "INSERT INTO email (content_hash, subject, from_address, to_address, n_attachments, "
            "raw_content, source_filename, processing_status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'parsed', ?)",
            (content_hash, parsed.subject, parsed.from_address, parsed.to_address,
             parsed.n_attachments, parsed.raw_content, parsed.source_filename, now_iso()))
        email_id = cur.lastrowid

    if bundle is not None:
        mv_id = register_model_version(conn, bundle)
    else:
        mv_id = ensure_baseline_version(conn)
        set_active(conn, mv_id)

    existing = conn.execute(
        "SELECT result_id FROM classification_result WHERE email_id = ? AND model_version_id = ?",
        (email_id, mv_id)).fetchone()
    if existing:
        conn.commit()
        return existing[0], True

    text = parsed.model_text
    start = time.perf_counter()
    if bundle is not None:
        p = bundle.predict_proba(text)
        predicted, confidence = decide(p, bundle.uncertain_below)
    else:
        predicted, _ = baseline.classify(text)
        p, confidence = None, None          # the rule has no probability
    latency_ms = int(round((time.perf_counter() - start) * 1000))

    cur = conn.execute(
        "INSERT INTO classification_result (email_id, model_version_id, predicted_class, "
        "phishing_probability, confidence_score, latency_ms, predicted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (email_id, mv_id, predicted, p, confidence, latency_ms, now_iso()))
    result_id = cur.lastrowid
    conn.execute("UPDATE email SET processing_status = 'classified' WHERE email_id = ?",
                 (email_id,))
    audit(conn, "classified", "classification_result", result_id,
          {"predicted_class": predicted, "confidence": confidence, "latency_ms": latency_ms})
    conn.commit()
    return result_id, False


def save_feedback(conn, result_id, user_label, comment):
    """Store the user's label (one per result; changing it updates the row)."""
    result = conn.execute(
        "SELECT predicted_class FROM classification_result WHERE result_id = ?",
        (result_id,)).fetchone()
    if result is None:
        raise KeyError(result_id)
    comment = (comment or "").strip() or None
    existing = conn.execute("SELECT feedback_id FROM feedback WHERE result_id = ? "
                            "ORDER BY feedback_id DESC LIMIT 1", (result_id,)).fetchone()
    if existing:
        conn.execute("UPDATE feedback SET user_label = ?, comment = ?, submitted_at = ? "
                     "WHERE feedback_id = ?", (user_label, comment, now_iso(), existing[0]))
        fb_id = existing[0]
    else:
        cur = conn.execute("INSERT INTO feedback (result_id, user_label, comment, submitted_at) "
                           "VALUES (?, ?, ?, ?)", (result_id, user_label, comment, now_iso()))
        fb_id = cur.lastrowid
    predicted = result[0]
    if predicted in ("phishing", "legitimate") and predicted != user_label:
        audit(conn, "misclassification", "classification_result", result_id,
              {"model_said": predicted, "user_said": user_label, "feedback_id": fb_id})
    else:
        audit(conn, "feedback", "classification_result", result_id,
              {"model_said": predicted, "user_said": user_label, "feedback_id": fb_id})
    conn.commit()
    return fb_id
