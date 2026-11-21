-- Inbox Investigator physical schema (SQLite, third normal form).
-- Dates are ISO-8601 text. Foreign keys are switched on by db.connect().

CREATE TABLE IF NOT EXISTS dataset (
    dataset_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name        TEXT    NOT NULL,
    source              TEXT    NOT NULL,
    total_samples       INTEGER NOT NULL CHECK (total_samples >= 0),
    phishing_samples    INTEGER NOT NULL CHECK (phishing_samples >= 0),
    legitimate_samples  INTEGER NOT NULL CHECK (legitimate_samples >= 0),
    collected_at        TEXT    NOT NULL,
    UNIQUE (dataset_name, collected_at)
);

CREATE TABLE IF NOT EXISTS model_version (
    model_version_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name        TEXT    NOT NULL UNIQUE,
    algorithm           TEXT    NOT NULL,
    artifact_path       TEXT,                    -- NULL for the keyword baseline
    dataset_id          INTEGER REFERENCES dataset (dataset_id),
    uncertain_below     REAL CHECK (uncertain_below IS NULL
                                    OR (uncertain_below >= 0.5 AND uncertain_below <= 1)),
    trained_at          TEXT    NOT NULL,
    is_active           INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS email (
    email_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash        TEXT    NOT NULL UNIQUE,  -- SHA-256 of the raw text
    subject             TEXT,
    from_address        TEXT,
    to_address          TEXT,
    n_attachments       INTEGER NOT NULL DEFAULT 0,
    raw_content         TEXT    NOT NULL,
    source_filename     TEXT,
    processing_status   TEXT    NOT NULL DEFAULT 'parsed'
                        CHECK (processing_status IN ('parsed', 'classified')),
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS classification_result (
    result_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id              INTEGER NOT NULL REFERENCES email (email_id) ON DELETE CASCADE,
    model_version_id      INTEGER NOT NULL REFERENCES model_version (model_version_id),
    predicted_class       TEXT    NOT NULL
                          CHECK (predicted_class IN ('phishing', 'legitimate', 'uncertain')),
    phishing_probability  REAL CHECK (phishing_probability IS NULL
                                      OR (phishing_probability >= 0 AND phishing_probability <= 1)),
    confidence_score      REAL CHECK (confidence_score IS NULL
                                      OR (confidence_score >= 0 AND confidence_score <= 1)),
    latency_ms            INTEGER NOT NULL CHECK (latency_ms >= 0),
    predicted_at          TEXT    NOT NULL,
    UNIQUE (email_id, model_version_id)
);
CREATE INDEX IF NOT EXISTS idx_result_email   ON classification_result (email_id);
CREATE INDEX IF NOT EXISTS idx_result_class   ON classification_result (predicted_class, predicted_at);
CREATE INDEX IF NOT EXISTS idx_result_model   ON classification_result (model_version_id);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES classification_result (result_id) ON DELETE CASCADE,
    user_label      TEXT    NOT NULL CHECK (user_label IN ('phishing', 'legitimate')),
    comment         TEXT,
    submitted_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_result ON feedback (result_id);

CREATE TABLE IF NOT EXISTS model_metric (
    metric_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id  INTEGER NOT NULL REFERENCES model_version (model_version_id) ON DELETE CASCADE,
    split_name        TEXT    NOT NULL CHECK (split_name IN ('validation', 'test')),
    metric_name       TEXT    NOT NULL,           -- e.g. accuracy, f1, fpr: <group>
    metric_value      REAL    NOT NULL,
    recorded_at       TEXT    NOT NULL,
    UNIQUE (model_version_id, split_name, metric_name)
);
CREATE INDEX IF NOT EXISTS idx_metric_model ON model_metric (model_version_id);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    action        TEXT    NOT NULL,
    entity_type   TEXT,
    entity_id     INTEGER,
    details       TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log (created_at);
