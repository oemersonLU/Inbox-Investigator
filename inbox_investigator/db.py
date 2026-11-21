"""SQLite helpers."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=None):
    config.ensure_dirs()
    conn = sqlite3.connect(str(path or config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn):
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.commit()


def audit(conn, action, entity_type=None, entity_id=None, details=None):
    if details is not None and not isinstance(details, str):
        details = json.dumps(details, default=str)
    conn.execute(
        "INSERT INTO audit_log (action, entity_type, entity_id, details, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (action, entity_type, entity_id, details, now_iso()),
    )
