"""Hand-made features (link count, suspicious phrases, sender domain, ...)."""
import math
import re

import numpy as np

from .preprocess import URL_RE

SUSPICIOUS_PHRASES = [
    "verify your account", "confirm your", "click here", "urgent", "suspended",
    "password", "log in", "login", "update your", "security alert", "limited time",
    "act now", "winner", "congratulations", "bank account", "wire transfer",
    "social security", "unusual activity", "will expire", "immediately",
    "dear customer", "dear user", "gift card", "claim your", "lottery",
    "inheritance", "account has been", "verify your identity", "final notice",
]
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".tk", ".click", ".ru", ".cn", ".work", ".zip", ".gq", ".ml")
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
FROM_RE = re.compile(r"^From:\s*(.*)$", re.M)

# name -> plain-language label used when explaining a result
FEATURES = [
    ("n_links", "number of links"),
    ("ip_link", "link that uses a raw IP address"),
    ("shortener", "shortened link"),
    ("n_phrases", "suspicious phrases (e.g. 'verify your account')"),
    ("n_exclaim", "exclamation marks"),
    ("upper_ratio", "share of CAPITAL letters"),
    ("log_len", "message length"),
    ("digit_ratio", "share of digits"),
    ("has_sender", "sender address present"),
    ("sender_bad_tld", "sender domain ends in a risky suffix"),
    ("sender_digits", "digits in the sender domain"),
]
FEATURE_NAMES = [n for n, _ in FEATURES]
FEATURE_LABELS = dict(FEATURES)


def suspicious_phrases_in(text):
    low = text.lower()
    return [p for p in SUSPICIOUS_PHRASES if p in low]


def _row(text):
    low = text.lower()
    links = URL_RE.findall(text)
    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    n = max(len(text), 1)
    m = FROM_RE.match(text) or FROM_RE.search(text[:500])
    domain = ""
    if m:
        addr = re.search(r"@([\w.-]+)", m.group(1))
        domain = addr.group(1).lower() if addr else ""
    return [
        math.log1p(len(links)),
        1.0 if IP_URL_RE.search(text) else 0.0,
        1.0 if any(s in low for s in SHORTENERS) else 0.0,
        float(len(suspicious_phrases_in(text))),
        math.log1p(text.count("!")),
        upper / max(len(letters), 1),
        math.log1p(len(text)),
        sum(c.isdigit() for c in text) / n,
        1.0 if domain else 0.0,
        1.0 if domain.endswith(SUSPICIOUS_TLDS) else 0.0,
        1.0 if re.search(r"\d", domain) else 0.0,
    ]


def hand_features(texts):
    """Used inside the sklearn pipeline (must stay a module-level function for pickling)."""
    return np.array([_row(t) for t in texts], dtype=float)
