"""The simple keyword rule used as the baseline (and as the fallback when no model loads)."""
BASELINE_KEYWORDS = [
    "verify your account", "click here", "urgent", "suspended", "password",
    "confirm your", "login", "log in", "bank account", "winner", "act now",
    "security alert", "update your",
]


def matched_keywords(text):
    low = text.lower()
    return [k for k in BASELINE_KEYWORDS if k in low]


def classify(text):
    """Return (label, matched keywords). Two or more different keywords = phishing."""
    hits = matched_keywords(text)
    return ("phishing" if len(hits) >= 2 else "legitimate"), hits
