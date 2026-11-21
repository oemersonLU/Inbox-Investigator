"""Reading uploaded emails and cleaning text for the model."""
import re
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser, Parser
from typing import Optional

from . import config

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
TAG_RE = re.compile(r"<[^>]+>")
STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
HEADER_START_RE = re.compile(
    r"\s*(From|To|Subject|Date|Reply-To|Return-Path|Received|MIME-Version|"
    r"Content-Type|Message-ID):", re.I)
PNG_MAGIC = b"\x89PNG"


class EmailReadError(Exception):
    """Raised when an upload cannot be read as an email. The message is shown to the user."""


@dataclass
class ParsedEmail:
    subject: Optional[str]
    from_address: Optional[str]
    to_address: Optional[str]
    n_attachments: int
    body: str
    raw_content: str
    source_filename: Optional[str]

    @property
    def model_text(self):
        parts = []
        if self.from_address:
            parts.append(f"From: {self.from_address}")
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        parts.append(self.body)
        return "\n".join(parts)[: config.MAX_CHARS]


def strip_html(s):
    s = STYLE_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)
    return re.sub(r"&nbsp;|&#160;", " ", s)


def clean_for_tfidf(text):
    """TF-IDF preprocessor: drop the From line, mask links/addresses, lowercase."""
    if text.startswith("From: "):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    text = strip_html(text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def _from_message(msg, raw_text, filename):
    if not any(msg.get(h) for h in ("From", "Subject", "To")):
        raise EmailReadError(
            "This file does not look like an email (no From, To, or Subject header).")
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if body_part is not None:
        try:
            body = body_part.get_content()
        except (LookupError, UnicodeError):
            body = str(body_part.get_payload(decode=False))
        if body_part.get_content_type() == "text/html":
            body = strip_html(body)
    elif not msg.is_multipart():
        body = str(msg.get_payload())
    attachments = sum(1 for _ in msg.iter_attachments())
    body = body.strip()
    if not body and not msg.get("Subject"):
        raise EmailReadError("The email has no readable text.")
    return ParsedEmail(
        subject=str(msg.get("Subject") or "") or None,
        from_address=str(msg.get("From") or "") or None,
        to_address=str(msg.get("To") or "") or None,
        n_attachments=attachments,
        body=body,
        raw_content=raw_text,
        source_filename=filename,
    )


def parse_text(text, filename=None):
    """Parse pasted or .txt text. Header lines are used when the text starts with them."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise EmailReadError("The email is empty.")
    if HEADER_START_RE.match(text):
        msg = Parser(policy=policy.default).parsestr(text)
        return _from_message(msg, text, filename)
    return ParsedEmail(None, None, None, 0, text, text, filename)


def read_upload(filename, data):
    """Turn an uploaded file into a ParsedEmail or raise EmailReadError."""
    name = (filename or "").lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        raise EmailReadError(
            f"Cannot read '{filename}'. Please upload a .eml or .txt file or paste the text.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise EmailReadError("The file is larger than 2 MB.")
    if not data.strip():
        raise EmailReadError("The file is empty.")
    if data.startswith(PNG_MAGIC) or b"\x00" in data[:4096]:
        raise EmailReadError(f"'{filename}' is a binary file, not an email.")
    try:
        if ext == ".eml":
            msg = BytesParser(policy=policy.default).parsebytes(data)
            raw = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
            return _from_message(msg, raw, filename)
        return parse_text(data.decode("utf-8", errors="replace"), filename)
    except EmailReadError:
        raise
    except Exception as exc:  # malformed MIME, bad charset, etc.
        raise EmailReadError(f"Could not read '{filename}': {exc}") from exc
