"""End-to-end tests on a tiny synthetic dataset in a temporary folder.

Run:  python -m pytest -q
"""
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOME = tempfile.mkdtemp(prefix="ii_test_")
os.environ["INBOX_INVESTIGATOR_HOME"] = HOME
sys.path.insert(0, str(REPO))

from inbox_investigator import baseline, config, db, preprocess  # noqa: E402
from inbox_investigator.app import create_app  # noqa: E402


def run_script(name):
    env = dict(os.environ, INBOX_INVESTIGATOR_HOME=HOME)
    subprocess.run([sys.executable, str(REPO / "scripts" / name)], check=True, env=env,
                   capture_output=True)


@pytest.fixture(scope="module")
def trained():
    run_script("make_demo_data.py")
    run_script("train.py")
    run_script("evaluate.py")


def sample(name):
    return (REPO / "sample_emails" / name).read_bytes()


def test_parse_eml_and_txt():
    p = preprocess.read_upload("a.eml", sample("phishing_paypal.eml"))
    assert p.subject.startswith("URGENT") and "paypa1" in p.from_address
    t = preprocess.read_upload("b.txt", sample("legit_newsletter.txt"))
    assert t.subject == "Library hours for Thanksgiving week"


def test_bad_files_rejected():
    with pytest.raises(preprocess.EmailReadError):
        preprocess.read_upload("bad.png", sample("bad_file_not_an_email.png"))
    with pytest.raises(preprocess.EmailReadError):
        preprocess.read_upload("x.txt", b"\x89PNG\x00\x00")
    with pytest.raises(preprocess.EmailReadError):
        preprocess.read_upload("x.eml", b"just some words, no headers")
    with pytest.raises(preprocess.EmailReadError):
        preprocess.parse_text("   ")


def test_baseline_rule():
    assert baseline.classify("Click here to verify your account urgent")[0] == "phishing"
    assert baseline.classify("Lunch on Friday?")[0] == "legitimate"


def test_train_outputs(trained):
    for f in ("train.csv", "validation.csv", "test.csv"):
        assert (config.SPLITS_DIR / f).exists()
    assert (config.CURRENT_MODEL_FILE).exists()
    assert (config.RESULTS_DIR / "results_table.md").exists()
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM model_metric WHERE split_name='test'").fetchone()[0] > 5
    assert conn.execute("SELECT COUNT(*) FROM model_version WHERE is_active=1").fetchone()[0] == 1


def test_web_flow(trained):
    client = create_app().test_client()
    assert client.get("/").status_code == 200

    # phishing upload -> result page with verdict + confidence
    r = client.post("/check", data={"file": (io.BytesIO(sample("phishing_paypal.eml")), "p.eml")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200 and b"Likely phishing" in r.data and b"Confidence" in r.data
    assert b"Pushed toward phishing" in r.data

    # same email again -> saved result, no duplicate row
    r2 = client.post("/check", data={"file": (io.BytesIO(sample("phishing_paypal.eml")), "p.eml")},
                     content_type="multipart/form-data", follow_redirects=True)
    assert b"already checked" in r2.data

    # paste text
    r3 = client.post("/check", data={"text": "Hi, are we still meeting for lunch on Friday?"},
                     follow_redirects=True)
    assert r3.status_code == 200 and b"Likely legitimate" in r3.data

    # unreadable file -> error message, not a crash
    bad = client.post("/check", data={"file": (io.BytesIO(sample("bad_file_not_an_email.png")), "x.png")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and b"could not be read" in bad.data
    assert client.post("/check", data={}).status_code == 400

    # correction that disagrees with the model is logged as a misclassification
    conn = db.connect()
    rid = conn.execute("SELECT result_id FROM classification_result "
                       "WHERE predicted_class='phishing' LIMIT 1").fetchone()[0]
    fb = client.post(f"/result/{rid}/feedback", data={"user_label": "legitimate", "comment": "mine"},
                     follow_redirects=True)
    assert fb.status_code == 200 and b"legitimate" in fb.data
    assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='misclassification'").fetchone()[0] == 1
    client.post(f"/result/{rid}/feedback", data={"user_label": "phishing"})   # change it later
    assert conn.execute("SELECT COUNT(*) FROM feedback WHERE result_id=?", (rid,)).fetchone()[0] == 1
    assert client.post(f"/result/{rid}/feedback", data={"user_label": "bogus"}).status_code == 400

    # other pages
    assert client.get("/review").status_code == 200
    assert client.get("/history").status_code == 200
    csv_resp = client.get("/history.csv")
    assert csv_resp.mimetype == "text/csv" and b"model_verdict" in csv_resp.data
    m = client.get("/metrics")
    assert m.status_code == 200 and b"Accuracy" in m.data
    assert client.get("/result/99999").status_code == 404


def test_fallback_when_model_missing(trained):
    """If the model file will not load the app still runs, on the keyword rule."""
    meta = config.CURRENT_MODEL_FILE
    saved = meta.read_text()
    try:
        meta.write_text(saved.replace(".joblib", "_missing.joblib"))
        client = create_app().test_client()
        page = client.get("/")
        assert b"No trained model is loaded" in page.data
        r = client.post("/check", data={"text": "URGENT click here to verify your account password"},
                        follow_redirects=True)
        assert b"Likely phishing" in r.data and b"unavailable" in r.data
        conn = db.connect()
        n = conn.execute("SELECT COUNT(*) FROM classification_result r JOIN model_version m "
                         "USING (model_version_id) WHERE m.version_name = ? "
                         "AND r.confidence_score IS NULL", (config.BASELINE_VERSION,)).fetchone()[0]
        assert n == 1
    finally:
        meta.write_text(saved)
        create_app()   # re-activate the real model


def test_uncertain_goes_to_review_queue(trained):
    from inbox_investigator import service
    from inbox_investigator.model import decide
    assert decide(0.6, 0.75)[0] == "uncertain"
    assert decide(0.9, 0.75)[0] == "phishing"
    assert decide(0.1, 0.75)[0] == "legitimate"
