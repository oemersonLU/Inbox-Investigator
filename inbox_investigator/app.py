"""The local web app.  Run with:  python -m inbox_investigator.app   (or run.bat)"""
import csv
import io

from flask import (Flask, Response, abort, flash, g, redirect, render_template, request,
                   url_for)

from . import baseline, config, db, model, service
from .preprocess import EmailReadError, parse_text, read_upload

VERDICTS = {
    "phishing": ("Likely phishing", "phish"),
    "legitimate": ("Likely legitimate", "legit"),
    "uncertain": ("Not sure", "unsure"),
}


def create_app():
    config.ensure_dirs()
    app = Flask(__name__)
    app.secret_key = "local-only-no-accounts"       # the app is single-user and never leaves this PC
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES + 100_000

    bundle, reason = model.load_current()
    app.config["MODEL"] = bundle
    app.config["MODEL_PROBLEM"] = reason
    conn = db.connect()
    db.init_db(conn)
    if bundle is not None:
        service.register_model_version(conn, bundle)
    else:
        service.set_active(conn, service.ensure_baseline_version(conn))
    conn.commit()
    conn.close()

    def get_db():
        if "db" not in g:
            g.db = db.connect()
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn_ = g.pop("db", None)
        if conn_ is not None:
            conn_.close()

    @app.context_processor
    def inject():
        return {"model_problem": app.config["MODEL_PROBLEM"],
                "queue_count": get_db().execute(
                    "SELECT COUNT(*) FROM classification_result r WHERE r.predicted_class = "
                    "'uncertain' AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.result_id = "
                    "r.result_id)").fetchone()[0]}

    def error_page(message, status=400):
        db.audit(get_db(), "upload_error", details=message)
        get_db().commit()
        return render_template("error.html", message=message), status

    @app.errorhandler(413)
    def too_big(_e):
        return error_page("The file is larger than 2 MB.", 413)

    # ---------- check an email ----------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/check")
    def check():
        upload = request.files.get("file")
        pasted = (request.form.get("text") or "").strip()
        try:
            if upload is not None and upload.filename:
                parsed = read_upload(upload.filename, upload.read())
            elif pasted:
                parsed = parse_text(pasted)
            else:
                raise EmailReadError("Choose a file or paste the email text first.")
            result_id, already = service.classify_and_store(get_db(), parsed,
                                                            app.config["MODEL"])
        except EmailReadError as exc:
            return error_page(str(exc))
        if already:
            flash("This email was already checked with the current model. Showing the saved result.")
        return redirect(url_for("result", result_id=result_id))

    # ---------- result ----------
    def load_result(result_id):
        row = get_db().execute(
            "SELECT r.*, e.subject, e.from_address, e.to_address, e.n_attachments, "
            "e.raw_content, e.source_filename, m.version_name, m.algorithm, m.uncertain_below, "
            "(SELECT user_label FROM feedback WHERE result_id = r.result_id "
            " ORDER BY feedback_id DESC LIMIT 1) AS user_label, "
            "(SELECT comment FROM feedback WHERE result_id = r.result_id "
            " ORDER BY feedback_id DESC LIMIT 1) AS user_comment "
            "FROM classification_result r JOIN email e USING (email_id) "
            "JOIN model_version m USING (model_version_id) WHERE r.result_id = ?",
            (result_id,)).fetchone()
        if row is None:
            abort(404)
        return row

    @app.get("/result/<int:result_id>")
    def result(result_id):
        row = load_result(result_id)
        label, css = VERDICTS[row["predicted_class"]]
        parsed_text = row["raw_content"]
        explain = None
        bundle_ = app.config["MODEL"]
        if row["version_name"] == config.BASELINE_VERSION:
            explain = {"kind": "keywords", "hits": baseline.matched_keywords(parsed_text)}
        elif bundle_ is not None and bundle_.version_name == row["version_name"]:
            try:
                p = parse_text(parsed_text, row["source_filename"])
                toward_phish, toward_legit = bundle_.explain(p.model_text)
                explain = {"kind": "model", "phish": toward_phish, "legit": toward_legit}
            except EmailReadError:
                explain = None
        return render_template(
            "result.html", r=row, verdict=label, css=css, explain=explain,
            body=(parsed_text[:6000] + ("\n... (shortened)" if len(parsed_text) > 6000 else "")))

    @app.post("/result/<int:result_id>/feedback")
    def feedback(result_id):
        load_result(result_id)
        label = request.form.get("user_label")
        if label not in ("phishing", "legitimate"):
            abort(400)
        service.save_feedback(get_db(), result_id, label, request.form.get("comment"))
        flash(f"Saved. You marked this email as {label}.")
        if request.form.get("next") == "review":
            return redirect(url_for("review"))
        return redirect(url_for("result", result_id=result_id))

    # ---------- review queue / history ----------
    @app.get("/review")
    def review():
        rows = get_db().execute(
            "SELECT r.result_id, r.phishing_probability, r.confidence_score, r.predicted_at, "
            "e.subject, e.from_address, substr(e.raw_content, 1, 220) AS preview "
            "FROM classification_result r JOIN email e USING (email_id) "
            "WHERE r.predicted_class = 'uncertain' AND NOT EXISTS "
            "(SELECT 1 FROM feedback f WHERE f.result_id = r.result_id) "
            "ORDER BY r.confidence_score ASC, r.predicted_at DESC").fetchall()
        return render_template("review.html", rows=rows)

    HISTORY_SQL = (
        "SELECT r.result_id, r.predicted_at, r.predicted_class, r.confidence_score, "
        "e.subject, e.from_address, m.version_name, "
        "(SELECT user_label FROM feedback WHERE result_id = r.result_id "
        " ORDER BY feedback_id DESC LIMIT 1) AS user_label, "
        "(SELECT comment FROM feedback WHERE result_id = r.result_id "
        " ORDER BY feedback_id DESC LIMIT 1) AS user_comment "
        "FROM classification_result r JOIN email e USING (email_id) "
        "JOIN model_version m USING (model_version_id) ORDER BY r.predicted_at DESC, r.result_id DESC")

    @app.get("/history")
    def history():
        return render_template("history.html", rows=get_db().execute(HISTORY_SQL).fetchall(),
                               verdicts=VERDICTS)

    @app.get("/history.csv")
    def history_csv():
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["checked_at", "subject", "from", "model_version", "model_verdict",
                    "confidence", "your_label", "your_comment", "disagrees_with_model"])
        for r in get_db().execute(HISTORY_SQL).fetchall():
            conf = "" if r["confidence_score"] is None else f"{r['confidence_score']:.4f}"
            disagrees = (r["user_label"] is not None and r["predicted_class"] != "uncertain"
                         and r["user_label"] != r["predicted_class"])
            w.writerow([r["predicted_at"], r["subject"] or "", r["from_address"] or "",
                        r["version_name"], r["predicted_class"], conf, r["user_label"] or "",
                        r["user_comment"] or "", "yes" if disagrees else "no"])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=history.csv"})

    # ---------- developer view ----------
    @app.get("/metrics")
    def metrics():
        c = get_db()
        mrow = c.execute(
            "SELECT mv.model_version_id, mv.version_name FROM model_version mv "
            "WHERE mv.version_name != ? AND EXISTS (SELECT 1 FROM model_metric mm WHERE "
            "mm.model_version_id = mv.model_version_id AND mm.split_name = 'test') "
            "ORDER BY mv.trained_at DESC, mv.model_version_id DESC LIMIT 1",
            (config.BASELINE_VERSION,)).fetchone()
        table, fpr = None, []
        if mrow:
            def get(mv, name):
                r = c.execute("SELECT metric_value FROM model_metric WHERE model_version_id = ? "
                              "AND split_name = 'test' AND metric_name = ?", (mv, name)).fetchone()
                return None if r is None else r[0]
            base = c.execute("SELECT model_version_id FROM model_version WHERE version_name = ?",
                             (config.BASELINE_VERSION,)).fetchone()
            base_id = base[0] if base else None
            table = [(n, get(base_id, k) if base_id else None, get(mrow[0], k))
                     for n, k in (("Accuracy", "accuracy"), ("Precision", "precision"),
                                  ("Recall", "recall"), ("F1 score", "f1"))]
            fpr = c.execute(
                "SELECT substr(metric_name, 6) AS grp, metric_value FROM model_metric WHERE "
                "model_version_id = ? AND split_name = 'test' AND metric_name LIKE 'fpr: %' "
                "ORDER BY metric_value DESC", (mrow[0],)).fetchall()
        return render_template("metrics.html", version=mrow["version_name"] if mrow else None,
                               table=table, fpr=fpr, gap=get(mrow[0], "fpr_gap") if mrow else None,
                               latency=(get(mrow[0], "latency_mean_s"), get(mrow[0], "latency_max_s"))
                               if mrow else (None, None))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
