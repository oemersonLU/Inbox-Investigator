# Inbox Investigator

A local phishing-email checker built for CIS 321 (Milestone 3 working prototype) by Nolan Clark and Olivia Emerson.

You upload or paste an email. The app cleans the text, builds features, and a model trained on the Kaggle Phishing Email Dataset says **Likely phishing**, **Likely legitimate**, or **Not sure**, with a confidence percentage. Everything runs on your own computer. Email text is never sent anywhere.

- **Model:** TF-IDF words and word pairs + hand-made features (link count, suspicious phrases, sender domain, ...) -> logistic regression (scikit-learn), saved as a file and loaded when the app starts.
- **Uncertain emails** (confidence below 75% by default) are marked "Not sure" and go to a review queue where you decide.
- **Corrections:** you can mark any result as phishing or legitimate. Disagreements are logged as misclassifications.
- **Storage:** a local SQLite database (`data/inbox_investigator.db`). Schema: `inbox_investigator/schema.sql`.
- **Out of scope:** connecting to a real mailbox, scanning attachments or following links, user accounts, retraining from the web page.

## Run it (Windows)

1. Download this repository (**Code > Download ZIP**) or `git clone` it, and unzip it.
2. Install **Python 3.10 or newer** from python.org. Tick **"Add Python to PATH"** during install.
3. Double-click **`setup.bat`**. It creates a virtual environment and installs the packages in `requirements.txt`.
4. Make a free Kaggle account and download the [Phishing Email Dataset](https://www.kaggle.com/datasets/charanrajasekhar/phishing-email-dataset). Unzip it and put the CSV file(s) in the `data\raw` folder.
5. Double-click **`train.bat`**. It trains the model, evaluates it on the held-out test split, and prints the metrics. The results are saved in `results\` (the table for the report is `results\results_table.md`).
6. Double-click **`run.bat`**, then open <http://127.0.0.1:5000> in a browser.
7. Upload a file from `sample_emails\`, or paste your own email text. To see the error message, upload `bad_file_not_an_email.png`.

If you start the app before step 5 it still works: it shows a yellow banner and uses the simple keyword rule (the project's baseline) with confidence shown as "unavailable".

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# put the Kaggle CSV file(s) in data/raw/
python scripts/train.py
python scripts/evaluate.py
python -m inbox_investigator.app      # then open http://127.0.0.1:5000
```

## What the scripts do

| Command | What it does |
|---|---|
| `python scripts/train.py` | Loads the CSV(s) in `data/raw`, removes duplicates, splits 80/10/10 (stratified, seed 42) **before anything else**, balances the training set only if the classes are more than 1.5 : 1, trains, scores the validation split, saves `models/<version>.joblib` and `models/current.json`, and records the dataset, model version and validation metrics in the database. Option: `--uncertain-below 0.75`. |
| `python scripts/evaluate.py` | Scores the model **and** the keyword baseline on the untouched 10% test split. Writes `results/results_table.md`, `results.json`, `fpr_by_group.csv` and the `model_metric` table. Includes time to classify one email and false-positive rates across writing styles. |
| `python scripts/make_demo_data.py` | Optional. Makes a small **synthetic** dataset so you can try the app without Kaggle. **Do not report results from it**, and delete `data/raw/demo_emails.csv` before training on the real data. |
| `python -m pytest -q` | Runs the tests (uses a temporary folder and synthetic data; does not touch your real database or model). |

## Folder map

```
inbox_investigator/   app.py (Flask pages)  service.py (classify + save)  model.py  features.py
                      preprocess.py  baseline.py  data.py  db.py  schema.sql  templates/  static/
scripts/              train.py  evaluate.py  make_demo_data.py
sample_emails/        a few emails to try, plus a file that triggers the error message
data/raw/             put the Kaggle CSV(s) here (not committed)
data/splits/          train / validation / test CSVs written by train.py (not committed)
models/               trained model files (not committed)
results/              evaluation output
tests/                pytest tests
```

## Pages

- **Check an email** - upload `.eml` / `.txt` or paste text.
- **Result** - verdict in words, confidence bar with the "not sure" cutoff marked, the email, the words that pushed the result each way, and a correction form.
- **Review queue** - only the "Not sure" emails, each with Phishing / Legitimate buttons.
- **History** - every email checked, the model's answer next to yours, export to CSV.
- **Model metrics** - baseline vs model scores and the false-positive table (developer view).

## Troubleshooting

- **`python` is not recognized:** reinstall Python and tick "Add Python to PATH", then run `setup.bat` again.
- **"No CSV files found in data\raw":** step 4 was skipped, or the CSV is inside a sub-folder. Put the `.csv` file directly in `data\raw`.
- **"could not find the text and label columns":** the loader looks for columns such as `Email Text` / `Email Type`. If the Kaggle file uses other names, rename them to `text` and `label` or add the names to `TEXT_COLUMNS` / `LABEL_COLUMNS` in `inbox_investigator/data.py`.
- **Yellow "No trained model" banner:** run `train.bat`, then restart `run.bat`.
- **Port 5000 already in use:** close the other program, or change the port at the bottom of `inbox_investigator/app.py`.
- **Start over:** delete `data\inbox_investigator.db`, `data\splits\*` and `models\*.joblib` and `models\current.json`, then run `train.bat` again.

## Limits

This is a class prototype for one user at a time. A simple model does not understand language the way a large hosted model does, and it can be fooled. It only advises: it never deletes, moves, or blocks an email.
