"""Make a SMALL SYNTHETIC dataset so you can try the app without downloading Kaggle data.

    python scripts/make_demo_data.py

Writes data/raw/demo_emails.csv. These emails are generated from templates, so the model will
score almost perfectly on them. DO NOT report results from this file. Delete it before
training on the real Kaggle dataset.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from inbox_investigator import config  # noqa: E402

random.seed(config.RANDOM_SEED)
BANKS = ["PayPal", "Chase", "Wells Fargo", "Amazon", "Netflix", "Microsoft", "Apple", "DHL"]
PHISH = [
    "URGENT: Your {b} account has been suspended. Click here to verify your account immediately: "
    "http://{ip}/login or your account will be closed. Confirm your password now!",
    "Dear customer, we noticed unusual activity on your {b} account. Please log in and update "
    "your payment details at http://{b2}-secure.{tld}/verify to avoid suspension. Act now!",
    "Congratulations winner! You have been selected to claim your {n} dollar gift card. Click "
    "here: http://bit.ly/{code} Limited time offer, confirm your details immediately.",
    "Security alert from {b}: your password will expire today. Verify your identity at "
    "http://{ip}/reset. Final notice.",
]
LEGIT = [
    "Hi team, the weekly meeting is moved to {d} at {t}. Please bring your project updates and "
    "let me know if you have conflicts. Thanks, {p}",
    "Hello {p}, attached are the notes from our discussion about the budget. Can you review "
    "section {n} before Friday? I think the numbers look fine overall. Best, Sam",
    "Your order {code} has shipped and should arrive on {d}. You can track the package from your "
    "account page on our website. Thank you for shopping with us.",
    "Reminder: the library book you borrowed is due on {d}. You can renew it online or at the "
    "front desk. Have a great week!",
    "Hey {p}, are we still on for lunch on {d}? I was thinking about the new place near campus. "
    "Let me know what time works for you.",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Chris", "Riley"]


def fill(t):
    return t.format(b=random.choice(BANKS), b2=random.choice(BANKS).lower().replace(" ", ""),
                    ip=".".join(str(random.randint(1, 250)) for _ in range(4)),
                    tld=random.choice(["xyz", "top", "com", "net"]),
                    code="".join(random.choice("abcdef0123456789") for _ in range(7)),
                    n=random.randint(2, 900), d=random.choice(DAYS),
                    t=f"{random.randint(1, 5)} PM", p=random.choice(NAMES))


rows = []
for _ in range(400):
    rows.append((fill(random.choice(PHISH)), "Phishing Email"))
    rows.append((fill(random.choice(LEGIT)), "Safe Email"))
config.ensure_dirs()
out = config.RAW_DIR / "demo_emails.csv"
pd.DataFrame(rows, columns=["Email Text", "Email Type"]).sample(
    frac=1, random_state=1).to_csv(out, index=False)
print(f"Wrote {out} (synthetic - for trying the app only, not for your results).")
