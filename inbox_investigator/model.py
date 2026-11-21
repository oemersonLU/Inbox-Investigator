"""The self-hosted model: TF-IDF + hand-made features -> logistic regression."""
import json
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config
from .features import FEATURE_LABELS, FEATURE_NAMES, hand_features
from .preprocess import clean_for_tfidf

ALGORITHM = "TF-IDF + hand-made features + logistic regression"


def build_pipeline():
    tfidf = TfidfVectorizer(preprocessor=clean_for_tfidf, ngram_range=(1, 2),
                            min_df=2, max_features=50000, sublinear_tf=True)
    hand = Pipeline([("calc", FunctionTransformer(hand_features)),
                     ("scale", StandardScaler(with_mean=False))])
    return Pipeline([
        ("features", FeatureUnion([("tfidf", tfidf), ("hand", hand)])),
        ("clf", LogisticRegression(max_iter=2000, C=3.0, random_state=config.RANDOM_SEED)),
    ])


def decide(p_phishing, uncertain_below):
    """Return (predicted_class, confidence). Confidence is the probability of the chosen side."""
    confidence = max(p_phishing, 1.0 - p_phishing)
    if confidence < uncertain_below:
        return "uncertain", confidence
    return ("phishing" if p_phishing >= 0.5 else "legitimate"), confidence


@dataclass
class ModelBundle:
    pipeline: Pipeline
    version_name: str
    uncertain_below: float
    artifact_path: str

    def predict_proba(self, text):
        """Probability that the email is phishing."""
        return float(self.pipeline.predict_proba([text])[0][1])

    def predict_proba_many(self, texts):
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def explain(self, text, k=6):
        """Top features pushing toward phishing and toward legitimate for this email."""
        union = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        x = union.transform([text])
        tfidf = union.transformer_list[0][1]
        names = list(tfidf.get_feature_names_out())
        labels = names + [FEATURE_LABELS[n] for n in FEATURE_NAMES]
        contrib = np.asarray(x.multiply(clf.coef_[0]).todense()).ravel()
        order = np.argsort(contrib)
        toward_legit = [(labels[i], contrib[i]) for i in order[:k] if contrib[i] < 0]
        toward_phish = [(labels[i], contrib[i]) for i in order[::-1][:k] if contrib[i] > 0]
        return toward_phish, toward_legit


def save_model(pipeline, version_name, uncertain_below):
    config.ensure_dirs()
    path = config.MODELS_DIR / f"{version_name}.joblib"
    joblib.dump(pipeline, path)
    config.CURRENT_MODEL_FILE.write_text(json.dumps({
        "version_name": version_name,
        "artifact": path.name,
        "uncertain_below": uncertain_below,
    }, indent=2), encoding="utf-8")
    return path


def load_current():
    """Load the active model. Returns (bundle_or_None, reason_if_none)."""
    if not config.CURRENT_MODEL_FILE.exists():
        return None, "No trained model found (models/current.json is missing)."
    try:
        meta = json.loads(config.CURRENT_MODEL_FILE.read_text(encoding="utf-8"))
        path = config.MODELS_DIR / meta["artifact"]
        pipeline = joblib.load(path)
        return ModelBundle(pipeline, meta["version_name"],
                           float(meta["uncertain_below"]),
                           f"models/{meta['artifact']}"), None
    except Exception as exc:
        return None, f"The model file could not be loaded ({exc})."
