"""Small classifier distilled from the large model's scores: TF-IDF + MiniLM embeddings,
one logistic regression per component. Runs on a CPU in seconds, no API key."""

import re

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import dataset as ds

MIN_FIT = 5        # a score of 5 or more means "fits the component"
MIN_POSITIVES = 5  # components with fewer labelled positives are not trained


def embed(texts):
    """float32 MiniLM vectors, normalized, then int8-quantized like the stored ones."""
    from fastembed import TextEmbedding
    vectors = np.array(list(TextEmbedding(ds.EMBEDDING_MODEL).embed(texts, batch_size=32)))
    return ds.quantize(vectors)


def features(vectorizer, texts, rows):
    return hstack([vectorizer.transform(texts), csr_matrix(ds.dequantize(rows))]).tocsr()


def fit(texts, rows, scores, component_ids):
    """scores: array (videos x components) of 0-10 labels."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000).fit(texts)
    X = features(vectorizer, texts, rows)
    heads = {}
    for j, cid in enumerate(component_ids):
        y = (scores[:, j] >= MIN_FIT).astype(int)
        if y.sum() >= MIN_POSITIVES:
            heads[cid] = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000).fit(X, y)
    return {"vectorizer": vectorizer, "heads": heads, "embedding": ds.EMBEDDING_MODEL}


def to_score(p):
    """Probability -> 0-10 score, with the 0.5 decision boundary landing exactly on MIN_FIT."""
    if p >= 0.5:
        return int(min(10, MIN_FIT + round((10 - MIN_FIT) * (p - 0.5) / 0.5)))
    return int(min(MIN_FIT - 1, round(10 * p)))


def predict(model, videos, rows, components):
    """Fill `scores` of each video for every component the model has a head for."""
    if not videos:
        return
    X = features(model["vectorizer"], [ds.text_of(v) for v in videos], rows)
    for c in components:
        head = model["heads"].get(c["id"])
        rule = c["classifier"]
        probs = head.predict_proba(X)[:, 1] if head else np.zeros(len(videos))
        pattern = re.compile(rule["candidates"], re.I) if rule["mode"] == "candidates" else None
        title_rule = re.compile(rule["title_max"], re.I) if rule.get("title_max") else None
        for v, p in zip(videos, probs):
            hay = " ".join([v["title"], v["description"], " ".join(v.get("keywords", [])),
                            " ".join(ch["title"] for ch in v["chapters"])])
            score = to_score(p) if head and (not pattern or pattern.search(hay)) else 0
            if title_rule and title_rule.search(v["title"]):
                score = 10
            if score >= ds.MIN_STORED:
                v["scores"][c["id"]] = score


def save(model):
    ds.MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, ds.MODEL_FILE, compress=3)


def load():
    return joblib.load(ds.MODEL_FILE)
