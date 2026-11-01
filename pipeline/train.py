#!/usr/bin/env python3
"""Train the small classifier on the teacher's (Sonnet) labels and measure it.

    python3 train.py

1. 5-fold cross-validation: each fold trains on 80% and predicts the held-out 20%.
   Per component it reports precision, recall, F1 and "top 20" (share of the 20
   highest-ranked held-out talks that the teacher also rates as a fit).
2. Fits the final model on every teacher-labelled talk and saves it to model/.
"""

import datetime
import json
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

import classifier
import dataset as ds


def labelled(dataset):
    """Talks scored by the teacher; the classifier's own guesses are never used as labels."""
    return [v for v in dataset["videos"] if v.get("scored_by", "sonnet") == "sonnet"]


def cross_validate(texts, rows, scores, ids):
    oof = np.full(scores.shape, np.nan)
    for train, test in KFold(5, shuffle=True, random_state=0).split(texts):
        model = classifier.fit([texts[i] for i in train], rows[train], scores[train], ids)
        X = classifier.features(model["vectorizer"], [texts[i] for i in test], rows[test])
        for j, cid in enumerate(ids):
            if cid in model["heads"]:
                oof[test, j] = model["heads"][cid].predict_proba(X)[:, 1]
    report = {}
    for j, cid in enumerate(ids):
        if np.isnan(oof[:, j]).any():
            continue
        y, p = (scores[:, j] >= classifier.MIN_FIT).astype(int), oof[:, j]
        pred = p >= 0.5
        report[cid] = {"positives": int(y.sum()),
                       "precision": round(precision_score(y, pred, zero_division=0), 3),
                       "recall": round(recall_score(y, pred, zero_division=0), 3),
                       "f1": round(f1_score(y, pred, zero_division=0), 3),
                       "top20": round(float(y[np.argsort(-p)[:20]].mean()), 3)}
    macro = {k: round(float(np.mean([r[k] for r in report.values()])), 3) for k in ("precision", "recall", "f1", "top20")}
    return {"macro": macro, "components": report}


def main():
    dataset = ds.load()
    videos = labelled(dataset)
    ids = [c["id"] for c in dataset["components"]]
    vectors = ds.stored_vectors()
    missing = [v for v in videos if ds.video_id(v) not in vectors]
    if missing:
        vectors.update(zip([ds.video_id(v) for v in missing], classifier.embed([ds.text_of(v) for v in missing])))
    texts = [ds.text_of(v) for v in videos]
    rows = np.stack([vectors[ds.video_id(v)] for v in videos])
    scores = np.array([[v["scores"].get(c, 0) for c in ids] for v in videos], dtype=float)

    print(f"cross-validating on {len(videos)} teacher-labelled talks", file=sys.stderr)
    metrics = cross_validate(texts, rows, scores, ids)
    print(json.dumps(metrics["macro"]), file=sys.stderr)
    model = classifier.fit(texts, rows, scores, ids)
    classifier.save(model)
    metrics.update({"trained": datetime.date.today().isoformat(), "talks": len(videos), "embedding": ds.EMBEDDING_MODEL})
    ds.METRICS.write_text(json.dumps(metrics, indent=1))
    print(f"saved {ds.MODEL_FILE.name} ({len(model['heads'])} components) and {ds.METRICS.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
