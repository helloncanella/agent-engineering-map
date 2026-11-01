"""Paths and helpers for the dataset, the slim page copy and the search vectors."""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "videos.json"     # source of truth: every field of every talk
PAGE_DATA = ROOT / "data" / "mapa.json"     # slim copy the page reads
VECTORS = ROOT / "data" / "vectors.i8"      # one int8 embedding row per video, in PAGE_DATA order
MODEL_FILE = ROOT / "model" / "classifier.joblib"
METRICS = ROOT / "model" / "metrics.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BROWSER_MODEL = "Xenova/all-MiniLM-L6-v2"   # same weights, ONNX build for transformers.js
DIM = 384
MIN_STORED = 4                              # scores below this are not kept


def video_id(video):
    return video["url"].split("=")[1]


def text_of(video):
    """What both classifiers read: title, description and chapter titles."""
    chapters = "; ".join(c["title"] for c in video["chapters"])
    return f"{video['title']}. {video['description'][:900]} Chapters: {chapters[:400]}"


def load():
    return json.loads(DATASET.read_text())


def write_atomic(path, data):
    """Write to a temp file and swap it in, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    (tmp.write_bytes if isinstance(data, bytes) else tmp.write_text)(data)
    os.replace(tmp, path)


def save(dataset):
    write_atomic(DATASET, json.dumps(dataset, ensure_ascii=False, indent=1))


def stored_vectors():
    """Vectors from the last run, keyed by video id (empty on the first run)."""
    if not (VECTORS.exists() and PAGE_DATA.exists()):
        return {}
    ids = [v["id"] for v in json.loads(PAGE_DATA.read_text())["videos"]]
    rows = np.fromfile(VECTORS, dtype=np.int8).reshape(-1, DIM)
    return dict(zip(ids, rows)) if len(ids) == len(rows) else {}


def quantize(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.clip(np.round(vectors / np.maximum(norm, 1e-9) * 127), -127, 127).astype(np.int8)


def dequantize(rows):
    return rows.astype(np.float32) / 127


def write_outputs(dataset, vectors_by_id):
    """Write the full dataset, the slim page copy and the vectors in the same order."""
    save(dataset)
    slim = {k: dataset[k] for k in ("channel", "collected", "sources", "ranking")}
    slim["components"] = [{k: c[k] for k in ("id", "layer", "name", "desc")} for c in dataset["components"]]
    slim["search"] = {"model": BROWSER_MODEL, "dim": DIM, "vectors": VECTORS.name}
    slim["videos"] = []
    for v in dataset["videos"]:
        row = {"id": video_id(v), "t": v["title"], "d": v["published"], "s": v["length_s"], "v": v["views"], "vd": v["views_date"],
               "k": v["keywords"], "c": v["scores"], "ch": [c["title"] for c in v["chapters"]][:12]}
        if v.get("scored_by") == "model":
            row["e"] = 1
        slim["videos"].append(row)
    # Vectors first, then the page data: the page checks that both have the same row count.
    write_atomic(VECTORS, np.stack([vectors_by_id[video_id(v)] for v in dataset["videos"]]).astype(np.int8).tobytes())
    write_atomic(PAGE_DATA, json.dumps(slim, ensure_ascii=False, separators=(",", ":")))
