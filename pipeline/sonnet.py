"""Optional teacher: scores talks with Claude Sonnet through the `claude` CLI.

Used locally to label a new topic or to refresh labels before retraining the small
classifier. The scheduled update never calls it.
"""

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import dataset as ds

BATCH = 40


def model_input(video):
    return {"id": ds.video_id(video), "title": video["title"], "description": video["description"][:900],
            "chapters": "; ".join(c["title"] for c in video["chapters"])[:400]}


def ask(prompt, items):
    """Run one batch; retry until every id comes back."""
    wanted = {x["id"] for x in items}
    for _ in range(3):
        result = subprocess.run(["claude", "-p", "--model", "sonnet", prompt], input=json.dumps(items, ensure_ascii=False),
                                capture_output=True, text=True, timeout=600)
        text = result.stdout
        try:
            rows = json.loads(text[text.find("["):text.rfind("]") + 1])
            if {r["id"] for r in rows} >= wanted:
                return rows
        except ValueError:
            pass
    raise RuntimeError(f"model batch failed: {result.stderr[:300] or text[:300]}")


def in_batches(prompt, videos):
    items = [model_input(v) for v in videos]
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        return [row for rows in pool.map(lambda b: ask(prompt, b), batches) for row in rows]


def score_all_mode(videos, components):
    """One call per batch covers every 'all' component, plus keywords."""
    listing = "\n".join(f"{c['id']}: {c['classifier']['criterion']}" for c in components)
    prompt = f"""You classify conference talk videos from the "AI Engineer" YouTube channel into agent-engineering components.

Components:
{listing}

For each video in the JSON array on stdin, output how well the talk's MAIN subject teaches each component, as an integer 0-10 (10 = the talk is squarely about it; 0 = unrelated). Only list components scoring >= {ds.MIN_STORED}. A talk can match several. Talks not about building with LLMs/agents (e.g. pure business, hiring, hardware, generic ML) get an empty list.
Also give 3-6 short English keywords for the talk.

Output ONLY a JSON array, no prose, no code fence:
[{{"id": "<video id>", "scores": {{"<component id>": 8}}, "keywords": ["..."]}}]
Include every input id exactly once."""
    by_id = {ds.video_id(v): v for v in videos}
    ids = {c["id"] for c in components}
    for row in in_batches(prompt, videos):
        video = by_id[row["id"]]
        for cid, score in (row.get("scores") or {}).items():
            if cid in ids and score >= ds.MIN_STORED:
                video["scores"][cid] = score
        if row.get("keywords"):
            video["keywords"] = row["keywords"]


def score_candidates_mode(videos, component):
    rule = component["classifier"]
    pattern = re.compile(rule["candidates"], re.I)
    candidates = [v for v in videos if pattern.search(" ".join(
        [v["title"], v["description"], " ".join(v.get("keywords", [])), " ".join(c["title"] for c in v["chapters"])]))]
    if candidates:
        prompt = (f"Score how much each conference talk's MAIN subject is about {rule['criterion']}. "
                  "Integer 0-10 (10 = squarely about it). "
                  'Output ONLY a JSON array: [{"id": "...", "score": 7}] with every input id once.')
        by_id = {ds.video_id(v): v for v in candidates}
        for row in in_batches(prompt, candidates):
            if row["score"] >= ds.MIN_STORED:
                by_id[row["id"]]["scores"][component["id"]] = row["score"]
    if rule.get("title_max"):
        title_rule = re.compile(rule["title_max"], re.I)
        for v in videos:
            if title_rule.search(v["title"]):
                v["scores"][component["id"]] = 10
    print(f"  {component['id']}: {len(candidates)} candidates", file=sys.stderr)


def score(videos, components):
    if not videos or not components:
        return
    print(f"sonnet: scoring {len(videos)} videos for {[c['id'] for c in components]}", file=sys.stderr)
    all_mode = [c for c in components if c["classifier"]["mode"] == "all"]
    if all_mode:
        score_all_mode(videos, all_mode)
    for c in components:
        if c["classifier"]["mode"] == "candidates":
            score_candidates_mode(videos, c)
