#!/usr/bin/env python3
"""Keep the talk map current. Safe to run as often as you like.

    python3 update.py                    # new uploads, scored by the small classifier (no API key)
    python3 update.py --labeler sonnet   # new uploads and new topics, scored by Claude Sonnet (local)
    python3 update.py --refresh-views    # also refresh views of every stored talk (heavy)

Each run:
  1. Lists the channel through InnerTube (anonymous, no cookies), newest first, and stops
     at the first talk already stored: only new uploads are fetched. Details come from the
     player endpoint; when YouTube refuses it (it does for cloud runners), from the channel's
     public RSS feed, which covers the 15 latest uploads but has no length. Talks YouTube refused
     on an earlier run wait in `pending` and are retried. Talks from the last 30 days get
     fresh views, so a talk is not frozen at the view count of its first hours.
  2. Scores the new talks: with the small classifier by default (marked scored_by="model"),
     or with Sonnet when asked (scored_by="sonnet").
  3. With --labeler sonnet, also scores every talk for components not in `scored_topics`
     (a topic just added to the JSON). Then run train.py so the classifier learns it.
  4. Writes data/videos.json, the slim data/mapa.json the page reads, and data/vectors.i8
     (one MiniLM vector per talk, used by the page's "ask" search).

Exits with status 1 after writing when some talk is still refused or when the listing never
reached a stored talk (YouTube blocking or changing its pages), so a scheduled run that
silently stops adding talks shows up as a failed job.
"""

import argparse
import datetime
import sys
import time

RECENT_DAYS = 30  # talks this young get their views refreshed on every run
GONE = {"ERROR", "UNPLAYABLE"}  # removed or private: stop retrying
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import classifier
import dataset as ds
import innertube


def safe_details(video_id):
    """A network failure on one talk must not end the run: it is reported as refused."""
    try:
        return innertube.video_details(video_id)
    except Exception as e:
        print(f"warning: {video_id}: {e}", file=sys.stderr)
        return {"id": video_id, "title": "", "status": "NETWORK"}


def fetch_details(ids):
    """Fetch details, retrying the ones YouTube refused (they come back without a title).
    Returns (details by id, ids of talks that are gone: removed or private)."""
    details, pending, gone = {}, list(ids), set()
    for attempt in range(5):
        with ThreadPoolExecutor(max_workers=6) as pool:
            for d in pool.map(safe_details, pending):
                if d["title"]:
                    details[d["id"]] = d
                elif d["status"] in GONE:
                    gone.add(d["id"])
        pending = [i for i in pending if i not in details and i not in gone]
        if not pending or (not details and attempt >= 1):  # nothing gets through: stop asking
            break
        time.sleep(2 ** attempt)
    if gone:
        print(f"note: {len(gone)} talks removed or private, not retried: {sorted(gone)[:5]}", file=sys.stderr)
    if pending:
        print(f"warning: {len(pending)} talks still without details, skipped: {pending[:5]}", file=sys.stderr)
    return details, gone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeler", choices=["model", "sonnet"], default="model")
    parser.add_argument("--refresh-views", action="store_true", help="re-fetch views of every stored talk")
    args = parser.parse_args()

    today = datetime.date.today().isoformat()
    dataset = ds.load()
    components = dataset["components"]
    known = {ds.video_id(v): v for v in dataset["videos"]}
    handle = dataset["channel"].rsplit("/", 1)[-1]

    channel_id, listed, reached = innertube.list_video_ids(handle, stop_at=frozenset(known))
    new_ids = list(dict.fromkeys(listed + [i for i in dataset.get("pending", []) if i not in known]))
    details, gone = fetch_details(new_ids)
    gone -= set(listed)  # a talk on the channel's videos tab right now is public, whatever player says
    try:
        feed = innertube.feed_details(channel_id)
    except Exception as e:  # the feed is only a fallback
        feed = {}
        print(f"warning: RSS feed unavailable: {e}", file=sys.stderr)
    for i in new_ids:
        if i not in details and i in feed:
            details[i] = feed[i]  # no length until a run where the player answers
    dataset["pending"] = [i for i in new_ids if i not in details and i not in gone]
    new_videos = [{"url": d["url"], "title": d["title"], "published": d["published"], "length_s": d["length_s"],
                   "views": d["views"], "scores": {}, "keywords": [], "chapters": d["chapters"],
                   "description": d["description"], "scored_by": args.labeler, "views_date": today}
                  for d in (details.get(i) for i in new_ids) if d]
    print(f"{len(new_ids)} new since {dataset['collected']}, {len(new_videos)} fetched", file=sys.stderr)

    cutoff = (datetime.date.today() - datetime.timedelta(days=RECENT_DAYS)).isoformat()
    refresh = list(known) if args.refresh_views else [i for i, v in known.items() if v["published"] >= cutoff]
    fresh = fetch_details(refresh)[0]
    fresh.update({i: feed[i] for i in refresh if i not in fresh and i in feed})
    for i, d in fresh.items():
        known[i].update(views=d["views"], views_date=today, length_s=d["length_s"] or known[i]["length_s"],
                        published=d["published"] or known[i]["published"])

    vectors = ds.stored_vectors()
    missing = [v for v in new_videos + list(known.values()) if ds.video_id(v) not in vectors]
    if missing:
        vectors.update(zip([ds.video_id(v) for v in missing], classifier.embed([ds.text_of(v) for v in missing])))

    unscored = [c for c in components if c["id"] not in dataset["scored_topics"]]
    if args.labeler == "sonnet":
        import sonnet
        sonnet.score(new_videos, components)
        sonnet.score(list(known.values()), unscored)
        dataset["scored_topics"] = [c["id"] for c in components]
    else:
        if unscored:
            print(f"note: {[c['id'] for c in unscored]} have no labels yet; run with --labeler sonnet, then train.py",
                  file=sys.stderr)
        if new_videos:
            rows = np.stack([vectors[ds.video_id(v)] for v in new_videos])
            classifier.predict(classifier.load(), new_videos, rows, components)

    # Newest first, as the channel lists them.
    dataset["videos"] = sorted(new_videos + list(known.values()), key=lambda v: v["published"], reverse=True)
    # Views were refreshed on this run, so the page's age and traction reference is today.
    dataset["collected"] = today
    ds.write_outputs(dataset, vectors)
    print(f"wrote {len(dataset['videos'])} talks", file=sys.stderr)
    problems = []
    if dataset["pending"]:
        problems.append(f"{len(dataset['pending'])} talks refused by YouTube, kept in `pending` for the next run")
    if refresh and not fresh:
        problems.append(f"none of the {len(refresh)} stored talks got fresh views")
    if known and not reached:
        problems.append("the channel listing never reached a stored talk; YouTube may be blocking or changed its pages")
    if problems:
        sys.exit("error: " + "; ".join(problems))


if __name__ == "__main__":
    main()
