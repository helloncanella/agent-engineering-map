#!/usr/bin/env python3
"""Read a YouTube channel through InnerTube (youtubei/v1), anonymously.

    innertube.py <@handle> <out.json>   # dump every video of a channel

For each video: id, title, publish date, length, views, description and the
chapters parsed from the description timestamps. No cookies, no account.
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://www.youtube.com/youtubei/v1"
CONTEXT = {"client": {"clientName": "WEB", "clientVersion": "2.20251020.01.00", "hl": "en", "gl": "US"}}
VIDEOS_TAB = "EgZ2aWRlb3PyBgQKAjoA"
CHAPTER_RE = re.compile(r"^\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{2})\)?\s*[-–—:|]?\s*(.+)$")


def post(endpoint, body, tries=4):
    data = json.dumps({"context": CONTEXT, **body}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(f"{BASE}/{endpoint}?prettyPrint=false", data=data,
                                         headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def walk(node, key):
    """Yield every value stored under `key` anywhere in a nested JSON tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from walk(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item, key)


def list_video_ids(handle, stop_at=frozenset()):
    """Channel uploads, newest first. Stops paging at the first id in `stop_at`,
    so an incremental run only lists what was published since the last one.
    Returns (browse_id, ids, reached), where `reached` says whether a stored id was seen."""
    resolved = post("navigation/resolve_url", {"url": f"https://www.youtube.com/{handle}"})
    browse_id = next(walk(resolved, "browseId"))
    page = post("browse", {"browseId": browse_id, "params": VIDEOS_TAB})
    ids = []
    while True:
        page_ids = [r["videoId"] for r in walk(page, "videoRenderer")]
        for renderer in walk(page, "richItemRenderer"):
            for vid in walk(renderer, "videoId"):
                page_ids.append(vid)
                break
        for vid in dict.fromkeys(page_ids):
            if vid in stop_at:
                return browse_id, list(dict.fromkeys(ids)), True
            ids.append(vid)
        tokens = [c["continuationCommand"]["token"] for c in walk(page, "continuationEndpoint") if "continuationCommand" in c]
        if not tokens:
            break
        page = post("browse", {"continuation": tokens[0]})
    return browse_id, list(dict.fromkeys(ids)), False


def chapters_from(description):
    chapters = []
    for line in description.splitlines():
        match = CHAPTER_RE.match(line)
        if match:
            chapters.append({"t": match.group(1), "title": match.group(2).strip()})
    return chapters if len(chapters) >= 2 else []


def video_details(video_id):
    player = post("player", {"videoId": video_id})
    details = player.get("videoDetails", {})
    micro = player.get("microformat", {}).get("playerMicroformatRenderer", {})
    description = details.get("shortDescription", "")
    return {
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": details.get("title", ""),
        "published": micro.get("publishDate", "")[:10],
        "length_s": int(details.get("lengthSeconds") or 0),
        "views": int(details.get("viewCount") or 0),
        "description": description,
        "chapters": chapters_from(description),
        # OK, LOGIN_REQUIRED (bot check), ERROR (removed), UNPLAYABLE (private)...
        "status": player.get("playabilityStatus", {}).get("status", ""),
    }


def main():
    handle, out = sys.argv[1], sys.argv[2]
    browse_id, ids, _ = list_video_ids(handle)
    print(f"{handle} ({browse_id}): {len(ids)} videos", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=6) as pool:
        videos = list(pool.map(video_details, ids))
    json.dump({"channel": handle, "browse_id": browse_id, "videos": videos}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {len(videos)} to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
