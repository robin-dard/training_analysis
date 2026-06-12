"""
sync/fetch_strava_activity_meta.py

Fetch activity metadata (date, sport, name) for all Strava activities
already in .strava_streams_state.json and save to data/strava_streams/metadata.json.

Safe to re-run — only fetches IDs not yet in metadata.json.
Also fetches any full years given as arguments.

Usage
-----
python sync/fetch_strava_activity_meta.py          # fill gaps only
python sync/fetch_strava_activity_meta.py 2018     # ensure all 2018 activities present
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from parsers.strava_client import StravaClient

_ROOT        = Path(__file__).resolve().parent.parent
STATE_FILE   = _ROOT / ".strava_streams_state.json"
META_FILE    = _ROOT / "data/strava_streams/metadata.json"

_SLEEP = 0.5


def _load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def _load_meta() -> dict[str, dict]:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}


def _save_meta(meta: dict) -> None:
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_year_list(client: StravaClient, year: int) -> list[dict]:
    """Fetch all activity summaries for a given year."""
    from_date = f"{year}-01-01"
    to_date   = f"{year}-12-31"
    activities, page = [], 1
    while True:
        batch = client.list_activities(after=from_date, before=to_date,
                                       per_page=100, page=page)
        if not batch:
            break
        activities.extend(batch)
        print(f"  page {page}: {len(batch)} activities")
        if len(batch) < 100:
            break
        page += 1
        time.sleep(_SLEEP)
    return activities


def run(years: list[int] | None = None) -> None:
    meta    = _load_meta()
    known   = _load_state()
    client  = StravaClient()

    # --- Year-level fetch (get all activity summaries for requested years) ---
    if years:
        for year in years:
            print(f"\nFetching {year} activity list ...")
            activities = _fetch_year_list(client, year)
            added = 0
            for a in activities:
                sid = str(a["id"])
                meta[sid] = {
                    "date":       a.get("start_date_local", "")[:10],
                    "sport_type": a.get("sport_type") or a.get("type", "unknown"),
                    "name":       a.get("name", ""),
                    "distance_m": a.get("distance", 0),
                }
                added += 1
            print(f"  {added} activities stored for {year}")

    # --- Gap fill: fetch individual summaries for known IDs not yet in meta ---
    missing = known - set(meta.keys())
    if missing:
        print(f"\nFetching metadata for {len(missing)} individual activities ...")
        for i, sid in enumerate(sorted(missing), 1):
            try:
                a = client.get_activity(int(sid))
                meta[sid] = {
                    "date":       (a.get("start_date_local") or "")[:10],
                    "sport_type": a.get("sport_type") or a.get("type", "unknown"),
                    "name":       a.get("name", ""),
                    "distance_m": a.get("distance", 0),
                }
                if i % 20 == 0:
                    _save_meta(meta)
                    print(f"  {i}/{len(missing)} ...")
            except Exception as e:
                print(f"  [skip] {sid}: {e}")
            time.sleep(_SLEEP)

    _save_meta(meta)
    print(f"\nSaved {len(meta)} entries to {META_FILE}")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else None
    run(years)
