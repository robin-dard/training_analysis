"""
sync/fetch_strava_training.py

Fetch all Strava activities for a date range and save their streams
as parquet files in data/strava_streams/.

Designed for years where no Garmin FIT files exist (e.g. 2018).

Re-running is safe — already fetched activities are skipped.
Respects Strava rate limit: 100 req / 15 min.

Usage
-----
python sync/fetch_strava_training.py              # defaults to 2018
python sync/fetch_strava_training.py 2017 2018    # custom range
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from parsers.strava_client import StravaClient

_ROOT        = Path(__file__).resolve().parent.parent
STREAMS_DIR  = _ROOT / "data/strava_streams"
STATE_FILE   = _ROOT / ".strava_streams_state.json"
META_FILE    = _ROOT / "data/strava_streams/metadata.json"

STREAMS_DIR.mkdir(parents=True, exist_ok=True)

# Strava → our column convention
_RENAME = {
    "velocity_smooth": "speed_ms",
    "grade_smooth":    "grade_pct",
    "time":            "elapsed_s",
    "distance":        "distance_m",
    "altitude":        "altitude_m",
    "heart_rate":      "heart_rate",
    "watts":           "power_w",
}

# Stay well under 100 req/15min
_SLEEP_BETWEEN = 1.0   # seconds between stream fetches
_SLEEP_PAGE    = 0.5   # seconds between activity list pages


def _load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def _save_state(ids: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(ids)))


def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}


def _save_meta(meta: dict) -> None:
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(from_date: str, to_date: str) -> None:
    downloaded = _load_state()
    client = StravaClient()

    # --- Fetch activity list ---
    print(f"Fetching activity list {from_date} → {to_date} ...")
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
        time.sleep(_SLEEP_PAGE)

    to_fetch = [a for a in activities if str(a["id"]) not in downloaded]
    print(f"\n{len(activities)} total — {len(to_fetch)} to fetch "
          f"({len(activities) - len(to_fetch)} already done)\n")

    # --- Fetch streams ---
    meta = _load_meta()
    for i, a in enumerate(to_fetch, 1):
        sid  = a["id"]
        date = a["start_date_local"][:10]
        name = a.get("name", "")
        out  = STREAMS_DIR / f"{sid}.parquet"

        print(f"[{i}/{len(to_fetch)}] {date} — {name} ...", end=" ", flush=True)
        try:
            df = client.streams_to_dataframe(sid)
            df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
            df.to_parquet(out, index=False)
            downloaded.add(str(sid))
            meta[str(sid)] = {
                "date":       date,
                "sport_type": a.get("sport_type") or a.get("type", "unknown"),
                "name":       name,
                "distance_m": a.get("distance", 0),
            }
            print(f"{len(df)} pts")
        except Exception as e:
            print(f"ERROR: {e}")

        # save state periodically so a crash doesn't lose progress
        if i % 20 == 0:
            _save_state(downloaded)
            _save_meta(meta)

        time.sleep(_SLEEP_BETWEEN)

    _save_state(downloaded)
    _save_meta(meta)
    print(f"\nDone — {len(to_fetch)} streams saved to {STREAMS_DIR}/")


if __name__ == "__main__":
    from_date = sys.argv[1] if len(sys.argv) > 1 else "2018-01-01"
    to_date   = sys.argv[2] if len(sys.argv) > 2 else "2018-12-31"
    fetch(from_date, to_date)
