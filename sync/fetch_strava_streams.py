"""
sync/fetch_strava_streams.py

For races in data/races/races.json that have no FIT file,
fetch streams from Strava and save as parquet in data/strava_streams/.

Re-running is safe — already fetched streams are skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from parsers.strava_client import StravaClient

RACES_FILE    = Path("data/races/races.json")
STREAMS_DIR   = Path("data/strava_streams")

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


def _stream_path(strava_id: int | str) -> Path:
    return STREAMS_DIR / f"{strava_id}.parquet"


def fetch_missing(dry_run: bool = False) -> None:
    races = json.loads(RACES_FILE.read_text())
    targets = [r for r in races if not r.get("fit_file") and r.get("strava_id")]

    if not targets:
        print("No races missing FIT — nothing to fetch.")
        return

    print(f"{len(targets)} race(s) without FIT file:")
    client = StravaClient()
    updated = 0

    for r in targets:
        sid = r["strava_id"]
        out = _stream_path(sid)

        if out.exists():
            print(f"  [skip] {r['date']} — {r['name']} (already fetched)")
            if not r.get("strava_stream_file"):
                r["strava_stream_file"] = out.name
                updated += 1
            continue

        print(f"  [fetch] {r['date']} — {r['name']} ({r['distance_km']}km) ...", end=" ")
        try:
            df = client.streams_to_dataframe(sid)
            df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
            if not dry_run:
                df.to_parquet(out, index=False)
            r["strava_stream_file"] = out.name
            updated += 1
            print(f"{len(df)} pts → {out.name}")
            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR: {e}")

    if not dry_run and updated:
        RACES_FILE.write_text(json.dumps(races, indent=2, ensure_ascii=False))
        print(f"\nUpdated races.json with strava_stream_file references.")


if __name__ == "__main__":
    fetch_missing()
