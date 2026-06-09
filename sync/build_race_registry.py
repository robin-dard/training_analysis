"""
sync/build_race_registry.py

Fetch races from Strava (workout_type=1), match to local FIT files,
and write stubs to data/races.json for manual classification.

Run once, then edit data/races.json to fill in:
  - "type": "objective" or "preparatory"
  - "score": 0–4 (objectives only, null for preparatory)

Re-running is safe — existing entries are preserved, only new races are added.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from parsers.strava_client import StravaClient

RACES_FILE = Path("data/races/races.json")
FIT_DIR    = Path("data/fit")


def _load_registry() -> list[dict]:
    if RACES_FILE.exists():
        return json.loads(RACES_FILE.read_text())
    return []


def _save_registry(races: list[dict]) -> None:
    RACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    races = sorted(races, key=lambda r: r.get("date", ""))
    RACES_FILE.write_text(json.dumps(races, indent=2, ensure_ascii=False))


def _fit_files_by_date() -> dict[str, list[Path]]:
    """Index FIT files by date string YYYY-MM-DD."""
    index: dict[str, list[Path]] = {}
    for f in FIT_DIR.glob("*.fit"):
        date_str = f.name[:10]
        index.setdefault(date_str, []).append(f)
    return index


def _find_fit(date_str: str, fit_index: dict[str, list[Path]]) -> str | None:
    """Return best matching FIT filename for a date, preferring running sports."""
    candidates = fit_index.get(date_str, [])
    if not candidates:
        return None
    running = [f for f in candidates if "running" in f.name or "trail" in f.name]
    best = running[0] if running else candidates[0]
    return best.name


def _fetch_all_races(client: StravaClient) -> list[dict]:
    """Fetch all Strava activities with workout_type=1 (race) since 2019."""
    races, page = [], 1
    while True:
        batch = client.list_activities(after="2018-01-01", per_page=100, page=page)
        if not batch:
            break
        for a in batch:
            if a.get("workout_type") == 1:
                races.append(a)
        if len(batch) < 100:
            break
        page += 1
    return races


def build(dry_run: bool = False) -> None:
    registry = _load_registry()
    existing_ids = {str(r["strava_id"]) for r in registry}
    fit_index = _fit_files_by_date()

    client = StravaClient()
    strava_races = _fetch_all_races(client)
    print(f"Found {len(strava_races)} races on Strava")

    new_count = 0
    for a in sorted(strava_races, key=lambda x: x["start_date_local"]):
        sid = str(a["id"])
        if sid in existing_ids:
            continue

        date_str = a["start_date_local"][:10]
        fit_file = _find_fit(date_str, fit_index)

        entry = {
            "strava_id": a["id"],
            "name": a.get("name", ""),
            "date": date_str,
            "sport_type": a.get("sport_type") or a.get("type", ""),
            "distance_km": round((a.get("distance") or 0) / 1000, 1),
            "dplus_m": round(a.get("total_elevation_gain") or 0, 0),
            "duration_s": a.get("moving_time"),
            "fit_file": fit_file,
            "type": None,    # fill in: "objective" or "preparatory"
            "score": None,   # fill in: 0–4 for objectives, null for preparatory
            "notes": "",
        }

        registry.append(entry)
        existing_ids.add(sid)
        new_count += 1
        print(f"  + {date_str} — {entry['name']} ({entry['distance_km']}km, "
              f"D+{entry['dplus_m']:.0f}m)"
              + (f" → {fit_file}" if fit_file else " → no FIT match"))

    if not dry_run:
        _save_registry(registry)
        print(f"\n{new_count} new race(s) added → {RACES_FILE}")
    else:
        print(f"\nDry run — {new_count} race(s) would be added")


if __name__ == "__main__":
    build()
