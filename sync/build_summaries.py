"""
sync/build_summaries.py

Build data/summaries.parquet — one row per activity, all sources.

Sources
-------
  - Garmin FIT files  → data/fit/*.fit       (uses fit_metadata, fast)
  - Strava streams    → data/strava_streams/  (aggregated from parquet)

Re-running rebuilds from scratch (fast — no stream parsing).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from parsers.fit_parser import fit_metadata

_WORKERS = max(1, (os.cpu_count() or 4) - 1)

FIT_DIR      = Path("data/fit")
STREAMS_DIR  = Path("data/strava_streams")
OUT_FILE     = Path("data/summaries.parquet")

# Map Garmin FIT sport enum to readable string
_SPORT_MAP = {
    "running":          "running",
    "trail_running":    "trail_running",
    "cycling":          "cycling",
    "swimming":         "swimming",
    "transition":       "transition",
}


def _parse_one(f: Path) -> dict | None:
    try:
        m = fit_metadata(f)
        parts = f.stem.split("_")
        date_str  = parts[0]
        sport_str = "_".join(parts[1:-1]) if len(parts) > 2 else (m.get("sport") or "unknown")
        return {
            "source":       "garmin",
            "file":         f.name,
            "date":         date_str,
            "sport":        sport_str,
            "distance_km":  round((m.get("total_distance_m") or 0) / 1000, 2),
            "duration_s":   m.get("total_elapsed_time_s"),
            "dplus_m":      m.get("total_ascent_m"),
            "dminus_m":     m.get("total_descent_m"),
            "avg_hr":       m.get("avg_heart_rate"),
            "max_hr":       m.get("max_heart_rate"),
            "avg_speed_ms": m.get("avg_speed_ms"),
            "calories":     m.get("total_calories"),
            "start_time":   str(m.get("start_time") or ""),
        }
    except Exception as e:
        print(f"  [skip] {f.name}: {e}")
        return None


def _garmin_rows() -> list[dict]:
    files = sorted(FIT_DIR.glob("*.fit"))
    print(f"Parsing {len(files)} FIT files ({_WORKERS} workers) ...")

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_parse_one, f): f for f in files}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(files)}")
            result = fut.result()
            if result:
                rows.append(result)

    return rows


def _strava_rows() -> list[dict]:
    files = sorted(STREAMS_DIR.glob("*.parquet"))
    if not files:
        return []

    print(f"Aggregating {len(files)} Strava stream files ...")
    rows = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue

            dplus = 0.0
            if "altitude_m" in df.columns:
                alt_diff = df["altitude_m"].diff().clip(lower=0)
                dplus = alt_diff.sum()

            rows.append({
                "source":       "strava",
                "file":         f.name,
                "date":         "",        # filled from races.json if needed
                "sport":        "unknown",
                "distance_km":  round(df["distance_m"].max() / 1000, 2) if "distance_m" in df.columns else None,
                "duration_s":   df["elapsed_s"].max() if "elapsed_s" in df.columns else None,
                "dplus_m":      round(dplus, 0),
                "dminus_m":     None,
                "avg_hr":       round(df["heart_rate"].mean(), 1) if "heart_rate" in df.columns else None,
                "max_hr":       df["heart_rate"].max() if "heart_rate" in df.columns else None,
                "avg_speed_ms": df["speed_ms"].mean() if "speed_ms" in df.columns else None,
                "calories":     None,
                "start_time":   "",
            })
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")

    return rows


def _enrich_strava(rows: list[dict]) -> list[dict]:
    """Fill date + sport for Strava rows from metadata.json, then races.json fallback."""
    import json

    # Primary: metadata.json with date + sport for all downloaded activities
    meta: dict = {}
    meta_file = Path("data/strava_streams/metadata.json")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # Fallback: races.json (covers races even without metadata.json)
    races_file = Path("data/races/races.json")
    stream_to_race: dict = {}
    if races_file.exists():
        races = json.loads(races_file.read_text(encoding="utf-8"))
        stream_to_race = {
            r["strava_stream_file"]: r
            for r in races
            if r.get("strava_stream_file")
        }

    for row in rows:
        sid = row["file"].replace(".parquet", "")
        m = meta.get(sid)
        if m:
            row["date"]  = m.get("date", "")
            row["sport"] = m.get("sport_type", "unknown")
        else:
            race = stream_to_race.get(row["file"])
            if race:
                row["date"]  = race.get("date", "")
                row["sport"] = race.get("sport_type", "unknown")

    return rows


def build() -> None:
    garmin = _garmin_rows()
    strava = _strava_rows()
    strava = _enrich_strava(strava)

    all_rows = garmin + strava
    df = pd.DataFrame(all_rows)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved {len(df)} activities to {OUT_FILE}")
    print(f"  Garmin : {len(garmin)}")
    print(f"  Strava : {len(strava)}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor
    from multiprocessing import freeze_support
    freeze_support()
    build()
