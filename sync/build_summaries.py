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
from tqdm import tqdm

from parsers.fit_parser import fit_metadata

_WORKERS = max(1, (os.cpu_count() or 4) - 1)
_ROOT    = Path(__file__).resolve().parent.parent

FIT_DIR      = _ROOT / "data/fit"
STREAMS_DIR  = _ROOT / "data/strava_streams"
OUT_FILE     = _ROOT / "data/summaries.parquet"

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


def _garmin_rows(known_files: set[str]) -> list[dict]:
    all_files = sorted(FIT_DIR.glob("*.fit"))
    new_files = [f for f in all_files if f.name not in known_files]

    if not new_files:
        print("FIT files: 0 new files.")
        return []

    rows = []
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_parse_one, f): f for f in new_files}
        with tqdm(as_completed(futures), total=len(new_files), desc=f"FIT files (+{len(new_files)} new)", unit="file") as bar:
            for fut in bar:
                result = fut.result()
                if result:
                    rows.append(result)
    return rows


def _strava_rows(known_files: set[str]) -> list[dict]:
    all_files = sorted(STREAMS_DIR.glob("*.parquet"))
    files = [f for f in all_files if f.name not in known_files]
    if not files:
        print("Strava streams: 0 new files.")
        return []

    rows = []

    for f in tqdm(files, desc=f"Strava streams (+{len(files)} new)", unit="file"):
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
    meta_file = _ROOT / "data/strava_streams/metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # Fallback: races.json (covers races even without metadata.json)
    races_file = _ROOT / "data/races/races.json"
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


def build(rebuild: bool = False) -> None:
    import time
    t0 = time.perf_counter()

    existing = pd.DataFrame()
    known_files: set[str] = set()
    if not rebuild and OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
        known_files = set(existing["file"].dropna())
        print(f"Existing parquet: {len(existing)} activities — checking for new files...")

    garmin = _garmin_rows(known_files)
    t1 = time.perf_counter()

    strava = _strava_rows(known_files)
    strava = _enrich_strava(strava)
    t2 = time.perf_counter()

    new_rows = garmin + strava
    if not new_rows and not existing.empty:
        print("Nothing new — parquet is up to date.")
        return

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(OUT_FILE, index=False)
    t3 = time.perf_counter()

    print(f"\nSaved {len(df)} activities to {OUT_FILE} (+{len(new_rows)} new)")
    print(f"  Garmin new : {len(garmin)}")
    print(f"  Strava new : {len(strava)}")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nTimings:")
    print(f"  FIT parsing : {t1 - t0:.1f}s")
    print(f"  Strava      : {t2 - t1:.1f}s")
    print(f"  Write       : {t3 - t2:.1f}s")
    print(f"  Total       : {t3 - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing parquet and rebuild from scratch")
    args = parser.parse_args()
    build(rebuild=args.rebuild)
