"""
analysis/build_taper_pattern.py

Compute weekly build and daily taper stats for a race
from data/summaries.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

SUMMARIES_FILE = Path("data/summaries.parquet")

# Sport classification — covers both Garmin typeKeys and Strava sport_type
_TRAIL = {
    "trail_running", "running", "track_running", "treadmill_running",
    "Run", "TrailRun", "VirtualRun",
}
_BIKE = {
    "road_biking", "cycling", "mountain_biking", "indoor_cycling",
    "virtual_ride", "gravel_cycling",
    "Ride", "VirtualRide", "GravelRide", "MountainBikeRide", "EBikeRide",
}
_SKI = {
    "backcountry_skiing", "resort_skiing", "nordic_skiing",
    "AlpineSki", "BackcountrySki", "NordicSki",
}


def _classify(sport: str) -> str:
    if sport in _TRAIL:
        return "trail"
    if sport in _BIKE:
        return "bike"
    if sport in _SKI:
        return "ski"
    return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeeklyStats:
    week_offset: int       # -8 = 8 weeks before race, -1 = week before race
    trail_km: float = 0.0
    trail_dplus: float = 0.0
    trail_h: float = 0.0
    bike_km: float = 0.0
    bike_h: float = 0.0
    ski_h: float = 0.0
    total_h: float = 0.0
    n_sessions: int = 0


@dataclass
class DailyStats:
    day_offset: int        # -21 = 21 days before race, -1 = day before
    trail_km: float = 0.0
    trail_dplus: float = 0.0
    trail_h: float = 0.0
    bike_km: float = 0.0
    bike_h: float = 0.0
    n_sessions: int = 0


@dataclass
class RaceWindow:
    race_name: str
    race_date: date
    score: Optional[int]
    distance_km: float
    dplus_m: float
    build: list[WeeklyStats] = field(default_factory=list)
    taper: list[DailyStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_race_window(
    summaries: pd.DataFrame,
    race: dict,
    n_build_weeks: int = 8,
    n_taper_days: int = 21,
) -> RaceWindow:
    race_dt  = pd.Timestamp(race["date"])
    race_d   = race_dt.date()
    window   = RaceWindow(
        race_name=race["name"],
        race_date=race_d,
        score=race.get("score"),
        distance_km=race.get("distance_km", 0),
        dplus_m=race.get("dplus_m", 0),
    )

    build_start = race_dt - pd.Timedelta(weeks=n_build_weeks)
    sub = summaries[
        (summaries["date"] >= build_start) &
        (summaries["date"] <  race_dt)
    ].copy()
    sub["cat"]         = sub["sport"].apply(_classify)
    sub["days_before"] = (race_dt - sub["date"]).dt.days
    sub["week_offset"] = -((sub["days_before"] - 1) // 7 + 1)

    # --- Weekly build ---
    for w in range(-n_build_weeks, 0):
        wdf   = sub[sub["week_offset"] == w]
        trail = wdf[wdf["cat"] == "trail"]
        bike  = wdf[wdf["cat"] == "bike"]
        ski   = wdf[wdf["cat"] == "ski"]
        ws = WeeklyStats(
            week_offset=w,
            trail_km    = trail["distance_km"].sum(),
            trail_dplus = trail["dplus_m"].fillna(0).sum(),
            trail_h     = trail["duration_s"].fillna(0).sum() / 3600,
            bike_km     = bike["distance_km"].sum(),
            bike_h      = bike["duration_s"].fillna(0).sum() / 3600,
            ski_h       = ski["duration_s"].fillna(0).sum() / 3600,
        )
        ws.total_h     = ws.trail_h + ws.bike_h + ws.ski_h
        ws.n_sessions  = len(wdf[wdf["cat"] != "other"])
        window.build.append(ws)

    # --- Daily taper ---
    for d in range(-n_taper_days, 0):
        ddf   = sub[sub["days_before"] == -d]
        trail = ddf[ddf["cat"] == "trail"]
        bike  = ddf[ddf["cat"] == "bike"]
        ds = DailyStats(
            day_offset  = d,
            trail_km    = trail["distance_km"].sum(),
            trail_dplus = trail["dplus_m"].fillna(0).sum(),
            trail_h     = trail["duration_s"].fillna(0).sum() / 3600,
            bike_km     = bike["distance_km"].sum(),
            bike_h      = bike["duration_s"].fillna(0).sum() / 3600,
            n_sessions  = len(ddf[ddf["cat"] != "other"]),
        )
        window.taper.append(ds)

    return window


def load_summaries() -> pd.DataFrame:
    return pd.read_parquet(SUMMARIES_FILE)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def avg_build(windows: list[RaceWindow]) -> list[dict]:
    """Average weekly build stats across multiple races."""
    n = len(windows)
    if n == 0:
        return []
    n_weeks = len(windows[0].build)
    rows = []
    for i in range(n_weeks):
        ws_list = [w.build[i] for w in windows if i < len(w.build)]
        rows.append({
            "week": ws_list[0].week_offset,
            "trail_km":    round(sum(w.trail_km    for w in ws_list) / len(ws_list), 1),
            "trail_dplus": round(sum(w.trail_dplus for w in ws_list) / len(ws_list), 0),
            "trail_h":     round(sum(w.trail_h     for w in ws_list) / len(ws_list), 1),
            "bike_km":     round(sum(w.bike_km     for w in ws_list) / len(ws_list), 1),
            "bike_h":      round(sum(w.bike_h      for w in ws_list) / len(ws_list), 1),
            "total_h":     round(sum(w.total_h     for w in ws_list) / len(ws_list), 1),
            "n":           len(ws_list),
        })
    return rows


def avg_taper(windows: list[RaceWindow]) -> list[dict]:
    """Average daily taper stats across multiple races."""
    if not windows:
        return []
    n_days = len(windows[0].taper)
    rows = []
    for i in range(n_days):
        ds_list = [w.taper[i] for w in windows if i < len(w.taper)]
        rows.append({
            "day":          ds_list[0].day_offset,
            "trail_km":     round(sum(d.trail_km    for d in ds_list) / len(ds_list), 1),
            "trail_dplus":  round(sum(d.trail_dplus for d in ds_list) / len(ds_list), 0),
            "trail_h":      round(sum(d.trail_h     for d in ds_list) / len(ds_list), 2),
            "bike_km":      round(sum(d.bike_km     for d in ds_list) / len(ds_list), 1),
            "bike_h":       round(sum(d.bike_h      for d in ds_list) / len(ds_list), 2),
            "n_sessions":   round(sum(d.n_sessions  for d in ds_list) / len(ds_list), 1),
        })
    return rows
