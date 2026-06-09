"""
analysis/build_taper.py

Weekly build and taper aggregation.
Works from a list of Garmin activity dicts (from GarminClient.sync or
garminconnect.get_activities).

Produces the same format as the HTML tables we built manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Sport categories
# ---------------------------------------------------------------------------

TRAIL_SPORTS = {"trail_running", "running", "track_running", "treadmill_running", "hiking"}
BIKE_SPORTS  = {"road_biking", "cycling", "mountain_biking", "indoor_cycling", "gravel_cycling"}
SKI_SPORTS   = {"backcountry_skiing", "resort_skiing", "nordic_skiing"}


def _classify_sport(sport_type: str) -> str:
    if sport_type in TRAIL_SPORTS:
        return "trail"
    if sport_type in BIKE_SPORTS:
        return "bike"
    if sport_type in SKI_SPORTS:
        return "ski"
    return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeekStats:
    week_start: date
    trail_km: float = 0.0
    trail_dplus: float = 0.0
    trail_h: float = 0.0
    bike_km: float = 0.0
    bike_dplus: float = 0.0
    bike_h: float = 0.0
    ski_km: float = 0.0
    ski_dplus: float = 0.0
    ski_h: float = 0.0
    total_h: float = 0.0
    n_activities: int = 0

    @property
    def week_end(self) -> date:
        return self.week_start + timedelta(days=6)

    @property
    def label(self) -> str:
        return f"{self.week_start.strftime('%-d %b')}–{self.week_end.strftime('%-d %b')}"

    @property
    def trail_dplus_per_km(self) -> float:
        return self.trail_dplus / self.trail_km if self.trail_km > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.label:16s} | "
            f"trail {self.trail_km:5.1f}km {self.trail_dplus:5.0f}m {self.trail_h:4.1f}h | "
            f"bike {self.bike_km:5.1f}km {self.bike_h:4.1f}h"
        )


@dataclass
class BuildTaper:
    race_name: str
    race_date: date
    weeks: list[WeekStats] = field(default_factory=list)
    taper_days: list[dict] = field(default_factory=list)

    @property
    def peak_trail_dplus(self) -> float:
        return max((w.trail_dplus for w in self.weeks), default=0.0)

    @property
    def total_trail_dplus(self) -> float:
        return sum(w.trail_dplus for w in self.weeks)

    @property
    def total_trail_km(self) -> float:
        return sum(w.trail_km for w in self.weeks)

    @property
    def total_h(self) -> float:
        return sum(w.total_h for w in self.weeks)

    def summary(self) -> str:
        lines = [
            f"\n=== Build/Taper: {self.race_name} ({self.race_date}) ===",
            f"  {'Week':16s}   Trail km  Trail D+  Trail h   Bike km  Bike h",
            f"  {'':16s}   {'─'*8}  {'─'*8}  {'─'*7}   {'─'*7}  {'─'*6}",
        ]
        for w in self.weeks:
            lines.append(
                f"  {w.label:16s} | "
                f"{w.trail_km:7.1f}   {w.trail_dplus:7.0f}   {w.trail_h:5.1f}h  | "
                f"{w.bike_km:6.1f}   {w.bike_h:5.1f}h"
            )
        lines.append(f"\n  Peak D+ week : {self.peak_trail_dplus:.0f}m")
        lines.append(f"  Total trail D+: {self.total_trail_dplus:.0f}m")
        lines.append(f"  Total trail km: {self.total_trail_km:.0f}km")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "week": w.label,
                "trail_km": round(w.trail_km, 1),
                "trail_dplus": round(w.trail_dplus, 0),
                "trail_h": round(w.trail_h, 1),
                "bike_km": round(w.bike_km, 1),
                "bike_dplus": round(w.bike_dplus, 0),
                "bike_h": round(w.bike_h, 1),
                "ski_dplus": round(w.ski_dplus, 0),
                "total_h": round(w.total_h, 1),
            }
            for w in self.weeks
        ])


# ---------------------------------------------------------------------------
# From Garmin activities
# ---------------------------------------------------------------------------

def build_taper_from_garmin(
    activities: list[dict],
    race_name: str,
    race_date: date,
    n_build_weeks: int = 8,
    week_start_day: int = 0,   # 0 = Monday
) -> BuildTaper:
    """
    Build a BuildTaper object from a list of Garmin activity dicts.

    Parameters
    ----------
    activities : list of activity dicts from GarminClient or
                 garminconnect.Garmin.get_activities()
    race_name  : label
    race_date  : date of race
    n_build_weeks : how many weeks to look back
    week_start_day : 0 = Monday (ISO default)

    Returns
    -------
    BuildTaper
    """
    # Convert to DataFrame for easier filtering
    df = _activities_to_df(activities)
    if df.empty:
        return BuildTaper(race_name=race_name, race_date=race_date)

    # Compute build window
    build_start = race_date - timedelta(weeks=n_build_weeks)

    # Get Monday of race week as cutoff
    race_monday = race_date - timedelta(days=race_date.weekday())
    window_df = df[
        (df["date"] >= build_start) & (df["date"] < race_date)
    ].copy()

    # Group by week
    weeks = _group_by_week(window_df, build_start, race_monday)

    return BuildTaper(
        race_name=race_name,
        race_date=race_date,
        weeks=weeks,
    )


def _activities_to_df(activities: list[dict]) -> pd.DataFrame:
    rows = []
    for a in activities:
        sport = a.get("activityType", {}).get("typeKey", "")
        start = a.get("startTimeLocal", "")
        try:
            dt = pd.to_datetime(start).date()
        except Exception:
            continue
        rows.append({
            "date": dt,
            "sport_type": sport,
            "sport_cat": _classify_sport(sport),
            "distance_m": a.get("distance", 0) or 0,
            "dplus_m": a.get("elevationGain", 0) or 0,
            "moving_time_s": a.get("duration", 0) or 0,
            "name": a.get("activityName", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _group_by_week(
    df: pd.DataFrame,
    build_start: date,
    build_end: date,
) -> list[WeekStats]:
    """Aggregate activities into ISO weeks."""
    weeks: list[WeekStats] = []
    current = build_start - timedelta(days=build_start.weekday())  # rewind to Monday

    while current < build_end:
        week_end = current + timedelta(days=7)
        mask = (df["date"] >= current) & (df["date"] < week_end)
        week_df = df[mask]

        w = WeekStats(week_start=current)
        w.n_activities = len(week_df)

        for cat, prefix in [("trail", "trail"), ("bike", "bike"), ("ski", "ski")]:
            sub = week_df[week_df["sport_cat"] == cat]
            setattr(w, f"{prefix}_km", sub["distance_m"].sum() / 1000)
            setattr(w, f"{prefix}_dplus", sub["dplus_m"].sum())
            setattr(w, f"{prefix}_h", sub["moving_time_s"].sum() / 3600)

        w.total_h = week_df["moving_time_s"].sum() / 3600
        weeks.append(w)
        current = week_end

    return weeks


# ---------------------------------------------------------------------------
# Season totals comparison (Nolio format)
# ---------------------------------------------------------------------------

def compare_seasons(
    seasons: dict[str, list[WeekStats]],
) -> pd.DataFrame:
    """
    Compare multiple seasons side by side.

    Parameters
    ----------
    seasons : dict mapping season label → list of WeekStats

    Returns
    -------
    DataFrame with one row per season and columns for key totals
    """
    rows = []
    for label, weeks in seasons.items():
        rows.append({
            "season": label,
            "trail_km": round(sum(w.trail_km for w in weeks), 0),
            "trail_dplus": round(sum(w.trail_dplus for w in weeks), 0),
            "trail_h": round(sum(w.trail_h for w in weeks), 1),
            "bike_km": round(sum(w.bike_km for w in weeks), 0),
            "bike_h": round(sum(w.bike_h for w in weeks), 1),
            "ski_dplus": round(sum(w.ski_dplus for w in weeks), 0),
            "total_h": round(sum(w.total_h for w in weeks), 1),
            "peak_dplus_week": round(max((w.trail_dplus for w in weeks), default=0), 0),
        })
    return pd.DataFrame(rows)
