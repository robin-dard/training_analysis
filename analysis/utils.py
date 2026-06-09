"""
analysis/utils.py

Shared utilities for all analysis modules.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grade-based segment classification
# ---------------------------------------------------------------------------

GradeCategory = Literal["gentle_uphill", "steep_uphill", "very_steep_uphill", "flat", "downhill"]

GRADE_THRESHOLDS = {
    "gentle_uphill":    (3.0,  10.0),   # runnable uphill
    "steep_uphill":     (10.0, 20.0),   # power-hike territory
    "very_steep_uphill":(20.0, None),   # hands-on-knees / technical
    "flat":             (-3.0, 3.0),
    "downhill":         (None, -3.0),   # any grade < -3%
}


def classify_grade(grade_series: pd.Series) -> pd.Series:
    """
    Classify each point by grade category.
    Categories (in priority order): very_steep_uphill, steep_uphill,
    gentle_uphill, downhill, flat.
    """
    cat = pd.Series("flat", index=grade_series.index, dtype="object")
    cat[grade_series < -3.0]  = "downhill"
    cat[grade_series >= 3.0]  = "gentle_uphill"
    cat[grade_series >= 10.0] = "steep_uphill"
    cat[grade_series >= 20.0] = "very_steep_uphill"
    return cat


# ---------------------------------------------------------------------------
# Phase splitting
# ---------------------------------------------------------------------------

def split_into_phases(df: pd.DataFrame, n: int = 3) -> list[pd.DataFrame]:
    """
    Split DataFrame into n roughly equal phases by distance.
    Returns a list of n DataFrames.
    """
    if "distance_m" not in df.columns:
        raise ValueError("DataFrame must have 'distance_m' column")
    total = df["distance_m"].max()
    boundaries = [total * i / n for i in range(n + 1)]
    phases = []
    for i in range(n):
        mask = (df["distance_m"] >= boundaries[i]) & (df["distance_m"] < boundaries[i + 1])
        phases.append(df[mask].copy())
    return phases


# ---------------------------------------------------------------------------
# Ascensional speed
# ---------------------------------------------------------------------------

def ascensional_speed_mh(df: pd.DataFrame, min_grade: float = 3.0) -> float:
    """
    Compute ascensional speed (m/h) for a DataFrame segment.

    Only considers points with grade > min_grade (uphill).
    Uses the actual altitude gain between first and last uphill point
    per continuous uphill segment to avoid double-counting.

    Parameters
    ----------
    df : DataFrame with 'altitude_m', 'elapsed_s', 'grade_pct'
    min_grade : minimum % grade to consider as uphill (default 3%)

    Returns
    -------
    float : ascensional speed in m/h, or NaN if no uphill data
    """
    required = {"altitude_m", "elapsed_s", "grade_pct"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    uphill = df[df["grade_pct"] > min_grade].copy()
    if uphill.empty:
        return float("nan")

    # Total D+ from uphill segments only
    alt_diff = uphill["altitude_m"].diff().clip(lower=0)
    total_dplus = alt_diff.sum()

    # Total time spent on uphill segments
    time_diff = uphill["elapsed_s"].diff().clip(lower=0)
    total_time_h = time_diff.sum() / 3600.0

    if total_time_h < 1e-6:
        return float("nan")

    return total_dplus / total_time_h


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_hr_mean(df: pd.DataFrame, window_s: int = 300) -> pd.Series:
    """
    Rolling mean heart rate with a time-based window (seconds).
    Requires 'elapsed_s' and 'heart_rate' columns.
    """
    if "elapsed_s" not in df.columns or "heart_rate" not in df.columns:
        raise ValueError("Need 'elapsed_s' and 'heart_rate' columns")
    df_indexed = df.set_index("elapsed_s")["heart_rate"]
    return df_indexed.rolling(window=window_s, min_periods=10).mean()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def format_pace(speed_ms: float) -> str:
    """Format m/s as MM:SS/km pace string."""
    if speed_ms <= 0:
        return "--:--/km"
    secs_per_km = 1000.0 / speed_ms
    m = int(secs_per_km) // 60
    s = int(secs_per_km) % 60
    return f"{m}:{s:02d}/km"
