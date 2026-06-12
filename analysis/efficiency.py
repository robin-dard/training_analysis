"""
analysis/efficiency.py

Efficiency analysis for structured training sessions.

Track intervals  : speed_ms / mean_hr  (higher = more economical)
Hill repeats     : asc_speed_mh / mean_hr
Zone speed       : median speed at each HR zone (track sessions only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .hr_analysis import DEFAULT_HR_ZONES, DEFAULT_MAX_HR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    start_s: float
    end_s: float
    is_work: bool
    mean_speed_ms: float
    mean_hr: Optional[float]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def speed_kmh(self) -> float:
        return round(self.mean_speed_ms * 3.6, 2)

    @property
    def efficiency(self) -> Optional[float]:
        if self.mean_hr and self.mean_hr > 0:
            return round(self.mean_speed_ms / self.mean_hr, 5)
        return None


@dataclass
class HillRepeat:
    repeat_num: int
    start_s: float
    end_s: float
    dplus_m: float
    dist_m: float
    avg_grade_pct: float
    asc_speed_mh: float
    mean_hr: Optional[float]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def efficiency(self) -> Optional[float]:
        if self.mean_hr and self.mean_hr > 0:
            return round(self.asc_speed_mh / self.mean_hr, 3)
        return None


# ---------------------------------------------------------------------------
# Track interval detection
# ---------------------------------------------------------------------------

def detect_track_intervals(
    df: pd.DataFrame,
    speed_quantile: float = 0.58,
    min_work_s: float = 25.0,
    min_recovery_s: float = 15.0,
) -> list[Interval]:
    """
    Split track session into work/recovery blocks using adaptive speed threshold.

    The 58th-percentile threshold typically falls in the valley between the
    recovery cluster and the work cluster for standard interval sessions.
    """
    required = {"elapsed_s", "speed_ms"}
    if not required.issubset(df.columns) or df.empty:
        return []

    speeds = df["speed_ms"].dropna()
    if len(speeds) < 20:
        return []

    threshold = float(speeds.quantile(speed_quantile))

    df = df.copy().reset_index(drop=True)
    df["_work"]  = df["speed_ms"].fillna(0) >= threshold
    df["_block"] = (df["_work"] != df["_work"].shift()).cumsum()

    intervals: list[Interval] = []
    for _, grp in df.groupby("_block", sort=False):
        is_work  = bool(grp["_work"].iloc[0])
        t_start  = float(grp["elapsed_s"].min())
        t_end    = float(grp["elapsed_s"].max())
        duration = t_end - t_start
        if duration < (min_work_s if is_work else min_recovery_s):
            continue

        mean_speed = float(grp["speed_ms"].mean())
        hr_col = grp["heart_rate"].dropna() if "heart_rate" in grp.columns else pd.Series(dtype=float)
        mean_hr = float(hr_col.mean()) if len(hr_col) > 3 else None

        intervals.append(Interval(
            start_s=t_start,
            end_s=t_end,
            is_work=is_work,
            mean_speed_ms=round(mean_speed, 3),
            mean_hr=round(mean_hr, 1) if mean_hr is not None else None,
        ))

    return intervals


# ---------------------------------------------------------------------------
# Hill repeat detection
# ---------------------------------------------------------------------------

def detect_hill_repeats(
    df: pd.DataFrame,
    min_dplus_m: float = 400.0,
    max_dplus_m: float = 900.0,
    min_avg_grade_pct: float = 10.0,
    min_repeats: int = 3,
    similarity_cv: float = 0.30,
    gap_tolerance_m: float = 300.0,
) -> list[HillRepeat]:
    """
    Detect structured hill repeats from a trail session.

    Returns a list of HillRepeat objects if the session contains >= min_repeats
    climbs with similar D+ (CV < similarity_cv), else empty list.
    """
    required = {"grade_pct", "distance_m", "altitude_m", "elapsed_s"}
    if not required.issubset(df.columns) or df.empty:
        return []

    # Detect climbs within D+ range
    climbs = _extract_climbs(
        df,
        min_dplus_m=min_dplus_m,
        min_avg_grade_pct=min_avg_grade_pct,
        gap_tolerance_m=gap_tolerance_m,
    )
    climbs = [c for c in climbs if c["dplus_m"] <= max_dplus_m]

    if len(climbs) < min_repeats:
        return []

    dplus = np.array([c["dplus_m"] for c in climbs])
    cv = dplus.std() / dplus.mean() if dplus.mean() > 0 else 1.0
    if cv > similarity_cv:
        return []

    repeats: list[HillRepeat] = []
    for i, climb in enumerate(climbs, 1):
        seg = df[
            (df["distance_m"] >= climb["dist_start_m"]) &
            (df["distance_m"] <= climb["dist_end_m"])
        ]
        if seg.empty:
            continue

        hr_col = seg["heart_rate"].dropna() if "heart_rate" in seg.columns else pd.Series(dtype=float)
        mean_hr = float(hr_col.mean()) if len(hr_col) > 5 else None

        # Time on uphill only
        if "grade_pct" in seg.columns:
            uphill = seg[seg["grade_pct"] > 5.0]
        else:
            uphill = seg
        uphill_h = uphill["elapsed_s"].diff().clip(lower=0).sum() / 3600 if not uphill.empty else 0
        asc_speed = climb["dplus_m"] / uphill_h if uphill_h > 1e-4 else 0.0

        repeats.append(HillRepeat(
            repeat_num=i,
            start_s=float(seg["elapsed_s"].min()),
            end_s=float(seg["elapsed_s"].max()),
            dplus_m=round(float(climb["dplus_m"]), 0),
            dist_m=round(float(climb["dist_end_m"] - climb["dist_start_m"]), 0),
            avg_grade_pct=round(float(climb["avg_grade_pct"]), 1),
            asc_speed_mh=round(float(asc_speed), 0),
            mean_hr=round(float(mean_hr), 1) if mean_hr is not None else None,
        ))

    return repeats


def _extract_climbs(
    df: pd.DataFrame,
    min_dplus_m: float,
    min_avg_grade_pct: float,
    gap_tolerance_m: float,
) -> list[dict]:
    """Extract significant uphill segments from a time series."""
    df = df.copy().reset_index(drop=True)
    df["_up"] = df["grade_pct"] > 5.0

    segments: list[tuple[int, int]] = []
    in_climb, seg_start, last_up_idx = False, 0, -1

    for idx, row in df.iterrows():
        if row["_up"]:
            if not in_climb:
                if (last_up_idx >= 0 and segments and
                        row["distance_m"] - df.loc[last_up_idx, "distance_m"] <= gap_tolerance_m):
                    seg_start = segments.pop()[0]
                else:
                    seg_start = idx
                in_climb = True
            last_up_idx = idx
        else:
            if in_climb:
                in_climb = False
                segments.append((seg_start, idx - 1))
    if in_climb:
        segments.append((seg_start, len(df) - 1))

    climbs = []
    for start, end in segments:
        seg = df.iloc[start:end + 1]
        if seg.empty:
            continue
        dplus = float(seg["altitude_m"].diff().clip(lower=0).sum())
        dist  = float(seg["distance_m"].max() - seg["distance_m"].min())
        if dist < 1 or dplus < min_dplus_m:
            continue
        avg_grade = (dplus / dist) * 100
        if avg_grade < min_avg_grade_pct:
            continue
        climbs.append({
            "dist_start_m": float(seg["distance_m"].min()),
            "dist_end_m":   float(seg["distance_m"].max()),
            "dplus_m":      dplus,
            "avg_grade_pct": avg_grade,
        })

    return climbs


# ---------------------------------------------------------------------------
# Zone speed (track only)
# ---------------------------------------------------------------------------

def zone_speed_ms(
    df: pd.DataFrame,
    max_hr: int = DEFAULT_MAX_HR,
    zones: Optional[dict] = None,
    min_points: int = 10,
) -> dict[str, Optional[float]]:
    """
    For each HR zone, compute median speed (m/s).
    Returns {zone_name: median_speed_ms} or None when insufficient data.
    """
    zones = zones or DEFAULT_HR_ZONES
    if "heart_rate" not in df.columns or "speed_ms" not in df.columns:
        return {z: None for z in zones}

    result: dict[str, Optional[float]] = {}
    for name, (lo, hi) in zones.items():
        mask = (df["heart_rate"] >= lo * max_hr) & (df["heart_rate"] < hi * max_hr)
        speeds = df.loc[mask, "speed_ms"].dropna()
        result[name] = round(float(speeds.median()), 3) if len(speeds) >= min_points else None

    return result
