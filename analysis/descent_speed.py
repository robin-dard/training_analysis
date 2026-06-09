"""
analysis/descent_speed.py

Descent and flat speed analysis — measures finishing freshness.

Key metric: speed on negative-grade sections in the last 20% of race
vs the first 20%. A ratio ≥ 85% is the target (ref: TN 2019 = 86%).

Why it matters: downhill speed late in a race is a direct proxy for
leg freshness. Quadriceps are the limiting factor — if they're blown,
descent speed drops sharply regardless of aerobic capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import classify_grade, format_pace


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DescentProfile:
    race_name: str
    early_descent_speed_ms: float     # avg descent speed, first 20%
    late_descent_speed_ms: float      # avg descent speed, last 20%
    early_flat_speed_ms: float
    late_flat_speed_ms: float

    @property
    def descent_retention(self) -> Optional[float]:
        """Late / early descent speed ratio (target ≥ 0.85)."""
        if self.early_descent_speed_ms <= 0:
            return None
        return self.late_descent_speed_ms / self.early_descent_speed_ms

    @property
    def flat_retention(self) -> Optional[float]:
        if self.early_flat_speed_ms <= 0:
            return None
        return self.late_flat_speed_ms / self.early_flat_speed_ms

    @property
    def descent_quality(self) -> str:
        r = self.descent_retention
        if r is None:
            return "no data"
        if r >= 0.90:
            return "excellent — legs intact"
        if r >= 0.85:
            return "good — target met"
        if r >= 0.75:
            return "acceptable — some fatigue"
        return "warning — significant leg fatigue"

    def summary(self) -> str:
        lines = [f"\n=== Descent Profile: {self.race_name} ==="]
        lines.append(
            f"  Descent speed — early: {self.early_descent_speed_ms:.2f} m/s  "
            f"late: {self.late_descent_speed_ms:.2f} m/s  "
            f"retention: {self.descent_retention * 100:.1f}%"
            if self.descent_retention else "  Descent: no data"
        )
        lines.append(
            f"  Flat speed    — early: {self.early_flat_speed_ms:.2f} m/s  "
            f"late: {self.late_flat_speed_ms:.2f} m/s  "
            f"retention: {self.flat_retention * 100:.1f}%"
            if self.flat_retention else "  Flat: no data"
        )
        lines.append(f"  Quality: {self.descent_quality}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "race_name": self.race_name,
            "early_descent_ms": round(self.early_descent_speed_ms, 3),
            "late_descent_ms": round(self.late_descent_speed_ms, 3),
            "descent_retention": round(self.descent_retention, 3) if self.descent_retention else None,
            "early_flat_ms": round(self.early_flat_speed_ms, 3),
            "late_flat_ms": round(self.late_flat_speed_ms, 3),
            "flat_retention": round(self.flat_retention, 3) if self.flat_retention else None,
            "quality": self.descent_quality,
        }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def descent_speed_profile(
    df: pd.DataFrame,
    race_name: str = "Race",
    window_fraction: float = 0.20,
    max_grade_pct: float = -3.0,
    flat_grade_range: tuple[float, float] = (-3.0, 3.0),
) -> DescentProfile:
    """
    Compute descent and flat speed in early vs late race windows.

    Parameters
    ----------
    df : DataFrame with 'distance_m', 'speed_ms', 'grade_pct'
    race_name : label
    window_fraction : fraction of total distance to use as early/late window
                      (default 0.20 = first/last 20%)
    max_grade_pct : maximum grade to classify as downhill (default -3%)
    flat_grade_range : (min, max) grade to classify as flat

    Returns
    -------
    DescentProfile
    """
    required = {"distance_m", "speed_ms", "grade_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    total_dist = df["distance_m"].max()
    cutoff_early = total_dist * window_fraction
    cutoff_late = total_dist * (1 - window_fraction)

    early = df[df["distance_m"] <= cutoff_early]
    late = df[df["distance_m"] >= cutoff_late]

    return DescentProfile(
        race_name=race_name,
        early_descent_speed_ms=_mean_speed_on_terrain(
            early, "downhill", max_grade_pct, flat_grade_range
        ),
        late_descent_speed_ms=_mean_speed_on_terrain(
            late, "downhill", max_grade_pct, flat_grade_range
        ),
        early_flat_speed_ms=_mean_speed_on_terrain(
            early, "flat", max_grade_pct, flat_grade_range
        ),
        late_flat_speed_ms=_mean_speed_on_terrain(
            late, "flat", max_grade_pct, flat_grade_range
        ),
    )


def speed_by_grade_bin(
    df: pd.DataFrame,
    grade_bins: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Average speed per grade bin — useful for building a personal
    speed/grade curve and identifying where time is lost.

    Returns a DataFrame with columns: grade_bin, avg_speed_ms, count
    """
    if "speed_ms" not in df.columns or "grade_pct" not in df.columns:
        raise ValueError("Need 'speed_ms' and 'grade_pct' columns")

    grade_bins = grade_bins or [-60, -30, -20, -10, -5, -3, 0, 3, 5, 10, 20, 30, 50, 80]

    df = df.copy()
    df["grade_bin"] = pd.cut(df["grade_pct"], bins=grade_bins)
    result = (
        df.groupby("grade_bin")["speed_ms"]
        .agg(avg_speed_ms="mean", count="count")
        .reset_index()
    )
    result["avg_pace"] = result["avg_speed_ms"].apply(
        lambda v: format_pace(v) if v > 0 else "--:--"
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_speed_on_terrain(
    df: pd.DataFrame,
    terrain: str,
    max_grade_pct: float,
    flat_grade_range: tuple[float, float],
) -> float:
    if df.empty or "speed_ms" not in df.columns:
        return 0.0
    if terrain == "downhill":
        mask = df["grade_pct"] < max_grade_pct
    elif terrain == "flat":
        mask = (df["grade_pct"] >= flat_grade_range[0]) & (df["grade_pct"] <= flat_grade_range[1])
    else:
        return 0.0
    subset = df.loc[mask, "speed_ms"].dropna()
    return subset.mean() if not subset.empty else 0.0
