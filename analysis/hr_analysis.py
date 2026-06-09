"""
analysis/hr_analysis.py

Heart rate analysis:
  - HR drift across race phases (stability metric)
  - Zone distribution
  - Decoupling (HR vs pace divergence — aerobic efficiency indicator)
  - Cardiac drift rate (bpm/hour — key for ultra pacing)

Reference values from our race database:
  - Glaisins 2025 : T1→T3 drift = +1 bpm  → perfect distribution
  - Millefonts 2022: T1→T3 drift = +13 bpm → front-loaded, still excellent
  - Traversée Nord : slight HR decrease across 15h → normal for ultra/night
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import format_hms, split_into_phases


# ---------------------------------------------------------------------------
# HR zones (% of max HR — Karvonen approach optional)
# ---------------------------------------------------------------------------

# Default zones based on % of max HR.
# Override with set_hr_zones() or pass custom zones to functions.
DEFAULT_HR_ZONES = {
    "Z1_recovery":   (0.50, 0.60),
    "Z2_aerobic":    (0.60, 0.78),
    "Z3_tempo":      (0.79, 0.88),
    "Z4_threshold":  (0.89, 0.92),
    "Z5_vo2max":     (0.92, 1.00),
}

# Estimated max HR — update with your actual measured value
DEFAULT_MAX_HR = 182


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HRPhaseStats:
    phase_num: int
    mean_hr: float
    std_hr: float
    min_hr: float
    max_hr: float
    duration_s: float

    def __str__(self) -> str:
        return (
            f"T{self.phase_num}: "
            f"{self.mean_hr:.0f}±{self.std_hr:.0f} bpm "
            f"[{self.min_hr:.0f}–{self.max_hr:.0f}]"
        )


@dataclass
class HRProfile:
    race_name: str
    phases: list[HRPhaseStats] = field(default_factory=list)
    zones: dict = field(default_factory=dict)    # zone_name → % of time
    max_hr_used: int = DEFAULT_MAX_HR

    @property
    def drift_bpm(self) -> Optional[float]:
        """HR drift from T1 to last phase (positive = rising)."""
        if len(self.phases) < 2:
            return None
        return self.phases[-1].mean_hr - self.phases[0].mean_hr

    @property
    def drift_quality(self) -> str:
        d = self.drift_bpm
        if d is None:
            return "insufficient data"
        if abs(d) <= 3:
            return "excellent — near-flat HR"
        if 3 < d <= 10:
            return "good — controlled progressive load"
        if 10 < d <= 15:
            return "acceptable — front-loaded start"
        if d < -5:
            return "note — HR decrease (night/ultra drift)"
        return "warning — possible fade or overcooking"

    def summary(self) -> str:
        lines = [f"\n=== HR Profile: {self.race_name} ==="]
        for p in self.phases:
            lines.append(f"  {p}")
        if self.drift_bpm is not None:
            sign = "+" if self.drift_bpm >= 0 else ""
            lines.append(
                f"  Drift T1→T{len(self.phases)}: "
                f"{sign}{self.drift_bpm:.1f} bpm → {self.drift_quality}"
            )
        if self.zones:
            lines.append("  Zone distribution:")
            for z, pct in self.zones.items():
                bar = "█" * int(pct / 5)
                lines.append(f"    {z:<20} {pct:5.1f}%  {bar}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "race_name": self.race_name,
            "drift_bpm": self.drift_bpm,
            "drift_quality": self.drift_quality,
            "phases": [
                {
                    "phase": p.phase_num,
                    "mean_hr": round(p.mean_hr, 1),
                    "std_hr": round(p.std_hr, 1),
                }
                for p in self.phases
            ],
            "zones": {k: round(v, 1) for k, v in self.zones.items()},
        }


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def hr_profile_by_phase(
    df: pd.DataFrame,
    n_phases: int = 3,
    race_name: str = "Race",
    max_hr: int = DEFAULT_MAX_HR,
) -> HRProfile:
    """
    Compute HR statistics per race phase.

    Parameters
    ----------
    df : DataFrame with 'heart_rate', 'distance_m', 'elapsed_s'
    n_phases : number of phases
    race_name : label
    max_hr : athlete's max HR for zone calculation

    Returns
    -------
    HRProfile
    """
    if "heart_rate" not in df.columns:
        raise ValueError("DataFrame has no 'heart_rate' column — no HR data in this file")

    phases_df = split_into_phases(df, n_phases)
    profile = HRProfile(race_name=race_name, max_hr_used=max_hr)

    for i, phase_df in enumerate(phases_df):
        hr = phase_df["heart_rate"].dropna()
        if hr.empty:
            continue
        duration = (
            phase_df["elapsed_s"].max() - phase_df["elapsed_s"].min()
            if "elapsed_s" in phase_df.columns else float("nan")
        )
        profile.phases.append(HRPhaseStats(
            phase_num=i + 1,
            mean_hr=hr.mean(),
            std_hr=hr.std(),
            min_hr=hr.min(),
            max_hr=hr.max(),
            duration_s=duration,
        ))

    # Zone distribution for full race
    profile.zones = hr_zone_distribution(df["heart_rate"].dropna(), max_hr)
    return profile


def hr_zone_distribution(
    hr_series: pd.Series,
    max_hr: int = DEFAULT_MAX_HR,
    zones: Optional[dict] = None,
) -> dict[str, float]:
    """
    Compute percentage of time spent in each HR zone.

    Returns
    -------
    dict mapping zone name → % of total points
    """
    zones = zones or DEFAULT_HR_ZONES
    total = len(hr_series)
    if total == 0:
        return {}
    result = {}
    for name, (lo, hi) in zones.items():
        lo_bpm = lo * max_hr
        hi_bpm = hi * max_hr
        count = ((hr_series >= lo_bpm) & (hr_series < hi_bpm)).sum()
        result[name] = count / total * 100
    return result


def cardiac_drift_rate(
    df: pd.DataFrame,
    window_km: float = 5.0,
) -> pd.Series:
    """
    Compute rolling HR per km-window — shows how HR drifts
    as the race progresses at constant pace.

    Returns a Series indexed by distance_m with rolling mean HR.
    """
    if "heart_rate" not in df.columns or "distance_m" not in df.columns:
        raise ValueError("Need 'heart_rate' and 'distance_m' columns")

    df_sorted = df.set_index("distance_m")["heart_rate"].sort_index()
    window_m = window_km * 1000
    return df_sorted.rolling(window=window_m, min_periods=50).mean()


def hr_pace_decoupling(
    df: pd.DataFrame,
    split_at_fraction: float = 0.5,
) -> Optional[float]:
    """
    Aerobic decoupling: measures how much HR rises relative to pace
    from the first half to the second half of an activity.

    A low decoupling (<5%) indicates good aerobic fitness —
    the cardiovascular system is not working progressively harder
    to maintain the same pace.

    Formula: (HR/pace ratio second half) / (HR/pace ratio first half) - 1

    Returns
    -------
    float : decoupling percentage (0.05 = 5%), or None if data missing
    """
    required = {"heart_rate", "velocity_smooth", "distance_m"}
    if not required.issubset(df.columns):
        return None

    total_dist = df["distance_m"].max()
    split_dist = total_dist * split_at_fraction

    first = df[df["distance_m"] < split_dist]
    second = df[df["distance_m"] >= split_dist]

    for half in [first, second]:
        if half.empty:
            return None

    def hr_pace_ratio(half_df: pd.DataFrame) -> float:
        hr = half_df["heart_rate"].dropna().mean()
        pace = half_df["velocity_smooth"].dropna().mean()
        if pace <= 0:
            return float("nan")
        return hr / pace

    r1 = hr_pace_ratio(first)
    r2 = hr_pace_ratio(second)

    if np.isnan(r1) or np.isnan(r2) or r1 == 0:
        return None

    return (r2 / r1) - 1.0
