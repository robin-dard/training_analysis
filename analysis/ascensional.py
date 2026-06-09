"""
analysis/ascensional.py

Ascensional speed analysis — the primary performance metric for trail racing.

Key concept: not just average asc. speed, but how well it's MAINTAINED
across race phases. A 70%+ T1→T3 maintenance ratio is the target for
top-quality races (ref: Traversée Nord 2019 = 78%, Glaisins 2025 = 75%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import (
    ascensional_speed_mh,
    classify_grade,
    format_hms,
    format_pace,
    split_into_phases,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    phase_num: int
    dist_start_m: float
    dist_end_m: float
    duration_s: float
    uphill_duration_s: float       # time spent on uphill sections only
    dplus_m: float
    asc_speed_mh: float            # m/h on uphill sections
    avg_speed_ms: float            # overall avg speed (all terrain)
    avg_hr: Optional[float]
    avg_power_w: Optional[float]

    @property
    def dist_km(self) -> float:
        return (self.dist_end_m - self.dist_start_m) / 1000

    @property
    def duration_hms(self) -> str:
        return format_hms(self.duration_s)

    @property
    def pace(self) -> str:
        return format_pace(self.avg_speed_ms)

    def __str__(self) -> str:
        hr_str = f"{self.avg_hr:.0f} bpm" if self.avg_hr else "no HR"
        return (
            f"T{self.phase_num} [{self.dist_km:.1f}km | "
            f"{self.dplus_m:.0f}m D+ | "
            f"{self.asc_speed_mh:.0f} m/h asc | "
            f"{hr_str}]"
        )


@dataclass
class ClimbSegment:
    """A significant sustained climb extracted from a race."""
    segment_num: int
    dist_start_m: float
    dist_end_m: float
    dplus_m: float
    avg_grade_pct: float
    asc_speed_mh: float
    duration_s: float
    avg_hr: Optional[float]

    @property
    def distance_m(self) -> float:
        return self.dist_end_m - self.dist_start_m

    @property
    def dist_km(self) -> float:
        return self.distance_m / 1000

    def __str__(self) -> str:
        hr_str = f"{self.avg_hr:.0f} bpm" if self.avg_hr else "no HR"
        return (
            f"Climb {self.segment_num} [{self.dist_km:.1f}km | "
            f"{self.dplus_m:.0f}m D+ | "
            f"{self.avg_grade_pct:.1f}% avg | "
            f"{self.asc_speed_mh:.0f} m/h | "
            f"{hr_str}]"
        )


@dataclass
class AscensionalProfile:
    race_name: str
    n_phases: int
    phases: list[PhaseStats] = field(default_factory=list)
    climbs: list[ClimbSegment] = field(default_factory=list)

    @property
    def maintenance_ratio(self) -> Optional[float]:
        """T1→Tn ascensional speed maintenance ratio (0–1)."""
        if len(self.phases) < 2:
            return None
        t1 = self.phases[0].asc_speed_mh
        tn = self.phases[-1].asc_speed_mh
        if t1 <= 0 or np.isnan(t1):
            return None
        return tn / t1

    @property
    def phase_ratios(self) -> list[Optional[float]]:
        """Each phase's asc speed as a ratio of T1 (1.0 = same as start)."""
        if not self.phases:
            return []
        t1 = self.phases[0].asc_speed_mh
        if t1 <= 0 or np.isnan(t1):
            return [None] * len(self.phases)
        return [
            round(p.asc_speed_mh / t1, 3) if not np.isnan(p.asc_speed_mh) else None
            for p in self.phases
        ]

    @property
    def overall_asc_speed_mh(self) -> float:
        """Weighted average ascensional speed — D+ divided by uphill time only."""
        total_dplus = sum(p.dplus_m for p in self.phases)
        total_uphill_h = sum(p.uphill_duration_s for p in self.phases) / 3600
        if total_uphill_h < 1e-6:
            return float("nan")
        return total_dplus / total_uphill_h

    def summary(self) -> str:
        lines = [f"\n=== Ascensional Profile: {self.race_name} ==="]
        ratios = self.phase_ratios
        for p, r in zip(self.phases, ratios):
            ratio_str = f" ({r * 100:.0f}% of T1)" if r is not None and p.phase_num > 1 else ""
            lines.append(f"  {p}{ratio_str}")
        ratio = self.maintenance_ratio
        if ratio is not None:
            lines.append(
                f"  Maintenance T1→T{self.n_phases}: "
                f"{ratio * 100:.1f}% "
                f"({'✓ good' if ratio >= 0.70 else '⚠ below target'})"
            )
        lines.append(f"  Overall asc. speed: {self.overall_asc_speed_mh:.0f} m/h")
        if self.climbs:
            lines.append(f"\n  Significant climbs (D+≥400m, slope≥15%):")
            for c in self.climbs:
                lines.append(f"    {c}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "race_name": self.race_name,
            "n_phases": self.n_phases,
            "maintenance_ratio": self.maintenance_ratio,
            "phase_ratios": self.phase_ratios,
            "overall_asc_speed_mh": self.overall_asc_speed_mh,
            "phases": [
                {
                    "phase": p.phase_num,
                    "dist_km": round(p.dist_km, 2),
                    "dplus_m": round(p.dplus_m, 0),
                    "asc_speed_mh": round(p.asc_speed_mh, 0),
                    "avg_hr": round(p.avg_hr, 0) if p.avg_hr else None,
                    "duration_hms": p.duration_hms,
                    "ratio_vs_t1": self.phase_ratios[i] if i < len(self.phase_ratios) else None,
                }
                for i, p in enumerate(self.phases)
            ],
            "climbs": [
                {
                    "num": c.segment_num,
                    "dist_km": round(c.dist_km, 2),
                    "dplus_m": round(c.dplus_m, 0),
                    "avg_grade_pct": round(c.avg_grade_pct, 1),
                    "asc_speed_mh": round(c.asc_speed_mh, 0),
                    "avg_hr": round(c.avg_hr, 0) if c.avg_hr else None,
                }
                for c in self.climbs
            ],
        }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def ascensional_speed_by_phase(
    df: pd.DataFrame,
    n_phases: int = 3,
    race_name: str = "Race",
    min_grade_pct: float = 3.0,
    detect: bool = True,
) -> AscensionalProfile:
    """
    Compute ascensional speed for each phase of a race.

    Parameters
    ----------
    df : DataFrame from parse_fit_file() or streams_to_dataframe()
         Required columns: distance_m, altitude_m, elapsed_s, grade_pct
         Optional: heart_rate, power_w
    n_phases : number of phases to split into (default 3 = thirds)
    race_name : label for output
    min_grade_pct : minimum % grade to count as uphill
    detect : whether to detect significant climbs (D+≥400m, slope≥15%)
    """
    _validate_columns(df, ["distance_m", "altitude_m", "elapsed_s"])

    if "grade_pct" not in df.columns:
        from parsers.fit_parser import _compute_grade
        df = df.copy()
        df["grade_pct"] = _compute_grade(df)

    phases_df = split_into_phases(df, n_phases)
    profile = AscensionalProfile(race_name=race_name, n_phases=n_phases)

    for i, phase_df in enumerate(phases_df):
        if phase_df.empty:
            continue

        uphill_mask = phase_df["grade_pct"] > min_grade_pct
        uphill_df   = phase_df[uphill_mask]
        uphill_time_s = uphill_df["elapsed_s"].diff().clip(lower=0).sum() if not uphill_df.empty else 0.0

        dplus    = phase_df["altitude_m"].diff().clip(lower=0).sum()
        asc_spd  = ascensional_speed_mh(phase_df, min_grade=min_grade_pct)
        duration = phase_df["elapsed_s"].max() - phase_df["elapsed_s"].min()
        dist_delta = phase_df["distance_m"].max() - phase_df["distance_m"].min()
        avg_speed = dist_delta / duration if duration > 0 else 0.0
        avg_hr = (
            phase_df["heart_rate"].mean()
            if "heart_rate" in phase_df.columns else None
        )
        avg_power = (
            phase_df["power_w"].mean()
            if "power_w" in phase_df.columns else None
        )

        profile.phases.append(PhaseStats(
            phase_num=i + 1,
            dist_start_m=phase_df["distance_m"].min(),
            dist_end_m=phase_df["distance_m"].max(),
            duration_s=duration,
            uphill_duration_s=uphill_time_s,
            dplus_m=dplus,
            asc_speed_mh=asc_spd,
            avg_speed_ms=avg_speed,
            avg_hr=avg_hr if avg_hr is not None and not np.isnan(avg_hr) else None,
            avg_power_w=avg_power if avg_power is not None and not np.isnan(avg_power) else None,
        ))

    if detect:
        profile.climbs = detect_climbs(df, min_grade_pct=min_grade_pct)

    return profile


# ---------------------------------------------------------------------------
# Climb detection
# ---------------------------------------------------------------------------

def detect_climbs(
    df: pd.DataFrame,
    min_dplus_m: float = 400.0,
    min_avg_grade_pct: float = 15.0,
    gap_tolerance_m: float = 200.0,
    min_grade_pct: float = 3.0,
) -> list[ClimbSegment]:
    """
    Detect significant climbs: D+ ≥ 400m and avg slope ≥ 15%.

    Parameters
    ----------
    min_dplus_m : minimum D+ for a climb to be selected (default 400m)
    min_avg_grade_pct : minimum average grade % (default 15% — 400m D+ in ≤2.7km)
    gap_tolerance_m : flat/downhill gaps smaller than this are absorbed into
                      the current climb (handles brief flat sections)
    min_grade_pct : grade threshold to define "uphill" points
    """
    if "grade_pct" not in df.columns or "distance_m" not in df.columns:
        return []

    df = df.copy().reset_index(drop=True)
    df["_uphill"] = df["grade_pct"] > min_grade_pct

    # Group consecutive uphill rows, merge gaps < gap_tolerance_m
    segments: list[tuple[int, int]] = []  # (start_idx, end_idx)
    in_climb = False
    seg_start = 0
    last_uphill_idx = -1

    for idx, row in df.iterrows():
        if row["_uphill"]:
            if not in_climb:
                # Check gap from last uphill
                if (last_uphill_idx >= 0 and segments and
                        row["distance_m"] - df.loc[last_uphill_idx, "distance_m"] <= gap_tolerance_m):
                    # Merge into previous segment
                    seg_start = segments.pop()[0]
                else:
                    seg_start = idx
                in_climb = True
            last_uphill_idx = idx
        else:
            if in_climb:
                # Check if gap is too large to continue
                in_climb = False
                segments.append((seg_start, idx - 1))

    if in_climb:
        segments.append((seg_start, len(df) - 1))

    # Evaluate each segment
    climbs = []
    for num, (start, end) in enumerate(segments, 1):
        seg = df.iloc[start:end + 1]
        if seg.empty:
            continue

        dplus = seg["altitude_m"].diff().clip(lower=0).sum()
        distance = seg["distance_m"].max() - seg["distance_m"].min()
        if distance < 1:
            continue

        avg_grade = (dplus / distance) * 100
        if dplus < min_dplus_m or avg_grade < min_avg_grade_pct:
            continue

        duration = seg["elapsed_s"].max() - seg["elapsed_s"].min()
        asc_spd  = ascensional_speed_mh(seg, min_grade=min_grade_pct)
        avg_hr   = seg["heart_rate"].mean() if "heart_rate" in seg.columns else None

        climbs.append(ClimbSegment(
            segment_num=num,
            dist_start_m=seg["distance_m"].min(),
            dist_end_m=seg["distance_m"].max(),
            dplus_m=dplus,
            avg_grade_pct=avg_grade,
            asc_speed_mh=asc_spd,
            duration_s=duration,
            avg_hr=avg_hr if avg_hr is not None and not np.isnan(avg_hr) else None,
        ))

    return climbs


# ---------------------------------------------------------------------------
# Pacing targets
# ---------------------------------------------------------------------------

def target_asc_speed(
    profile: AscensionalProfile,
    target_maintenance: float = 0.70,
    floor_last: float = 0.75,
) -> dict:
    """
    Compute target ascensional speeds for a future race using linear decay.

    Linear decay from T1 (ratio=1.0) to Tn (ratio=target_maintenance),
    with a floor of floor_last on the final phase regardless of n_phases.

    Parameters
    ----------
    target_maintenance : desired T1→Tn ratio (default 0.70)
    floor_last : minimum ratio for the last phase (default 0.75)
                 prevents overly aggressive targets on long races
    """
    if not profile.phases:
        return {}

    t1_speed = profile.phases[0].asc_speed_mh
    n = profile.n_phases

    if n == 1:
        ratios = [1.0]
    else:
        ratios = [
            1.0 - (1.0 - target_maintenance) * i / (n - 1)
            for i in range(n)
        ]
        ratios[-1] = max(ratios[-1], floor_last)

    return {
        f"T{i + 1}_target_mh": round(t1_speed * r, 0)
        for i, r in enumerate(ratios)
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )
