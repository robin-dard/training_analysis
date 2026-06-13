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

_WORK_TARGET_TYPES = frozenset({"speed", "heart_rate", "power", "cadence",
                                  "heart_rate_lap", "power_3s", "power_10s",
                                  "power_lap", "speed_lap"})


def _natural_break_threshold(speeds: list[float]) -> float | None:
    """
    Find the midpoint of the largest gap in a sorted speed list.

    Returns None when:
    - fewer than 4 data points
    - largest gap < 1.5 m/s (no clear bimodal pattern)
    - upper cluster mean < 3.5 m/s (upper group is easy jogging, not intervals)
    """
    if len(speeds) < 4:
        return None
    s = sorted(speeds)
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    max_gap, idx = max(gaps, key=lambda x: x[0])
    if max_gap < 1.5:
        return None
    upper = s[idx + 1:]
    if sum(upper) / len(upper) < 3.5:  # upper cluster must be interval pace, not easy jog
        return None
    return (s[idx] + s[idx + 1]) / 2


def detect_track_intervals_from_laps(
    laps_df: "pd.DataFrame",
    workout_steps: "list[dict] | None" = None,
) -> list[Interval]:
    """
    Build Interval list from FIT lap messages, using workout step targets when available.

    Classification priority:
      1. session_end laps are discarded (activity-summary lap, not an interval).
      2. Workout step found for this lap: work iff intensity=active AND target_type is
         an explicit metric (speed/HR/power).  Active steps with target_type=open are
         warmup/cooldown freeform runs → not work.
      3. No step info (manual-lap sessions): find the natural speed gap between work
         and recovery laps using manual-lap speeds only, then apply that threshold to
         all laps.  Falls back to intensity field if no clear bimodal pattern.
    """
    if laps_df is None or laps_df.empty:
        return []

    # Drop session-end summary lap — it is not a real interval
    laps_df = laps_df[laps_df["trigger"] != "session_end"]
    if laps_df.empty:
        return []

    step_lookup: dict = {}
    if workout_steps:
        for s in workout_steps:
            step_lookup[int(s["step_index"])] = s

    # For manual-lap sessions (no step_lookup), derive work/rest threshold from the
    # bimodal speed distribution of manual laps (work ~5 m/s, recovery ~1.5 m/s).
    # Garmin does not set the intensity field on manually-pressed laps.
    speed_threshold: float | None = None
    if not step_lookup:
        manual_speeds = [
            float(lap.get("avg_speed_ms") or 0)
            for _, lap in laps_df.iterrows()
            if str(lap.get("trigger", "")) == "manual"
            and float(lap.get("duration_s") or 0) >= 20
        ]
        speed_threshold = _natural_break_threshold(manual_speeds)

    intervals: list[Interval] = []
    for _, lap in laps_df.iterrows():
        dur = float(lap.get("duration_s") or 0)
        if dur < 5:
            continue

        step_idx = lap.get("wkt_step_index")
        try:
            step = step_lookup.get(int(step_idx)) if step_idx is not None else None
        except (ValueError, TypeError):
            step = None

        if step:
            step_intensity = str(step.get("intensity", "")).lower()
            target_type    = str(step.get("target_type", "")).lower()
            target_value   = step.get("target_value")
            is_work = (step_intensity == "active" and target_type in _WORK_TARGET_TYPES)
            # Use actual target values to exclude easy-effort steps that share
            # active+HR or active+speed metadata with real work steps.
            # Garmin stores targets either as `target_value` (named zone) or
            # `custom_hr_high` / `custom_speed_high` (custom range).
            if is_work:
                if target_type == "heart_rate":
                    # custom_hr_high preferred; fall back to target_value
                    eff_hr = step.get("custom_hr_high")
                    if eff_hr is None:
                        eff_hr = target_value
                    if eff_hr is not None:
                        try:
                            # HR target below zone-3 entry (≈80 % HRmax) → warmup/recovery
                            if float(eff_hr) <= DEFAULT_MAX_HR * 0.80:
                                is_work = False
                        except (TypeError, ValueError):
                            pass
                elif target_type == "speed":
                    # Use custom_speed_high only (target_value for speed zones is a
                    # zone index, not m/s).  Upper bound ≤ 3.5 m/s (12.6 km/h) means
                    # the step targets easy/recovery pace, never a real work interval.
                    eff_spd = step.get("custom_speed_high")
                    if eff_spd is not None:
                        try:
                            if float(eff_spd) <= 3.5:
                                is_work = False
                        except (TypeError, ValueError):
                            pass
        elif speed_threshold is not None:
            is_work = float(lap.get("avg_speed_ms") or 0) >= speed_threshold
        elif step_lookup:
            # Workout session but this lap has no step_idx (warmup/transition lap).
            # Don't fall back to intensity — post-processing speed gap will handle it.
            is_work = False
        else:
            is_work = str(lap.get("intensity", "")).lower() == "active"

        # Walking-pace laps are never work regardless of step metadata.
        speed_ms = float(lap.get("avg_speed_ms") or 0)
        if is_work and speed_ms < 2.0:
            is_work = False

        start_s = float(lap.get("start_s") or 0)
        hr      = lap.get("avg_hr")
        intervals.append(Interval(
            start_s=start_s,
            end_s=float(lap.get("end_s") or start_s + dur),
            is_work=is_work,
            mean_speed_ms=round(speed_ms, 3),
            mean_hr=round(float(hr), 1) if hr else None,
        ))

    # For workout sessions: remove very short work intervals when they are a
    # minority among longer work intervals.  Strides (4) within a progressive
    # 8km session have fewer short reps than main work blocks; but in an
    # 8×200m or 10×30s session the short intervals ARE the main work (majority).
    if step_lookup:
        long_work = [i for i in intervals if i.is_work and i.duration_s >= 60.0]
        short_work = [i for i in intervals if i.is_work and i.duration_s < 60.0]
        if long_work and short_work:
            min_d = min(i.duration_s for i in short_work)
            max_d = max(i.duration_s for i in long_work)
            if max_d > 3.0 * min_d and len(short_work) < len(long_work):
                for iv in short_work:
                    iv.is_work = False

    # Post-process for workout sessions: recovery laps sometimes share identical
    # step metadata with work laps (e.g., both have intensity=active+speed).
    # Re-run a speed-gap check on classified-work intervals only — any work
    # interval whose speed falls in the lower cluster is actually recovery.
    if step_lookup:
        work_speeds = [i.mean_speed_ms for i in intervals
                       if i.is_work and i.mean_speed_ms > 0]
        if len(work_speeds) >= 4:
            ws = sorted(work_speeds)
            speed_range = ws[-1] - ws[0]
            if speed_range > 0:
                gaps = [(ws[j + 1] - ws[j], j) for j in range(len(ws) - 1)]
                max_gap, gap_idx = max(gaps, key=lambda x: x[0])
                # Use relative gap (fraction of total range) to catch tempo/HR-target
                # workouts where work and recovery speeds differ by less than 1.2 m/s.
                # Absolute minimum 0.5 m/s prevents splitting uniform work intervals.
                if max_gap >= 0.5 and max_gap / speed_range >= 0.20:
                    upper = ws[gap_idx + 1:]
                    lower = ws[:gap_idx + 1]
                    # Upper cluster must be true interval pace (≥ 3.5 m/s = 12.6 km/h).
                    # Lower cluster must be genuinely slow (< 3.5 m/s) — if the "lower"
                    # group is already running at tempo pace, it is a second tier of work
                    # (e.g. a session mixing 10'/15' tempo blocks with shorter faster
                    # intervals), not recovery jogs that should be demoted.
                    if (sum(upper) / len(upper) >= 3.5
                            and sum(lower) / len(lower) < 3.5):
                        thr = (ws[gap_idx] + ws[gap_idx + 1]) / 2
                        for iv in intervals:
                            if iv.is_work:
                                iv.is_work = iv.mean_speed_ms >= thr

    return intervals


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
    min_avg_grade_pct: float = 15.0,
    min_climbs: int = 2,
    gap_tolerance_m: float = 500.0,
) -> list[HillRepeat]:
    """
    Return all significant climbs from a session (trail run, race, or structured repeats).

    No D+ cap, no similarity check.  D+ >= 400m and avg grade >= threshold qualify.
    Close uphill segments are merged (gap_tolerance_m).  The caller interprets
    whether the pattern looks like structured repeats, a race, or a trail run.
    """
    required = {"grade_pct", "distance_m", "altitude_m", "elapsed_s"}
    if not required.issubset(df.columns) or df.empty:
        return []

    climbs = _extract_climbs(
        df,
        min_dplus_m=min_dplus_m,
        min_avg_grade_pct=min_avg_grade_pct,
        gap_tolerance_m=gap_tolerance_m,
    )

    if len(climbs) < min_climbs:
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

        # Grey-box time bounds: start at first uphill point, end at altitude peak.
        # This prevents post-summit descents from inflating the displayed interval.
        start_s = float(uphill["elapsed_s"].min()) if not uphill.empty else float(seg["elapsed_s"].min())
        peak_idx = seg["altitude_m"].idxmax()
        end_s    = float(seg.loc[peak_idx, "elapsed_s"])

        repeats.append(HillRepeat(
            repeat_num=i,
            start_s=start_s,
            end_s=end_s,
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
    in_climb, seg_start = False, 0
    # Track end distance of last saved segment (not last uphill point).
    # Using last uphill point causes GPS noise during descents to shorten the
    # apparent gap between real climbs, leading to spurious merges.
    last_seg_end_dist: float = -gap_tolerance_m - 1.0

    for idx, row in df.iterrows():
        if row["_up"]:
            if not in_climb:
                gap = row["distance_m"] - last_seg_end_dist
                if segments and gap <= gap_tolerance_m:
                    # Only merge if altitude hasn't dropped significantly since the
                    # last segment: a drop ≥ 30 m means a real descent occurred and
                    # the two segments should remain separate.
                    prev_end_alt = float(df.loc[segments[-1][1], "altitude_m"])
                    if prev_end_alt - float(row["altitude_m"]) >= 30.0:
                        seg_start = idx
                    else:
                        seg_start = segments.pop()[0]
                else:
                    seg_start = idx
                in_climb = True
        else:
            if in_climb:
                in_climb = False
                segments.append((seg_start, idx - 1))
                last_seg_end_dist = float(df.loc[idx - 1, "distance_m"])
    if in_climb:
        segments.append((seg_start, len(df) - 1))

    climbs = []
    for start, end in segments:
        seg = df.iloc[start:end + 1]
        if seg.empty:
            continue
        # Cap individual altitude gains at 15 m per sample to suppress GPS glitches
        # (e.g. altitude briefly dropping to 0 then jumping back).
        dplus = float(seg["altitude_m"].diff().fillna(0).clip(lower=0, upper=15).sum())
        if dplus < min_dplus_m:
            continue
        # Compute avg grade over uphill-only points so that flat traverses within
        # a climb don't dilute the grade below threshold.
        # Use per-point distance increments from the full segment so that
        # gaps between non-consecutive uphill points don't inflate uphill_dist.
        d_dist = seg["distance_m"].diff().fillna(0).clip(lower=0)
        uphill_dist = float(d_dist[seg["_up"]].sum())
        if uphill_dist < 1:
            continue
        avg_grade = (dplus / uphill_dist) * 100
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
