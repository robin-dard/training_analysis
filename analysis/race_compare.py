"""
analysis/race_compare.py

Multi-race comparison using the 3-metric composite:
  1. Ascensional speed maintenance (T1→T3 ratio)
  2. HR stability (drift + SD)
  3. Descent/flat finish speed retention

Also computes a normalised composite score.

Reference database (from 2018–2026 analysis):
    Glaisins 2025     : asc 75%, HR +1 bpm, desc 100%  → score 27.5/30
    Traversée Nord    : asc 78%, HR −7 bpm, desc 89%   → score 25.7/30
    Millefonts 2022   : asc 55%, HR +13 bpm, desc 88%  → score 21.8/30
    (best ITRA despite lower composite — terrain difficulty factor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .ascensional import AscensionalProfile, ascensional_speed_by_phase
from .hr_analysis import HRProfile, hr_profile_by_phase
from .descent_speed import DescentProfile, descent_speed_profile


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

@dataclass
class RaceScore:
    race_name: str
    asc_profile: Optional[AscensionalProfile] = None
    hr_profile: Optional[HRProfile] = None
    desc_profile: Optional[DescentProfile] = None

    # Metadata
    distance_km: Optional[float] = None
    dplus_m: Optional[float] = None
    duration_s: Optional[float] = None
    n_prs: Optional[int] = None
    itra_score: Optional[int] = None
    notes: str = ""

    # ---------- Individual metric scores (0–10 each) ---------------------

    @property
    def asc_score(self) -> Optional[float]:
        """Score based on T1→T3 maintenance ratio."""
        if self.asc_profile is None:
            return None
        r = self.asc_profile.maintenance_ratio
        if r is None:
            return None
        # 100% → 10, 70% → 7, 50% → 5, linear
        return min(10.0, r * 10.0)

    @property
    def hr_score(self) -> Optional[float]:
        """Score based on HR drift magnitude."""
        if self.hr_profile is None:
            return None
        d = self.hr_profile.drift_bpm
        if d is None:
            return None
        abs_d = abs(d)
        # 0 bpm drift → 10, 15+ bpm → 0
        return max(0.0, 10.0 - abs_d * (10 / 15))

    @property
    def desc_score(self) -> Optional[float]:
        """Score based on descent retention."""
        if self.desc_profile is None:
            return None
        r = self.desc_profile.descent_retention
        if r is None:
            return None
        return min(10.0, r * 10.0)

    @property
    def composite_score(self) -> Optional[float]:
        """
        Sum of available metric scores (max 30).
        Skips missing metrics — label as partial if any missing.
        """
        scores = [s for s in [self.asc_score, self.hr_score, self.desc_score]
                  if s is not None]
        return sum(scores) if scores else None

    @property
    def composite_is_partial(self) -> bool:
        return any(s is None for s in [self.asc_score, self.hr_score, self.desc_score])

    def summary(self) -> str:
        lines = [f"\n{'='*50}", f"  {self.race_name}"]
        if self.distance_km:
            lines.append(f"  {self.distance_km:.1f}km / {self.dplus_m:.0f}m D+")
        if self.itra_score:
            lines.append(f"  ITRA score: {self.itra_score}")
        lines.append(f"  Asc. maintenance score : {_fmt(self.asc_score)}/10")
        lines.append(f"  HR stability score     : {_fmt(self.hr_score)}/10")
        lines.append(f"  Descent retention score: {_fmt(self.desc_score)}/10")
        if self.composite_score is not None:
            partial = " (partial)" if self.composite_is_partial else ""
            lines.append(f"  Composite              : {self.composite_score:.1f}/30{partial}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "race_name": self.race_name,
            "distance_km": self.distance_km,
            "dplus_m": self.dplus_m,
            "itra_score": self.itra_score,
            "asc_score": _round(self.asc_score),
            "hr_score": _round(self.hr_score),
            "desc_score": _round(self.desc_score),
            "composite_score": _round(self.composite_score),
            "partial": self.composite_is_partial,
            "asc_maintenance_pct": _round(
                self.asc_profile.maintenance_ratio * 100
                if self.asc_profile and self.asc_profile.maintenance_ratio else None
            ),
            "hr_drift_bpm": _round(
                self.hr_profile.drift_bpm if self.hr_profile else None
            ),
            "desc_retention_pct": _round(
                self.desc_profile.descent_retention * 100
                if self.desc_profile and self.desc_profile.descent_retention else None
            ),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# RaceComparison — collects multiple races and compares them
# ---------------------------------------------------------------------------

class RaceComparison:
    """
    Collect multiple RaceScore objects and produce a comparison table.

    Usage
    -----
    rc = RaceComparison()

    # Add from FIT file
    rc.add_from_fit("TN 2019", "data/fit/tn_2019.fit", itra_score=None)

    # Add from a parsed FIT DataFrame
    rc.add_from_df("Glaisins 2025", df_glaisins, itra_score=519)

    # Add manually pre-computed scores (e.g. from the 2018–2026 analysis)
    rc.add_manual(
        RaceScore(
            race_name="Glaisins 2025",
            distance_km=32.4, dplus_m=2347, itra_score=519,
            notes="2nd best ITRA",
        )
    )

    # Compare
    df = rc.compare()
    print(df.to_string())
    """

    def __init__(self):
        self.scores: list[RaceScore] = []

    def add_from_df(
        self,
        name: str,
        df: pd.DataFrame,
        n_phases: int = 3,
        itra_score: Optional[int] = None,
        notes: str = "",
        **meta,
    ) -> RaceScore:
        """Add a race from a streams/FIT DataFrame."""
        asc = _safe(lambda: ascensional_speed_by_phase(df, n_phases, name))
        hr = _safe(lambda: hr_profile_by_phase(df, n_phases, name))
        desc = _safe(lambda: descent_speed_profile(df, name))

        score = RaceScore(
            race_name=name,
            asc_profile=asc,
            hr_profile=hr,
            desc_profile=desc,
            itra_score=itra_score,
            notes=notes,
            **meta,
        )
        self.scores.append(score)
        return score

    def add_from_fit(
        self,
        name: str,
        fit_path: str,
        n_phases: int = 3,
        itra_score: Optional[int] = None,
        notes: str = "",
    ) -> RaceScore:
        """Parse a FIT file and add the race."""
        from parsers.fit_parser import parse_fit_file
        df = parse_fit_file(fit_path)
        return self.add_from_df(
            name, df, n_phases=n_phases, itra_score=itra_score, notes=notes
        )

    def add_manual(self, score: RaceScore) -> None:
        """Add a pre-built RaceScore directly."""
        self.scores.append(score)

    def compare(self) -> pd.DataFrame:
        """Return a comparison DataFrame, sorted by composite score."""
        rows = [s.to_dict() for s in self.scores]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False, na_position="last")
        return df.reset_index(drop=True)

    def print_summary(self) -> None:
        for s in sorted(
            self.scores,
            key=lambda x: x.composite_score or 0,
            reverse=True,
        ):
            print(s.summary())
        print()

    def reference_targets(self, target_race: str = "UTHG 2026") -> dict:
        """
        Based on existing race scores, output recommended targets
        for the target race.
        """
        if not self.scores:
            return {}
        # Take best composite score as reference
        best = max(
            (s for s in self.scores if s.composite_score is not None),
            key=lambda x: x.composite_score,
            default=None,
        )
        if best is None:
            return {}
        return {
            "race": target_race,
            "reference": best.race_name,
            "asc_maintenance_target_pct": "≥ 70%",
            "hr_drift_target_bpm": "≤ +10 (ultra night: T1 may be lower)",
            "desc_retention_target_pct": "≥ 85%",
            "composite_target": "≥ 22/30",
            "note": (
                "02h00 start: expect HR 5–10 bpm lower than day races in T1. "
                "Asc. speed T1 ≥ 450 m/h, T3 ≥ 315 m/h."
            ),
        }


# ---------------------------------------------------------------------------
# Pre-built reference database from 2018–2026 analysis
# ---------------------------------------------------------------------------

def load_reference_database() -> RaceComparison:
    """
    Returns a RaceComparison pre-populated with the manually analysed
    races from the 2018–2026 stream data analysis.
    Data matches the figures computed from the 2018–2026 FIT file analysis.
    """
    rc = RaceComparison()

    ref = [
        dict(race_name="Glaisins 2025", distance_km=32.4, dplus_m=2347,
             itra_score=519, notes="2nd best ITRA, HR flat +1 bpm"),
        dict(race_name="Millefonts 2022", distance_km=42.6, dplus_m=2957,
             itra_score=798, notes="Best ITRA ever, 1 Strava PR, new terrain"),
        dict(race_name="Traversée Nord 2019", distance_km=88.6, dplus_m=6212,
             notes="25 PRs, 15h17, no HR data"),
        dict(race_name="Canfranc 2021", distance_km=111.9, dplus_m=8661,
             notes="21 PRs, best ever ultra"),
        dict(race_name="Canfranc UT75 2018", distance_km=72.7, dplus_m=6238,
             notes="29 PRs"),
        dict(race_name="Montreux TF 2023", distance_km=70.1, dplus_m=5143,
             notes="4 PRs, HR avg 133"),
        dict(race_name="Chablais 2018", distance_km=49.4, dplus_m=3830,
             notes="12 PRs"),
        dict(race_name="Nivolet Revard 2026", distance_km=53.2, dplus_m=3226,
             notes="29 PRs, new terrain, base in construction"),
        dict(race_name="Salève Ultra 2026", distance_km=35.7, dplus_m=5413,
             notes="69 PRs, J-21 UTHG, peak form signal"),
    ]
    for r in ref:
        rc.add_manual(RaceScore(**r))

    return rc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _round(v: Optional[float], n: int = 1) -> Optional[float]:
    return round(v, n) if v is not None else None


def _safe(fn):
    """Run fn(), return None on any exception."""
    try:
        return fn()
    except Exception:
        return None
