"""
analysis/race_registry.py

Load and query data/races.json.

Usage
-----
reg = RaceRegistry.load()

# All objective races
for r in reg.objectives:
    print(r["name"], r["score"])

# Races with score >= 3
top = reg.by_score(min_score=3)

# Get FIT path for a race
path = reg.fit_path(race)

# Full DataFrame
df = reg.to_dataframe()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT        = Path(__file__).resolve().parent.parent
RACES_FILE   = _ROOT / "data/races/races.json"
FIT_DIR      = _ROOT / "data/fit"
STREAMS_DIR  = _ROOT / "data/strava_streams"


class RaceRegistry:

    def __init__(self, races: list[dict]):
        self._races = races

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path = RACES_FILE) -> "RaceRegistry":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run sync/build_race_registry.py first"
            )
        races = json.loads(path.read_text())
        return cls(races)

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    @property
    def all(self) -> list[dict]:
        return list(self._races)

    @property
    def objectives(self) -> list[dict]:
        return [r for r in self._races if r.get("type") == "objective"]

    @property
    def preparatory(self) -> list[dict]:
        return [r for r in self._races if r.get("type") == "preparatory"]

    @property
    def red_flags(self) -> list[dict]:
        return [r for r in self._races if r.get("type") == "red_flag"]

    @property
    def unclassified(self) -> list[dict]:
        return [r for r in self._races if r.get("type") is None]

    @property
    def for_analysis(self) -> list[dict]:
        """All races excluding red flags."""
        return [r for r in self._races if r.get("type") != "red_flag"]

    def by_score(self, min_score: int = 0, max_score: int = 4) -> list[dict]:
        return [
            r for r in self.objectives
            if r.get("score") is not None
            and min_score <= r["score"] <= max_score
        ]

    def by_sport(self, sport: str) -> list[dict]:
        return [r for r in self._races if sport.lower() in (r.get("sport_type") or "").lower()]

    # ------------------------------------------------------------------
    # Data access — FIT preferred, Strava stream as fallback
    # ------------------------------------------------------------------

    def fit_path(self, race: dict) -> Optional[Path]:
        fname = race.get("fit_file")
        if not fname:
            return None
        p = FIT_DIR / fname
        return p if p.exists() else None

    def stream_path(self, race: dict) -> Optional[Path]:
        fname = race.get("strava_stream_file")
        if not fname:
            return None
        p = STREAMS_DIR / fname
        return p if p.exists() else None

    def load_dataframe(self, race: dict) -> "pd.DataFrame":
        """
        Load the best available data for a race.
        Returns a FIT-parsed DataFrame if available, else Strava stream parquet.
        Raises FileNotFoundError if neither exists.
        """
        fit = self.fit_path(race)
        if fit:
            from parsers.fit_parser import parse_fit_file
            return parse_fit_file(fit)

        stream = self.stream_path(race)
        if stream:
            return pd.read_parquet(stream)

        raise FileNotFoundError(
            f"No data available for race '{race['name']}' ({race['date']})"
        )

    # ------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._races)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        df = self.to_dataframe()
        total = len(df)
        obj = len(self.objectives)
        prep = len(self.preparatory)
        red = len(self.red_flags)
        unclass = len(self.unclassified)
        print(f"Race registry: {total} races")
        print(f"  Objectives  : {obj}")
        print(f"  Preparatory : {prep}")
        print(f"  Red flag    : {red}")
        print(f"  Unclassified: {unclass}")
        if obj > 0:
            scored = [r for r in self.objectives if r.get("score") is not None]
            if scored:
                avg = sum(r["score"] for r in scored) / len(scored)
                print(f"  Avg score (objectives): {avg:.1f}/4")
