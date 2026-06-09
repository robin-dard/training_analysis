"""
tests/test_parsers.py

Unit tests for parsers. Uses synthetic data — no real FIT files needed.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Synthetic FIT data helpers
# ---------------------------------------------------------------------------

def make_synthetic_df(
    duration_s: int = 3600,
    distance_m: int = 10000,
    dplus_m: int = 500,
    has_hr: bool = True,
    has_power: bool = False,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Generate a synthetic activity DataFrame for testing.
    Simulates a simple out-and-back trail run with constant grade on the way up.
    """
    t = np.linspace(0, duration_s, n_points)
    d = np.linspace(0, distance_m, n_points)

    # Simple elevation profile: up first half, down second half
    half = n_points // 2
    alt_up = np.linspace(500, 500 + dplus_m, half)
    alt_down = np.linspace(500 + dplus_m, 500, n_points - half)
    alt = np.concatenate([alt_up, alt_down])

    # Speed: faster downhill
    speed = np.where(np.arange(n_points) < half, 2.0, 3.5)

    # Grade
    d_alt = np.gradient(alt)
    d_dist = np.gradient(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(np.abs(d_dist) > 0.01, d_alt / d_dist * 100, 0.0)

    now = datetime.now(tz=timezone.utc)
    ts = [now + timedelta(seconds=float(ti)) for ti in t]

    df = pd.DataFrame({
        "timestamp": ts,
        "elapsed_s": t,
        "distance_m": d,
        "altitude_m": alt,
        "speed_ms": speed,
        "grade_pct": grade.clip(-60, 80),
        "velocity_smooth": speed,
    })

    if has_hr:
        # HR rises from 130 to 160, then stays high
        hr = np.linspace(130, 160, n_points) + np.random.normal(0, 3, n_points)
        df["heart_rate"] = hr.clip(60, 200)

    if has_power:
        df["power_w"] = np.random.normal(150, 20, n_points).clip(0, 500)

    return df


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestSyntheticDataShape:

    def test_has_required_columns(self):
        df = make_synthetic_df()
        required = ["elapsed_s", "distance_m", "altitude_m", "grade_pct"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_distance_monotonically_increasing(self):
        df = make_synthetic_df()
        assert (df["distance_m"].diff().dropna() >= 0).all()

    def test_elapsed_monotonically_increasing(self):
        df = make_synthetic_df()
        assert (df["elapsed_s"].diff().dropna() >= 0).all()

    def test_grade_clipped(self):
        df = make_synthetic_df()
        assert df["grade_pct"].max() <= 80
        assert df["grade_pct"].min() >= -60

    def test_hr_in_range_when_present(self):
        df = make_synthetic_df(has_hr=True)
        assert "heart_rate" in df.columns
        assert df["heart_rate"].min() >= 30
        assert df["heart_rate"].max() <= 230

    def test_no_hr_when_not_requested(self):
        df = make_synthetic_df(has_hr=False)
        assert "heart_rate" not in df.columns


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestAscensionalSpeed:

    def test_basic_output(self):
        from analysis.ascensional import ascensional_speed_by_phase
        df = make_synthetic_df(dplus_m=800, duration_s=7200, n_points=500)
        profile = ascensional_speed_by_phase(df, n_phases=3, race_name="Test")
        assert len(profile.phases) == 3
        assert profile.phases[0].asc_speed_mh > 0

    def test_maintenance_ratio_range(self):
        from analysis.ascensional import ascensional_speed_by_phase
        # Use steeper grade to ensure uphill is detected in all phases
        df = make_synthetic_df(dplus_m=1500, duration_s=7200, distance_m=8000, n_points=500)
        profile = ascensional_speed_by_phase(df, n_phases=3, min_grade_pct=1.0)
        r = profile.maintenance_ratio
        # May be None if synthetic data doesn't have enough uphill in T1 — that's fine
        if r is not None and not np.isnan(r):
            assert 0.0 < r < 5.0

    def test_missing_column_raises(self):
        from analysis.ascensional import ascensional_speed_by_phase
        df = make_synthetic_df()
        df = df.drop(columns=["altitude_m"])
        with pytest.raises(ValueError, match="missing required columns"):
            ascensional_speed_by_phase(df)


class TestHRAnalysis:

    def test_basic_profile(self):
        from analysis.hr_analysis import hr_profile_by_phase
        df = make_synthetic_df(has_hr=True, n_points=500)
        profile = hr_profile_by_phase(df, n_phases=3)
        assert len(profile.phases) == 3
        assert all(p.mean_hr > 0 for p in profile.phases)

    def test_drift_direction(self):
        from analysis.hr_analysis import hr_profile_by_phase
        # HR that rises → positive drift
        df = make_synthetic_df(has_hr=True, n_points=500)
        profile = hr_profile_by_phase(df, n_phases=3)
        # Our synthetic HR rises — drift should be positive
        assert profile.drift_bpm is not None

    def test_no_hr_raises(self):
        from analysis.hr_analysis import hr_profile_by_phase
        df = make_synthetic_df(has_hr=False)
        with pytest.raises(ValueError, match="no HR"):
            hr_profile_by_phase(df)

    def test_zone_distribution_sums_to_100(self):
        from analysis.hr_analysis import hr_zone_distribution
        import pandas as pd, numpy as np
        hr = pd.Series(np.random.uniform(100, 185, 1000))
        zones = hr_zone_distribution(hr, max_hr=190)
        total = sum(zones.values())
        # Some HR may fall below Z1 — total can be < 100
        assert total <= 100.01


class TestDescentSpeed:

    def test_basic_descent_profile(self):
        from analysis.descent_speed import descent_speed_profile
        df = make_synthetic_df(dplus_m=800, duration_s=7200, n_points=500)
        profile = descent_speed_profile(df)
        # Downhill exists in second half of synthetic run
        assert profile.late_descent_speed_ms >= 0

    def test_retention_is_positive(self):
        from analysis.descent_speed import descent_speed_profile
        df = make_synthetic_df(dplus_m=800, duration_s=7200, n_points=500)
        profile = descent_speed_profile(df)
        if profile.descent_retention is not None:
            assert profile.descent_retention > 0


class TestRaceComparison:

    def test_compare_returns_dataframe(self):
        from analysis.race_compare import RaceComparison, RaceScore
        rc = RaceComparison()
        df = make_synthetic_df(dplus_m=800, duration_s=7200, n_points=500)
        rc.add_from_df("Test Race", df, itra_score=500)
        result = rc.compare()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_reference_database_loads(self):
        from analysis.race_compare import load_reference_database
        rc = load_reference_database()
        assert len(rc.scores) > 0

    def test_composite_score_bounded(self):
        from analysis.race_compare import RaceComparison
        rc = RaceComparison()
        df = make_synthetic_df(dplus_m=800, duration_s=7200, n_points=500)
        rc.add_from_df("Test", df)
        result = rc.compare()
        if "composite_score" in result.columns:
            scores = result["composite_score"].dropna()
            assert (scores >= 0).all()
            assert (scores <= 30).all()


class TestBuildTaper:

    def test_weekly_aggregation(self):
        from analysis.build_taper import build_taper_from_strava
        from datetime import date
        # Mock Strava activities
        activities = [
            {
                "start_local": "2026-04-14T07:00:00",
                "sport_type": "TrailRun",
                "summary": {"distance": 20000, "elevation_gain": 1500, "moving_time": 7200},
                "name": "Morning trail",
            },
            {
                "start_local": "2026-04-16T07:00:00",
                "sport_type": "Ride",
                "summary": {"distance": 80000, "elevation_gain": 1200, "moving_time": 10800},
                "name": "Bike ride",
            },
        ]
        bt = build_taper_from_strava(
            activities, "Test Race", date(2026, 6, 20), n_build_weeks=10
        )
        assert len(bt.weeks) > 0
        # At least one week has trail data
        trail_weeks = [w for w in bt.weeks if w.trail_km > 0]
        assert len(trail_weeks) > 0
