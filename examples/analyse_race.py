"""
examples/analyse_race.py

Entry point — analyse a race or a build/taper window from Garmin FIT files.
"""

from datetime import date


# ---------------------------------------------------------------------------
# Analyse a single FIT file
# ---------------------------------------------------------------------------

def analyse_fit_file(fit_path: str, race_name: str = "Race"):
    from parsers.fit_parser import parse_fit_file, fit_metadata
    from analysis.race_compare import RaceComparison

    meta = fit_metadata(fit_path)
    print(f"\nFile: {fit_path}")
    print(f"  Sport: {meta.get('sport')}, Start: {meta.get('start_time')}")
    print(f"  Distance: {meta.get('total_distance_m', 0) / 1000:.1f}km  "
          f"D+: {meta.get('total_ascent_m', 0):.0f}m  "
          f"Time: {meta.get('total_elapsed_time_s', 0) / 3600:.2f}h")

    df = parse_fit_file(fit_path)
    print(f"  Parsed {len(df)} data points")

    rc = RaceComparison()
    rc.add_from_df(race_name, df)
    rc.print_summary()
    return rc


# ---------------------------------------------------------------------------
# Compare two races from FIT files
# ---------------------------------------------------------------------------

def compare_two_races(
    fit_path_1: str, name_1: str,
    fit_path_2: str, name_2: str,
):
    from analysis.race_compare import RaceComparison

    rc = RaceComparison()
    rc.add_from_fit(name_1, fit_path_1)
    rc.add_from_fit(name_2, fit_path_2)
    rc.print_summary()

    comparison_df = rc.compare()
    print("\nComparison table:")
    print(comparison_df[["race_name", "asc_score", "hr_score",
                          "desc_score", "composite_score"]].to_string())
    return rc


# ---------------------------------------------------------------------------
# Build/taper from Garmin activity list
# ---------------------------------------------------------------------------

def build_taper_analysis(
    race_name: str,
    race_date: date,
    n_weeks: int = 8,
):
    from parsers.garmin_client import GarminClient
    from analysis.build_taper import build_taper_from_garmin
    from datetime import timedelta

    client = GarminClient()
    start = (race_date - timedelta(weeks=n_weeks)).isoformat()

    activities = client._fetch_all_activities(from_date=start)
    activities = [a for a in activities if a["startTimeLocal"][:10] <= race_date.isoformat()]

    print(f"Fetched {len(activities)} activities for build window")
    bt = build_taper_from_garmin(activities, race_name, race_date, n_weeks)
    print(bt.summary())
    return bt


# ---------------------------------------------------------------------------
# Season comparison from reference database
# ---------------------------------------------------------------------------

def season_comparison():
    from analysis.race_compare import load_reference_database

    rc = load_reference_database()
    print("Reference database — all races 2018–2026:")
    rc.print_summary()

    df = rc.compare()
    print("\nSorted by composite score:")
    cols = ["race_name", "distance_km", "dplus_m", "itra_score",
            "composite_score", "asc_maintenance_pct",
            "hr_drift_bpm", "desc_retention_pct"]
    cols_avail = [c for c in cols if c in df.columns]
    print(df[cols_avail].to_string())

    targets = rc.reference_targets("UTHG 2026")
    print("\nTargets for UTHG 2026:")
    for k, v in targets.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# TN 2019 vs UTHG 2026 (both from FIT files)
# ---------------------------------------------------------------------------

def tn_vs_uthg(uthg_fit_path: str, tn_fit_path: str = "data/fit/tn_2019.fit"):
    from analysis.race_compare import RaceComparison

    rc = RaceComparison()
    rc.add_from_fit(
        "Traversée Nord 2019", tn_fit_path,
        notes="25 PRs, 15h17"
    )
    rc.add_from_fit(
        "UTHG 2026", uthg_fit_path,
        notes="Post-race analysis"
    )

    rc.print_summary()
    targets = rc.reference_targets("UTHG 2026")
    print("\nActual vs targets:")
    uthg = next((s for s in rc.scores if "UTHG" in s.race_name), None)
    if uthg:
        print(f"  Asc. maintenance: "
              f"{uthg.asc_profile.maintenance_ratio * 100:.1f}% "
              f"(target ≥ 70%)")
        if uthg.hr_profile:
            print(f"  HR drift: {uthg.hr_profile.drift_bpm:+.1f} bpm (target ≤ +10)")
        if uthg.desc_profile and uthg.desc_profile.descent_retention:
            print(f"  Descent retention: "
                  f"{uthg.desc_profile.descent_retention * 100:.1f}% (target ≥ 85%)")

    return rc


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python examples/analyse_race.py fit <path_to.fit> [race_name]")
        print("  python examples/analyse_race.py compare <fit1> <name1> <fit2> <name2>")
        print("  python examples/analyse_race.py seasons")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "fit" and len(sys.argv) >= 3:
        name = sys.argv[3] if len(sys.argv) > 3 else "Race"
        analyse_fit_file(sys.argv[2], name)

    elif cmd == "compare" and len(sys.argv) >= 6:
        compare_two_races(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif cmd == "seasons":
        season_comparison()

    else:
        print(f"Unknown command: {cmd}")
