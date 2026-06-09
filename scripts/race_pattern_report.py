"""
scripts/race_pattern_report.py

Build/taper pattern comparison: good objective races (score >= 3)
vs bad objective races (score < 3).

Usage:
    python scripts/race_pattern_report.py
    python scripts/race_pattern_report.py --taper-only
    python scripts/race_pattern_report.py --race "Traversee Nord 2019"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from analysis.build_taper_pattern import (
    avg_build,
    avg_taper,
    compute_race_window,
    load_summaries,
)

RACES_FILE = Path("data/races/races.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_trail(race: dict, min_dplus_per_km: float = 30.0) -> bool:
    """True if the race has enough vert to qualify as trail (not road)."""
    km = race.get("distance_km") or 0
    dp = race.get("dplus_m") or 0
    if km <= 0:
        return True   # unknown — keep by default
    return (dp / km) >= min_dplus_per_km


def _load_objectives(trail_only: bool = False) -> tuple[list[dict], list[dict]]:
    """Return (good_races, bad_races) from objectives with a score."""
    races = json.loads(RACES_FILE.read_text(encoding="utf-8"))
    objectives = [
        r for r in races
        if r.get("type") == "objective" and r.get("score") is not None
    ]
    if trail_only:
        objectives = [r for r in objectives if _is_trail(r)]
    good = [r for r in objectives if r["score"] >= 3]
    bad  = [r for r in objectives if r["score"] < 3]
    return good, bad


def _check_data_coverage(summaries: pd.DataFrame, race: dict) -> float:
    """Return fraction of 8-week build window that has data (0–1)."""
    race_dt  = pd.Timestamp(race["date"])
    start    = race_dt - pd.Timedelta(weeks=8)
    window   = summaries[
        (summaries["date"] >= start) & (summaries["date"] < race_dt)
    ]
    # At least 1 activity per week → 8 expected; flag if too sparse
    expected_days = 56
    if window.empty:
        return 0.0
    span = (window["date"].max() - window["date"].min()).days
    return min(1.0, span / expected_days)


def _fmt_row(label: str, vals: list, fmts: list[str]) -> str:
    cols = [f"{v:{f}}" for v, f in zip(vals, fmts)]
    return f"  {label:<6}  " + "  ".join(cols)


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_build_table(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label}: no data")
        return
    header = f"  {'Wk':<6}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  {'BikeKm':>7}  {'BikeH':>6}  {'TotH':>6}"
    print(f"\n{label}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        lbl = f"W{r['week']:+d}"
        print(f"  {lbl:<6}  {r['trail_km']:7.1f}  {r['trail_dplus']:7.0f}  "
              f"{r['trail_h']:6.1f}  {r['bike_km']:7.1f}  {r['bike_h']:6.1f}  {r['total_h']:6.1f}")


def _print_taper_table(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label}: no data")
        return
    header = f"  {'Day':<5}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  {'BikeKm':>7}  {'BikeH':>6}"
    print(f"\n{label}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if r["trail_km"] + r["bike_km"] + r["trail_h"] + r["bike_h"] == 0:
            continue  # skip rest days
        print(f"  {r['day']:5d}  {r['trail_km']:7.1f}  {r['trail_dplus']:7.0f}  "
              f"{r['trail_h']:6.2f}  {r['bike_km']:7.1f}  {r['bike_h']:6.2f}")


def _print_taper_table_full(label: str, rows: list[dict]) -> None:
    """Show all days including rest."""
    if not rows:
        print(f"  {label}: no data")
        return
    header = f"  {'Day':<5}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  {'BikeKm':>7}  {'BikeH':>6}"
    print(f"\n{label}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        rest = " (rest)" if r["trail_km"] + r["bike_km"] == 0 else ""
        print(f"  {r['day']:5d}  {r['trail_km']:7.1f}  {r['trail_dplus']:7.0f}  "
              f"{r['trail_h']:6.2f}  {r['bike_km']:7.1f}  {r['bike_h']:6.2f}{rest}")


def _print_diff_build(good_rows: list[dict], bad_rows: list[dict]) -> None:
    if not good_rows or not bad_rows:
        return
    print("\nDIFFERENCE (good minus bad):")
    print(f"  {'Wk':<6}  {'dTrailKm':>9}  {'dTrailD+':>9}  {'dTrailH':>8}  {'dBikeKm':>9}  {'dBikeH':>8}  {'dTotH':>8}")
    print("  " + "-" * 65)
    for g, b in zip(good_rows, bad_rows):
        lbl = f"W{g['week']:+d}"
        def d(k):
            return g[k] - b[k]
        print(f"  {lbl:<6}  {d('trail_km'):+8.1f}  {d('trail_dplus'):+8.0f}  "
              f"{d('trail_h'):+7.1f}  {d('bike_km'):+8.1f}  {d('bike_h'):+7.1f}  {d('total_h'):+7.1f}")


def _print_single_race(race: dict, summaries: pd.DataFrame) -> None:
    w = compute_race_window(summaries, race)
    cov = _check_data_coverage(summaries, race)
    print(f"\n{'='*70}")
    print(f"  {race['name']}  [{race['date']}]  score={race.get('score')}  "
          f"dist={race.get('distance_km', '?')}km  D+={race.get('dplus_m', '?')}m")
    if cov < 0.5:
        print(f"  ** WARNING: low data coverage ({cov:.0%}) — Strava-only period or incomplete sync")
    _print_build_table("BUILD (8 weeks, W-8=oldest):", [
        {"week": ws.week_offset, "trail_km": ws.trail_km, "trail_dplus": ws.trail_dplus,
         "trail_h": ws.trail_h, "bike_km": ws.bike_km, "bike_h": ws.bike_h, "total_h": ws.total_h}
        for ws in w.build
    ])
    _print_taper_table_full("TAPER (21 days, day=-21 oldest):", [
        {"day": ds.day_offset, "trail_km": ds.trail_km, "trail_dplus": ds.trail_dplus,
         "trail_h": ds.trail_h, "bike_km": ds.bike_km, "bike_h": ds.bike_h}
        for ds in w.taper
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────

def _last_hard_day(taper: list, trail_km_thresh: float = 15.0,
                   trail_h_thresh: float = 2.0) -> int | None:
    """
    Return the day_offset of the last genuinely hard trail day.
    Default: ≥15km OR ≥2h trail — excludes activation runs (~1h/8km before race).
    """
    for ds in reversed(taper):
        if ds.trail_km >= trail_km_thresh or ds.trail_h >= trail_h_thresh:
            return ds.day_offset
    return None


def report(taper_only: bool = False, race_name: str | None = None,
           trail_only: bool = False) -> None:
    summaries = load_summaries()
    summaries["date"] = pd.to_datetime(summaries["date"], errors="coerce")
    summaries = summaries.dropna(subset=["date"])

    good_races, bad_races = _load_objectives(trail_only=trail_only)

    if race_name:
        # Single-race deep dive
        matches = [r for r in good_races + bad_races if race_name.lower() in r["name"].lower()]
        if not matches:
            print(f"Race not found: {race_name}")
            print("Available:", [r["name"] for r in good_races + bad_races])
            return
        for r in matches:
            _print_single_race(r, summaries)
        return

    # ── Headers
    print("\n" + "=" * 70)
    print("BUILD/TAPER PATTERN ANALYSIS")
    print("=" * 70)

    good_names = ", ".join(f"{r['name']} ({r['score']})" for r in good_races)
    bad_names  = ", ".join(f"{r['name']} ({r['score']})" for r in bad_races)
    print(f"\nGOOD races (score >= 3, n={len(good_races)}):")
    for r in good_races:
        cov = _check_data_coverage(summaries, r)
        flag = "  [!low data]" if cov < 0.5 else ""
        print(f"  {r['date']}  {r['name']:<40}  score={r['score']}{flag}")
    print(f"\nBAD races (score < 3, n={len(bad_races)}):")
    for r in bad_races:
        cov = _check_data_coverage(summaries, r)
        flag = "  [!low data]" if cov < 0.5 else ""
        print(f"  {r['date']}  {r['name']:<40}  score={r['score']}{flag}")

    # ── Compute windows
    print("\nComputing build/taper windows ...")
    good_windows = []
    for r in good_races:
        cov = _check_data_coverage(summaries, r)
        w = compute_race_window(summaries, r)
        good_windows.append(w)

    bad_windows = []
    for r in bad_races:
        w = compute_race_window(summaries, r)
        bad_windows.append(w)

    # ── Build comparison
    if not taper_only:
        good_build = avg_build(good_windows)
        bad_build  = avg_build(bad_windows)
        print("\n" + "=" * 70)
        print("BUILD PHASE — avg weekly load (8 weeks pre-race)")
        print("W-8 = oldest week, W-1 = week before race")
        _print_build_table(f"GOOD races (n={len(good_windows)}):", good_build)
        _print_build_table(f"BAD races  (n={len(bad_windows)}):",  bad_build)
        _print_diff_build(good_build, bad_build)

        # Peak build week
        if good_build:
            peak_good = max(good_build, key=lambda x: x["total_h"])
            print(f"\n  Peak build week (good): W{peak_good['week']:+d}  "
                  f"{peak_good['trail_km']:.0f}km trail / {peak_good['trail_dplus']:.0f}m D+ / "
                  f"{peak_good['trail_h']:.1f}h trail / {peak_good['total_h']:.1f}h total")
        if bad_build:
            peak_bad = max(bad_build, key=lambda x: x["total_h"])
            print(f"  Peak build week (bad):  W{peak_bad['week']:+d}  "
                  f"{peak_bad['trail_km']:.0f}km trail / {peak_bad['trail_dplus']:.0f}m D+ / "
                  f"{peak_bad['trail_h']:.1f}h trail / {peak_bad['total_h']:.1f}h total")

    # ── Taper comparison
    good_taper = avg_taper(good_windows)
    bad_taper  = avg_taper(bad_windows)
    print("\n" + "=" * 70)
    print("TAPER PHASE — avg daily load (21 days pre-race)")
    print("Day -21 = 3 weeks before race, day -1 = day before race")
    _print_taper_table(f"GOOD races (n={len(good_windows)}) — active days only:", good_taper)
    _print_taper_table(f"BAD races  (n={len(bad_windows)}) — active days only:", bad_taper)

    # Taper weekly sums
    print("\n  Taper — weekly totals (trail km / D+ / hours):")
    for group_name, taper in [("GOOD", good_taper), ("BAD ", bad_taper)]:
        weeks = {-3: [], -2: [], -1: []}
        for r in taper:
            wk = -3 if r["day"] <= -15 else (-2 if r["day"] <= -8 else -1)
            weeks[wk].append(r)
        print(f"  {group_name}  ", end="")
        for wk in [-3, -2, -1]:
            rows = weeks[wk]
            tkm  = sum(r["trail_km"]    for r in rows)
            td   = sum(r["trail_dplus"] for r in rows)
            th   = sum(r["trail_h"]     for r in rows)
            bkm  = sum(r["bike_km"]     for r in rows)
            print(f"  W{wk:+d}: {tkm:.0f}km/{td:.0f}m/{th:.1f}h+{bkm:.0f}bkm", end="")
        print()

    # ── Taper start analysis: last hard day per race
    print("\n" + "=" * 70)
    print("TAPER START — last day with significant trail training (>=15km OR >=2h)")
    print(f"  {'Race':<42}  {'Score':>5}  {'LastHardDay':>12}  {'DaysBefore':>11}")
    print("  " + "-" * 75)
    for w in sorted(good_windows + bad_windows, key=lambda x: x.race_date):
        last = _last_hard_day(w.taper)
        flag = " *" if (w.score or 0) >= 3 else "  "
        lbl  = str(last) if last is not None else "none in 21d"
        print(f"  {flag}{w.race_name:<40}  {w.score or '?':>5}  {lbl:>12}")
    # Average last hard day
    good_lhd = [d for w in good_windows if (d := _last_hard_day(w.taper)) is not None]
    bad_lhd  = [d for w in bad_windows  if (d := _last_hard_day(w.taper)) is not None]
    if good_lhd:
        print(f"\n  Avg last hard day — GOOD: {sum(good_lhd)/len(good_lhd):.1f}  "
              f"(median {sorted(good_lhd)[len(good_lhd)//2]})")
    if bad_lhd:
        print(f"  Avg last hard day — BAD:  {sum(bad_lhd)/len(bad_lhd):.1f}  "
              f"(median {sorted(bad_lhd)[len(bad_lhd)//2]})")
    print("  (* = good race, score >= 3)")

    # ── Individual race detail
    print("\n" + "=" * 70)
    print("INDIVIDUAL RACE BUILD TOTALS (8-week sums)")
    print(f"  {'Race':<42}  {'Score':>5}  {'TrailKm':>7}  {'TrailD+':>7}  "
          f"{'TrailH':>6}  {'BikeKm':>7}  {'BikeH':>6}")
    print("  " + "-" * 82)
    for w in sorted(good_windows + bad_windows, key=lambda x: x.race_date):
        tkm  = sum(ws.trail_km    for ws in w.build)
        td   = sum(ws.trail_dplus for ws in w.build)
        th   = sum(ws.trail_h     for ws in w.build)
        bkm  = sum(ws.bike_km     for ws in w.build)
        bh   = sum(ws.bike_h      for ws in w.build)
        flag = " *" if (w.score or 0) >= 3 else "  "
        print(f"  {flag}{w.race_name:<40}  {w.score or '?':>5}  {tkm:7.0f}  "
              f"{td:7.0f}  {th:6.1f}  {bkm:7.0f}  {bh:6.1f}")
    print("  (* = good race, score >= 3)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/taper pattern report")
    parser.add_argument("--taper-only", action="store_true",
                        help="Show only taper section (skip build tables)")
    parser.add_argument("--trail-only", action="store_true",
                        help="Exclude road/low-vert races (D+/km < 30)")
    parser.add_argument("--race", type=str, default=None,
                        help="Deep-dive into a specific race by name (partial match)")
    parser.add_argument("--all-races", action="store_true",
                        help="Print individual build+taper for every objective race")
    args = parser.parse_args()

    if args.all_races:
        summaries = load_summaries()
        summaries["date"] = pd.to_datetime(summaries["date"], errors="coerce")
        summaries = summaries.dropna(subset=["date"])
        good, bad = _load_objectives(trail_only=args.trail_only)
        for r in sorted(good + bad, key=lambda x: x["date"]):
            _print_single_race(r, summaries)
    else:
        report(taper_only=args.taper_only, race_name=args.race, trail_only=args.trail_only)
