"""
scripts/pre_race_check.py

Check current build/taper status for an upcoming race.
Compares vs historical good-race patterns.

Usage
-----
python scripts/pre_race_check.py "UTHG"
python scripts/pre_race_check.py "UTHG" --long-only   # compare vs long races (>60km) only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from analysis.build_taper_pattern import (
    avg_build, avg_taper, compute_race_window, load_summaries,
)

RACES_FILE = Path("data/races/races.json")
TODAY = date.today()

# ─────────────────────────────────────────────────────────────────────────────

def _load_all() -> tuple[dict | None, list[dict], list[dict]]:
    """Return (upcoming_race, good_races, bad_races) from races.json."""
    races = json.loads(RACES_FILE.read_text(encoding="utf-8"))
    upcoming = [r for r in races if r.get("score") is None and
                r.get("type") == "objective" and
                date.fromisoformat(r["date"]) >= TODAY]
    good = [r for r in races if r.get("type") == "objective" and
            (r.get("score") or 0) >= 3]
    bad  = [r for r in races if r.get("type") == "objective" and
            r.get("score") is not None and r["score"] < 3]
    return upcoming, good, bad


def _last_hard_day(taper_list, km=15.0, h=2.0) -> int | None:
    for ds in reversed(taper_list):
        if ds.trail_km >= km or ds.trail_h >= h:
            return ds.day_offset
    return None


def _status(val, ref, pct_warn=25, pct_ok=10, label="") -> str:
    if ref <= 0:
        return ""
    delta = (val - ref) / ref * 100
    if abs(delta) <= pct_ok:
        return f"  OK  ({delta:+.0f}%)"
    if delta < -pct_warn:
        return f"  LOW ({delta:+.0f}%)"
    if delta > pct_warn:
        return f"  HIGH({delta:+.0f}%)"
    return f"      ({delta:+.0f}%)"


def _sep(char="=", n=70):
    print(char * n)


# ─────────────────────────────────────────────────────────────────────────────

def check(race_name: str, long_only: bool = False) -> None:
    summaries = load_summaries()
    summaries["date"] = pd.to_datetime(summaries["date"], errors="coerce")
    summaries = summaries.dropna(subset=["date"])

    upcoming, good_races, bad_races = _load_all()

    # Find target race
    target = next(
        (r for r in upcoming if race_name.lower() in r["name"].lower()),
        None,
    )
    if target is None:
        print(f"Race '{race_name}' not found in upcoming objectives.")
        print("Upcoming:", [r["name"] for r in upcoming])
        return

    race_dt   = date.fromisoformat(target["date"])
    days_left = (race_dt - TODAY).days

    # Filter comparable good races
    if long_only:
        ref_races = [r for r in good_races if (r.get("distance_km") or 0) >= 60]
        ref_label = f"long good races (>60km, n={len(ref_races)})"
    else:
        ref_races = good_races
        ref_label = f"all good races (n={len(ref_races)})"

    # Most similar individual race (closest distance among good)
    own_dist = target.get("distance_km") or 0
    similar = sorted(
        good_races,
        key=lambda r: abs((r.get("distance_km") or 0) - own_dist),
    )[:3]

    # Compute windows
    target_win  = compute_race_window(summaries, target)
    ref_windows = [compute_race_window(summaries, r) for r in ref_races]
    sim_windows = [compute_race_window(summaries, r) for r in similar]
    ref_build   = avg_build(ref_windows)
    ref_taper   = avg_taper(ref_windows)

    # Current week offset from race
    current_week_offset = -((days_left - 1) // 7 + 1) if days_left > 0 else 0
    completed_weeks = [ws for ws in target_win.build if ws.week_offset < current_week_offset]
    current_week    = next((ws for ws in target_win.build if ws.week_offset == current_week_offset), None)

    # ── Header ───────────────────────────────────────────────────────────────
    _sep()
    print(f"PRE-RACE CHECK: {target['name'].upper()}")
    print(f"  {target['distance_km']:.0f}km / {target['dplus_m']:.0f}m D+  |  "
          f"Race: {race_dt}  |  Today: {TODAY}  |  D-{days_left}")
    print(f"  Reference: {ref_label}")
    _sep()

    # ── BUILD: completed weeks ────────────────────────────────────────────────
    last_complete = current_week_offset - 1  # -3 when current is W-2
    print(f"\nBUILD — completed weeks (W-8 to W{last_complete:+d})")
    print(f"  {'Wk':<5}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  "
          f"{'BikeKm':>7}  {'TotH':>5}  {'vs ref':>10}  {'vs ref D+':>10}")
    print("  " + "-" * 68)
    for ws in completed_weeks:
        ref_row = next((r for r in ref_build if r["week"] == ws.week_offset), None)
        status  = _status(ws.total_h, ref_row["total_h"]) if ref_row else ""
        dst     = _status(ws.trail_dplus, ref_row["trail_dplus"]) if ref_row else ""
        print(f"  W{ws.week_offset:+d}  {ws.trail_km:7.1f}  {ws.trail_dplus:7.0f}  "
              f"{ws.trail_h:6.1f}  {ws.bike_km:7.1f}  {ws.total_h:5.1f}  "
              f"{status:<10}  {dst:<10}")

    # Current (partial) week
    if current_week and days_left > 0:
        elapsed_days = (days_left - 1) % 7 + 1
        print(f"\n  W{current_week_offset:+d}  {current_week.trail_km:7.1f}  "
              f"{current_week.trail_dplus:7.0f}  {current_week.trail_h:6.1f}  "
              f"{current_week.bike_km:7.1f}  {current_week.total_h:5.1f}  "
              f"  [partial — {elapsed_days}d elapsed]")

    # 8-week totals
    your_tkm = sum(ws.trail_km    for ws in target_win.build)
    your_td  = sum(ws.trail_dplus for ws in target_win.build)
    your_th  = sum(ws.trail_h     for ws in target_win.build)
    your_bkm = sum(ws.bike_km     for ws in target_win.build)
    ref_tkm  = sum(r["trail_km"]    for r in ref_build)
    ref_td   = sum(r["trail_dplus"] for r in ref_build)
    ref_th   = sum(r["trail_h"]     for r in ref_build)

    print(f"\n  8-WEEK TOTALS (projected, including current partial week):")
    print(f"    Yours  : {your_tkm:.0f}km trail / {your_td:.0f}m D+ / {your_th:.1f}h trail / {your_bkm:.0f}km bike")
    print(f"    Ref avg: {ref_tkm:.0f}km trail / {ref_td:.0f}m D+ / {ref_th:.1f}h trail")
    tkm_st = _status(your_tkm, ref_tkm)
    td_st  = _status(your_td,  ref_td)
    th_st  = _status(your_th,  ref_th)
    print(f"    Delta  : trail km{tkm_st}  D+{td_st}  hours{th_st}")

    # ── Similar races comparison ──────────────────────────────────────────────
    print(f"\nMOST SIMILAR GOOD RACES (by distance):")
    print(f"  {'Race':<42}  {'Dist':>5}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  {'LastHard':>9}")
    print("  " + "-" * 78)
    for sw, sr in zip(sim_windows, similar):
        lhd  = _last_hard_day(sw.taper)
        tkm  = sum(ws.trail_km    for ws in sw.build)
        td   = sum(ws.trail_dplus for ws in sw.build)
        th   = sum(ws.trail_h     for ws in sw.build)
        lhdl = f"D{lhd}" if lhd else "-"
        print(f"  {sr['name']:<42}  {sr.get('distance_km',0):>5.0f}  {tkm:7.0f}  {td:7.0f}  {th:6.1f}  {lhdl:>9}")

    # ── Taper recommendation ──────────────────────────────────────────────────
    _sep("-")
    print(f"TAPER STATUS  (D-{days_left} — {days_left // 7} weeks + {days_left % 7} days remaining)")

    lhd_so_far = _last_hard_day(target_win.taper)
    if lhd_so_far is not None:
        days_since = abs(lhd_so_far) - days_left     # days since that session from today
        print(f"  Last hard day so far : D{lhd_so_far}  ({days_since} days ago from today)")
    else:
        print(f"  Last hard day so far : none in window (very conservative so far)")

    ref_lhd = [-14, -10]  # good-race target range
    print(f"  Target last hard day : D-14 to D-10  (good race median: D-10)")

    # Show remaining target weeks from ref
    remaining = [r for r in ref_build if r["week"] >= current_week_offset]
    if remaining:
        print(f"\n  RECOMMENDED REMAINING WEEKS (based on {ref_label}):")
        print(f"  {'Wk':<5}  {'Dates':<14}  {'TrailKm':>7}  {'TrailD+':>7}  {'TrailH':>6}  {'BikeKm':>7}  {'TotH':>5}")
        print("  " + "-" * 58)
        for r in remaining:
            wk_start = race_dt - __import__('datetime').timedelta(days=(-r["week"]) * 7 - 1)
            wk_end   = race_dt - __import__('datetime').timedelta(days=(-r["week"] - 1) * 7)
            dates    = f"{wk_start.strftime('%m/%d')}–{wk_end.strftime('%m/%d')}"
            print(f"  W{r['week']:+d}  {dates:<14}  {r['trail_km']:7.1f}  {r['trail_dplus']:7.0f}  "
                  f"{r['trail_h']:6.1f}  {r['bike_km']:7.1f}  {r['total_h']:5.1f}")

    # ── Red flags ────────────────────────────────────────────────────────────
    _sep("-")
    print("FLAGS:")
    flags = []

    # Check if any completed week was very low
    for ws in completed_weeks:
        ref_row = next((r for r in ref_build if r["week"] == ws.week_offset), None)
        if ref_row and ref_row["total_h"] > 2 and ws.total_h < ref_row["total_h"] * 0.6:
            flags.append(f"  [!] W{ws.week_offset:+d}: only {ws.total_h:.1f}h total "
                         f"vs ref {ref_row['total_h']:.1f}h (-{(1-ws.total_h/ref_row['total_h'])*100:.0f}%)")

    # Last hard day check
    if lhd_so_far is not None and lhd_so_far > -7:
        flags.append(f"  [!] Last hard day was D{lhd_so_far} — too close, risk of arriving tired")
    elif lhd_so_far is None and days_left < 10:
        flags.append(f"  [!] No hard day detected in last 21 days — possibly too little stimulus")

    # Taper start timing
    if days_left <= 14 and lhd_so_far is not None and lhd_so_far > -14:
        flags.append(f"  [!] Still training hard at D{lhd_so_far} with only {days_left} days to race")

    if not flags:
        flags.append("  [OK] No red flags detected based on available data")

    for f in flags:
        print(f)

    _sep()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-race build/taper check")
    parser.add_argument("race", type=str, help="Race name (partial match)")
    parser.add_argument("--long-only", action="store_true",
                        help="Compare vs long races (>60km) only")
    args = parser.parse_args()
    check(args.race, long_only=args.long_only)
