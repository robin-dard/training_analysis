"""
scripts/ytd_compare.py

Year-to-date training comparison across years.
Default: Jan 1 to today, years 2019 / 2025 / 2026.

Usage
-----
python scripts/ytd_compare.py
python scripts/ytd_compare.py --years 2023 2024 2025 2026
python scripts/ytd_compare.py --end 2026-04-15
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from analysis.build_taper_pattern import _classify, load_summaries

TODAY = date.today()


def run(years: list[int], end_md: tuple[int, int]) -> None:
    df = load_summaries()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["cat"]   = df["sport"].apply(_classify)
    df["h"]     = df["duration_s"].fillna(0) / 3600
    df["dplus"] = df["dplus_m"].fillna(0)
    df["km"]    = df["distance_km"].fillna(0)

    mm, dd = end_md

    # Aggregate per (sport, year)
    agg: dict[str, dict[int, dict]] = {}
    for cat in ["trail", "bike", "ski"]:
        agg[cat] = {}
        for yr in years:
            sub = df[
                (df["cat"] == cat) &
                (df["date"] >= f"{yr}-01-01") &
                (df["date"] <= f"{yr}-{mm:02d}-{dd:02d}")
            ]
            agg[cat][yr] = {
                "h":     round(sub["h"].sum(), 1),
                "km":    round(sub["km"].sum(), 0),
                "dplus": round(sub["dplus"].sum(), 0),
            }

    def fmt(x, unit):
        if unit == "h":  return f"{x:.1f}h"
        if unit == "km": return f"{int(x)}km"
        return f"{int(x)}m"

    def pct(a, b):
        return f"{(a - b) / b * 100:+.0f}%" if b > 0 else "n/a"

    col_w = 11
    yr_hdr = "".join(f"{str(yr):>{col_w}}" for yr in years)
    delta_hdr = "".join(("  vs " + str(years[i-1])).rjust(col_w+2) for i in range(1, len(years)))

    print()
    print(f"  YTD Jan 1 – {mm:02d}/{dd:02d}       {yr_hdr}{delta_hdr}")
    print("  " + "-" * (22 + col_w * len(years) + (col_w + 2) * (len(years) - 1)))

    for cat in ["trail", "bike", "ski"]:
        print(f"\n  [{cat.upper()}]")
        for metric, unit, label in [("h", "h", "hours"), ("km", "km", "distance"), ("dplus", "m", "elevation")]:
            vals = [agg[cat][yr][metric] for yr in years]
            line = f"  {label:<16}"
            for v in vals:
                line += f"{fmt(v, unit):>{col_w}}"
            for i in range(1, len(years)):
                line += f"  {pct(vals[i], vals[i-1]):>{col_w}}"
            print(line)

    # Summary: total aerobic
    print(f"\n  [TOTAL ALL SPORTS]")
    for metric, unit, label in [("h", "h", "hours"), ("dplus", "m", "elevation")]:
        vals = [sum(agg[cat][yr][metric] for cat in ["trail", "bike", "ski"]) for yr in years]
        line = f"  {label:<16}"
        for v in vals:
            line += f"{fmt(v, unit):>{col_w}}"
        for i in range(1, len(years)):
            line += f"  {pct(vals[i], vals[i-1]):>{col_w}}"
        print(line)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Year-to-date training comparison")
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2025, 2026],
                        metavar="YEAR", help="Years to compare (default: 2019 2025 2026)")
    parser.add_argument("--end", type=str, default=TODAY.strftime("%Y-%m-%d"),
                        metavar="MM-DD or YYYY-MM-DD",
                        help="End date as MM-DD or YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    raw = args.end
    if len(raw) == 5:  # MM-DD
        mm, dd = int(raw[:2]), int(raw[3:])
    else:              # YYYY-MM-DD
        mm, dd = int(raw[5:7]), int(raw[8:10])

    run(args.years, (mm, dd))
