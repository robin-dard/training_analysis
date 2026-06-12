"""
scripts/best_ascent_speed.py

Best ascensional speed (m/h) over 10', 20', 60' windows.
Top 3 trail efforts per year.

Results cached to data/best_ascent_cache.json — rerun is instant.
Delete the cache to force a full recompute.

Usage
-----
python scripts/best_ascent_speed.py
python scripts/best_ascent_speed.py --top 5
python scripts/best_ascent_speed.py --rebuild-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from analysis.build_taper_pattern import _classify, load_summaries
from parsers.fit_parser import parse_fit_file

CACHE_FILE   = _ROOT / "data/best_ascent_cache.json"
STREAMS_DIR  = _ROOT / "data/strava_streams"
FIT_DIR      = _ROOT / "data/fit"
WINDOWS      = {"10'": 600, "20'": 1200, "60'": 3600}

# Only analyse activities likely to have real climbs
MIN_DPLUS_M  = 200
MIN_DURATION_S = 1800   # 30 min


# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────

def best_ascent_mh(t: np.ndarray, alt: np.ndarray, window_s: int) -> float:
    """
    Maximum average ascent rate (m/h) over any window of exactly `window_s` seconds.
    Uses vectorised numpy — fast on 10k-point arrays.
    """
    t   = np.asarray(t,   dtype=float)
    alt = np.asarray(alt, dtype=float)
    n   = len(t)
    if n < 10:
        return 0.0

    # Robust smoothing: 11-point rolling median kills GPS spikes
    if n > 12:
        alt = (pd.Series(alt.astype(float))
               .rolling(11, center=True, min_periods=1)
               .median()
               .values)

    # Cumulative elevation gain
    dalt  = np.diff(alt)
    cgain = np.zeros(n)
    cgain[1:] = np.cumsum(np.maximum(dalt, 0.0))

    # For every start index i, find end index j = last point within window
    end_t   = t + window_s
    j_idx   = np.searchsorted(t, end_t, side="right") - 1
    j_idx   = np.minimum(j_idx, n - 1)

    durations = t[j_idx] - t
    valid     = (j_idx > np.arange(n)) & (durations >= window_s * 0.85)

    if not valid.any():
        return 0.0

    i_v = np.where(valid)[0]
    j_v = j_idx[i_v]
    gains = cgain[j_v] - cgain[i_v]
    rates = gains / (t[j_v] - t[i_v]) * 3600.0
    return float(round(rates.max(), 0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_strava(file_id: str) -> dict | None:
    """Load Strava stream parquet and compute best efforts."""
    path = STREAMS_DIR / f"{file_id}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "elapsed_s" not in df.columns or "altitude_m" not in df.columns:
            return None
        df = df.dropna(subset=["elapsed_s", "altitude_m"]).sort_values("elapsed_s")
        t   = df["elapsed_s"].values
        alt = df["altitude_m"].values
        return {w: best_ascent_mh(t, alt, s) for w, s in WINDOWS.items()}
    except Exception:
        return None


def _process_fit(filename: str) -> dict | None:
    """Parse FIT file and compute best efforts."""
    path = FIT_DIR / filename
    if not path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = parse_fit_file(path)
        if "elapsed_s" not in df.columns or "altitude_m" not in df.columns:
            return None
        df = df.dropna(subset=["elapsed_s", "altitude_m"]).sort_values("elapsed_s")
        t   = df["elapsed_s"].values
        alt = df["altitude_m"].values
        return {w: best_ascent_mh(t, alt, s) for w, s in WINDOWS.items()}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run(top_n: int = 3, rebuild: bool = False, skip_years: list[int] | None = None) -> None:
    summaries = load_summaries()
    summaries["date"] = pd.to_datetime(summaries["date"], errors="coerce")
    summaries = summaries.dropna(subset=["date"])
    summaries["cat"] = summaries["sport"].apply(_classify)

    # Filter to trail activities with meaningful elevation
    candidates = summaries[
        (summaries["cat"] == "trail") &
        (summaries["dplus_m"].fillna(0) >= MIN_DPLUS_M) &
        (summaries["duration_s"].fillna(0) >= MIN_DURATION_S)
    ].copy()

    print(f"Candidate trail activities: {len(candidates)}  (D+ >= {MIN_DPLUS_M}m, >= {MIN_DURATION_S//60}min)")

    cache = {} if rebuild else _load_cache()
    new_entries = 0

    for _, row in candidates.iterrows():
        key = row["file"]
        if key in cache:
            continue

        if row["source"] == "strava":
            file_id = key.replace(".parquet", "")
            result = _process_strava(file_id)
        else:
            result = _process_fit(key)

        if result:
            cache[key] = {
                "date":  str(row["date"].date()),
                "year":  int(row["date"].year),
                "sport": row["sport"],
                "dplus": float(row["dplus_m"] or 0),
                **result,
            }
        else:
            cache[key] = None

        new_entries += 1
        if new_entries % 50 == 0:
            _save_cache(cache)
            print(f"  processed {new_entries} new files ...")

    _save_cache(cache)
    if new_entries:
        print(f"  computed {new_entries} new entries, cache saved.")

    # Build results from cache
    records = [v for v in cache.values() if v]
    if not records:
        print("No results found.")
        return

    df = pd.DataFrame(records)
    if skip_years:
        df = df[~df["year"].isin(skip_years)]
    years = sorted(df["year"].unique())

    # ── Display ──────────────────────────────────────────────────────────────
    print()
    for win_label in WINDOWS:
        print(f"{'─'*65}")
        print(f"  BEST ASCENSIONAL SPEED — {win_label} window")
        print(f"{'─'*65}")

        sub = df[df[win_label] > 0].copy()
        sub = sub.sort_values(win_label, ascending=False)

        print(f"  {'Year':<6}  {'Rank':<5}  {'m/h':>6}  {'m/min':>6}  {'Date':<12}  Activity")
        print(f"  {'─'*5}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*11}  {'─'*30}")

        for yr in years:
            yr_sub = sub[sub["year"] == yr].head(top_n)
            if yr_sub.empty:
                continue
            for rank, (_, r) in enumerate(yr_sub.iterrows(), 1):
                mh  = int(r[win_label])
                mmn = round(mh / 60, 1)
                print(f"  {yr:<6}  #{rank:<4}  {mh:>6}  {mmn:>6}  {r['date']:<12}  D+{int(r['dplus'])}m")
        print()

    # ── Progress summary: best-ever per year per window ──────────────────────
    print(f"{'─'*65}")
    print("  YEAR-ON-YEAR PROGRESSION (best effort per year)")
    print(f"{'─'*65}")
    header = f"  {'Year':<6}" + "".join(f"  {w:>8}" for w in WINDOWS)
    print(header)
    print("  " + "─" * (6 + 12 * len(WINDOWS)))
    prev = {w: None for w in WINDOWS}
    for yr in years:
        yr_sub = df[df["year"] == yr]
        line = f"  {yr:<6}"
        for w in WINDOWS:
            best = yr_sub[w].max() if not yr_sub.empty else 0
            if best > 0:
                delta = f" ({best-prev[w]:+.0f})" if prev[w] else ""
                line += f"  {int(best):>5}m/h{delta:<7}"
                prev[w] = best
            else:
                line += f"  {'—':>8}  "
        print(line)
    print()


HTML_OUT = _ROOT / "data/best_ascent_report.html"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Best Ascensional Speed</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f1f5f9;padding:24px;font-size:14px}
h1{font-size:20px;font-weight:800;color:#0f172a;margin-bottom:4px}
.subtitle{font-size:12px;color:#64748b;margin-bottom:20px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}
@media(max-width:900px){.grid3{grid-template-columns:1fr}}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:18px}
.card h2{font-size:13px;font-weight:700;color:#0f172a;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.cwrap{position:relative;height:200px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{text-align:left;padding:5px 6px;background:#f8fafc;color:#64748b;font-weight:600;border-bottom:2px solid #e2e8f0}
td{padding:4px 6px;border-bottom:1px solid #f1f5f9;color:#334155;white-space:nowrap}
tr:hover td{background:#f8fafc}
.yr-hdr td{background:#f1f5f9;font-weight:700;color:#0f172a;padding:5px 6px}
.rank1 td:first-child{border-left:3px solid #16a34a}
.rank2 td:first-child{border-left:3px solid #3b82f6}
.rank3 td:first-child{border-left:3px solid #94a3b8}
.prog-card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:18px}
.prog-card h2{font-size:13px;font-weight:700;color:#0f172a;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.prog-wrap{position:relative;height:240px}
.delta-pos{color:#16a34a;font-weight:600}
.delta-neg{color:#dc2626;font-weight:600}
</style>
</head>
<body>
<h1>Best Ascensional Speed — Trail</h1>
<p class="subtitle">Top 3 efforts per year &nbsp;·&nbsp; 10' · 20' · 60' windows &nbsp;·&nbsp; m/h from GPS/baro altitude</p>

<div class="grid3" id="cards"></div>

<div class="prog-card">
  <h2>Year-on-year progression — best effort per year</h2>
  <div class="prog-wrap"><canvas id="prog-ch"></canvas></div>
</div>

<script>
const D = __DATA__;

const COLORS = {"10'":'rgba(234,88,12,.85)',"20'":'rgba(59,130,246,.85)',"60'":'rgba(22,163,74,.85)'};
const FILLS  = {"10'":'rgba(234,88,12,.12)',"20'":'rgba(59,130,246,.12)',"60'":'rgba(22,163,74,.12)'};

/* ── Per-window card ── */
D.windows.forEach(w => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<h2>${w.label} window</h2>
    <div class="cwrap"><canvas id="ch-${w.key}"></canvas></div>
    <table>
      <thead><tr><th>Year</th><th>Rank</th><th>m/h</th><th>m/min</th><th>Date</th><th>D+</th></tr></thead>
      <tbody id="tb-${w.key}"></tbody>
    </table>`;
  document.getElementById('cards').appendChild(card);

  /* bar chart: best per year */
  const years = w.best_per_year.map(r => r.year);
  const bests = w.best_per_year.map(r => r.mh);
  const barColors = bests.map((v,i) => {
    if(i===0) return COLORS[w.label];
    return v >= bests[i-1] ? 'rgba(22,163,74,.8)' : 'rgba(220,38,38,.7)';
  });
  new Chart(document.getElementById('ch-'+w.key).getContext('2d'), {
    type:'bar',
    data:{labels:years,datasets:[{
      label:'Best m/h',data:bests,backgroundColor:barColors,
      borderRadius:4,borderSkipped:false
    }]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},
        tooltip:{callbacks:{label:ctx=>{
          const r=w.best_per_year[ctx.dataIndex];
          const d=r.delta!=null?(r.delta>=0?' (+'+r.delta+')':' (-'+Math.abs(r.delta)+')'):'';
          return `${ctx.parsed.y} m/h${d} — ${r.date}`;
        }}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'m/h',font:{size:10}},
           ticks:{font:{size:10}},
           suggestedMin: Math.min(...bests)*0.9}
      }}
  });

  /* table: top 3 per year */
  const tbody = document.getElementById('tb-'+w.key);
  w.top3.forEach(yr => {
    const hdr = document.createElement('tr');
    hdr.className='yr-hdr';
    const delta = yr.delta!=null
      ? `<span class="${yr.delta>=0?'delta-pos':'delta-neg'}">${yr.delta>=0?'+':''}${yr.delta} m/h</span>`
      : '';
    hdr.innerHTML=`<td colspan="6">${yr.year} &nbsp; ${delta}</td>`;
    tbody.appendChild(hdr);
    yr.efforts.forEach((e,i) => {
      const tr = document.createElement('tr');
      tr.className=`rank${i+1}`;
      tr.innerHTML=`<td></td><td>#${i+1}</td><td><strong>${e.mh}</strong></td><td>${e.mmn}</td><td>${e.date}</td><td>D+${e.dplus}m</td>`;
      tbody.appendChild(tr);
    });
  });
});

/* ── Progression line chart ── */
const allYears = D.windows[0].best_per_year.map(r=>r.year);
new Chart(document.getElementById('prog-ch').getContext('2d'),{
  type:'line',
  data:{labels:allYears,datasets:D.windows.map(w=>({
    label:w.label,
    data:w.best_per_year.map(r=>r.mh),
    borderColor:COLORS[w.label],backgroundColor:FILLS[w.label],
    fill:true,tension:.3,borderWidth:2,pointRadius:4,pointHoverRadius:6
  }))},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{font:{size:11},boxWidth:12}}},
    scales:{
      x:{ticks:{font:{size:11}}},
      y:{title:{display:true,text:'m/h',font:{size:11}},ticks:{font:{size:11}}},
    }}
});
</script>
</body>
</html>
"""


def _build_html_data(df: pd.DataFrame, top_n: int) -> dict:
    years = sorted(df["year"].unique())
    windows_out = []
    for win_label, _ in WINDOWS.items():
        best_per_year = []
        top3_per_year = []
        prev_best = None
        for yr in years:
            sub = df[(df["year"] == yr) & (df[win_label] > 0)].sort_values(win_label, ascending=False)
            if sub.empty:
                continue
            best = int(sub.iloc[0][win_label])
            delta = (best - prev_best) if prev_best is not None else None
            best_per_year.append({
                "year": yr,
                "mh": best,
                "date": sub.iloc[0]["date"],
                "delta": delta,
            })
            efforts = []
            for _, r in sub.head(top_n).iterrows():
                mh = int(r[win_label])
                efforts.append({
                    "mh": mh,
                    "mmn": round(mh / 60, 1),
                    "date": r["date"],
                    "dplus": int(r["dplus"]),
                })
            top3_per_year.append({"year": yr, "delta": delta, "efforts": efforts})
            prev_best = best
        windows_out.append({
            "key": win_label.replace("'", "m"),
            "label": win_label,
            "best_per_year": best_per_year,
            "top3": top3_per_year,
        })
    return {"windows": windows_out}


def generate_html(df: pd.DataFrame, top_n: int, open_browser: bool) -> None:
    data = _build_html_data(df, top_n)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, default=int))
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Saved to {HTML_OUT.resolve()}")
    if open_browser:
        import webbrowser
        webbrowser.open(HTML_OUT.resolve().as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best ascensional speed per year")
    parser.add_argument("--top", type=int, default=3, metavar="N",
                        help="Show top N efforts per year (default: 3)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force recompute all activities (ignores cache)")
    parser.add_argument("--skip-years", type=int, nargs="+", default=[],
                        metavar="YEAR", help="Exclude years from output (e.g. --skip-years 2018)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML report instead of terminal output")
    parser.add_argument("--open", action="store_true",
                        help="Open HTML report in browser after generation")
    args = parser.parse_args()

    skip = args.skip_years or None
    if args.html or args.open:
        # Load cache directly, skip terminal output
        cache = _load_cache()
        records = [v for v in cache.values() if v]
        df = pd.DataFrame(records)
        if skip:
            df = df[~df["year"].isin(skip)]
        generate_html(df, args.top, args.open)
    else:
        run(top_n=args.top, rebuild=args.rebuild_cache, skip_years=skip)
