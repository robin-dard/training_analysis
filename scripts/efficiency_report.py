"""
scripts/efficiency_report.py

Generate data/efficiency_report.html.

Analyses two session types:
  - Track: detect work intervals, compute speed/HR efficiency per block
  - Hills: detect structured repeats (D+ 400-900m, >= 3 reps), compute asc_speed/HR

Also produces zone-speed progression (track only) — median speed per HR zone by year.

Cache at data/efficiency_cache.json prevents re-parsing FIT files on subsequent runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.efficiency import (
    detect_hill_repeats,
    detect_track_intervals,
    zone_speed_ms,
)
from analysis.hr_analysis import DEFAULT_HR_ZONES, DEFAULT_MAX_HR
from parsers.fit_parser import parse_fit_file

SUMMARIES    = _ROOT / "data/summaries.parquet"
FIT_DIR      = _ROOT / "data/fit"
CACHE_FILE   = _ROOT / "data/efficiency_cache.json"
OUT_HTML     = _ROOT / "data/efficiency_report.html"

# Trail sessions need at least 3 x 400m = 1200m D+ to be candidates
_HILL_MIN_DPLUS = 1200.0

# Downsample time series to this resolution for embedded chart data
_TS_STEP_S = 10


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {"track": {}, "hills": {}}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def _downsample(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every _TS_STEP_S row to reduce embedded data size."""
    if df.empty or "elapsed_s" not in df.columns:
        return df
    step = _TS_STEP_S
    df = df.copy()
    df["_bucket"] = (df["elapsed_s"] / step).astype(int)
    return df.groupby("_bucket", sort=True).last().reset_index(drop=True)


def _ts_arrays(df: pd.DataFrame) -> dict:
    """Extract time series arrays for chart embedding."""
    ds = _downsample(df)
    t = ds["elapsed_s"].round(0).astype(int).tolist() if "elapsed_s" in ds.columns else []
    speed = [round(v * 3.6, 2) if pd.notna(v) else None
             for v in ds.get("speed_ms", pd.Series())] if "speed_ms" in ds.columns else []
    hr = [round(v, 0) if pd.notna(v) else None
          for v in ds.get("heart_rate", pd.Series())] if "heart_rate" in ds.columns else []
    return {"t": t, "speed_kmh": speed, "hr": hr}


def _process_track(fit_path: Path) -> dict | None:
    """Parse FIT, detect intervals, compute zone speeds. Returns None on error."""
    try:
        df = parse_fit_file(fit_path)
    except Exception as e:
        print(f"  [skip] {fit_path.name}: {e}")
        return None

    intervals = detect_track_intervals(df)
    work = [i for i in intervals if i.is_work]
    if not work:
        return None

    zone_sp = zone_speed_ms(df, max_hr=DEFAULT_MAX_HR, zones=DEFAULT_HR_ZONES)

    return {
        "intervals": [
            {
                "start_s": i.start_s, "end_s": i.end_s, "is_work": i.is_work,
                "duration_s": round(i.duration_s, 0),
                "speed_kmh": i.speed_kmh,
                "mean_hr": i.mean_hr,
                "efficiency": i.efficiency,
            }
            for i in intervals
        ],
        "work_count": len(work),
        "mean_efficiency": (
            round(sum(i.efficiency for i in work if i.efficiency) / len(work), 5)
            if work else None
        ),
        "zone_speeds_kmh": {
            k: round(v * 3.6, 2) if v else None for k, v in zone_sp.items()
        },
        "ts": _ts_arrays(df),
    }


def _process_hills(fit_path: Path) -> dict | None:
    """Parse FIT, detect hill repeats. Returns None on error or no repeats."""
    try:
        df = parse_fit_file(fit_path)
    except Exception as e:
        print(f"  [skip] {fit_path.name}: {e}")
        return None

    repeats = detect_hill_repeats(df)
    if not repeats:
        return {"no_repeats": True}  # cached as negative result

    eff_vals = [r.efficiency for r in repeats if r.efficiency is not None]

    return {
        "repeats": [
            {
                "repeat_num": r.repeat_num,
                "start_s": r.start_s, "end_s": r.end_s,
                "duration_s": round(r.duration_s, 0),
                "dplus_m": r.dplus_m,
                "dist_m": r.dist_m,
                "avg_grade_pct": r.avg_grade_pct,
                "asc_speed_mh": r.asc_speed_mh,
                "mean_hr": r.mean_hr,
                "efficiency": r.efficiency,
            }
            for r in repeats
        ],
        "repeat_count": len(repeats),
        "mean_efficiency": round(sum(eff_vals) / len(eff_vals), 3) if eff_vals else None,
        "ts": _ts_arrays(df),
    }


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def build_cache(rebuild: bool = False) -> dict:
    """Process all track and trail-repeat sessions, update cache."""
    cache = {} if rebuild else _load_cache()
    if "track" not in cache:
        cache["track"] = {}
    if "hills" not in cache:
        cache["hills"] = {}

    if not SUMMARIES.exists():
        print(f"ERROR: {SUMMARIES} not found — run sync/build_summaries.py first.")
        sys.exit(1)

    df = pd.read_parquet(SUMMARIES)

    # Track sessions
    track_df = df[df["sport"] == "track_running"].copy()
    track_new = [
        r for _, r in track_df.iterrows()
        if r.get("file") and r["file"] not in cache["track"]
    ]
    if track_new:
        print(f"Track sessions: {len(track_new)} new to process ({len(track_df)} total)...")
        for row in tqdm(track_new, desc="Track", unit="session"):
            fit_path = FIT_DIR / row["file"]
            if not fit_path.exists():
                continue
            result = _process_track(fit_path)
            date_str = str(row.get("date", ""))[:10]
            cache["track"][row["file"]] = {
                "date": date_str,
                "distance_km": row.get("distance_km"),
                "duration_s": row.get("duration_s"),
                **(result or {"no_intervals": True}),
            }
    else:
        print(f"Track sessions: 0 new (cache has {len(cache['track'])} sessions).")

    # Hill repeat candidates: trail sessions with enough D+
    hill_candidates = df[
        (df["sport"] == "trail_running") &
        (df["dplus_m"].fillna(0) >= _HILL_MIN_DPLUS)
    ].copy()
    hills_new = [
        r for _, r in hill_candidates.iterrows()
        if r.get("file") and r["file"] not in cache["hills"]
    ]
    if hills_new:
        print(f"Hill candidates: {len(hills_new)} new to check ({len(hill_candidates)} total)...")
        for row in tqdm(hills_new, desc="Hills", unit="session"):
            fit_path = FIT_DIR / row["file"]
            if not fit_path.exists():
                continue
            result = _process_hills(fit_path)
            date_str = str(row.get("date", ""))[:10]
            cache["hills"][row["file"]] = {
                "date": date_str,
                "distance_km": row.get("distance_km"),
                "duration_s": row.get("duration_s"),
                "dplus_m": row.get("dplus_m"),
                **(result or {"no_repeats": True}),
            }
    else:
        print(f"Hill sessions: 0 new (cache has {len(cache['hills'])} sessions).")

    _save_cache(cache)
    return cache


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _fmt_min(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}" if h else f"{m}:{s:02d}"


def _zone_color(zone: str) -> str:
    colors = {
        "Z1_recovery":  "#74b9ff",
        "Z2_aerobic":   "#00b894",
        "Z3_tempo":     "#fdcb6e",
        "Z4_threshold": "#e17055",
        "Z5_vo2max":    "#d63031",
    }
    return colors.get(zone, "#636e72")


def render_html(cache: dict) -> str:
    # Build track session objects (only those with work intervals)
    track_sessions = {}
    for fname, data in sorted(cache.get("track", {}).items()):
        if data.get("no_intervals") or not data.get("intervals"):
            continue
        work = [i for i in data["intervals"] if i["is_work"] and i.get("efficiency")]
        if not work:
            continue
        track_sessions[fname] = data

    # Build hill session objects (only those with repeats)
    hill_sessions = {}
    for fname, data in sorted(cache.get("hills", {}).items()):
        if data.get("no_repeats") or not data.get("repeats"):
            continue
        hill_sessions[fname] = data

    # Zone speed by year (track only)
    zone_by_year: dict[int, dict[str, list[float]]] = {}
    for fname, data in track_sessions.items():
        year = int(data["date"][:4]) if len(data.get("date", "")) >= 4 else 0
        if year < 2018:
            continue
        if year not in zone_by_year:
            zone_by_year[year] = {z: [] for z in DEFAULT_HR_ZONES}
        zs = data.get("zone_speeds_kmh", {})
        for z in DEFAULT_HR_ZONES:
            v = zs.get(z)
            if v:
                zone_by_year[year][z].append(v)

    zone_years = sorted(zone_by_year)
    zone_series: dict[str, list[float | None]] = {z: [] for z in DEFAULT_HR_ZONES}
    for yr in zone_years:
        for z in DEFAULT_HR_ZONES:
            vals = zone_by_year[yr].get(z, [])
            zone_series[z].append(round(sum(vals) / len(vals), 2) if vals else None)

    # Efficiency trend series
    track_trend = [
        {"date": d["date"], "eff": d["mean_efficiency"], "file": f,
         "n": d.get("work_count", 0)}
        for f, d in sorted(track_sessions.items(), key=lambda x: x[1]["date"])
        if d.get("mean_efficiency")
    ]
    hill_trend = [
        {"date": d["date"], "eff": d["mean_efficiency"], "file": f,
         "n": d.get("repeat_count", 0)}
        for f, d in sorted(hill_sessions.items(), key=lambda x: x[1]["date"])
        if d.get("mean_efficiency")
    ]

    # Sidebar items
    def track_item(fname: str, d: dict) -> str:
        eff = d.get("mean_efficiency")
        eff_str = f"{eff*1000:.2f}" if eff else "—"
        return (
            f'<div class="sess-item" data-type="track" data-file="{fname}" '
            f'onclick="selectSession(this,\'track\')">'
            f'<span class="sess-date">{d["date"]}</span>'
            f'<span class="sess-meta">{d.get("work_count",0)} blocs · {eff_str} eff</span>'
            f'</div>'
        )

    def hill_item(fname: str, d: dict) -> str:
        eff = d.get("mean_efficiency")
        eff_str = f"{eff:.2f}" if eff else "—"
        n = d.get("repeat_count", 0)
        return (
            f'<div class="sess-item" data-type="hills" data-file="{fname}" '
            f'onclick="selectSession(this,\'hills\')">'
            f'<span class="sess-date">{d["date"]}</span>'
            f'<span class="sess-meta">{n} reps · {eff_str} eff</span>'
            f'</div>'
        )

    track_sidebar = "\n".join(track_item(f, d) for f, d in sorted(track_sessions.items(), key=lambda x: x[1]["date"], reverse=True))
    hill_sidebar  = "\n".join(hill_item(f, d) for f, d in sorted(hill_sessions.items(), key=lambda x: x[1]["date"], reverse=True))

    zone_colors = {z: _zone_color(z) for z in DEFAULT_HR_ZONES}
    zone_labels = {
        "Z1_recovery": "Z1 Récup",
        "Z2_aerobic": "Z2 Aérobie",
        "Z3_tempo": "Z3 Tempo",
        "Z4_threshold": "Z4 Seuil",
        "Z5_vo2max": "Z5 VO2max",
    }

    sessions_js = json.dumps(
        {**{f"track:{k}": v for k, v in track_sessions.items()},
         **{f"hills:{k}": v for k, v in hill_sessions.items()}},
        ensure_ascii=False,
    )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Efficiency Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; height: 100vh; display: flex;
         flex-direction: column; overflow: hidden; }}
  header {{ background: #1a1d27; border-bottom: 1px solid #2d3142;
            padding: 12px 20px; display: flex; align-items: center; gap: 20px;
            flex-shrink: 0; }}
  header h1 {{ font-size: 16px; font-weight: 600; color: #fff; letter-spacing: .5px; }}
  .tab-bar {{ display: flex; gap: 4px; }}
  .tab {{ background: transparent; border: 1px solid #3d4263; border-radius: 6px;
          color: #9a9db8; padding: 5px 14px; font-size: 13px; cursor: pointer;
          transition: all .15s; }}
  .tab:hover {{ border-color: #6c7aff; color: #c5c8ff; }}
  .tab.active {{ background: #3d4775; border-color: #6c7aff; color: #fff; }}
  .layout {{ display: flex; flex: 1; overflow: hidden; }}

  /* Sidebar */
  .sidebar {{ width: 240px; border-right: 1px solid #2d3142; background: #13151f;
              display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }}
  .sidebar-header {{ padding: 10px 14px; font-size: 11px; font-weight: 600;
                     color: #6b7089; text-transform: uppercase; letter-spacing: .8px;
                     border-bottom: 1px solid #2d3142; }}
  .sidebar-list {{ flex: 1; overflow-y: auto; }}
  .sess-item {{ padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #1e2030;
                transition: background .1s; }}
  .sess-item:hover {{ background: #1e2133; }}
  .sess-item.active {{ background: #2a3050; border-left: 3px solid #6c7aff; }}
  .sess-date {{ display: block; font-size: 13px; font-weight: 500; color: #cdd0e8; }}
  .sess-meta {{ display: block; font-size: 11px; color: #5c6080; margin-top: 2px; }}

  /* Main panel */
  .main {{ flex: 1; overflow-y: auto; padding: 20px 24px; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .overview-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  @media (max-width: 900px) {{ .overview-grid {{ grid-template-columns: 1fr; }} }}

  /* Cards */
  .card {{ background: #1a1d27; border: 1px solid #2d3142; border-radius: 10px;
           padding: 16px 20px; }}
  .card-title {{ font-size: 12px; font-weight: 600; color: #6b7089;
                 text-transform: uppercase; letter-spacing: .7px; margin-bottom: 10px; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-tall {{ position: relative; height: 280px; }}
  h2.sess-heading {{ font-size: 18px; color: #fff; margin-bottom: 4px; }}
  .sess-subheading {{ font-size: 13px; color: #6b7089; margin-bottom: 16px; }}
  .stat-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat {{ text-align: center; min-width: 80px; }}
  .stat-val {{ font-size: 22px; font-weight: 700; color: #6c7aff; }}
  .stat-lbl {{ font-size: 11px; color: #5c6080; margin-top: 2px; }}
  .no-data {{ color: #5c6080; font-style: italic; padding: 40px 0; text-align: center; font-size: 14px; }}

  /* Interval table */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th {{ text-align: left; padding: 6px 10px; color: #6b7089; font-weight: 600;
        font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
        border-bottom: 1px solid #2d3142; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e2030; color: #cdd0e8; }}
  tr.work-row {{ background: rgba(108,122,255,.06); }}
  tr:hover {{ background: rgba(255,255,255,.03); }}
  .eff-badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px;
                font-size: 12px; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>Efficiency Report</h1>
  <div class="tab-bar">
    <button class="tab active" onclick="switchTab('track')">Track</button>
    <button class="tab" onclick="switchTab('hills')">Hills</button>
    <button class="tab" onclick="switchTab('zones')">Zone Speed</button>
  </div>
</header>
<div class="layout">
  <!-- Sidebar -->
  <nav class="sidebar">
    <div class="sidebar-header" id="sidebar-label">Track Sessions ({len(track_sessions)})</div>
    <div class="sidebar-list" id="sidebar-track">
{track_sidebar}
    </div>
    <div class="sidebar-list" id="sidebar-hills" style="display:none">
{hill_sidebar}
    </div>
  </nav>

  <!-- Main panel -->
  <main class="main">
    <!-- TRACK panel -->
    <div id="panel-track" class="panel active">
      <div class="overview-grid">
        <div class="card">
          <div class="card-title">Efficiency — Track (speed/HR ×1000)</div>
          <div class="chart-wrap"><canvas id="chart-track-trend"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title">Session Detail — click a session in the sidebar</div>
          <div id="track-detail">
            <p class="no-data">Sélectionner une session dans la barre latérale</p>
          </div>
        </div>
      </div>
      <div id="track-ts-card" class="card" style="display:none; margin-bottom:16px">
        <div class="card-title">Vitesse &amp; FC — <span id="track-ts-title"></span></div>
        <div class="chart-wrap-tall"><canvas id="chart-track-ts"></canvas></div>
      </div>
      <div id="track-eff-card" class="card" style="display:none">
        <div class="card-title">Efficacité par bloc</div>
        <div class="chart-wrap"><canvas id="chart-track-eff"></canvas></div>
      </div>
    </div>

    <!-- HILLS panel -->
    <div id="panel-hills" class="panel">
      <div class="overview-grid">
        <div class="card">
          <div class="card-title">Efficiency — Hills (asc_speed/HR)</div>
          <div class="chart-wrap"><canvas id="chart-hills-trend"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title">Session Detail — click a session in the sidebar</div>
          <div id="hills-detail">
            <p class="no-data">Sélectionner une session dans la barre latérale</p>
          </div>
        </div>
      </div>
      <div id="hills-ts-card" class="card" style="display:none; margin-bottom:16px">
        <div class="card-title">Asc. speed &amp; FC — <span id="hills-ts-title"></span></div>
        <div class="chart-wrap-tall"><canvas id="chart-hills-ts"></canvas></div>
      </div>
      <div id="hills-eff-card" class="card" style="display:none">
        <div class="card-title">Efficacité par répétition</div>
        <div class="chart-wrap"><canvas id="chart-hills-eff"></canvas></div>
      </div>
    </div>

    <!-- ZONES panel -->
    <div id="panel-zones" class="panel">
      <div class="card" style="margin-bottom:16px">
        <div class="card-title">Vitesse par zone HR — évolution annuelle (sessions track)</div>
        <div class="chart-wrap-tall"><canvas id="chart-zones"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Tableau — vitesse médiane (km/h) par zone et par année</div>
        <div id="zones-table"></div>
      </div>
    </div>
  </main>
</div>

<script>
// ── embedded data ──────────────────────────────────────────────────────────
const SESSIONS = {sessions_js};
const TRACK_TREND = {json.dumps(track_trend)};
const HILL_TREND  = {json.dumps(hill_trend)};
const ZONE_YEARS  = {json.dumps(zone_years)};
const ZONE_SERIES = {json.dumps(zone_series)};
const ZONE_COLORS = {json.dumps(zone_colors)};
const ZONE_LABELS = {json.dumps(zone_labels)};

// ── chart helpers ──────────────────────────────────────────────────────────
let chartInstances = {{}};

function destroyChart(id) {{
  if (chartInstances[id]) {{ chartInstances[id].destroy(); delete chartInstances[id]; }}
}}

function mkChart(id, cfg) {{
  destroyChart(id);
  chartInstances[id] = new Chart(document.getElementById(id), cfg);
}}

const darkGrid = {{ color: 'rgba(255,255,255,.06)' }};
const darkTick = {{ color: '#5c6080' }};

// ── trend charts ───────────────────────────────────────────────────────────
function buildTrendChart(canvasId, trend, multiplier, yLabel, color) {{
  const labels = trend.map(d => d.date);
  const data   = trend.map(d => d.eff ? +(d.eff * multiplier).toFixed(3) : null);
  mkChart(canvasId, {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: yLabel,
        data: trend.map(d => ({{ x: d.date, y: d.eff ? +(d.eff * multiplier).toFixed(3) : null }})),
        backgroundColor: color,
        pointRadius: 5, pointHoverRadius: 7,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{
        callbacks: {{ label: ctx => `${{ctx.raw.x}}: ${{ctx.raw.y}} (${{trend[ctx.dataIndex]?.n}} blocs)` }}
      }} }},
      scales: {{
        x: {{ type: 'category', ticks: {{ ...darkTick, maxTicksLimit: 6, maxRotation: 0 }}, grid: darkGrid }},
        y: {{ title: {{ display: true, text: yLabel, color: '#6b7089' }}, ticks: darkTick, grid: darkGrid }},
      }},
    }}
  }});
}}

buildTrendChart('chart-track-trend', TRACK_TREND, 1000, 'speed/HR ×1000', 'rgba(108,122,255,.75)');
buildTrendChart('chart-hills-trend', HILL_TREND,  1,    'asc_speed/HR',    'rgba(46,204,113,.75)');

// ── zone speed chart ───────────────────────────────────────────────────────
(function() {{
  const zones = ['Z2_aerobic', 'Z3_tempo', 'Z4_threshold'];
  const ds = zones.map(z => ({{
    label: ZONE_LABELS[z],
    data: ZONE_SERIES[z],
    borderColor: ZONE_COLORS[z],
    backgroundColor: ZONE_COLORS[z] + '33',
    tension: .3,
    fill: false,
    pointRadius: 4,
  }}));
  mkChart('chart-zones', {{
    type: 'line',
    data: {{ labels: ZONE_YEARS.map(String), datasets: ds }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#cdd0e8', font: {{ size: 12 }} }} }} }},
      scales: {{
        x: {{ ticks: darkTick, grid: darkGrid }},
        y: {{ title: {{ display: true, text: 'Vitesse (km/h)', color: '#6b7089' }},
              ticks: darkTick, grid: darkGrid }},
      }},
    }}
  }});

  // Table
  const zAll = Object.keys(ZONE_LABELS);
  let html = '<table><tr><th>Année</th>' + zAll.map(z => `<th>${{ZONE_LABELS[z]}}</th>`).join('') + '</tr>';
  ZONE_YEARS.forEach((yr, i) => {{
    html += `<tr><td>${{yr}}</td>` + zAll.map(z => {{
      const v = ZONE_SERIES[z]?.[i]; return `<td>${{v != null ? v : '—'}}</td>`;
    }}).join('') + '</tr>';
  }});
  html += '</table>';
  document.getElementById('zones-table').innerHTML = html;
}})();

// ── session selection ──────────────────────────────────────────────────────
let activeEl = null;

function selectSession(el, type) {{
  if (activeEl) activeEl.classList.remove('active');
  el.classList.add('active');
  activeEl = el;
  const fname = el.dataset.file;
  const key   = `${{type}}:${{fname}}`;
  const data  = SESSIONS[key];
  if (!data) return;
  if (type === 'track') renderTrack(fname, data);
  else renderHills(fname, data);
}}

// ── track detail ───────────────────────────────────────────────────────────
function renderTrack(fname, data) {{
  const intervals = data.intervals || [];
  const work = intervals.filter(i => i.is_work);
  const dur  = data.duration_s ? fmtHms(data.duration_s) : '—';
  const dist = data.distance_km ? data.distance_km + ' km' : '—';
  const eff  = data.mean_efficiency ? (data.mean_efficiency * 1000).toFixed(2) : '—';

  document.getElementById('track-detail').innerHTML = `
    <h2 class="sess-heading">${{data.date}}</h2>
    <div class="sess-subheading">${{dist}} · ${{dur}} · ${{work.length}} blocs de travail</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">${{eff}}</div><div class="stat-lbl">eff ×1000</div></div>
      <div class="stat"><div class="stat-val">${{work.length}}</div><div class="stat-lbl">blocs</div></div>
      <div class="stat"><div class="stat-val">${{work.length > 0 ? work[0].speed_kmh.toFixed(1) : '—'}}</div><div class="stat-lbl">moy 1er bloc km/h</div></div>
    </div>
    <table>
      <tr><th>#</th><th>Durée</th><th>Vitesse km/h</th><th>FC moy</th><th>Efficacité ×1000</th></tr>
      ${{intervals.filter(i=>i.is_work).map((iv,i) => `
        <tr class="work-row">
          <td>${{i+1}}</td><td>${{fmtMin(iv.duration_s)}}</td>
          <td>${{iv.speed_kmh.toFixed(2)}}</td>
          <td>${{iv.mean_hr ? iv.mean_hr.toFixed(0) : '—'}}</td>
          <td><span class="eff-badge" style="background:${{effColor(iv.efficiency, 'track')}}">${{iv.efficiency ? (iv.efficiency*1000).toFixed(2) : '—'}}</span></td>
        </tr>`).join('')}}
    </table>`;

  // Time series
  const ts = data.ts || {{}};
  if (ts.t && ts.t.length) {{
    document.getElementById('track-ts-card').style.display = '';
    document.getElementById('track-ts-title').textContent = data.date;
    destroyChart('chart-track-ts');
    const annots = intervals.filter(i => i.is_work).map(iv => ({{
      type: 'box', xMin: fmtElapsed(iv.start_s), xMax: fmtElapsed(iv.end_s),
      backgroundColor: 'rgba(108,122,255,.12)', borderColor: 'transparent',
    }}));
    buildTsChart('chart-track-ts', ts, 'Vitesse (km/h)', 'rgba(108,122,255,.8)', intervals);
  }} else {{
    document.getElementById('track-ts-card').style.display = 'none';
  }}

  // Efficiency bars
  const workWithEff = intervals.filter(i => i.is_work && i.efficiency);
  if (workWithEff.length) {{
    document.getElementById('track-eff-card').style.display = '';
    mkChart('chart-track-eff', {{
      type: 'bar',
      data: {{
        labels: workWithEff.map((_,i) => `Bloc ${{i+1}}`),
        datasets: [{{
          label: 'Efficacité ×1000',
          data: workWithEff.map(iv => +(iv.efficiency * 1000).toFixed(3)),
          backgroundColor: workWithEff.map(iv => effColor(iv.efficiency, 'track')),
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: darkTick, grid: darkGrid }},
          y: {{ title: {{ display: true, text: 'speed/HR ×1000', color: '#6b7089' }},
                ticks: darkTick, grid: darkGrid }},
        }}
      }}
    }});
  }} else {{
    document.getElementById('track-eff-card').style.display = 'none';
  }}
}}

// ── hill detail ────────────────────────────────────────────────────────────
function renderHills(fname, data) {{
  const reps = data.repeats || [];
  const dur  = data.duration_s ? fmtHms(data.duration_s) : '—';
  const dist = data.distance_km ? data.distance_km + ' km' : '—';
  const eff  = data.mean_efficiency ? data.mean_efficiency.toFixed(2) : '—';
  const dplus = data.dplus_m ? Math.round(data.dplus_m) + ' m D+' : '—';

  document.getElementById('hills-detail').innerHTML = `
    <h2 class="sess-heading">${{data.date}}</h2>
    <div class="sess-subheading">${{dist}} · ${{dur}} · ${{dplus}} · ${{reps.length}} répétitions</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">${{eff}}</div><div class="stat-lbl">eff (asc/FC)</div></div>
      <div class="stat"><div class="stat-val">${{reps.length}}</div><div class="stat-lbl">reps</div></div>
      <div class="stat"><div class="stat-val">${{reps.length > 0 ? Math.round(reps[0].dplus_m) : '—'}}</div><div class="stat-lbl">D+ rep1 (m)</div></div>
    </div>
    <table>
      <tr><th>#</th><th>Durée</th><th>D+</th><th>Pente %</th><th>Vit. asc</th><th>FC moy</th><th>Efficacité</th></tr>
      ${{reps.map((r,i) => `
        <tr class="work-row">
          <td>${{r.repeat_num}}</td><td>${{fmtMin(r.duration_s)}}</td>
          <td>${{Math.round(r.dplus_m)}} m</td>
          <td>${{r.avg_grade_pct.toFixed(1)}}%</td>
          <td>${{Math.round(r.asc_speed_mh)}} m/h</td>
          <td>${{r.mean_hr ? r.mean_hr.toFixed(0) : '—'}}</td>
          <td><span class="eff-badge" style="background:${{effColor(r.efficiency, 'hills')}}">${{r.efficiency ? r.efficiency.toFixed(2) : '—'}}</span></td>
        </tr>`).join('')}}
    </table>`;

  const ts = data.ts || {{}};
  if (ts.t && ts.t.length) {{
    document.getElementById('hills-ts-card').style.display = '';
    document.getElementById('hills-ts-title').textContent = data.date;
    buildTsChart('chart-hills-ts', ts, 'FC', 'rgba(46,204,113,.8)', []);
  }} else {{
    document.getElementById('hills-ts-card').style.display = 'none';
  }}

  const repsWithEff = reps.filter(r => r.efficiency);
  if (repsWithEff.length) {{
    document.getElementById('hills-eff-card').style.display = '';
    mkChart('chart-hills-eff', {{
      type: 'bar',
      data: {{
        labels: repsWithEff.map(r => `Rep ${{r.repeat_num}}`),
        datasets: [{{
          label: 'asc_speed / FC',
          data: repsWithEff.map(r => +r.efficiency.toFixed(3)),
          backgroundColor: repsWithEff.map(r => effColor(r.efficiency, 'hills')),
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: darkTick, grid: darkGrid }},
          y: {{ title: {{ display: true, text: 'asc_speed/FC', color: '#6b7089' }},
                ticks: darkTick, grid: darkGrid }},
        }}
      }}
    }});
  }} else {{
    document.getElementById('hills-eff-card').style.display = 'none';
  }}
}}

// ── dual-axis time series chart ────────────────────────────────────────────
function buildTsChart(canvasId, ts, speedLabel, speedColor, intervals) {{
  const labels = ts.t.map(s => Math.round(s / 60) + 'min');
  const maxLabel = Math.ceil(ts.t[ts.t.length-1] / 60);

  // Work interval background via annotation plugin (not available — use dataset approach)
  const workBands = intervals.filter(i => i && i.is_work).map(iv => ({{
    start: Math.round(iv.start_s / 60), end: Math.round(iv.end_s / 60)
  }}));

  mkChart(canvasId, {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{
          label: speedLabel,
          data: ts.speed_kmh,
          yAxisID: 'y',
          borderColor: speedColor,
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: .2,
          spanGaps: true,
        }},
        {{
          label: 'FC',
          data: ts.hr,
          yAxisID: 'y2',
          borderColor: 'rgba(231,76,60,.75)',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: .2,
          spanGaps: true,
        }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ labels: {{ color: '#cdd0e8', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ ...darkTick, maxTicksLimit: 8, maxRotation: 0 }}, grid: darkGrid }},
        y:  {{ position: 'left',  title: {{ display: true, text: speedLabel, color: '#6b7089' }},
               ticks: darkTick, grid: darkGrid }},
        y2: {{ position: 'right', title: {{ display: true, text: 'FC (bpm)', color: '#6b7089' }},
               ticks: darkTick, grid: {{ drawOnChartArea: false }} }},
      }}
    }}
  }});
}}

// ── tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  document.getElementById('sidebar-track').style.display = tab === 'track' ? '' : 'none';
  document.getElementById('sidebar-hills').style.display = tab === 'hills' ? '' : 'none';
  const labels = {{ track: `Track Sessions ({len(track_sessions)})`, hills: `Hill Sessions ({len(hill_sessions)})`, zones: 'Zone Speed' }};
  document.getElementById('sidebar-label').textContent = labels[tab] || '';
}}

// ── helpers ────────────────────────────────────────────────────────────────
function fmtMin(s) {{
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${{m}}:${{String(sec).padStart(2,'0')}}`;
}}
function fmtHms(s) {{
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h ? `${{h}}h${{String(m).padStart(2,'0')}}` : `${{m}}min`;
}}
function fmtElapsed(s) {{
  return Math.round(s/60) + 'min';
}}

// Colour an efficiency value relative to all sessions
const trackEffs = TRACK_TREND.map(d => d.eff * 1000).filter(Boolean).sort((a,b)=>a-b);
const hillEffs  = HILL_TREND.map(d => d.eff).filter(Boolean).sort((a,b)=>a-b);

function effColor(eff, type) {{
  if (!eff) return '#3d4263';
  const arr = type === 'track' ? trackEffs : hillEffs;
  const v = type === 'track' ? eff * 1000 : eff;
  const p = arr.filter(x => x <= v).length / arr.length;
  if (p >= .8) return 'rgba(46,204,113,.55)';
  if (p >= .5) return 'rgba(253,203,110,.55)';
  return 'rgba(214,63,49,.45)';
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    import argparse
    parser = argparse.ArgumentParser(description="Generate efficiency_report.html")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore cache and re-parse all sessions")
    args = parser.parse_args()

    cache = build_cache(rebuild=args.rebuild_cache)

    # Filter to sessions with actual results
    track_ok = {k: v for k, v in cache["track"].items()
                if not v.get("no_intervals") and v.get("intervals")}
    hill_ok  = {k: v for k, v in cache["hills"].items()
                if not v.get("no_repeats") and v.get("repeats")}

    print(f"\nSessions with intervals: track={len(track_ok)}, hills={len(hill_ok)}")

    html = render_html(cache)
    OUT_HTML.write_text(html, encoding="utf-8")

    t1 = time.perf_counter()
    print(f"Written: {OUT_HTML}  ({len(html)//1024} KB)  [{t1-t0:.1f}s]")


if __name__ == "__main__":
    main()
