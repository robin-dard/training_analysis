"""
scripts/efficiency_report.py

Generate data/efficiency_report.html.

Two session types:
  - Track  : detect work intervals (FIT laps preferred, speed threshold fallback)
  - Hills  : detect structured repeats (D+ 400-900m, >= 3, similar D+)

Zone-speed progression (track only) — median speed per HR zone, by year.
Cache at data/efficiency_cache.json; first run ~10 min, subsequent runs instant.

Run with --rebuild-cache to force re-parse of all sessions (needed after any
change to detection parameters or when altitude was missing from old cache).
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
    detect_track_intervals_from_laps,
    zone_speed_ms,
)
from analysis.hr_analysis import DEFAULT_HR_ZONES, DEFAULT_MAX_HR
from parsers.fit_parser import fit_metadata, parse_fit_file, parse_fit_laps

SUMMARIES    = _ROOT / "data/summaries.parquet"
FIT_DIR      = _ROOT / "data/fit"
CACHE_FILE   = _ROOT / "data/efficiency_cache.json"
OUT_HTML     = _ROOT / "data/efficiency_report.html"

_HILL_MIN_DPLUS = 1200.0
_TS_STEP_S      = 10

# Cache schema version — bump when cache format changes to force rebuild
_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        c = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if c.get("_version") == _CACHE_VERSION:
            return c
    return {"_version": _CACHE_VERSION, "track": {}, "hills": {}}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Strava name lookup (date-based)
# ---------------------------------------------------------------------------

def _load_strava_names() -> dict[str, list[str]]:
    """
    Load Strava metadata and build {date_str: [name, ...]} lookup.

    Multiple activities can occur on the same date, so we keep a list.
    Date matching is the primary join key between Garmin FIT and Strava.
    """
    meta_file = _ROOT / "data/strava_streams/metadata.json"
    if not meta_file.exists():
        return {}
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        result: dict[str, list[str]] = {}
        for _, m in meta.items():
            date = str(m.get("date", ""))[:10]
            name = (m.get("name") or "").strip()
            if date and name:
                result.setdefault(date, []).append(name)
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Time-series downsampling
# ---------------------------------------------------------------------------

def _downsample(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "elapsed_s" not in df.columns:
        return df
    df = df.copy()
    df["_bucket"] = (df["elapsed_s"] / _TS_STEP_S).astype(int)
    return df.groupby("_bucket", sort=True).last().reset_index(drop=True)


def _ts_arrays(df: pd.DataFrame, include_alt: bool = False) -> dict:
    ds = _downsample(df)
    t     = ds["elapsed_s"].round(0).astype(int).tolist() if "elapsed_s" in ds.columns else []
    speed = ([round(v * 3.6, 2) if pd.notna(v) else None for v in ds["speed_ms"]]
             if "speed_ms" in ds.columns else [])
    hr    = ([round(v, 0) if pd.notna(v) else None for v in ds["heart_rate"]]
             if "heart_rate" in ds.columns else [])
    out: dict = {"t": t, "speed_kmh": speed, "hr": hr}
    if include_alt and "altitude_m" in ds.columns:
        out["alt"] = [round(v, 0) if pd.notna(v) else None for v in ds["altitude_m"]]
    return out


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def _session_name(fit_path: Path, date_str: str, strava_names: dict[str, list[str]]) -> str:
    """Return the best available session name: FIT workout name > Strava name."""
    try:
        meta = fit_metadata(fit_path)
        wkt = (meta.get("workout_name") or "").strip()
        if wkt:
            return wkt
    except Exception:
        pass
    names = strava_names.get(date_str, [])
    return names[0] if names else ""


def _process_track(fit_path: Path, strava_names: dict) -> dict | None:
    """Parse FIT, detect intervals (laps preferred), compute zone speeds."""
    try:
        df = parse_fit_file(fit_path)
    except Exception as e:
        print(f"  [skip] {fit_path.name}: {e}")
        return None

    # Try structured lap data first
    used_laps = False
    try:
        laps_df = parse_fit_laps(fit_path)
        if not laps_df.empty and (laps_df["intensity"] == "active").any():
            intervals = detect_track_intervals_from_laps(laps_df)
            used_laps = True
        else:
            intervals = detect_track_intervals(df)
    except Exception:
        intervals = detect_track_intervals(df)

    work = [i for i in intervals if i.is_work]
    if not work:
        return None

    eff_vals = [i.efficiency for i in work if i.efficiency]
    zone_sp  = zone_speed_ms(df, max_hr=DEFAULT_MAX_HR, zones=DEFAULT_HR_ZONES)
    date_str = fit_path.stem[:10]

    return {
        "name": _session_name(fit_path, date_str, strava_names),
        "used_laps": used_laps,
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
        "mean_efficiency": round(sum(eff_vals) / len(eff_vals), 5) if eff_vals else None,
        "zone_speeds_kmh": {k: round(v * 3.6, 2) if v else None for k, v in zone_sp.items()},
        "ts": _ts_arrays(df),
    }


def _process_hills(fit_path: Path, strava_names: dict) -> dict | None:
    """Parse FIT, detect hill repeats (with altitude in ts)."""
    try:
        df = parse_fit_file(fit_path)
    except Exception as e:
        print(f"  [skip] {fit_path.name}: {e}")
        return None

    repeats = detect_hill_repeats(df)
    if not repeats:
        return {"no_repeats": True}

    eff_vals = [r.efficiency for r in repeats if r.efficiency is not None]
    date_str = fit_path.stem[:10]

    return {
        "name": _session_name(fit_path, date_str, strava_names),
        "repeats": [
            {
                "repeat_num": r.repeat_num,
                "start_s": r.start_s, "end_s": r.end_s,
                "duration_s": round(r.duration_s, 0),
                "dplus_m": r.dplus_m, "dist_m": r.dist_m,
                "avg_grade_pct": r.avg_grade_pct,
                "asc_speed_mh": r.asc_speed_mh,
                "mean_hr": r.mean_hr,
                "efficiency": r.efficiency,
            }
            for r in repeats
        ],
        "repeat_count": len(repeats),
        "mean_efficiency": round(sum(eff_vals) / len(eff_vals), 3) if eff_vals else None,
        "ts": _ts_arrays(df, include_alt=True),
    }


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def build_cache(rebuild: bool = False) -> dict:
    cache = {} if rebuild else _load_cache()
    if "track" not in cache:
        cache["track"] = {}
    if "hills" not in cache:
        cache["hills"] = {}

    if not SUMMARIES.exists():
        print(f"ERROR: {SUMMARIES} not found — run sync/build_summaries.py first.")
        sys.exit(1)

    df = pd.read_parquet(SUMMARIES)
    strava_names = _load_strava_names()

    # Track sessions
    track_df = df[df["sport"] == "track_running"].copy()
    track_new = [
        r for _, r in track_df.iterrows()
        if r.get("file") and r["file"] not in cache["track"]
    ]
    if track_new:
        print(f"Track sessions: {len(track_new)} new to process ({len(track_df)} total)...")
        for row in tqdm(track_new, desc="Track", unit="session"):
            fp = FIT_DIR / row["file"]
            if not fp.exists():
                continue
            result = _process_track(fp, strava_names)
            date_str = str(row.get("date", ""))[:10]
            cache["track"][row["file"]] = {
                "date": date_str,
                "distance_km": row.get("distance_km"),
                "duration_s": row.get("duration_s"),
                **(result or {"no_intervals": True}),
            }
    else:
        print(f"Track sessions: 0 new (cache has {len(cache['track'])} sessions).")

    # Hill candidates
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
            fp = FIT_DIR / row["file"]
            if not fp.exists():
                continue
            result = _process_hills(fp, strava_names)
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
# HTML rendering
# ---------------------------------------------------------------------------

def _fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}" if h else f"{m}:{s:02d}"


def _zone_color(zone: str) -> str:
    return {
        "Z1_recovery":  "#74b9ff",
        "Z2_aerobic":   "#00b894",
        "Z3_tempo":     "#fdcb6e",
        "Z4_threshold": "#e17055",
        "Z5_vo2max":    "#d63031",
    }.get(zone, "#636e72")


def render_html(cache: dict) -> str:
    # Filter sessions with real results
    track_sessions = {
        f: d for f, d in sorted(cache.get("track", {}).items())
        if not d.get("no_intervals") and d.get("intervals")
        and any(i["is_work"] and i.get("efficiency") for i in d["intervals"])
    }
    hill_sessions = {
        f: d for f, d in sorted(cache.get("hills", {}).items())
        if not d.get("no_repeats") and d.get("repeats")
    }

    # Zone speed by year (track only)
    zone_by_year: dict[int, dict[str, list[float]]] = {}
    for d in track_sessions.values():
        yr = int(d["date"][:4]) if len(d.get("date", "")) >= 4 else 0
        if yr < 2018:
            continue
        zone_by_year.setdefault(yr, {z: [] for z in DEFAULT_HR_ZONES})
        for z, v in d.get("zone_speeds_kmh", {}).items():
            if v:
                zone_by_year[yr][z].append(v)

    zone_years = sorted(zone_by_year)
    zone_series = {
        z: [
            (round(sum(zone_by_year[yr][z]) / len(zone_by_year[yr][z]), 2)
             if zone_by_year[yr].get(z) else None)
            for yr in zone_years
        ]
        for z in DEFAULT_HR_ZONES
    }

    track_trend = [
        {"date": d["date"], "eff": d["mean_efficiency"], "file": f, "n": d.get("work_count", 0),
         "name": d.get("name", "")}
        for f, d in sorted(track_sessions.items(), key=lambda x: x[1]["date"])
        if d.get("mean_efficiency")
    ]
    hill_trend = [
        {"date": d["date"], "eff": d["mean_efficiency"], "file": f, "n": d.get("repeat_count", 0),
         "name": d.get("name", "")}
        for f, d in sorted(hill_sessions.items(), key=lambda x: x[1]["date"])
        if d.get("mean_efficiency")
    ]

    zone_colors = {z: _zone_color(z) for z in DEFAULT_HR_ZONES}
    zone_labels = {
        "Z1_recovery": "Z1 Récup", "Z2_aerobic": "Z2 Aérobie",
        "Z3_tempo": "Z3 Tempo",   "Z4_threshold": "Z4 Seuil",
        "Z5_vo2max": "Z5 VO2max",
    }

    # Sidebar items
    def track_item(fname, d):
        eff = d.get("mean_efficiency")
        eff_s = f"{eff*1000:.2f}" if eff else "—"
        laps_mark = "●" if d.get("used_laps") else "○"
        name = d.get("name", "")
        name_line = f'<span class="sess-name">{name}</span>' if name else ""
        return (
            f'<div class="sess-item" data-file="{fname}" onclick="selectSession(this,\'track\')">'
            f'<span class="sess-date">{d["date"]} {laps_mark}</span>'
            f'{name_line}'
            f'<span class="sess-meta">{d.get("work_count",0)} blocs · {eff_s} vit/FC</span>'
            f'</div>'
        )

    def hill_item(fname, d):
        eff = d.get("mean_efficiency")
        eff_s = f"{eff:.2f}" if eff else "—"
        name = d.get("name", "")
        name_line = f'<span class="sess-name">{name}</span>' if name else ""
        return (
            f'<div class="sess-item" data-file="{fname}" onclick="selectSession(this,\'hills\')">'
            f'<span class="sess-date">{d["date"]}</span>'
            f'{name_line}'
            f'<span class="sess-meta">{d.get("repeat_count",0)} reps · {eff_s} asc/FC</span>'
            f'</div>'
        )

    track_sidebar = "\n".join(
        track_item(f, d)
        for f, d in sorted(track_sessions.items(), key=lambda x: x[1]["date"], reverse=True)
    )
    hill_sidebar = "\n".join(
        hill_item(f, d)
        for f, d in sorted(hill_sessions.items(), key=lambda x: x[1]["date"], reverse=True)
    )

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

  .sidebar {{ width: 240px; border-right: 1px solid #2d3142; background: #13151f;
              display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }}
  .sidebar-header {{ padding: 10px 14px; font-size: 11px; font-weight: 600;
                     color: #6b7089; text-transform: uppercase; letter-spacing: .8px;
                     border-bottom: 1px solid #2d3142; }}
  .sidebar-list {{ flex: 1; overflow-y: auto; }}
  .sess-item {{ padding: 8px 14px; cursor: pointer; border-bottom: 1px solid #1e2030;
                transition: background .1s; }}
  .sess-item:hover {{ background: #1e2133; }}
  .sess-item.active {{ background: #2a3050; border-left: 3px solid #6c7aff; }}
  .sess-date {{ display: block; font-size: 12px; font-weight: 500; color: #cdd0e8; }}
  .sess-name {{ display: block; font-size: 12px; color: #a0a8d8; margin-top: 2px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .sess-meta {{ display: block; font-size: 11px; color: #5c6080; margin-top: 2px; }}

  .main {{ flex: 1; overflow-y: auto; padding: 20px 24px; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .overview-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}

  .card {{ background: #1a1d27; border: 1px solid #2d3142; border-radius: 10px;
           padding: 16px 20px; margin-bottom: 16px; }}
  .card-title {{ font-size: 12px; font-weight: 600; color: #6b7089;
                 text-transform: uppercase; letter-spacing: .7px; margin-bottom: 10px; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-tall {{ position: relative; height: 280px; }}

  h2.sess-heading {{ font-size: 17px; color: #fff; margin-bottom: 2px; }}
  .sess-label {{ font-size: 13px; color: #a0a8d8; margin-bottom: 4px; }}
  .sess-subheading {{ font-size: 12px; color: #6b7089; margin-bottom: 14px; }}
  .stat-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 18px; }}
  .stat {{ text-align: center; min-width: 80px; }}
  .stat-val {{ font-size: 22px; font-weight: 700; color: #6c7aff; }}
  .stat-lbl {{ font-size: 11px; color: #5c6080; margin-top: 2px; }}
  .no-data {{ color: #5c6080; font-style: italic; padding: 40px 0; text-align: center; font-size: 14px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th {{ text-align: left; padding: 6px 10px; color: #6b7089; font-weight: 600;
        font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
        border-bottom: 1px solid #2d3142; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e2030; color: #cdd0e8; }}
  tr.work-row {{ background: rgba(108,122,255,.06); }}
  tr:hover {{ background: rgba(255,255,255,.03); }}
  .eff-badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px;
                font-size: 12px; font-weight: 600; }}
  .laps-badge {{ font-size: 10px; padding: 1px 6px; border-radius: 3px;
                 background: rgba(108,122,255,.25); color: #a0b0ff; margin-left: 6px; }}
</style>
</head>
<body>
<header>
  <h1>Efficiency Report</h1>
  <div class="tab-bar">
    <button class="tab active" onclick="switchTab('track',event)">Track</button>
    <button class="tab" onclick="switchTab('hills',event)">Hills</button>
    <button class="tab" onclick="switchTab('zones',event)">Zone Speed</button>
  </div>
</header>
<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-header" id="sidebar-label">Track ({len(track_sessions)})</div>
    <div class="sidebar-list" id="sidebar-track">{track_sidebar}</div>
    <div class="sidebar-list" id="sidebar-hills" style="display:none">{hill_sidebar}</div>
  </nav>
  <main class="main">
    <!-- TRACK -->
    <div id="panel-track" class="panel active">
      <div class="overview-grid">
        <div class="card">
          <div class="card-title">Économie track — vit/FC ×1000 (tous sessions)</div>
          <div class="chart-wrap"><canvas id="chart-track-trend"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title">Détail session</div>
          <div id="track-detail"><p class="no-data">← Sélectionner une session</p></div>
        </div>
      </div>
      <div id="track-ts-card" class="card" style="display:none">
        <div class="card-title">Vitesse &amp; fréq. cardiaque — <span id="track-ts-title"></span></div>
        <div class="chart-wrap-tall"><canvas id="chart-track-ts"></canvas></div>
      </div>
      <div id="track-eff-card" class="card" style="display:none">
        <div class="card-title">Économie par bloc (vit. m/s ÷ FC)</div>
        <div class="chart-wrap"><canvas id="chart-track-eff"></canvas></div>
      </div>
    </div>

    <!-- HILLS -->
    <div id="panel-hills" class="panel">
      <div class="overview-grid">
        <div class="card">
          <div class="card-title">Économie montée — vit.asc/FC (tous sessions)</div>
          <div class="chart-wrap"><canvas id="chart-hills-trend"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title">Détail session</div>
          <div id="hills-detail"><p class="no-data">← Sélectionner une session</p></div>
        </div>
      </div>
      <div id="hills-ts-card" class="card" style="display:none">
        <div class="card-title">Profil altimétrique &amp; fréq. cardiaque — <span id="hills-ts-title"></span></div>
        <div class="chart-wrap-tall"><canvas id="chart-hills-ts"></canvas></div>
      </div>
      <div id="hills-eff-card" class="card" style="display:none">
        <div class="card-title">Économie par répétition (vit.ascens. m/h ÷ FC)</div>
        <div class="chart-wrap"><canvas id="chart-hills-eff"></canvas></div>
      </div>
    </div>

    <!-- ZONES -->
    <div id="panel-zones" class="panel">
      <div class="card">
        <div class="card-title">Vitesse (km/h) par zone FC — évolution annuelle (track)</div>
        <div class="chart-wrap-tall"><canvas id="chart-zones"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Tableau — vitesse médiane par zone et par année</div>
        <div id="zones-table"></div>
      </div>
    </div>
  </main>
</div>

<script>
// ── embedded data ──────────────────────────────────────────────────────────
const SESSIONS    = {sessions_js};
const TRACK_TREND = {json.dumps(track_trend, ensure_ascii=False)};
const HILL_TREND  = {json.dumps(hill_trend,  ensure_ascii=False)};
const ZONE_YEARS  = {json.dumps(zone_years)};
const ZONE_SERIES = {json.dumps(zone_series)};
const ZONE_COLORS = {json.dumps(zone_colors)};
const ZONE_LABELS = {json.dumps(zone_labels, ensure_ascii=False)};

// ── custom plugin: grey band overlay (work intervals / hill repeats) ───────
const workBandsPlugin = {{
  id: 'workBands',
  beforeDraw(chart) {{
    const bands = chart.options.plugins?.workBands;
    if (!bands?.length) return;
    const {{ctx, chartArea: {{top, bottom, left, right}}, scales: {{x}}}} = chart;
    const h = bottom - top;
    ctx.save();
    bands.forEach((b) => {{
      const x0 = Math.max(left,  x.getPixelForValue(b.s));
      const x1 = Math.min(right, x.getPixelForValue(b.e));
      if (x1 <= x0) return;
      ctx.fillStyle = 'rgba(255,255,255,0.07)';
      ctx.fillRect(x0, top, x1 - x0, h);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, top, x1 - x0, h);
      if (b.lbl) {{
        ctx.fillStyle = 'rgba(180,190,255,0.75)';
        ctx.font = 'bold 10px system-ui';
        ctx.fillText(b.lbl, x0 + 3, top + 12);
      }}
    }});
    ctx.restore();
  }}
}};
Chart.register(workBandsPlugin);

// ── chart registry ─────────────────────────────────────────────────────────
const _charts = {{}};
function destroyChart(id) {{ if (_charts[id]) {{ _charts[id].destroy(); delete _charts[id]; }} }}
function mkChart(id, cfg) {{ destroyChart(id); _charts[id] = new Chart(document.getElementById(id), cfg); }}

const darkGrid = {{ color: 'rgba(255,255,255,.05)' }};
const darkTick = {{ color: '#5c6080' }};

// ── helper: ts.t indices for a time in seconds ─────────────────────────────
function tIdx(sec, tArr) {{
  let i = tArr.findIndex(t => t >= sec);
  return i < 0 ? tArr.length - 1 : i;
}}
function computeBands(segs, tArr, prefix) {{
  let workNum = 0;
  return segs.map(s => {{
    if (!s.is_work && prefix === 'B') return null;
    workNum++;
    return {{
      s: tIdx(s.start_s, tArr),
      e: tIdx(s.end_s,   tArr),
      lbl: prefix + workNum,
    }};
  }}).filter(Boolean);
}}
function computeHillBands(reps, tArr) {{
  return reps.map(r => ({{
    s: tIdx(r.start_s, tArr),
    e: tIdx(r.end_s,   tArr),
    lbl: 'R' + r.repeat_num,
  }}));
}}

// ── trend charts ───────────────────────────────────────────────────────────
function buildTrend(canvasId, trend, mult, yLabel, color) {{
  mkChart(canvasId, {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: yLabel,
        data: trend.map(d => ({{ x: d.date, y: d.eff ? +(d.eff * mult).toFixed(3) : null }})),
        backgroundColor: color, pointRadius: 5, pointHoverRadius: 7,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.raw.x}} — ${{ctx.raw.y}} (${{trend[ctx.dataIndex]?.n}} × | ${{trend[ctx.dataIndex]?.name || ''}})` }} }},
      }},
      scales: {{
        x: {{ type: 'category', ticks: {{ ...darkTick, maxTicksLimit: 6, maxRotation: 0 }}, grid: darkGrid }},
        y: {{ title: {{ display: true, text: yLabel, color: '#6b7089' }}, ticks: darkTick, grid: darkGrid }},
      }},
    }}
  }});
}}
buildTrend('chart-track-trend', TRACK_TREND, 1000, 'vit/FC ×1000', 'rgba(108,122,255,.75)');
buildTrend('chart-hills-trend', HILL_TREND,  1,    'asc/FC',        'rgba(46,204,113,.75)');

// ── zone speed chart ───────────────────────────────────────────────────────
(function() {{
  const z3 = ['Z2_aerobic','Z3_tempo','Z4_threshold'];
  mkChart('chart-zones', {{
    type: 'line',
    data: {{
      labels: ZONE_YEARS.map(String),
      datasets: z3.map(z => ({{
        label: ZONE_LABELS[z], data: ZONE_SERIES[z],
        borderColor: ZONE_COLORS[z], backgroundColor: ZONE_COLORS[z]+'33',
        tension: .3, fill: false, pointRadius: 4,
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#cdd0e8', font: {{ size: 12 }} }} }} }},
      scales: {{
        x: {{ ticks: darkTick, grid: darkGrid }},
        y: {{ title: {{ display: true, text: 'Vitesse (km/h)', color: '#6b7089' }},
              ticks: darkTick, grid: darkGrid }},
      }}
    }}
  }});
  const allZ = Object.keys(ZONE_LABELS);
  let html = '<table><tr><th>Année</th>' + allZ.map(z => `<th>${{ZONE_LABELS[z]}}</th>`).join('') + '</tr>';
  ZONE_YEARS.forEach((yr, i) => {{
    html += `<tr><td>${{yr}}</td>` + allZ.map(z => `<td>${{ZONE_SERIES[z]?.[i] ?? '—'}}</td>`).join('') + '</tr>';
  }});
  document.getElementById('zones-table').innerHTML = html + '</table>';
}})();

// ── session selection ──────────────────────────────────────────────────────
let _activeEl = null;
function selectSession(el, type) {{
  if (_activeEl) _activeEl.classList.remove('active');
  el.classList.add('active');
  _activeEl = el;
  const data = SESSIONS[type + ':' + el.dataset.file];
  if (!data) return;
  type === 'track' ? renderTrack(data) : renderHills(data);
}}

// ── track detail ───────────────────────────────────────────────────────────
function renderTrack(data) {{
  const ivs  = data.intervals || [];
  const work = ivs.filter(i => i.is_work);
  const eff  = data.mean_efficiency ? (data.mean_efficiency*1000).toFixed(2) : '—';
  const lbl  = data.used_laps ? '<span class="laps-badge">laps GPS</span>' : '';
  document.getElementById('track-detail').innerHTML = `
    ${{data.name ? `<div class="sess-label">${{data.name}}</div>` : ''}}
    <h2 class="sess-heading">${{data.date}}${{lbl}}</h2>
    <div class="sess-subheading">${{data.distance_km || '—'}} km · ${{fmtHms(data.duration_s)}} · ${{work.length}} blocs travail</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">${{eff}}</div><div class="stat-lbl">économie ×1000</div></div>
      <div class="stat"><div class="stat-val">${{work.length}}</div><div class="stat-lbl">blocs</div></div>
    </div>
    <table>
      <tr><th>#</th><th>Durée</th><th>Vit. km/h</th><th>FC moy (bpm)</th><th>Éco = vit/FC ×1000</th></tr>
      ${{work.map((iv,i) => `<tr class="work-row">
        <td>${{i+1}}</td><td>${{fmtMin(iv.duration_s)}}</td><td>${{iv.speed_kmh.toFixed(2)}}</td>
        <td>${{iv.mean_hr ? iv.mean_hr.toFixed(0) : '—'}}</td>
        <td><span class="eff-badge" style="background:${{effColor(iv.efficiency,'track')}}">${{iv.efficiency ? (iv.efficiency*1000).toFixed(2) : '—'}}</span></td>
      </tr>`).join('')}}
    </table>`;

  const ts = data.ts || {{}};
  if (ts.t?.length) {{
    document.getElementById('track-ts-title').textContent = data.name || data.date;
    document.getElementById('track-ts-card').style.display = '';
    const bands = computeBands(ivs, ts.t, 'B');
    buildTrackTs('chart-track-ts', ts, bands);
  }} else {{
    document.getElementById('track-ts-card').style.display = 'none';
  }}

  const workEff = work.filter(i => i.efficiency);
  if (workEff.length) {{
    document.getElementById('track-eff-card').style.display = '';
    mkChart('chart-track-eff', {{
      type: 'bar',
      data: {{
        labels: workEff.map((_,i) => 'B'+(i+1)),
        datasets: [{{ label: 'Éco ×1000', data: workEff.map(i => +(i.efficiency*1000).toFixed(3)),
                      backgroundColor: workEff.map(i => effColor(i.efficiency,'track')), }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: darkTick, grid: darkGrid }},
          y: {{ title: {{ display: true, text: 'vit.(m/s) ÷ FC (bpm)', color: '#6b7089' }},
                ticks: darkTick, grid: darkGrid }},
        }}
      }}
    }});
  }} else {{
    document.getElementById('track-eff-card').style.display = 'none';
  }}
}}

// ── hills detail ───────────────────────────────────────────────────────────
function renderHills(data) {{
  const reps = data.repeats || [];
  const eff  = data.mean_efficiency ? data.mean_efficiency.toFixed(2) : '—';
  document.getElementById('hills-detail').innerHTML = `
    ${{data.name ? `<div class="sess-label">${{data.name}}</div>` : ''}}
    <h2 class="sess-heading">${{data.date}}</h2>
    <div class="sess-subheading">${{data.distance_km || '—'}} km · ${{fmtHms(data.duration_s)}} · ${{data.dplus_m ? Math.round(data.dplus_m)+'m D+' : ''}} · ${{reps.length}} reps</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">${{eff}}</div><div class="stat-lbl">éco (asc/FC)</div></div>
      <div class="stat"><div class="stat-val">${{reps.length}}</div><div class="stat-lbl">répétitions</div></div>
      <div class="stat"><div class="stat-val">${{reps.length ? Math.round(reps[0].dplus_m) : '—'}}</div><div class="stat-lbl">D+ rep1 (m)</div></div>
    </div>
    <table>
      <tr><th>#</th><th>Durée</th><th>D+</th><th>Pente</th><th>Vit.asc (m/h)</th><th>FC moy (bpm)</th><th>Éco = vit.asc ÷ FC</th></tr>
      ${{reps.map(r => `<tr class="work-row">
        <td>R${{r.repeat_num}}</td><td>${{fmtMin(r.duration_s)}}</td>
        <td>${{Math.round(r.dplus_m)}}m</td><td>${{r.avg_grade_pct.toFixed(1)}}%</td>
        <td>${{Math.round(r.asc_speed_mh)}}</td>
        <td>${{r.mean_hr ? r.mean_hr.toFixed(0) : '—'}}</td>
        <td><span class="eff-badge" style="background:${{effColor(r.efficiency,'hills')}}">${{r.efficiency ? r.efficiency.toFixed(2) : '—'}}</span></td>
      </tr>`).join('')}}
    </table>`;

  const ts = data.ts || {{}};
  if (ts.t?.length) {{
    document.getElementById('hills-ts-title').textContent = data.name || data.date;
    document.getElementById('hills-ts-card').style.display = '';
    const bands = computeHillBands(reps, ts.t);
    buildHillsTs('chart-hills-ts', ts, bands);
  }} else {{
    document.getElementById('hills-ts-card').style.display = 'none';
  }}

  const repsEff = reps.filter(r => r.efficiency);
  if (repsEff.length) {{
    document.getElementById('hills-eff-card').style.display = '';
    mkChart('chart-hills-eff', {{
      type: 'bar',
      data: {{
        labels: repsEff.map(r => 'R'+r.repeat_num),
        datasets: [{{ label: 'vit.asc ÷ FC', data: repsEff.map(r => +r.efficiency.toFixed(3)),
                      backgroundColor: repsEff.map(r => effColor(r.efficiency,'hills')), }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: darkTick, grid: darkGrid }},
          y: {{ title: {{ display: true, text: 'vit.ascens.(m/h) ÷ FC (bpm)', color: '#6b7089' }},
                ticks: darkTick, grid: darkGrid }},
        }}
      }}
    }});
  }} else {{
    document.getElementById('hills-eff-card').style.display = 'none';
  }}
}}

// ── time series charts ─────────────────────────────────────────────────────
function buildTrackTs(id, ts, bands) {{
  const labels = ts.t.map(s => fmtMin(s));
  mkChart(id, {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Vitesse (km/h)', data: ts.speed_kmh, yAxisID: 'y',
           borderColor: 'rgba(108,122,255,.85)', backgroundColor: 'transparent',
           borderWidth: 1.5, pointRadius: 0, tension: .2, spanGaps: true }},
        {{ label: 'Fréq. cardiaque (bpm)', data: ts.hr, yAxisID: 'y2',
           borderColor: 'rgba(231,76,60,.75)', backgroundColor: 'transparent',
           borderWidth: 1.5, pointRadius: 0, tension: .2, spanGaps: true }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#cdd0e8', font: {{ size: 11 }} }} }},
        workBands: bands,
      }},
      scales: {{
        x:  {{ ticks: {{ ...darkTick, maxTicksLimit: 10, maxRotation: 0 }}, grid: darkGrid }},
        y:  {{ position: 'left', title: {{ display: true, text: 'Vitesse (km/h)', color: '#6b7089' }},
               ticks: darkTick, grid: darkGrid }},
        y2: {{ position: 'right', title: {{ display: true, text: 'Fréq. cardiaque (bpm)', color: '#6b7089' }},
               ticks: darkTick, grid: {{ drawOnChartArea: false }} }},
      }},
    }},
  }});
}}

function buildHillsTs(id, ts, bands) {{
  const labels = ts.t.map(s => fmtMin(s));
  const hasAlt = ts.alt?.length > 0;
  const datasets = [];
  if (hasAlt) {{
    datasets.push({{ label: 'Altitude (m)', data: ts.alt, yAxisID: 'y',
      borderColor: 'rgba(46,204,113,.8)', backgroundColor: 'rgba(46,204,113,.08)',
      borderWidth: 1.5, pointRadius: 0, tension: .15, spanGaps: true, fill: true }});
  }} else {{
    datasets.push({{ label: 'Vitesse (km/h)', data: ts.speed_kmh, yAxisID: 'y',
      borderColor: 'rgba(108,122,255,.85)', backgroundColor: 'transparent',
      borderWidth: 1.5, pointRadius: 0, tension: .2, spanGaps: true }});
  }}
  datasets.push({{ label: 'Fréq. cardiaque (bpm)', data: ts.hr, yAxisID: 'y2',
    borderColor: 'rgba(231,76,60,.75)', backgroundColor: 'transparent',
    borderWidth: 1.5, pointRadius: 0, tension: .2, spanGaps: true }});

  mkChart(id, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#cdd0e8', font: {{ size: 11 }} }} }},
        workBands: bands,
      }},
      scales: {{
        x:  {{ ticks: {{ ...darkTick, maxTicksLimit: 10, maxRotation: 0 }}, grid: darkGrid }},
        y:  {{ position: 'left',
               title: {{ display: true, text: hasAlt ? 'Altitude (m)' : 'Vitesse (km/h)', color: '#6b7089' }},
               ticks: darkTick, grid: darkGrid }},
        y2: {{ position: 'right',
               title: {{ display: true, text: 'Fréq. cardiaque (bpm)', color: '#6b7089' }},
               ticks: darkTick, grid: {{ drawOnChartArea: false }} }},
      }},
    }},
  }});
}}

// ── tab switch ─────────────────────────────────────────────────────────────
function switchTab(tab, ev) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  ev.target.classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
  document.getElementById('sidebar-track').style.display = tab === 'track' ? '' : 'none';
  document.getElementById('sidebar-hills').style.display = tab === 'hills' ? '' : 'none';
  document.getElementById('sidebar-label').textContent = {{
    track: 'Track ({len(track_sessions)})',
    hills: 'Hills ({len(hill_sessions)})',
    zones: 'Zone Speed',
  }}[tab] || '';
}}

// ── formatting helpers ─────────────────────────────────────────────────────
function fmtMin(s) {{
  const m = Math.floor(s/60), sec = Math.round(s%60);
  return `${{m}}:${{String(sec).padStart(2,'0')}}`;
}}
function fmtHms(s) {{
  if (!s) return '—';
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h ? `${{h}}h${{String(m).padStart(2,'0')}}` : `${{m}}min`;
}}

// ── efficiency colour (percentile-based, red/yellow/green) ─────────────────
const _trkEffs = TRACK_TREND.map(d => d.eff*1000).filter(Boolean).sort((a,b)=>a-b);
const _hlsEffs = HILL_TREND.map(d => d.eff).filter(Boolean).sort((a,b)=>a-b);
function effColor(eff, type) {{
  if (!eff) return '#3d4263';
  const arr = type==='track' ? _trkEffs : _hlsEffs;
  const v   = type==='track' ? eff*1000 : eff;
  const p   = arr.filter(x => x <= v).length / arr.length;
  return p>=.8 ? 'rgba(46,204,113,.5)' : p>=.5 ? 'rgba(253,203,110,.5)' : 'rgba(214,63,49,.4)';
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
    p = argparse.ArgumentParser(description="Generate efficiency_report.html")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Re-parse all sessions (needed after parameter or format changes)")
    args = p.parse_args()

    cache = build_cache(rebuild=args.rebuild_cache)

    n_trk = sum(1 for d in cache["track"].values() if not d.get("no_intervals") and d.get("intervals"))
    n_hls = sum(1 for d in cache["hills"].values() if not d.get("no_repeats") and d.get("repeats"))
    print(f"\nSessions with results: track={n_trk}, hills={n_hls}")

    html = render_html(cache)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Written: {OUT_HTML}  ({len(html)//1024} KB)  [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
