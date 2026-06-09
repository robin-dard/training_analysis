"""
scripts/generate_html_report.py

Generate a self-contained HTML race build/taper analysis report.

Usage
-----
python scripts/generate_html_report.py
python scripts/generate_html_report.py --trail-only
python scripts/generate_html_report.py --min-dist 60 --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.build_taper_pattern import (
    avg_build, avg_taper, compute_race_window, load_summaries,
)

RACES_FILE = Path("data/races/races.json")
OUT_FILE   = Path("data/race_report.html")
TODAY      = date.today()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_trail(race: dict, min_dplus_per_km: float = 30.0) -> bool:
    km = race.get("distance_km") or 0
    dp = race.get("dplus_m") or 0
    return km <= 0 or (dp / km) >= min_dplus_per_km


def _load_objectives(trail_only: bool = False, min_dist: float = 0
                     ) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (good_races, bad_races, upcoming_races)."""
    races = json.loads(RACES_FILE.read_text(encoding="utf-8"))
    upcoming = [r for r in races
                if r.get("type") == "objective"
                and r.get("score") is None
                and date.fromisoformat(r["date"]) >= TODAY]
    objs = [r for r in races
            if r.get("type") == "objective" and r.get("score") is not None]
    if trail_only:
        objs = [r for r in objs if _is_trail(r)]
    if min_dist > 0:
        objs = [r for r in objs if (r.get("distance_km") or 0) >= min_dist]
    good = [r for r in objs if r["score"] >= 3]
    bad  = [r for r in objs if r["score"] < 3]
    return good, bad, upcoming


def _last_hard_day(taper_list, km_thresh: float = 15.0, h_thresh: float = 2.0) -> int | None:
    for ds in reversed(taper_list):
        if ds.trail_km >= km_thresh or ds.trail_h >= h_thresh:
            return ds.day_offset
    return None


def _window_to_dict(w, is_good: bool,
                    is_upcoming: bool = False, plan: dict | None = None) -> dict:
    lhd = _last_hard_day(w.taper)
    peak_wk = max(w.build, key=lambda ws: ws.total_h, default=None)
    return {
        "name":          w.race_name,
        "date":          str(w.race_date),
        "score":         w.score,
        "dist_km":       w.distance_km,
        "dplus_m":       w.dplus_m,
        "is_good":       is_good,
        "is_upcoming":   is_upcoming,
        "plan":          plan,
        "last_hard":     lhd,
        "peak_week":     peak_wk.week_offset if peak_wk else None,
        "peak_total_h":  round(peak_wk.total_h, 1) if peak_wk else 0,
        "build": [
            {"week": ws.week_offset,
             "trail_km": round(ws.trail_km, 1),
             "trail_dplus": round(ws.trail_dplus, 0),
             "trail_h": round(ws.trail_h, 2),
             "bike_km": round(ws.bike_km, 1),
             "bike_h": round(ws.bike_h, 2),
             "total_h": round(ws.total_h, 2)}
            for ws in w.build
        ],
        "taper": [
            {"day": ds.day_offset,
             "trail_km": round(ds.trail_km, 1),
             "trail_dplus": round(ds.trail_dplus, 0),
             "trail_h": round(ds.trail_h, 2),
             "bike_km": round(ds.bike_km, 1),
             "bike_h": round(ds.bike_h, 2)}
            for ds in w.taper
        ],
        "build_trail_km":  round(sum(ws.trail_km    for ws in w.build), 0),
        "build_dplus":     round(sum(ws.trail_dplus for ws in w.build), 0),
        "build_trail_h":   round(sum(ws.trail_h     for ws in w.build), 1),
        "build_bike_km":   round(sum(ws.bike_km     for ws in w.build), 0),
        "build_bike_h":    round(sum(ws.bike_h      for ws in w.build), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Race Build/Taper Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--good:#16a34a;--bad:#dc2626;--upcoming:#d97706;--sidebar:265px;--bg:#f1f5f9}
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;height:100vh;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;background:var(--bg);font-size:14px}

#sidebar{width:var(--sidebar);background:#0f172a;color:#e2e8f0;overflow-y:auto;flex-shrink:0;display:flex;flex-direction:column}
#sidebar-header{padding:14px 16px 10px;border-bottom:1px solid #1e293b;flex-shrink:0}
#sidebar-header h1{font-size:13px;font-weight:700;letter-spacing:.05em;color:#f8fafc}
#sidebar-header p{font-size:10px;color:#64748b;margin-top:2px}
#overview-btn{display:block;width:100%;text-align:left;padding:10px 16px;background:none;border:none;border-bottom:1px solid #1e293b;color:#94a3b8;cursor:pointer;font-size:12px;transition:background .15s}
#overview-btn:hover,#overview-btn.active{background:#1e293b;color:#e2e8f0}
.grp-hdr{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#64748b;padding:10px 16px 4px}
.race-item{padding:8px 16px;cursor:pointer;border-left:3px solid transparent;transition:all .12s;display:flex;gap:8px;align-items:flex-start}
.race-item:hover{background:#1e293b}
.race-item.active{background:#1e293b}
.race-item.good.active,.race-item.good:hover{border-left-color:var(--good)}
.race-item.bad.active,.race-item.bad:hover{border-left-color:var(--bad)}
.race-item.upcoming.active,.race-item.upcoming:hover{border-left-color:var(--upcoming)}
.sbadge{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:1px}
.good .sbadge{background:#14532d;color:#86efac}
.bad .sbadge{background:#7f1d1d;color:#fca5a5}
.upcoming .sbadge{background:#92400e;color:#fcd34d}
.sname{font-size:11px;line-height:1.35;color:#cbd5e1}
.smeta{font-size:10px;color:#475569;margin-top:1px}

#content{flex:1;overflow-y:auto;padding:20px}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:18px;margin-bottom:16px}
.card h2{font-size:14px;font-weight:700;margin-bottom:14px;color:#0f172a}
.card h3{font-size:12px;font-weight:600;margin-bottom:10px;color:#334155}

.rh-wrap{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.rh-title{font-size:20px;font-weight:800;color:#0f172a}
.rh-pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.pill{background:#f1f5f9;border-radius:99px;padding:3px 10px;font-size:11px;color:#475569}
.score-big{font-size:32px;font-weight:800;line-height:1}
.score-big.good{color:var(--good)}.score-big.bad{color:var(--bad)}.score-big.upcoming{color:var(--upcoming)}
.lhd-tag{font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;margin-top:4px;display:inline-block}
.lhd-tag.good{background:#dcfce7;color:#14532d}.lhd-tag.bad{background:#fee2e2;color:#7f1d1d}.lhd-tag.ok{background:#fef9c3;color:#713f12}

.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.cgrid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.cwrap{position:relative;height:240px}
.chart-note{font-size:10px;color:#94a3b8;margin-top:6px}
@media(max-width:900px){.cgrid,.cgrid3{grid-template-columns:1fr}}

table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:7px 8px;background:#f8fafc;color:#64748b;font-weight:600;border-bottom:2px solid #e2e8f0;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#f1f5f9}th.sorted{color:#0f172a}
td{padding:6px 8px;border-bottom:1px solid #f1f5f9;color:#334155;white-space:nowrap}
tr:hover td{background:#f8fafc}
tr.grow td:first-child{border-left:3px solid var(--good)}
tr.badr td:first-child{border-left:3px solid var(--bad)}
tr.upcr td:first-child{border-left:3px solid var(--upcoming)}
.sc{font-weight:700}.sc.s4{color:#15803d}.sc.s3{color:var(--good)}.sc.s2{color:#ea580c}.sc.s1{color:var(--bad)}
.lhd-cell.early{color:#15803d;font-weight:600}.lhd-cell.late{color:var(--bad);font-weight:600}

/* plan card */
.plan-card{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:16px;margin-bottom:16px}
.plan-card h3{color:#92400e;font-size:13px;font-weight:700;margin-bottom:12px}
.plan-table{width:100%;border-collapse:collapse;font-size:12px}
.plan-table th{background:#fef3c7;color:#78350f;padding:6px 8px;text-align:left;font-weight:600}
.plan-table td{padding:6px 8px;border-bottom:1px solid #fde68a;vertical-align:top}
.plan-ok{color:#15803d;font-weight:600}.plan-warn{color:#b45309;font-weight:600}
.sessions-row{margin-top:10px;font-size:11px;color:#78350f}
.session-badge{display:inline-block;background:#fde68a;border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-weight:600}
</style>
</head>
<body>
<nav id="sidebar">
  <div id="sidebar-header">
    <h1>Build / Taper Analysis</h1>
    <p id="sb-subtitle"></p>
  </div>
  <button id="overview-btn" onclick="showOverview()">Overview &amp; Compare</button>
  <div id="upcoming-section" style="display:none">
    <div class="grp-hdr" style="color:#d97706">Upcoming</div>
    <div id="upcoming-list"></div>
  </div>
  <div class="grp-hdr">Good races (score &ge; 3)</div>
  <div id="good-list"></div>
  <div class="grp-hdr" style="margin-top:6px">Bad races (score &lt; 3)</div>
  <div id="bad-list"></div>
</nav>
<main id="content"><div id="view"></div></main>

<script>
const DATA = __DATA__;

/* ── Sidebar ── */
function buildSidebar(){
  const nGood=DATA.races.filter(r=>r.is_good&&!r.is_upcoming).length;
  const nBad=DATA.races.filter(r=>!r.is_good&&!r.is_upcoming).length;
  document.getElementById('sb-subtitle').textContent=`${nGood} good · ${nBad} bad`;

  const upcoming=DATA.races.filter(r=>r.is_upcoming);
  if(upcoming.length>0){
    document.getElementById('upcoming-section').style.display='';
    document.getElementById('upcoming-list').innerHTML=upcoming.map(r=>{
      const i=DATA.races.indexOf(r);
      return `<div class="race-item upcoming" id="ri${i}" onclick="showRace(${i})">
        <span class="sbadge">&#9654;</span>
        <span><div class="sname">${r.name}</div>
        <div class="smeta">${r.date.slice(0,7)} · ${r.dist_km}km D+${r.dplus_m}m</div></span>
      </div>`;
    }).join('');
  }

  ['good','bad'].forEach(g=>{
    const el=document.getElementById(g+'-list');
    el.innerHTML=DATA.races.filter(r=>r.is_good===(g==='good')&&!r.is_upcoming).map(r=>{
      const i=DATA.races.indexOf(r);
      return `<div class="race-item ${g}" id="ri${i}" onclick="showRace(${i})">
        <span class="sbadge">${r.score}</span>
        <span><div class="sname">${r.name}</div>
        <div class="smeta">${r.date.slice(0,7)} · ${r.dist_km}km D+${r.dplus_m}m</div></span>
      </div>`;
    }).join('');
  });
}

/* ── Chart helpers ── */
const _ch={};
function _kill(id){if(_ch[id]){_ch[id].destroy();delete _ch[id];}}

/* Overview chart 1: avg trail hours good vs bad per build week */
function ovTrailHChart(id){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lbl=DATA.good_build_avg.map(w=>'W'+w.week);
  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:lbl,datasets:[
      {label:'GOOD trail h',data:DATA.good_build_avg.map(w=>w.trail_h),backgroundColor:'rgba(22,163,74,.75)'},
      {label:'BAD trail h', data:DATA.bad_build_avg.map(w=>w.trail_h), backgroundColor:'rgba(220,38,38,.65)'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'avg trail hours',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Overview chart 2: avg trail D+ good vs bad per build week */
function ovDplusChart(id){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lbl=DATA.good_build_avg.map(w=>'W'+w.week);
  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:lbl,datasets:[
      {label:'GOOD D+',data:DATA.good_build_avg.map(w=>w.trail_dplus),backgroundColor:'rgba(22,163,74,.65)'},
      {label:'BAD D+', data:DATA.bad_build_avg.map(w=>w.trail_dplus), backgroundColor:'rgba(220,38,38,.55)'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'avg trail D+ (m)',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Overview chart 3: taper avg daily km */
function ovTaperChart(id){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  _ch[id]=new Chart(ctx,{type:'line',
    data:{labels:DATA.good_taper_avg.map(d=>'D'+d.day),datasets:[
      {label:'GOOD trail km',data:DATA.good_taper_avg.map(d=>d.trail_km),
       borderColor:'rgba(22,163,74,.9)',backgroundColor:'rgba(22,163,74,.1)',
       fill:true,tension:.3,borderWidth:2,pointRadius:2},
      {label:'BAD trail km', data:DATA.bad_taper_avg.map(d=>d.trail_km),
       borderColor:'rgba(220,38,38,.9)',backgroundColor:'rgba(220,38,38,.08)',
       fill:true,tension:.3,borderWidth:2,pointRadius:2},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:9},maxRotation:45}},
        y:{title:{display:true,text:'avg trail km',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Overview chart 4: avg bike hours good vs bad */
function ovBikeChart(id){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lbl=DATA.good_build_avg.map(w=>'W'+w.week);
  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:lbl,datasets:[
      {label:'GOOD bike h',data:DATA.good_build_avg.map(w=>w.bike_h),backgroundColor:'rgba(59,130,246,.75)'},
      {label:'BAD bike h', data:DATA.bad_build_avg.map(w=>w.bike_h), backgroundColor:'rgba(99,102,241,.6)'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'avg bike hours',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Race detail: trail build chart (trail_h bars + D+ line + avg dashed) */
function trailBuildChart(id, race, avgData){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lbl=race.build.map(w=>'W'+w.week);
  const acc=race.is_upcoming?'rgba(217,119,6,.8)':race.is_good?'rgba(22,163,74,.75)':'rgba(220,38,38,.7)';

  // For upcoming races: replace W-2/W-1 bars with plan data if plan exists
  const trailH=race.build.map(w=>{
    if(race.is_upcoming&&race.plan&&race.plan.weeks&&race.plan.weeks[String(w.week)])
      return race.plan.weeks[String(w.week)].trail_h;
    return w.trail_h;
  });
  const trailDp=race.build.map(w=>{
    if(race.is_upcoming&&race.plan&&race.plan.weeks&&race.plan.weeks[String(w.week)])
      return race.plan.weeks[String(w.week)].trail_dplus;
    return w.trail_dplus;
  });
  const barColors=race.build.map(w=>{
    if(race.is_upcoming&&race.plan&&race.plan.weeks&&race.plan.weeks[String(w.week)])
      return 'rgba(217,119,6,.4)';
    return acc;
  });

  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:lbl,datasets:[
      {label:'Trail h',data:trailH,backgroundColor:barColors,
       borderColor:barColors.map(c=>c.replace('.4','1').replace('.75','1').replace('.7','1').replace('.8','1')),
       borderWidth:barColors.map(c=>c.includes('.4')?2:0)},
      {label:'Grp avg trail h',data:avgData.map(w=>w.trail_h),type:'line',
       borderColor:'rgba(148,163,184,.8)',borderDash:[4,3],borderWidth:1.5,
       pointRadius:0,fill:false,tension:.3},
      {label:'D+ (m)',data:trailDp,type:'line',
       borderColor:'rgba(234,179,8,.9)',borderWidth:2,pointRadius:3,
       fill:false,yAxisID:'y2',tension:.3},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'Trail hours',font:{size:10}},ticks:{font:{size:10}}},
        y2:{position:'right',title:{display:true,text:'D+ (m)',font:{size:10}},
            ticks:{font:{size:10}},grid:{display:false}},
      }}
  });
}

/* Race detail: bike build chart (bike_h bars + avg dashed) */
function bikeBuildChart(id, race, avgData){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lbl=race.build.map(w=>'W'+w.week);

  const bikeH=race.build.map(w=>{
    if(race.is_upcoming&&race.plan&&race.plan.weeks&&race.plan.weeks[String(w.week)])
      return race.plan.weeks[String(w.week)].bike_h;
    return w.bike_h;
  });
  const barColors=race.build.map(w=>{
    if(race.is_upcoming&&race.plan&&race.plan.weeks&&race.plan.weeks[String(w.week)])
      return 'rgba(59,130,246,.35)';
    return 'rgba(59,130,246,.65)';
  });

  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:lbl,datasets:[
      {label:'Bike h',data:bikeH,backgroundColor:barColors,
       borderColor:barColors.map(c=>c.includes('.35')?'rgba(59,130,246,1)':'transparent'),
       borderWidth:barColors.map(c=>c.includes('.35')?2:0)},
      {label:'Grp avg bike h',data:avgData.map(w=>w.bike_h),type:'line',
       borderColor:'rgba(148,163,184,.8)',borderDash:[4,3],borderWidth:1.5,
       pointRadius:0,fill:false,tension:.3},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}}},
      scales:{
        x:{ticks:{font:{size:10}}},
        y:{title:{display:true,text:'Bike hours',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Taper chart — shows daily trail km + bike km + avg line */
function taperChart(id, race, avgData){
  _kill(id);
  const ctx=document.getElementById(id).getContext('2d');
  const lhd=race.last_hard;

  // For upcoming: inject planned sessions from plan.key_sessions
  const planSessions={};
  if(race.is_upcoming&&race.plan&&race.plan.key_sessions){
    race.plan.key_sessions.forEach(s=>{ planSessions[s.day]={km:s.trail_km,label:s.label}; });
  }

  const trailColors=race.taper.map(d=>{
    if(planSessions[d.day]) return 'rgba(217,119,6,.85)';
    if(d.day===lhd) return 'rgba(239,68,68,.85)';
    return race.is_upcoming?'rgba(148,163,184,.5)':race.is_good?'rgba(22,163,74,.7)':'rgba(220,38,38,.65)';
  });
  const trailData=race.taper.map(d=>{
    if(race.is_upcoming&&planSessions[d.day]) return planSessions[d.day].km;
    return d.trail_km;
  });

  _ch[id]=new Chart(ctx,{type:'bar',
    data:{labels:race.taper.map(d=>'D'+d.day),datasets:[
      {label:'Trail km',data:trailData,backgroundColor:trailColors},
      {label:'Bike km', data:race.taper.map(d=>d.bike_km),backgroundColor:'rgba(59,130,246,.45)'},
      {label:'Grp avg km',data:avgData.map(d=>d.trail_km),type:'line',
       borderColor:'rgba(148,163,184,.7)',borderDash:[3,3],borderWidth:1.5,
       pointRadius:0,fill:false,tension:.3},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{font:{size:10},boxWidth:10}},
        tooltip:{callbacks:{title:ttips=>{
          const d=race.taper[ttips[0].dataIndex];
          const ps=planSessions[d.day];
          if(ps) return `Day ${d.day} — PLANNED: ${ps.label}`;
          return `Day ${d.day}  (D+ ${d.trail_dplus}m)`;
        }}}},
      scales:{
        x:{ticks:{font:{size:9},maxRotation:45}},
        y:{title:{display:true,text:'km',font:{size:10}},ticks:{font:{size:10}}},
      }}
  });
}

/* Taper plan comparison card HTML for upcoming races */
function planCardHTML(race, goodAvg){
  if(!race.plan||!race.plan.weeks) return '';
  const pw=race.plan.weeks;
  const rows=['-2','-1'].map(k=>{
    const p=pw[k];if(!p)return '';
    const wk=parseInt(k);
    const ref=goodAvg.find(w=>w.week===wk);
    if(!ref) return '';
    const dtPct=(v,r)=>r>0?Math.round((v-r)/r*100):0;
    const pClass=(pct)=>Math.abs(pct)<=20?'plan-ok':'plan-warn';
    const tPct=dtPct(p.trail_h,ref.trail_h);
    const dPct=dtPct(p.trail_dplus,ref.trail_dplus);
    const bPct=dtPct(p.bike_h,ref.bike_h);
    return `<tr>
      <td><strong>W${k}</strong></td>
      <td class="${pClass(tPct)}">${p.trail_km}km / ${p.trail_dplus}m D+ / ${p.trail_h}h
        <small>(${tPct>0?'+':''}${tPct}% vs ref trail h)</small></td>
      <td>${ref.trail_km}km / ${ref.trail_dplus}m D+ / ${ref.trail_h}h</td>
      <td class="${pClass(dPct)}">${dPct>0?'+':''}${dPct}%</td>
      <td class="${pClass(bPct)}">${p.bike_km}km bike / ${p.bike_h}h
        <small>(${bPct>0?'+':''}${bPct}% vs ref)</small></td>
      <td>${ref.bike_km}km / ${ref.bike_h}h</td>
    </tr>`;
  }).join('');

  const sessions=(race.plan.key_sessions||[]).map(s=>
    `<span class="session-badge">D${s.day}: ${s.label}</span>`
  ).join('');

  return `<div class="plan-card">
    <h3>&#9654; UTHG Taper Plan vs Good-Race Reference</h3>
    <table class="plan-table">
      <thead><tr>
        <th>Week</th><th>Your Plan (trail)</th><th>Good Avg (trail ref)</th><th>D+ delta</th>
        <th>Your Plan (bike)</th><th>Good Avg (bike ref)</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="sessions-row"><strong>Key sessions:</strong> ${sessions}</div>
  </div>`;
}

/* ── Views ── */
let _sortCol=9,_sortDir=1;

function showOverview(){
  document.querySelectorAll('.race-item').forEach(e=>e.classList.remove('active'));
  document.getElementById('overview-btn').classList.add('active');
  document.getElementById('view').innerHTML=`
    <div class="cgrid3">
      <div class="card"><h2>Build — avg trail hours / week</h2>
        <div class="cwrap"><canvas id="ov-th"></canvas></div>
        <p class="chart-note">Green = good races &nbsp;|&nbsp; Red = bad races</p></div>
      <div class="card"><h2>Build — avg trail D+ / week</h2>
        <div class="cwrap"><canvas id="ov-dp"></canvas></div>
        <p class="chart-note">Elevation load in meters: good vs bad</p></div>
      <div class="card"><h2>Taper — avg daily trail km</h2>
        <div class="cwrap"><canvas id="ov-t"></canvas></div></div>
    </div>
    <div class="card"><h2>All races — click row to view detail</h2>
      <table id="rt">
        <thead><tr>
          <th onclick="sortTable(0)">Date</th>
          <th onclick="sortTable(1)">Race</th>
          <th onclick="sortTable(2)">Score</th>
          <th onclick="sortTable(3)">Dist</th>
          <th onclick="sortTable(4)">D+</th>
          <th onclick="sortTable(5)">Trail km</th>
          <th onclick="sortTable(6)">Trail D+</th>
          <th onclick="sortTable(7)">Trail h</th>
          <th onclick="sortTable(8)">Bike km</th>
          <th onclick="sortTable(9)">Last hard day</th>
          <th onclick="sortTable(10)">Peak week</th>
        </tr></thead>
        <tbody id="rtbody"></tbody>
      </table>
    </div>`;
  ovTrailHChart('ov-th');
  ovDplusChart('ov-dp');
  ovTaperChart('ov-t');
  renderTable();
}

function renderTable(){
  const rows=[...DATA.races].sort((a,b)=>{
    const v=r=>[r.date,r.name,r.score??-1,r.dist_km,r.dplus_m,
                 r.build_trail_km,r.build_dplus,r.build_trail_h,
                 r.build_bike_km, r.last_hard??99, r.peak_week??0];
    const av=v(a)[_sortCol], bv=v(b)[_sortCol];
    return av<bv?-_sortDir:av>bv?_sortDir:0;
  });
  document.getElementById('rtbody').innerHTML=rows.map(r=>{
    const i=DATA.races.indexOf(r);
    const gr=r.is_upcoming?'upcr':r.is_good?'grow':'badr';
    const lhd=r.last_hard;
    const lhdTxt=lhd!=null?`D${lhd}`:'-';
    const lhdCls=lhd==null?'':lhd<=-14?'early':lhd>=-8?'late':'';
    const scoreTxt=r.score!=null?r.score:'–';
    const scClass=r.score!=null?`sc s${r.score}`:'';
    return `<tr class="${gr}" onclick="showRace(${i})" style="cursor:pointer">
      <td>${r.date}</td><td>${r.name}</td>
      <td class="${scClass}">${scoreTxt}</td>
      <td>${r.dist_km}km</td><td>${r.dplus_m}m</td>
      <td>${r.build_trail_km}</td><td>${r.build_dplus}</td>
      <td>${r.build_trail_h}h</td><td>${r.build_bike_km}</td>
      <td class="lhd-cell ${lhdCls}">${lhdTxt}</td>
      <td>W${r.peak_week} (${r.peak_total_h}h)</td>
    </tr>`;
  }).join('');
}

function sortTable(col){
  _sortDir=(_sortCol===col)?-_sortDir:1;
  _sortCol=col;
  document.querySelectorAll('#rt th').forEach((th,i)=>th.classList.toggle('sorted',i===col));
  renderTable();
}

function showRace(idx){
  const r=DATA.races[idx];
  const avg = r.is_upcoming ? DATA.good_build_avg : (r.is_good?DATA.good_build_avg:DATA.bad_build_avg);
  const atap= r.is_upcoming ? DATA.good_taper_avg : (r.is_good?DATA.good_taper_avg:DATA.bad_taper_avg);
  const sc=r.is_upcoming?'upcoming':r.is_good?'good':'bad';

  document.querySelectorAll('.race-item').forEach(e=>e.classList.remove('active'));
  document.getElementById('overview-btn').classList.remove('active');
  const ri=document.getElementById('ri'+idx);
  if(ri){ri.classList.add('active');ri.scrollIntoView({block:'nearest'});}

  const lhd=r.last_hard;
  const lhdTxt=r.is_upcoming?(lhd!=null?`Last hard so far: D${lhd}`:'No hard day recorded yet')
                             :(lhd!=null?`Last hard day: D${lhd}`:'No hard day in 21 days');
  const lhdCls=lhd==null?'ok':lhd<=-14?'good':lhd>=-8?'bad':'ok';
  const scoreDisp=r.score!=null?`${r.score}/4`:'UPCOMING';
  const planNote=r.is_upcoming?'<p class="chart-note" style="color:#d97706"><strong>Lighter bars = planned (W-2/W-1)</strong></p>':'';

  document.getElementById('view').innerHTML=`
    <div class="card">
      <div class="rh-wrap">
        <div style="flex:1">
          <div class="rh-title">${r.name}</div>
          <div class="rh-pills">
            <span class="pill">${r.date}</span>
            <span class="pill">${r.dist_km} km</span>
            <span class="pill">D+ ${r.dplus_m} m</span>
            <span class="pill">8wk trail: ${r.build_trail_km}km / ${r.build_dplus}m D+ / ${r.build_trail_h}h</span>
            <span class="pill">8wk bike: ${r.build_bike_km}km / ${r.build_bike_h}h</span>
            <span class="pill">Peak: W${r.peak_week} (${r.peak_total_h}h total)</span>
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div class="score-big ${sc}">${scoreDisp}</div>
          <div class="lhd-tag ${lhdCls}">${lhdTxt}</div>
        </div>
      </div>
    </div>
    ${r.is_upcoming?planCardHTML(r,DATA.good_build_avg):''}
    <div class="card">
      <h3>Build — Trail (8 weeks)</h3>
      <div class="cwrap"><canvas id="rc-trail"></canvas></div>
      <p class="chart-note">Bars = trail hours &nbsp;|&nbsp; Dashed = group avg &nbsp;|&nbsp; Yellow line = D+ (m, right axis)${r.is_upcoming?' &nbsp;|&nbsp; <strong style="color:#d97706">Light bars = planned W-2/W-1</strong>':''}</p>
    </div>
    <div style="margin-bottom:12px">
      <button id="bike-btn" onclick="toggleBike()"
        style="font-size:11px;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:4px 14px;cursor:pointer">
        &#9660; Show bike chart
      </button>
    </div>
    <div id="bike-section" style="display:none">
      <div class="card" style="margin-bottom:12px">
        <h3>Build — Bike (8 weeks)</h3>
        <div class="cwrap"><canvas id="rc-bike"></canvas></div>
        <p class="chart-note">Bars = bike hours &nbsp;|&nbsp; Dashed = group avg${r.is_upcoming?' &nbsp;|&nbsp; <strong style="color:#d97706">Light bars = planned W-2/W-1</strong>':''}</p>
      </div>
    </div>
    <div class="card">
      <h3>Taper — 21 days before race</h3>
      <div class="cwrap" style="height:220px"><canvas id="rc-t"></canvas></div>
      <p class="chart-note">${r.is_upcoming?'Orange bars = planned sessions &nbsp;|&nbsp;':'Red bar = last hard day (D'+(lhd??'–')+') &nbsp;|&nbsp;'} Dashed = group avg trail km</p>
    </div>`;

  _curRace=r; _curAvg=avg;
  trailBuildChart('rc-trail', r, avg);
  taperChart('rc-t', r, atap);
}

let _curRace=null, _curAvg=null;
function toggleBike(){
  const s=document.getElementById('bike-section');
  const b=document.getElementById('bike-btn');
  const show=s.style.display==='none';
  s.style.display=show?'':'none';
  b.innerHTML=(show?'&#9650; Hide':'&#9660; Show')+' bike chart';
  if(show) bikeBuildChart('rc-bike',_curRace,_curAvg);
}

/* ── Init ── */
buildSidebar();
showOverview();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML race build/taper report")
    parser.add_argument("--trail-only", action="store_true",
                        help="Exclude road/low-vert races (D+/km < 30)")
    parser.add_argument("--min-dist", type=float, default=0, metavar="KM",
                        help="Only include races with distance >= KM")
    parser.add_argument("--open", action="store_true",
                        help="Open report in browser after generation")
    args = parser.parse_args()

    summaries = load_summaries()
    summaries["date"] = pd.to_datetime(summaries["date"], errors="coerce")
    summaries = summaries.dropna(subset=["date"])

    good_races, bad_races, upcoming_races = _load_objectives(
        trail_only=args.trail_only, min_dist=args.min_dist
    )
    print(f"Computing {len(good_races)} good + {len(bad_races)} bad + "
          f"{len(upcoming_races)} upcoming race windows ...")

    good_windows     = [compute_race_window(summaries, r) for r in good_races]
    bad_windows      = [compute_race_window(summaries, r) for r in bad_races]
    upcoming_windows = [compute_race_window(summaries, r) for r in upcoming_races]

    data = {
        "good_build_avg": avg_build(good_windows),
        "bad_build_avg":  avg_build(bad_windows),
        "good_taper_avg": avg_taper(good_windows),
        "bad_taper_avg":  avg_taper(bad_windows),
        "races": (
            [_window_to_dict(w, True)  for w in good_windows] +
            [_window_to_dict(w, False) for w in bad_windows]  +
            [_window_to_dict(w, True, is_upcoming=True, plan=r.get("taper_plan"))
             for w, r in zip(upcoming_windows, upcoming_races)]
        ),
    }

    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved to {OUT_FILE.resolve()}")

    if args.open:
        webbrowser.open(OUT_FILE.resolve().as_uri())


if __name__ == "__main__":
    main()
