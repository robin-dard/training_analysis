"""
parsers/strava_zip.py

Read GPS time-series from a Strava bulk-export zip for activities that
have no Garmin FIT file (Strava-only sessions).

The zip's activities.csv maps Strava activity IDs to the actual file
names inside the zip (which may differ).  File formats supported:
  - activities/<name>.fit.gz  — preferred, richer data
  - activities/<name>.tcx.gz  — Garmin TCX (older activities)
  - activities/<name>.gpx     — fallback

The returned DataFrame has the same schema as parse_fit_file():
    elapsed_s, distance_m, altitude_m, speed_ms, heart_rate,
    grade_pct, lat, lon
"""
from __future__ import annotations

import gzip
import io
import math
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Zip index: activity_id -> zip entry path  (cached per zip path)
# ---------------------------------------------------------------------------

_ZIP_INDEX: dict[str, dict[str, str]] = {}  # zip_path_str -> {activity_id: zip_entry}


def _norm_col(s: str) -> str:
    """Normalise a CSV column name: strip accents, lowercase, strip whitespace."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()


def index_zip(zip_path: Path) -> dict[str, str]:
    """
    Return {strava_activity_id: zip_entry_path} for all activities.

    Uses activities.csv inside the zip to map Strava IDs to their actual
    filenames (they can differ, especially for older TCX exports).
    Falls back to filename-based ID extraction if CSV lookup fails.
    """
    key = str(zip_path)
    if key in _ZIP_INDEX:
        return _ZIP_INDEX[key]

    result: dict[str, str] = {}

    with zipfile.ZipFile(str(zip_path)) as z:
        all_names = set(z.namelist())

        # Build a set of valid activity entries keyed by their base name (no ext)
        activity_files: dict[str, str] = {}  # base_name -> zip_entry
        for name in all_names:
            if not name.startswith("activities/"):
                continue
            base = name[len("activities/"):]
            for ext in (".fit.gz", ".tcx.gz", ".gpx"):
                if base.endswith(ext):
                    stem = base[: -len(ext)]
                    # Prefer fit.gz > tcx.gz > gpx
                    existing = activity_files.get(stem)
                    if existing is None or (
                        ext == ".fit.gz" or
                        (ext == ".tcx.gz" and not existing.endswith(".fit.gz"))
                    ):
                        activity_files[stem] = name
                    break

        # activities.csv maps Strava activity ID -> filename
        if "activities.csv" in all_names:
            try:
                csv_bytes = z.read("activities.csv")
                acts = pd.read_csv(io.BytesIO(csv_bytes))
                acts.columns = [_norm_col(c) for c in acts.columns]

                id_col   = next((c for c in acts.columns if "id" in c and "activ" in c), None)
                file_col = next((c for c in acts.columns if "fichier" in c or "filename" in c or "file" in c), None)

                if id_col and file_col:
                    for _, row in acts.iterrows():
                        aid  = str(row[id_col]).strip()
                        fval = str(row[file_col]).strip() if pd.notna(row[file_col]) else ""
                        if not fval or fval == "nan":
                            continue
                        # fval is the zip entry path (e.g. "activities/12345.tcx.gz")
                        if fval in all_names:
                            result[aid] = fval
            except Exception:
                pass  # fall through to filename-based index

        # Fallback: use the file stem as the activity ID
        for stem, entry in activity_files.items():
            if stem not in result:
                result[stem] = entry

    _ZIP_INDEX[key] = result
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_strava_activity(zip_path: Path, activity_id: str) -> pd.DataFrame:
    """
    Return a time-series DataFrame for the given Strava activity ID.
    Returns empty DataFrame if not found or parse fails.
    """
    idx  = index_zip(zip_path)
    entry = idx.get(str(activity_id))
    if entry is None:
        return pd.DataFrame()

    with zipfile.ZipFile(str(zip_path)) as z:
        raw = z.read(entry)

    if entry.endswith(".fit.gz"):
        fit_bytes = gzip.decompress(raw)
        from parsers.fit_parser import parse_fit_all
        df, _, _, _ = parse_fit_all(fit_bytes)
        return df
    elif entry.endswith(".tcx.gz"):
        return _parse_tcx(gzip.decompress(raw))
    else:
        return _parse_gpx(raw)


# ---------------------------------------------------------------------------
# TCX parser
# ---------------------------------------------------------------------------

_TCX_NS = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"


def _parse_tcx(data: bytes) -> pd.DataFrame:
    """Parse a TCX file into a time-series DataFrame."""
    import xml.etree.ElementTree as ET

    # Strip UTF-8 BOM and any leading whitespace before the XML declaration
    text = data.decode("utf-8-sig", errors="replace").lstrip()
    root = ET.fromstring(text)
    ns   = _TCX_NS

    rows: list[dict] = []
    for tp in root.iter(f"{{{ns}}}Trackpoint"):
        time_el = tp.find(f"{{{ns}}}Time")
        pos_el  = tp.find(f"{{{ns}}}Position")
        alt_el  = tp.find(f"{{{ns}}}AltitudeMeters")
        dist_el = tp.find(f"{{{ns}}}DistanceMeters")
        hr_el   = tp.find(f".//{{{ns}}}Value")  # HeartRateBpm/Value

        lat = lon = None
        if pos_el is not None:
            lat_el = pos_el.find(f"{{{ns}}}LatitudeDegrees")
            lon_el = pos_el.find(f"{{{ns}}}LongitudeDegrees")
            if lat_el is not None and lon_el is not None:
                try:
                    lat = float(lat_el.text)
                    lon = float(lon_el.text)
                except (TypeError, ValueError):
                    pass

        rows.append({
            "lat":        lat,
            "lon":        lon,
            "altitude_m": float(alt_el.text) if (alt_el is not None and alt_el.text) else None,
            "distance_m": float(dist_el.text) if (dist_el is not None and dist_el.text) else None,
            "_t":         time_el.text if time_el is not None else None,
            "heart_rate": int(float(hr_el.text)) if (hr_el is not None and hr_el.text) else None,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["_t"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["elapsed_s"] = (
        (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().astype(float)
    )

    # Use TCX cumulative distance when available; compute from GPS otherwise
    if df["distance_m"].notna().any():
        df["distance_m"] = df["distance_m"].interpolate(method="linear")
    elif df["lat"].notna().any():
        df["distance_m"] = _haversine_cumulative(df["lat"].values, df["lon"].values)

    return _add_speed_grade(df.drop(columns=["_t"], errors="ignore"))


# ---------------------------------------------------------------------------
# GPX parser
# ---------------------------------------------------------------------------

_GPX_NS    = "http://www.topografix.com/GPX/1/1"
_GARMIN_NS = "http://www.garmin.com/xmlschemas/TrackPointExtension/v1"


def _parse_gpx(data: bytes) -> pd.DataFrame:
    """Parse a GPX file into a time-series DataFrame."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(data.decode("utf-8", errors="replace"))
    rows: list[dict] = []

    for trkpt in root.iter(f"{{{_GPX_NS}}}trkpt"):
        try:
            lat = float(trkpt.get("lat", 0))
            lon = float(trkpt.get("lon", 0))
        except ValueError:
            continue
        ele_el  = trkpt.find(f"{{{_GPX_NS}}}ele")
        time_el = trkpt.find(f"{{{_GPX_NS}}}time")
        hr_el   = trkpt.find(f".//{{{_GARMIN_NS}}}hr")
        rows.append({
            "lat":        lat,
            "lon":        lon,
            "altitude_m": float(ele_el.text) if (ele_el is not None and ele_el.text) else None,
            "_t":         time_el.text if time_el is not None else None,
            "heart_rate": int(hr_el.text) if (hr_el is not None and hr_el.text) else None,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["_t"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["elapsed_s"] = (
        (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().astype(float)
    )
    df["distance_m"] = _haversine_cumulative(df["lat"].values, df["lon"].values)

    return _add_speed_grade(df.drop(columns=["_t"], errors="ignore"))


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _add_speed_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Add speed_ms and grade_pct columns from distance + altitude."""
    dists = df["distance_m"].values

    d_dist = pd.Series(dists).diff().fillna(0)
    d_time = df["elapsed_s"].diff().fillna(1).replace(0, 1)
    speed  = (d_dist / d_time).clip(0, 15)
    df["speed_ms"] = speed.rolling(3, center=True, min_periods=1).mean()

    if "altitude_m" in df.columns and df["altitude_m"].notna().any():
        alt    = df["altitude_m"].rolling(5, center=True, min_periods=1).mean()
        d_alt  = alt.diff()
        grade  = pd.Series(float("nan"), index=df.index)
        mask   = d_dist.abs() > 0.1
        grade[mask] = (d_alt[mask] / d_dist[mask]) * 100.0
        df["grade_pct"] = grade.clip(-60, 80)
    else:
        df["grade_pct"] = float("nan")

    return df


def _haversine_cumulative(lats: "np.ndarray", lons: "np.ndarray") -> "np.ndarray":
    """Compute cumulative distance in metres from lat/lon arrays."""
    import numpy as np
    R   = 6_371_000.0
    cum = [0.0]
    for i in range(1, len(lats)):
        if lats[i] is None or lons[i] is None or lats[i-1] is None or lons[i-1] is None:
            cum.append(cum[-1])
            continue
        dlat = math.radians(float(lats[i]) - float(lats[i - 1]))
        dlon = math.radians(float(lons[i]) - float(lons[i - 1]))
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(float(lats[i - 1])))
             * math.cos(math.radians(float(lats[i])))
             * math.sin(dlon / 2) ** 2)
        cum.append(cum[-1] + 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    return np.array(cum)
