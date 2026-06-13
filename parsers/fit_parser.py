"""
parsers/fit_parser.py

Parse a Garmin .fit file into a tidy pandas DataFrame.
Each row = one recorded data point (typically 1s resolution).

Columns produced (when available in the file):
    timestamp       datetime64[ns, UTC]
    elapsed_s       float  — seconds since activity start
    distance_m      float  — cumulative metres
    altitude_m      float  — GPS/barometric altitude
    speed_ms        float  — m/s
    heart_rate      float  — bpm
    power_w         float  — watts (if power meter present)
    cadence         float  — rpm/spm
    grade_pct       float  — % grade (computed from altitude + distance)
    lat             float  — degrees
    lon             float  — degrees
    temp_c          float  — °C (if sensor present)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

# fitparse is the most battle-tested FIT library for Python.
# Install: pip install fitparse
try:
    import fitparse
except ImportError:
    fitparse = None  # handled gracefully below


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_fit_file(path: str | Path) -> pd.DataFrame:
    """
    Parse a .fit file and return a cleaned DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .fit file.

    Returns
    -------
    pd.DataFrame
        One row per record message, sorted by timestamp.
        Grade column is computed from altitude + distance when not provided
        by the device.
    """
    _require_fitparse()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FIT file not found: {path}")

    records = _extract_records(path)
    df = _records_to_dataframe(records)
    df = _clean(df)
    return df


def fit_metadata(path: str | Path) -> dict:
    """
    Return high-level metadata from a FIT file (sport, start time, device).
    Useful for quickly inspecting a file before full parsing.
    """
    _require_fitparse()
    path = Path(path)
    meta: dict = {"path": str(path)}

    fitfile = fitparse.FitFile(str(path))
    for msg in fitfile.get_messages("session"):
        d = {f.name: f.value for f in msg.fields if f.value is not None}
        meta.update({
            "sport": d.get("sport"),
            "sub_sport": d.get("sub_sport"),
            "start_time": d.get("start_time"),
            "total_elapsed_time_s": d.get("total_elapsed_time"),
            "total_distance_m": d.get("total_distance"),
            "total_ascent_m": d.get("total_ascent"),
            "total_descent_m": d.get("total_descent"),
            "avg_heart_rate": d.get("avg_heart_rate"),
            "max_heart_rate": d.get("max_heart_rate"),
            "avg_speed_ms": d.get("avg_speed"),
            "total_calories": d.get("total_calories"),
        })
    for msg in fitfile.get_messages("device_info"):
        d = {f.name: f.value for f in msg.fields if f.value is not None}
        meta["manufacturer"] = d.get("manufacturer")
        meta["product"] = d.get("product")
        break
    for msg in fitfile.get_messages("workout"):
        d = {f.name: f.value for f in msg.fields if f.value is not None}
        meta["workout_name"] = d.get("wkt_name") or None
        break

    return meta


def parse_fit_all(path: "str | Path | bytes") -> tuple["pd.DataFrame", "pd.DataFrame", list[dict], str | None]:
    """
    Parse a FIT file in a single pass.

    Parameters
    ----------
    path : str, Path, or bytes
        Path to a .fit file, or raw FIT bytes (e.g. from a decompressed .fit.gz).

    Returns
    -------
    df            : record time-series DataFrame (same as parse_fit_file)
    laps_df       : lap DataFrame (same as parse_fit_laps)
    workout_steps : list of dicts (same as parse_fit_workout_steps)
    workout_name  : wkt_name string or None
    """
    import io as _io
    _require_fitparse()
    if isinstance(path, (bytes, bytearray)):
        fileish = _io.BytesIO(bytes(path))
    else:
        path = Path(path)
        fileish = str(path)
    fitfile = fitparse.FitFile(
        fileish, data_processor=fitparse.StandardUnitsDataProcessor()
    )

    records: list[dict] = []
    lap_rows: list[dict] = []
    wkt_steps: list[dict] = []
    workout_name: str | None = None
    t0 = None

    for msg in fitfile.get_messages():
        name = msg.name

        if name == "record":
            row: dict = {}
            for field in msg.fields:
                mapped = _FIELD_MAP.get(field.name)
                if mapped and field.value is not None:
                    row[mapped] = field.value
            if row:
                records.append(row)

        elif name == "lap":
            d = {f.name: f.value for f in msg.fields if f.value is not None}
            start_t = d.get("start_time")
            if t0 is None and start_t:
                t0 = start_t
            start_s = (start_t - t0).total_seconds() if (start_t and t0) else 0.0
            dur   = float(d.get("total_elapsed_time") or 0)
            dist  = float(d.get("total_distance") or 0)
            speed_ms = dist / dur if dur > 1 else 0.0
            step_idx = d.get("wkt_step_index")
            if hasattr(step_idx, "value"):
                step_idx = step_idx.value
            lap_rows.append({
                "start_s":        round(start_s, 1),
                "duration_s":     round(dur, 1),
                "end_s":          round(start_s + dur, 1),
                "distance_m":     round(dist, 1),
                "avg_speed_ms":   round(speed_ms, 3),
                "avg_hr":         d.get("avg_heart_rate"),
                "intensity":      str(d.get("intensity") or "").lower(),
                "trigger":        str(d.get("lap_trigger") or "").lower(),
                "wkt_step_index": step_idx,
            })

        elif name == "workout_step":
            d = {f.name: f.value for f in msg.fields if f.value is not None}
            idx = d.get("message_index")
            if hasattr(idx, "value"):
                idx = idx.value
            target_type = str(d.get("target_type") or "").lower().replace(" ", "_")
            wkt_steps.append({
                "step_index":        int(idx) if idx is not None else len(wkt_steps),
                "intensity":         str(d.get("intensity") or "").lower(),
                "target_type":       target_type,
                "target_value":      d.get("target_value"),
                # Garmin uses custom_target_* instead of target_value when the user
                # defines a custom zone range.  Capture high end of each range for use
                # as an "effective target" in work/recovery classification.
                "custom_hr_high":    d.get("custom_target_heart_rate_high"),
                "custom_speed_high": d.get("custom_target_speed_high"),
            })

        elif name == "workout" and workout_name is None:
            d = {f.name: f.value for f in msg.fields if f.value is not None}
            workout_name = d.get("wkt_name") or None

    del fitfile  # fitparse holds all messages in memory; release explicitly
    df      = _clean(_records_to_dataframe(records))
    laps_df = pd.DataFrame(lap_rows) if lap_rows else pd.DataFrame()
    return df, laps_df, wkt_steps, workout_name


def parse_fit_laps(path: str | Path) -> pd.DataFrame:
    """
    Return one row per lap message from a FIT file.

    Useful for track sessions where the watch records each interval/recovery
    as a distinct lap with an intensity label (active / warmup / rest / cooldown).
    Distances are in metres, speeds derived from distance ÷ elapsed time.
    wkt_step_index links each lap to its workout step (for target-based classification).
    """
    _require_fitparse()
    path = Path(path)
    fitfile = fitparse.FitFile(
        str(path), data_processor=fitparse.StandardUnitsDataProcessor()
    )

    t0 = None
    rows: list[dict] = []
    for msg in fitfile.get_messages("lap"):
        d = {f.name: f.value for f in msg.fields if f.value is not None}
        start_t = d.get("start_time")
        if t0 is None and start_t:
            t0 = start_t
        start_s = (start_t - t0).total_seconds() if (start_t and t0) else 0.0
        dur = float(d.get("total_elapsed_time") or 0)
        dist = float(d.get("total_distance") or 0)  # metres (no unit conversion needed)
        # avg_speed from Garmin is often 0 for track laps; compute from dist/time
        speed_ms = dist / dur if dur > 1 else 0.0
        step_idx = d.get("wkt_step_index")
        if hasattr(step_idx, "value"):
            step_idx = step_idx.value
        rows.append({
            "start_s":       round(start_s, 1),
            "duration_s":    round(dur, 1),
            "end_s":         round(start_s + dur, 1),
            "distance_m":    round(dist, 1),
            "avg_speed_ms":  round(speed_ms, 3),
            "avg_hr":        d.get("avg_heart_rate"),
            "intensity":     str(d.get("intensity") or "").lower(),
            "trigger":       str(d.get("lap_trigger") or "").lower(),
            "wkt_step_index": step_idx,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def parse_fit_workout_steps(path: str | Path) -> list[dict]:
    """
    Return one dict per workout_step message from a FIT file.

    Keys: step_index (int), intensity (str), target_type (str), target_value (float|None).
    Returns empty list if the file has no structured workout.
    """
    _require_fitparse()
    path = Path(path)
    fitfile = fitparse.FitFile(str(path))

    steps: list[dict] = []
    for msg in fitfile.get_messages("workout_step"):
        d = {f.name: f.value for f in msg.fields if f.value is not None}
        idx = d.get("message_index")
        if hasattr(idx, "value"):
            idx = idx.value
        target_type = str(d.get("target_type") or "").lower().replace(" ", "_")
        steps.append({
            "step_index":   int(idx) if idx is not None else len(steps),
            "intensity":    str(d.get("intensity") or "").lower(),
            "target_type":  target_type,
            "target_value": d.get("target_value"),
        })
    return steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FIELD_MAP = {
    "timestamp": "timestamp",
    "distance": "distance_m",
    "altitude": "altitude_m",
    "enhanced_altitude": "altitude_m",   # prefer enhanced when available
    "speed": "speed_ms",
    "enhanced_speed": "speed_ms",
    "heart_rate": "heart_rate",
    "power": "power_w",
    "cadence": "cadence",
    "fractional_cadence": "_frac_cadence",
    "position_lat": "lat_semicircles",
    "position_long": "lon_semicircles",
    "temperature": "temp_c",
    "grade": "grade_pct",
}

_SEMICIRCLE_TO_DEG = 180.0 / (2**31)


def _require_fitparse():
    if fitparse is None:
        raise ImportError(
            "fitparse is required: pip install fitparse"
        )


def _extract_records(path: Path) -> list[dict]:
    fitfile = fitparse.FitFile(
        str(path),
        data_processor=fitparse.StandardUnitsDataProcessor(),
    )
    rows = []
    for msg in fitfile.get_messages("record"):
        row: dict = {}
        for field in msg.fields:
            mapped = _FIELD_MAP.get(field.name)
            if mapped and field.value is not None:
                # enhanced_* fields overwrite the base field — desired
                row[mapped] = field.value
        if row:
            rows.append(row)
    return rows


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Convert semicircle coordinates to degrees
    for col, factor in [("lat_semicircles", _SEMICIRCLE_TO_DEG),
                        ("lon_semicircles", _SEMICIRCLE_TO_DEG)]:
        if col in df.columns:
            out = col.replace("_semicircles", "")
            df[out] = df[col] * factor
            df.drop(columns=[col], inplace=True)

    # Cadence: add fractional part when present
    if "_frac_cadence" in df.columns:
        if "cadence" in df.columns:
            df["cadence"] = df["cadence"].fillna(0) + df["_frac_cadence"].fillna(0)
        df.drop(columns=["_frac_cadence"], inplace=True)

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # StandardUnitsDataProcessor converts FIT native meters to km — restore to meters
    if "distance_m" in df.columns:
        df["distance_m"] = df["distance_m"] * 1000

    # StandardUnitsDataProcessor converts speed from m/s to km/h — restore to m/s
    if "speed_ms" in df.columns:
        df["speed_ms"] = df["speed_ms"] / 3.6

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = (
            (df["timestamp"] - df["timestamp"].iloc[0])
            .dt.total_seconds()
            .astype(float)
        )

    # Compute grade from altitude + distance if not supplied by device
    if "grade_pct" not in df.columns:
        df["grade_pct"] = _compute_grade(df)

    # Clip physiologically implausible values
    if "heart_rate" in df.columns:
        df.loc[df["heart_rate"] < 30, "heart_rate"] = float("nan")
        df.loc[df["heart_rate"] > 230, "heart_rate"] = float("nan")
    if "speed_ms" in df.columns:
        df.loc[df["speed_ms"] < 0, "speed_ms"] = 0.0
        df.loc[df["speed_ms"] > 15, "speed_ms"] = float("nan")  # > 54 km/h running
    if "power_w" in df.columns:
        df.loc[df["power_w"] < 0, "power_w"] = 0.0
        df.loc[df["power_w"] > 2000, "power_w"] = float("nan")

    return df


def _compute_grade(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Estimate % grade from altitude and distance using a rolling window
    to reduce GPS noise.
    """
    if "altitude_m" not in df.columns or "distance_m" not in df.columns:
        return pd.Series(float("nan"), index=df.index)

    alt = df["altitude_m"].rolling(window, center=True, min_periods=1).mean()
    dist = df["distance_m"]

    d_alt = alt.diff()
    d_dist = dist.diff()

    grade = pd.Series(float("nan"), index=df.index)
    mask = d_dist.abs() > 0.1
    grade[mask] = (d_alt[mask] / d_dist[mask]) * 100.0

    # Clip to physically reasonable range (−60% to +80%)
    grade = grade.clip(-60, 80)
    return grade
