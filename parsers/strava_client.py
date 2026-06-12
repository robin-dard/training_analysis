"""
parsers/strava_client.py

Strava API v3 client — token refresh + activity streams.

Covers the same data we've been pulling via the Claude MCP connector,
but as standalone Python for use in scripts.

Scopes needed: activity:read_all
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_CACHE  = _PROJECT_ROOT / ".strava_token_cache.json"
_BASE = "https://www.strava.com/api/v3"
_AUTH = "https://www.strava.com/oauth/token"


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def _load_cached_token() -> Optional[dict]:
    if _TOKEN_CACHE.exists():
        try:
            return json.loads(_TOKEN_CACHE.read_text())
        except Exception:
            pass
    return None


def _save_token(data: dict) -> None:
    _TOKEN_CACHE.write_text(json.dumps(data, indent=2))


def _refresh_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    resp = requests.post(_AUTH, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> str:
    """
    Return a valid Strava access token, refreshing if expired.
    Credentials fall back to environment variables.
    """
    client_id = client_id or os.getenv("STRAVA_CLIENT_ID", "")
    client_secret = client_secret or os.getenv("STRAVA_CLIENT_SECRET", "")
    refresh_token = refresh_token or os.getenv("STRAVA_REFRESH_TOKEN", "")

    cached = _load_cached_token()
    if cached and cached.get("expires_at", 0) > time.time() + 60:
        return cached["access_token"]

    data = _refresh_token(client_id, client_secret, refresh_token)
    _save_token(data)
    return data["access_token"]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class StravaClient:
    """
    Thin Strava API v3 client.

    Usage
    -----
    client = StravaClient()
    activities = client.list_activities(after="2026-01-01", before="2026-06-05")
    streams = client.get_streams("12345678901")
    """

    def __init__(self):
        self._token = get_access_token()
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self._token}"

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    def list_activities(
        self,
        after: Optional[str] = None,   # ISO date string "YYYY-MM-DD"
        before: Optional[str] = None,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """
        List activities. Handles basic pagination — call repeatedly
        incrementing `page` until empty list returned.
        """
        import datetime
        params: dict = {"per_page": per_page, "page": page}
        if after:
            params["after"] = int(datetime.datetime.fromisoformat(after).timestamp())
        if before:
            params["before"] = int(datetime.datetime.fromisoformat(before).timestamp())
        return self._get("/athlete/activities", params=params)

    def get_activity(self, activity_id: str | int) -> dict:
        return self._get(f"/activities/{activity_id}")

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------

    STREAM_KEYS = [
        "time", "distance", "altitude", "velocity_smooth",
        "heart_rate", "cadence", "watts", "grade_smooth",
        "moving", "temp", "latlng",
    ]

    def get_streams(
        self,
        activity_id: str | int,
        keys: Optional[list[str]] = None,
        resolution: Optional[str] = None,  # "low" | "medium" | "high"
    ) -> dict[str, list]:
        """
        Fetch activity streams. Returns a dict keyed by stream name,
        values are lists of data points.

        Parameters
        ----------
        keys : list of stream names (defaults to all available)
        resolution : optional downsampling ("low"=100pts, "medium"=1000pts,
                     "high"=max)
        """
        keys = keys or self.STREAM_KEYS
        params: dict = {"keys": ",".join(keys), "key_by_type": "true"}
        if resolution:
            params["resolution"] = resolution
        raw = self._get(f"/activities/{activity_id}/streams", params=params)
        # Flatten to {stream_name: [values]}
        return {k: v["data"] for k, v in raw.items() if "data" in v}

    def streams_to_dataframe(
        self,
        activity_id: str | int,
        keys: Optional[list[str]] = None,
        resolution: Optional[str] = None,
    ) -> "pd.DataFrame":
        """
        Fetch streams and return as a DataFrame.
        Requires pandas.
        """
        import pandas as pd
        streams = self.get_streams(activity_id, keys=keys, resolution=resolution)
        df = pd.DataFrame(streams)
        # latlng → lat, lon columns
        if "latlng" in df.columns:
            df[["lat", "lon"]] = pd.DataFrame(df["latlng"].tolist(), index=df.index)
            df.drop(columns=["latlng"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Laps
    # ------------------------------------------------------------------

    def get_laps(self, activity_id: str | int) -> list[dict]:
        return self._get(f"/activities/{activity_id}/laps")

    # ------------------------------------------------------------------
    # Raw HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = _BASE.rstrip("/") + path
        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            raise RuntimeError("Strava rate limit hit — wait 15 min")
        resp.raise_for_status()
        return resp.json()
