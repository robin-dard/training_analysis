"""
parsers/nolio_client.py

Nolio / Enduraw dashboard API client.

The Nolio API is not publicly documented. Endpoints and auth tokens are
reverse-engineered from browser Network tab requests while logged in to
dashboard.enduraw-data.com.

HOW TO UPDATE ENDPOINTS
-----------------------
1. Open dashboard.enduraw-data.com in Chrome/Firefox.
2. Open DevTools → Network → filter by XHR/Fetch.
3. Log in, navigate to dashboard, weekly view, etc.
4. Identify the relevant requests and copy:
   - URL pattern
   - Request headers (especially Authorization: Bearer ...)
   - Request/response body structure
5. Update the constants and methods below accordingly.

AUTH FLOW (most likely — update if different)
----------------------------------------------
POST /api/auth/login  →  { token: "..." }
Subsequent requests:  Authorization: Bearer <token>
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class NolioConfig:
    base_url: str = os.getenv("NOLIO_BASE_URL", "https://dashboard.enduraw-data.com")
    email: str = os.getenv("NOLIO_EMAIL", "")
    password: str = os.getenv("NOLIO_PASSWORD", "")
    timeout: int = 30


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NolioClient:
    """
    Authenticated HTTP client for the Nolio/Enduraw dashboard API.

    Usage
    -----
    client = NolioClient()
    client.login()
    weekly = client.get_weekly_summary(date(2026, 1, 1), date(2026, 6, 5))
    """

    def __init__(self, config: Optional[NolioConfig] = None):
        self.cfg = config or NolioConfig()
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._token: Optional[str] = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self) -> None:
        """
        Authenticate and store Bearer token.
        Update the endpoint + payload structure after inspecting Network tab.
        """
        resp = self._post(
            "/api/auth/login",
            json={"email": self.cfg.email, "password": self.cfg.password},
            auth=False,
        )
        # Update key name if different in actual response
        self._token = resp.get("token") or resp.get("access_token")
        if not self._token:
            raise ValueError(f"Login succeeded but no token in response: {resp}")
        self._session.headers["Authorization"] = f"Bearer {self._token}"

    def set_token(self, token: str) -> None:
        """Manually set a token (e.g. copied from browser DevTools)."""
        self._token = token
        self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Training data endpoints
    # ------------------------------------------------------------------

    def get_weekly_summary(
        self,
        start: date,
        end: date,
    ) -> list[dict]:
        """
        Fetch weekly training summaries between two dates.
        Returns a list of week objects — update parsing once endpoint
        structure is confirmed.

        Expected response shape (update as needed):
        [
          {
            "week_start": "2026-01-06",
            "sport_totals": {
              "trail": {"duration_s": ..., "distance_m": ..., "elevation_m": ...},
              "bike": {"duration_s": ..., "distance_m": ..., "elevation_m": ...},
              "ski": {"duration_s": ..., "distance_m": ..., "elevation_m": ...},
            },
            "load_trimp": ...,
            "fitness": ...,
            "fatigue": ...,
            "form": ...,
          },
          ...
        ]
        """
        return self._get("/api/training/weekly", params={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        })

    def get_activities(
        self,
        start: date,
        end: date,
        sport: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch activity list. `sport` filters by type (e.g. "trail", "bike").
        """
        params: dict = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        if sport:
            params["sport"] = sport
        return self._get("/api/activities", params=params)

    def get_activity_detail(self, activity_id: str | int) -> dict:
        """Fetch full details for a single activity including FIT metrics."""
        return self._get(f"/api/activities/{activity_id}")

    def get_training_load(
        self,
        start: date,
        end: date,
    ) -> dict:
        """
        Fetch TRIMP-based load curve (fitness / fatigue / form).
        Returns the time-series used for the Nolio dashboard chart.
        """
        return self._get("/api/training/load", params={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        })

    def get_sport_totals(
        self,
        start: date,
        end: date,
    ) -> dict:
        """
        Fetch totals per sport for a date range.
        Matches the "Dénivelé par sport" breakdown visible in the dashboard.
        """
        return self._get("/api/training/totals", params={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        })

    # ------------------------------------------------------------------
    # Raw HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        resp = self._session.get(url, params=params, timeout=self.cfg.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict, auth: bool = True) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        resp = self._session.post(url, json=json, timeout=self.cfg.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience: dump raw response for endpoint discovery
    # ------------------------------------------------------------------

    def raw_get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        """
        Return the raw Response object for a given path.
        Use this while reverse-engineering endpoints to inspect
        headers, status codes, and raw body.

        Example
        -------
        r = client.raw_get("/api/whatever")
        print(r.status_code, r.headers)
        print(r.text[:2000])
        """
        url = self.cfg.base_url.rstrip("/") + path
        return self._session.get(url, params=params, timeout=self.cfg.timeout)
