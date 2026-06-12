"""
parsers/garmin_client.py

Garmin Connect client — login + bulk/incremental FIT download.

Credentials (in .env):
    GARMIN_EMAIL
    GARMIN_PASSWORD

FIT files are saved to data/fit/<date>_<sport>_<activity_id>.fit
State is tracked in .garmin_state.json (last sync date + downloaded IDs).

Usage
-----
client = GarminClient()

# Initial bulk download from 2019
client.sync(from_date="2019-01-01")

# Incremental (uses last_sync from state file)
client.sync()
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_FILE   = _PROJECT_ROOT / ".garmin_state.json"
_FIT_DIR      = _PROJECT_ROOT / "data/fit"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_sync": "2019-01-01", "downloaded": []}


def _save_state(state: dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GarminClient:
    """
    Garmin Connect client.

    Usage
    -----
    client = GarminClient()
    client.sync(from_date="2019-01-01")   # bulk
    client.sync()                          # incremental
    """

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        fit_dir: Path = _FIT_DIR,
    ):
        try:
            from garminconnect import Garmin
        except ImportError:
            raise ImportError("garminconnect is required: pip install garminconnect")

        self._email    = email    or os.getenv("GARMIN_EMAIL", "")
        self._password = password or os.getenv("GARMIN_PASSWORD", "")
        self._fit_dir  = Path(fit_dir)
        self._fit_dir.mkdir(parents=True, exist_ok=True)

        self._api = Garmin(self._email, self._password)
        self._api.login()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, from_date: Optional[str] = None) -> list[Path]:
        """
        Download all activities since from_date (or last_sync if omitted).

        Returns the list of newly downloaded FIT file paths.
        """
        from datetime import date

        state = _load_state()
        from_date = from_date or state["last_sync"]
        downloaded_ids = set(state["downloaded"])

        activities = self._fetch_all_activities(from_date)
        to_download = [a for a in activities if a["activityId"] not in downloaded_ids]

        print(f"{len(to_download)} activité(s) à télécharger depuis {from_date}...")

        new_files: list[Path] = []
        for i, activity in enumerate(to_download, 1):
            path = self._download_one(activity, i, len(to_download))
            if path:
                downloaded_ids.add(activity["activityId"])
                new_files.append(path)

        state["downloaded"] = list(downloaded_ids)
        state["last_sync"]  = date.today().isoformat()
        _save_state(state)

        print(f"Sync terminée — {len(new_files)} nouveau(x) fichier(s).")
        return new_files

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_all_activities(self, from_date: str) -> list[dict]:
        activities: list[dict] = []
        start, limit = 0, 100

        while True:
            batch = self._api.get_activities(start, limit)
            if not batch:
                break
            activities.extend(batch)
            if len(batch) < limit:
                break
            start += limit
            time.sleep(1)

        return [a for a in activities if a["startTimeLocal"][:10] >= from_date]

    def _download_one(self, activity: dict, index: int, total: int) -> Optional[Path]:
        activity_id = activity["activityId"]
        date_str    = activity["startTimeLocal"][:10]
        sport       = activity.get("activityType", {}).get("typeKey", "unknown")
        filepath    = self._fit_dir / f"{date_str}_{sport}_{activity_id}.fit"

        if filepath.exists() and filepath.stat().st_size > 0:
            return filepath

        try:
            data = self._api.download_activity(
                activity_id,
                dl_fmt=self._api.ActivityDownloadFormat.ORIGINAL,
            )
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                fit_files = [n for n in z.namelist() if n.endswith(".fit")]
                if not fit_files:
                    print(f"  [{index}/{total}] Pas de .fit dans le ZIP pour {activity_id}")
                    return None
                with z.open(fit_files[0]) as src:
                    filepath.write_bytes(src.read())

            print(f"  [{index}/{total}] {filepath.name}")
            time.sleep(0.5)
            return filepath

        except Exception as e:
            print(f"  [{index}/{total}] Erreur {activity_id}: {e}")
            return None
