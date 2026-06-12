"""
scripts/edit_races.py

Terminal editor for data/races/races.json.
Navigate races, set type and score without touching the JSON directly.

Controls
--------
  Enter a number to edit that race
  q  — quit and save
  l  — list all races again
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent.parent
RACES_FILE = _ROOT / "data/races/races.json"
TYPES      = ("objective", "preparatory", "red_flag")


def _load() -> list[dict]:
    return json.loads(RACES_FILE.read_text())


def _save(races: list[dict]) -> None:
    RACES_FILE.write_text(json.dumps(races, indent=2, ensure_ascii=False))


def _type_label(t: str | None) -> str:
    return {"objective": "OBJ", "preparatory": "PREP", "red_flag": "RED "}.get(t or "", "  ? ")


def _score_label(s) -> str:
    return str(s) if s is not None else "-"


def _list(races: list[dict]) -> None:
    print(f"\n{'#':>3}  {'Date':10}  {'T':4}  {'S'}  {'Name'}")
    print("─" * 65)
    for i, r in enumerate(races):
        print(f"{i:>3}  {r['date']:10}  {_type_label(r.get('type')):4}  "
              f"{_score_label(r.get('score'))}  {r['name'][:40]}")
    print()


def _edit(race: dict) -> dict:
    print(f"\n── {race['date']} — {race['name']}")
    print(f"   {race['distance_km']}km  D+{race.get('dplus_m', '?')}m")
    print(f"   Current type : {race.get('type') or 'unset'}")
    print(f"   Current score: {race.get('score') if race.get('score') is not None else 'unset'}")

    # Type
    print(f"\n   Type: (o)bjective  (p)reparatory  (r)ed_flag  (blank=keep)")
    t = input("   > ").strip().lower()
    if t == "o":
        race["type"] = "objective"
    elif t == "p":
        race["type"] = "preparatory"
    elif t == "r":
        race["type"] = "red_flag"

    # Score (only if objective)
    if race.get("type") == "objective":
        print(f"   Score 0–4 (blank=keep): ", end="")
        s = input().strip()
        if s in {"0", "1", "2", "3", "4"}:
            race["score"] = int(s)
    else:
        race["score"] = None

    return race


def main() -> None:
    races = _load()
    _list(races)

    while True:
        cmd = input("# to edit, (l)ist, (q)uit > ").strip().lower()

        if cmd == "q":
            _save(races)
            print(f"Saved {len(races)} races.")
            break

        elif cmd == "l":
            _list(races)

        elif cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(races):
                races[idx] = _edit(races[idx])
                _save(races)
                r = races[idx]
                print(f"   → {_type_label(r.get('type'))}  score={_score_label(r.get('score'))}")
            else:
                print(f"   Invalid index (0–{len(races)-1})")

        else:
            print("   Unknown command.")


if __name__ == "__main__":
    main()
