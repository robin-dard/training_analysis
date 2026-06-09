# endurance_analysis

Personal trail running analysis toolkit — FIT files, Nolio API, Strava streams.

## Structure

```
data/
  fit/          → raw .fit files from Garmin (export from Garmin Connect)
  nolio/        → JSON exports from Nolio API
  strava/       → cached Strava stream JSON (optional)

parsers/
  fit_parser.py     → FIT file → structured DataFrame
  nolio_client.py   → Nolio API client (reverse-engineered endpoints)
  strava_client.py  → Strava API client (token-based)

analysis/
  ascensional.py    → ascensional speed by race phase
  hr_analysis.py    → HR drift, stability, zone distribution
  descent_speed.py  → descent/flat finish speed vs start
  race_compare.py   → multi-race comparison (the 3-metric composite)
  build_taper.py    → weekly build/taper aggregation

dashboard/
  report.py         → generate HTML report from analysis results

tests/
  test_parsers.py
  test_analysis.py
```

## Setup

```bash
pip install fitparse pandas numpy garmin-fit-sdk requests python-dotenv
```

## Config

Copy `.env.example` to `.env` and fill in your credentials:

```
STRAVA_CLIENT_ID=...
STRAVA_CLIENT_SECRET=...
STRAVA_REFRESH_TOKEN=...
NOLIO_EMAIL=...
NOLIO_PASSWORD=...
```

## Quick start

```python
from parsers.fit_parser import parse_fit_file
from analysis.ascensional import ascensional_speed_by_phase
from analysis.race_compare import RaceComparison

# From FIT file
df = parse_fit_file("data/fit/traversee_nord_2019.fit")
phases = ascensional_speed_by_phase(df, n_phases=3)

# Multi-race comparison
rc = RaceComparison()
rc.add_from_fit("TN 2019", "data/fit/traversee_nord_2019.fit")
rc.add_from_fit("UTHG 2026", "data/fit/uthg_2026.fit")
rc.compare()
```
