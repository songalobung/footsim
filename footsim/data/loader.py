"""Download, cache and clean football-data.co.uk season CSVs.

Every file is downloaded once into a local cache directory and read from there
on later runs. Team names are normalised with a single mapping file
(``team_names.csv`` next to this module).
"""

from __future__ import annotations

import io
import urllib.request
import warnings
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

#: Repo-root download cache. Completed seasons are never re-downloaded.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "football-data"

#: Team name mapping file (raw football-data name -> canonical name).
TEAM_NAMES_FILE = Path(__file__).resolve().parent / "team_names.csv"

#: The last 6 completed EPL seasons as of the 2026-27 season (fixed so that
#: results are reproducible; use :func:`last_completed_seasons` to recompute).
DEFAULT_SEASONS: tuple[str, ...] = ("2021", "2122", "2223", "2324", "2425", "2526")

GOAL_COLS = ["FTHG", "FTAG"]
EVENT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
ODDS_COLS = ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]
KEEP_COLS = (
    ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    + EVENT_COLS
    + ["Referee"]
    + ODDS_COLS
)


def season_code(start_year: int) -> str:
    """Return the football-data season code for a season starting in ``start_year``.

    >>> season_code(2025)
    '2526'
    """
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(d: pd.Timestamp | date) -> int:
    """Return the start year of the season a date belongs to.

    Seasons are taken to run from 1 July to 30 June (see docs/decisions.md).
    """
    return d.year if d.month >= 7 else d.year - 1


def last_completed_seasons(n: int = 6, today: date | None = None) -> list[str]:
    """Return codes of the last ``n`` seasons that ended before ``today``.

    A season is treated as completed once 1 July of its end year has passed.
    """
    today = today or date.today()
    current_start = season_start_year(pd.Timestamp(today))
    return [season_code(y) for y in range(current_start - n, current_start)]


def download_season(
    season: str,
    league: str = "E0",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Download one season CSV into the cache and return its path.

    The file is only fetched if it is not cached yet, or if ``refresh`` is True
    (useful for an in-progress season).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{league}_{season}.csv"
    if path.exists() and not refresh:
        return path
    url = BASE_URL.format(season=season, league=league)
    req = urllib.request.Request(url, headers={"User-Agent": "footsim/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)  # atomic: never leave a half-written cache file
    return path


def load_team_mapping(path: Path | str = TEAM_NAMES_FILE) -> dict[str, str]:
    """Load the raw -> canonical team name mapping."""
    m = pd.read_csv(path, comment="#", dtype=str)
    return dict(zip(m["raw"].str.strip(), m["canonical"].str.strip()))


def normalise_team_names(
    names: pd.Series, mapping: dict[str, str] | None = None
) -> pd.Series:
    """Map raw team names to canonical names.

    Names missing from the mapping are passed through unchanged with a warning,
    so a new promoted team never silently breaks loading, but is visible.
    """
    mapping = load_team_mapping() if mapping is None else mapping
    stripped = names.astype(str).str.strip()
    unknown = sorted(set(stripped) - set(mapping))
    if unknown:
        warnings.warn(
            f"Team names not in {TEAM_NAMES_FILE.name}, passed through unchanged: {unknown}",
            stacklevel=2,
        )
    return stripped.map(lambda s: mapping.get(s, s))


def parse_dates(raw: pd.Series) -> pd.Series:
    """Parse football-data dates explicitly.

    Older files use ``dd/mm/yy``, newer ones ``dd/mm/yyyy``. Each format is
    parsed with an explicit format string; anything else raises.
    """
    s = raw.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    long_mask = s.str.len() == 10
    short_mask = s.str.len() == 8
    out[long_mask] = pd.to_datetime(s[long_mask], format="%d/%m/%Y")
    out[short_mask] = pd.to_datetime(s[short_mask], format="%d/%m/%y")
    bad = ~(long_mask | short_mask)
    if bad.any():
        raise ValueError(f"Unrecognised date strings: {s[bad].unique()[:5].tolist()}")
    return out


def _read_csv_bytes(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def clean_season(df: pd.DataFrame, season: str, mapping: dict[str, str]) -> pd.DataFrame:
    """Clean a raw season frame: select columns, parse dates, normalise names.

    - Rows with missing goals are dropped.
    - Rows with missing event columns are kept and flagged in ``events_missing``.
    - Missing odds columns (e.g. closing odds in older files) are added as NaN.
    """
    df = df.dropna(how="all")
    df = df.loc[df["HomeTeam"].notna() & df["AwayTeam"].notna()].copy()
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[KEEP_COLS].copy()
    df = df.dropna(subset=GOAL_COLS)
    df["Date"] = parse_dates(df["Date"])
    df["HomeTeam"] = normalise_team_names(df["HomeTeam"], mapping)
    df["AwayTeam"] = normalise_team_names(df["AwayTeam"], mapping)
    df["Referee"] = df["Referee"].astype("string").str.strip().replace("", pd.NA)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    for col in EVENT_COLS + ODDS_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["events_missing"] = df[EVENT_COLS].isna().any(axis=1)
    df["season"] = season
    return df


def load_matches(
    seasons: Sequence[str] | Iterable[str] = DEFAULT_SEASONS,
    league: str = "E0",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load, clean and concatenate several seasons.

    Returns one row per played match, sorted by date (stable within a date),
    with a fresh integer index.
    """
    mapping = load_team_mapping()
    frames = []
    for season in seasons:
        path = download_season(season, league=league, cache_dir=cache_dir, refresh=refresh)
        frames.append(clean_season(_read_csv_bytes(path), season, mapping))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    return out
