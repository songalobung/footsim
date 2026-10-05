"""Data loading and cleaning for football-data.co.uk CSVs."""

from footsim.data.loader import (
    DEFAULT_SEASONS,
    load_matches,
    last_completed_seasons,
    season_code,
    season_start_year,
)

__all__ = [
    "DEFAULT_SEASONS",
    "load_matches",
    "last_completed_seasons",
    "season_code",
    "season_start_year",
]
