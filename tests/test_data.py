"""Tests for footsim.data.loader."""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
import pytest

from footsim.data.loader import (
    DEFAULT_SEASONS,
    clean_season,
    download_season,
    last_completed_seasons,
    load_team_mapping,
    normalise_team_names,
    parse_dates,
    season_start_year,
)


def test_parse_dates_handles_both_formats():
    out = parse_dates(pd.Series(["12/09/20", "13/08/2021", " 05/08/2022 "]))
    assert list(out) == [pd.Timestamp("2020-09-12"), pd.Timestamp("2021-08-13"),
                         pd.Timestamp("2022-08-05")]


def test_parse_dates_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_dates(pd.Series(["2021-08-13"]))


def test_season_helpers():
    assert season_start_year(pd.Timestamp("2025-01-11")) == 2024
    assert season_start_year(pd.Timestamp("2025-07-01")) == 2025
    from datetime import date
    assert last_completed_seasons(6, today=date(2026, 10, 3)) == list(DEFAULT_SEASONS)


def test_team_mapping_normalises_aliases_and_warns_on_unknown():
    mapping = load_team_mapping()
    s = pd.Series(["Manchester United", "Man United", "Spurs", "Nottingham Forest"])
    assert list(normalise_team_names(s, mapping)) == ["Man United", "Man United",
                                                      "Tottenham", "Nott'm Forest"]
    with pytest.warns(UserWarning, match="Atlantis FC"):
        out = normalise_team_names(pd.Series(["Atlantis FC"]), mapping)
    assert out.iloc[0] == "Atlantis FC"


def test_clean_season_drops_missing_goals_and_flags_missing_events():
    raw = pd.DataFrame({
        "Date": ["01/09/2023", "02/09/2023", "03/09/2023"],
        "HomeTeam": ["Arsenal", "Chelsea", "Wolves"],
        "AwayTeam": ["Chelsea", "Wolves", "Arsenal"],
        "FTHG": [1, np.nan, 2], "FTAG": [0, 1, 2], "FTR": ["H", None, "D"],
        "HY": [1, 2, np.nan], "AY": [2, 2, 1],
    })
    out = clean_season(raw, "2324", load_team_mapping())
    assert len(out) == 2                      # missing-goal row dropped
    assert out["events_missing"].tolist() == [True, True]  # many event cols absent
    assert {"PSCH", "PSCD", "PSCA", "Referee"} <= set(out.columns)
    assert out["season"].eq("2324").all()


def test_download_is_cached(tmp_path, monkeypatch):
    calls = []

    class FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            calls.append(1)
            return b"Div,Date\nE0,01/01/2024\n"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    p1 = download_season("2324", cache_dir=tmp_path)
    p2 = download_season("2324", cache_dir=tmp_path)
    assert p1 == p2 and p1.exists()
    assert len(calls) == 1                    # second call served from cache
    download_season("2324", cache_dir=tmp_path, refresh=True)
    assert len(calls) == 2


@pytest.mark.network
def test_real_data_shape(epl):
    assert sorted(epl["season"].unique()) == sorted(DEFAULT_SEASONS)
    assert epl.groupby("season").size().eq(380).all()
    assert epl["Date"].is_monotonic_increasing
    canonical = set(load_team_mapping().values())
    assert set(epl["HomeTeam"]) | set(epl["AwayTeam"]) <= canonical
    for _, g in epl.groupby("season"):
        assert g["HomeTeam"].nunique() == 20
    assert epl[["FTHG", "FTAG"]].notna().all().all()
