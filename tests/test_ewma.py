"""Unit tests for Phase C opponent-adjusted EWMA ratings and shot-level xG proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footsim.data.ewma import (
    OpponentAdjustedEWMA,
    compute_match_xg_proxy,
    get_current_ewma_ratings,
)
from footsim.goals.dixon_coles import DixonColes


def test_compute_match_xg_proxy() -> None:
    """Test shot-level xG proxy conversion calculations."""
    # 10 shots, 4 on target, 5 corners
    xg = compute_match_xg_proxy(shots=10, shots_on_target=4, corners=5, goals=1)
    # expected: 0.31 * 4 + 0.045 * 6 + 0.035 * 5 = 1.24 + 0.27 + 0.175 = 1.685
    assert pytest.approx(xg, rel=1e-2) == 1.685

    # Edge cases
    assert compute_match_xg_proxy(shots=0, shots_on_target=0, goals=2) == 2.0
    assert compute_match_xg_proxy(shots=np.nan, shots_on_target=np.nan, goals=0) == 0.0



def test_opponent_adjusted_ewma_no_leakage() -> None:
    """Verify that matchday 1 features do not have lookahead leakage."""
    dates = pd.date_range("2025-08-15", periods=4, freq="7D")
    df = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[2]],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Man City"],
            "AwayTeam": ["Everton", "Fulham", "Chelsea", "Arsenal"],
            "FTHG": [3, 1, 2, 0],
            "FTAG": [0, 1, 1, 2],
            "HS": [15, 10, 12, 8],
            "HST": [6, 3, 5, 2],
            "AS": [4, 8, 9, 14],
            "AST": [1, 2, 3, 6],
            "HC": [7, 4, 6, 3],
            "AC": [2, 3, 4, 7],
        }
    )

    engine = OpponentAdjustedEWMA(alpha=0.20)
    res_df = engine.compute_all_prematch_features(df)

    # First matchday fixtures MUST have pre-match baseline ratings = 1.0
    assert res_df.loc[0, "home_ewma_att"] == 1.0
    assert res_df.loc[0, "away_ewma_def"] == 1.0
    assert res_df.loc[1, "home_ewma_att"] == 1.0

    # Arsenal scored 3 goals against Everton in match 1, so Arsenal's attack rating
    # in match 2 (row 2) MUST be higher than 1.0!
    assert res_df.loc[2, "home_ewma_att"] > 1.0


def test_dixon_coles_with_ewma() -> None:
    """Test fitting and predicting with DixonColes incorporating ewma_weight."""
    dates = pd.date_range("2025-08-15", periods=6, freq="7D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Liverpool", "Man City", "Arsenal", "Liverpool", "Man City"],
            "FTHG": [2, 1, 3, 0, 1, 2],
            "FTAG": [1, 1, 1, 2, 2, 0],
            "HS": [12, 10, 15, 8, 11, 14],
            "HST": [5, 4, 7, 2, 4, 6],
            "AS": [8, 9, 6, 12, 10, 5],
            "AST": [3, 3, 2, 5, 4, 1],
            "HC": [6, 5, 8, 3, 5, 7],
            "AC": [3, 4, 2, 6, 4, 2],
        }
    )

    dc = DixonColes(ewma_weight=0.15, ewma_alpha=0.20).fit(df)
    assert hasattr(dc, "ewma_ratings_")
    assert len(dc.ewma_ratings_) > 0

    lam, mu = dc.expected_goals("Arsenal", "Chelsea")
    assert lam > 0
    assert mu > 0

    p_h, p_d, p_a = dc.predict_1x2("Arsenal", "Chelsea")
    assert pytest.approx(p_h + p_d + p_a, rel=1e-4) == 1.0
