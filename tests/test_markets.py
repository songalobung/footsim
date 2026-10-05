"""Tests for Market Query Layer (Milestone 5)."""

import pytest
import numpy as np
import pandas as pd

from footsim.sim.match import simulate_match
from footsim.markets.query import (
    wilson_score_interval,
    market,
    market_1x2,
    market_correct_score,
    market_correct_scores,
    market_over_under_goals,
    market_btts,
    market_clean_sheet,
    market_asian_handicap,
    market_total_cards,
    market_total_corners,
    market_red_card,
    market_first_scorer_team,
    market_ht_ft,
)


@pytest.fixture(scope="module")
def sample_sims():
    """Simulate 20,000 matches with seed=42 for market testing."""
    return simulate_match(
        home="Arsenal",
        away="Chelsea",
        n=20_000,
        seed=42,
        lam=1.60,
        mu=1.10,
        rho=-0.05,
    )


def test_wilson_score_interval():
    """Test Wilson score interval properties and bounds."""
    # 0 successes
    low, high = wilson_score_interval(0, 1000)
    assert low == 0.0
    assert 0.0 < high < 0.01

    # 1000 successes
    low, high = wilson_score_interval(1000, 1000)
    assert 0.99 < low < 1.0
    assert high == 1.0

    # 500 out of 1000 (p=0.5)
    low, high = wilson_score_interval(500, 1000)
    assert 0.46 < low < 0.50
    assert 0.50 < high < 0.54
    assert low < high


def test_arbitrary_market_query(sample_sims):
    """Test market evaluation with arbitrary string expressions."""
    # Query per spec example
    res = market(sample_sims, "home_goals == 2 and away_goals == 1 and home_reds >= 1")
    assert isinstance(res.prob, float)
    assert 0.0 <= res.prob <= 1.0
    assert res.count == np.count_nonzero(
        (sample_sims["home_goals"] == 2)
        & (sample_sims["away_goals"] == 1)
        & (sample_sims["home_reds"] >= 1)
    )
    assert res.total == 20_000
    assert res.ci_lower <= res.prob <= res.ci_upper

    # Float conversion
    assert float(res) == res.prob


def test_sample_size_rule(sample_sims):
    """Test 200-sample size reliability rule and suggestion."""
    # Frequent event: reliable
    res_frequent = market(sample_sims, "home_goals > away_goals")
    assert res_frequent.count >= 200
    assert res_frequent.reliable is True
    assert res_frequent.suggested_n is None
    assert "Reliable" in res_frequent.summary()

    # Rare event: unreliable
    res_rare = market(sample_sims, "home_goals == 6 and away_goals == 5")
    assert res_rare.count < 200
    assert res_rare.reliable is False
    assert res_rare.suggested_n is not None
    assert res_rare.suggested_n > len(sample_sims)
    assert "UNRELIABLE" in res_rare.summary()
    assert "suggested n >=" in res_rare.summary()


def test_market_1x2(sample_sims):
    """Test 1X2 market returns H, D, A summing to 1.0."""
    res_1x2 = market_1x2(sample_sims)
    assert "H" in res_1x2 and "D" in res_1x2 and "A" in res_1x2
    assert "1" in res_1x2 and "X" in res_1x2 and "2" in res_1x2

    p_sum = res_1x2["H"].prob + res_1x2["D"].prob + res_1x2["A"].prob
    assert pytest.approx(1.0, abs=1e-6) == p_sum
    assert res_1x2["H"].count + res_1x2["D"].count + res_1x2["A"].count == 20_000


def test_market_correct_scores(sample_sims):
    """Test correct score and top N correct scores."""
    cs_00 = market_correct_score(sample_sims, 0, 0)
    manual_count = np.count_nonzero((sample_sims["home_goals"] == 0) & (sample_sims["away_goals"] == 0))
    assert cs_00.count == manual_count

    top10 = market_correct_scores(sample_sims, top_n=10)
    assert len(top10) == 10
    # Check descending order
    probs = [res.prob for _, res in top10]
    assert probs == sorted(probs, reverse=True)


def test_market_totals_and_btts(sample_sims):
    """Test over/under goals and BTTS."""
    ou = market_over_under_goals(sample_sims, line=2.5)
    assert pytest.approx(1.0, abs=1e-6) == ou["over"].prob + ou["under"].prob

    btts = market_btts(sample_sims)
    assert pytest.approx(1.0, abs=1e-6) == btts["yes"].prob + btts["no"].prob


def test_market_cards_and_corners(sample_sims):
    """Test cards, corners, and red card markets."""
    cards = market_total_cards(sample_sims, line=3.5)
    assert pytest.approx(1.0, abs=1e-6) == cards["over"].prob + cards["under"].prob

    corners = market_total_corners(sample_sims, line=9.5)
    assert pytest.approx(1.0, abs=1e-6) == corners["over"].prob + corners["under"].prob

    red_match = market_red_card(sample_sims)
    assert red_match.count == np.count_nonzero(sample_sims["total_reds"] > 0)

    red_home = market_red_card(sample_sims, team="home")
    red_away = market_red_card(sample_sims, team="away")
    assert red_home.count == np.count_nonzero(sample_sims["home_reds"] > 0)
    assert red_away.count == np.count_nonzero(sample_sims["away_reds"] > 0)


def test_market_first_scorer_and_ht_ft(sample_sims):
    """Test first scorer team and HT/FT double result."""
    fs = market_first_scorer_team(sample_sims)
    assert pytest.approx(1.0, abs=1e-6) == fs["home"].prob + fs["away"].prob + fs["none"].prob

    ht_ft = market_ht_ft(sample_sims)
    assert len(ht_ft) == 9
    total_ht_ft_prob = sum(res.prob for res in ht_ft.values())
    assert pytest.approx(1.0, abs=1e-6) == total_ht_ft_prob


def test_market_asian_handicap(sample_sims):
    """Test Asian Handicap lines."""
    # Half line: -0.5 (must equal Home win)
    ah_half = market_asian_handicap(sample_sims, -0.5)
    h_win = market_1x2(sample_sims)["H"]
    assert pytest.approx(h_win.prob, abs=1e-6) == ah_half.home_win_prob
    assert ah_half.push_prob == 0.0

    # Level line: 0.0 (DNB)
    ah_zero = market_asian_handicap(sample_sims, 0.0)
    d_prob = market_1x2(sample_sims)["D"].prob
    assert pytest.approx(d_prob, abs=1e-6) == ah_zero.push_prob

    # Quarter line: -0.25
    ah_quarter = market_asian_handicap(sample_sims, -0.25)
    assert 0.0 <= ah_quarter.home_cover_rate <= 1.0
    assert 0.0 <= ah_quarter.away_cover_rate <= 1.0


def test_invalid_query_raises():
    """Test invalid expressions raise ValueError."""
    dummy_df = pd.DataFrame({"home_goals": [1, 2]})
    with pytest.raises(ValueError, match="Invalid market query expression"):
        market(dummy_df, "invalid >>> syntax error")
