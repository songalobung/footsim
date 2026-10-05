"""Unit tests for economic betting backtest, Edge/EV engine, and CLV tracking."""

from __future__ import annotations

import pandas as pd
import pytest

from footsim.eval.betting import (
    assert_feature_timestamp,
    calculate_edge_and_ev,
    calculate_stake,
    run_economic_backtest,
)


def test_assert_feature_timestamp_valid() -> None:
    """Valid timestamps (feature prior to kickoff) should not raise."""
    assert_feature_timestamp(
        "rolling_xg",
        "2026-10-10 12:00:00",
        "2026-10-10 15:00:00",
    )


def test_assert_feature_timestamp_leakage_detected() -> None:
    """Future or concurrent timestamps must trigger a data leakage ValueError."""
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        assert_feature_timestamp(
            "future_stat",
            "2026-10-10 15:00:00",
            "2026-10-10 15:00:00",
        )

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        assert_feature_timestamp(
            "post_match_result",
            "2026-10-10 17:00:00",
            "2026-10-10 15:00:00",
        )


def test_calculate_edge_and_ev() -> None:
    """Verify edge, EV, and closing line value calculations."""
    # Model prob = 0.50, market odds = 2.20 (implied ~0.4545)
    # devigged market prob = 0.45
    # closing odds = 2.00
    res = calculate_edge_and_ev(
        model_prob=0.50,
        market_odds=2.20,
        devigged_prob=0.45,
        closing_odds=2.00,
        min_edge=0.02,
        min_ev=0.02,
    )

    assert res.fair_odds == 2.0
    assert pytest.approx(res.edge, rel=1e-3) == 0.05
    assert pytest.approx(res.ev, rel=1e-3) == 0.10  # 0.50 * 2.20 - 1 = 0.10
    assert pytest.approx(res.clv, rel=1e-3) == 0.10  # 2.20 / 2.00 - 1 = 0.10
    assert res.is_value is True


def test_calculate_edge_and_ev_no_value() -> None:
    """Negative EV or edge below threshold should not qualify as value."""
    res = calculate_edge_and_ev(
        model_prob=0.40,
        market_odds=2.20,
        devigged_prob=0.45,
        min_edge=0.02,
        min_ev=0.02,
    )
    assert res.edge < 0
    assert res.ev < 0
    assert res.is_value is False


def test_calculate_stake_fractional_kelly() -> None:
    """Fractional Kelly sizing should correctly scale with edge and cap at max stake."""
    bankroll = 1000.0
    # b = 2.0 - 1 = 1.0, p = 0.60, q = 0.40
    # full Kelly: (1.0 * 0.6 - 0.4) / 1.0 = 0.20
    # 0.25 Kelly: 0.20 * 0.25 = 0.05 of bankroll -> $50
    # But max_stake_pct default is 0.03 -> max stake is $30
    stake = calculate_stake(
        bankroll=bankroll,
        model_prob=0.60,
        market_odds=2.0,
        edge=0.10,
        staking_mode="fractional_kelly",
        kelly_fraction=0.25,
        max_stake_pct=0.03,
    )
    assert pytest.approx(stake) == 30.0

    # Without max cap restriction
    stake_uncapped = calculate_stake(
        bankroll=bankroll,
        model_prob=0.60,
        market_odds=2.0,
        edge=0.10,
        staking_mode="fractional_kelly",
        kelly_fraction=0.25,
        max_stake_pct=0.10,
    )
    assert pytest.approx(stake_uncapped) == 50.0


def test_calculate_stake_negative_ev() -> None:
    """No stake should be placed if Kelly fraction is negative."""
    stake = calculate_stake(
        bankroll=1000.0,
        model_prob=0.30,
        market_odds=2.0,
        edge=-0.10,
        staking_mode="fractional_kelly",
    )
    assert stake == 0.0


def test_run_economic_backtest_simulation() -> None:
    """Test full simulation run with synthetic matchday data."""
    dates = pd.date_range("2025-08-15", periods=5, freq="7D")
    df = pd.DataFrame(
        {
            "date": dates,
            "season": ["2526"] * 5,
            "home_team": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs"],
            "away_team": ["Wolves", "Fulham", "Everton", "Brentford", "Bournemouth"],
            "actual_ftr": ["H", "H", "D", "A", "H"],
            # FootSim model win probabilities
            "model_H": [0.65, 0.60, 0.55, 0.70, 0.58],
            "model_D": [0.20, 0.25, 0.30, 0.20, 0.24],
            "model_A": [0.15, 0.15, 0.15, 0.10, 0.18],
            # Opening odds with value on Home
            "PSH": [1.90, 2.00, 2.10, 1.80, 2.05],
            "PSD": [3.60, 3.50, 3.40, 3.80, 3.50],
            "PSA": [4.50, 4.20, 3.90, 5.00, 4.00],
            # Closing odds
            "pin_H_odds": [1.80, 1.90, 2.00, 1.70, 1.95],
            "pin_D_odds": [3.70, 3.60, 3.50, 3.90, 3.60],
            "pin_A_odds": [4.80, 4.50, 4.10, 5.20, 4.20],
        }
    )

    sim = run_economic_backtest(
        backtest_df=df,
        initial_bankroll=1000.0,
        min_edge=0.02,
        min_ev=0.02,
        staking_mode="fractional_kelly",
        kelly_fraction=0.25,
    )

    assert sim.total_bets > 0
    assert sim.turnover > 0
    assert sim.initial_bankroll == 1000.0
    assert sim.final_bankroll > 0
    assert len(sim.bets_df) == sim.total_bets
    assert sim.avg_clv_pct is not None
    # Report formatting should succeed without error
    rep = sim.format_report()
    assert "ECONOMIC BETTING BACKTEST" in rep
    assert "Closing Line Value" in rep
