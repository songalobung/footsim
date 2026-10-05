"""Tests for footsim.eval (odds margin removal, metrics, and walk-forward backtest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footsim.eval.backtest import (
    compute_all_calibrations,
    compute_summary_metrics,
    matchday_blocks,
    plot_calibration,
    run_backtest_report,
    walk_forward_backtest,
)
from footsim.eval.metrics import (
    binary_log_loss,
    brier_score,
    calibration_table,
    multiclass_log_loss,
    ranked_probability_score,
)
from footsim.eval.odds import proportional_probs, remove_margin, shin_probs


# --------------------------------------------------------------------------
# Odds Margin Removal Tests
# --------------------------------------------------------------------------
def test_proportional_probs():
    odds = [2.0, 3.0, 6.0]  # sum(1/O) = 0.5 + 0.333 + 0.1667 = 1.0 (no margin)
    p = proportional_probs(odds)
    np.testing.assert_allclose(p, [0.5, 1.0 / 3.0, 1.0 / 6.0])

    # With margin
    odds_margin = [1.80, 3.50, 4.50]
    p_m = proportional_probs(odds_margin)
    assert p_m.sum() == pytest.approx(1.0)
    assert (p_m > 0).all()


def test_shin_probs():
    odds = [1.50, 4.00, 7.00]
    p_shin = shin_probs(odds)
    p_prop = proportional_probs(odds)

    # Shin probabilities sum to 1
    assert p_shin.sum() == pytest.approx(1.0, abs=1e-6)
    assert (p_shin > 0).all()

    # Shin addresses favorite-longshot bias:
    # Favorite gets higher probability under Shin than proportional,
    # and longshot gets lower probability.
    assert p_shin[0] > p_prop[0]
    assert p_shin[2] < p_prop[2]


def test_remove_margin_matrix():
    odds_matrix = np.array([
        [1.80, 3.60, 4.80],
        [2.20, 3.20, 3.40],
        [np.nan, 3.0, 4.0],
    ])
    res_shin = remove_margin(odds_matrix, method="shin")
    res_prop = remove_margin(odds_matrix, method="proportional")

    assert res_shin.shape == (3, 3)
    assert np.isnan(res_shin[2]).all()
    assert res_shin[0].sum() == pytest.approx(1.0, abs=1e-5)
    assert res_shin[1].sum() == pytest.approx(1.0, abs=1e-5)
    assert res_prop[0].sum() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# Metrics Tests
# --------------------------------------------------------------------------
def test_multiclass_log_loss():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outcomes = np.array([0, 1])
    assert multiclass_log_loss(probs, outcomes) == pytest.approx(0.0, abs=1e-8)

    probs_wrong = np.array([[0.0, 1.0, 0.0]])
    outcomes_wrong = np.array([0])
    assert multiclass_log_loss(probs_wrong, outcomes_wrong) > 10.0


def test_brier_score():
    probs = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    outcomes = np.array([0, 0])
    # match 0: (1-1)^2 = 0
    # match 1: (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5
    # mean: 0.25
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_ranked_probability_score():
    # Perfect forecast
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outcomes = np.array([0, 1])
    assert ranked_probability_score(probs, outcomes) == pytest.approx(0.0)

    # Distant forecast has higher penalty than close forecast
    # Home win happened (class 0)
    p_draw = np.array([[0.0, 1.0, 0.0]])  # predicted draw
    p_away = np.array([[0.0, 0.0, 1.0]])  # predicted away
    rps_draw = ranked_probability_score(p_draw, np.array([0]))
    rps_away = ranked_probability_score(p_away, np.array([0]))
    assert rps_draw < rps_away


def test_binary_log_loss():
    probs = np.array([0.9, 0.1])
    outcomes = np.array([1, 0])
    loss = binary_log_loss(probs, outcomes)
    expected = -0.5 * (np.log(0.9) + np.log(0.9))
    assert loss == pytest.approx(expected)


def test_calibration_table():
    probs = np.array([0.05, 0.15, 0.25, 0.85, 0.95])
    outcomes = np.array([0, 0, 0, 1, 1])
    table = calibration_table(probs, outcomes, n_bins=10)
    assert len(table) == 10
    assert table["count"].sum() == 5
    # Bin [0.8, 0.9) has 1 item, outcome 1 -> rate 1.0
    sub8 = table[(table["bin_lower"] == 0.8)]
    assert sub8["count"].iloc[0] == 1
    assert sub8["obs_frequency"].iloc[0] == 1.0


# --------------------------------------------------------------------------
# Matchday Blocking Tests
# --------------------------------------------------------------------------
def test_matchday_blocks():
    dates = pd.Series(pd.to_datetime([
        "2024-08-16",  # Friday
        "2024-08-17",  # Saturday
        "2024-08-18",  # Sunday
        "2024-08-24",  # Next Saturday (gap 6 days -> new block)
        "2024-08-25",  # Sunday
    ]))
    blks = matchday_blocks(dates, max_gap_days=2)
    assert blks.nunique() == 2
    assert blks.iloc[0] == blks.iloc[1] == blks.iloc[2]
    assert blks.iloc[3] == blks.iloc[4]
    assert blks.iloc[0] != blks.iloc[3]


# --------------------------------------------------------------------------
# Acceptance: Walk-Forward Backtest & Reports
# --------------------------------------------------------------------------
@pytest.mark.network
def test_acceptance_walk_forward_backtest_last_two_seasons(epl, tmp_path):
    """ACCEPTANCE: single command produces the report for the last two full seasons."""
    seasons = ["2425", "2526"]
    reports_dir = tmp_path / "reports"

    pred_df, summary_df, cal_df = run_backtest_report(
        seasons=seasons,
        reports_dir=reports_dir,
    )

    # Check that all 4 expected report artifacts exist
    assert (reports_dir / "backtest_predictions.csv").exists()
    assert (reports_dir / "backtest_summary.csv").exists()
    assert (reports_dir / "calibration_table.csv").exists()
    assert (reports_dir / "calibration_plot.png").exists()

    # Check predictions dataframe
    assert len(pred_df) == 760  # 380 matches * 2 seasons
    assert set(pred_df["season"].unique()) == {"2425", "2526"}
    for col in ["dc_pH", "dc_pD", "dc_pA", "dc_pOver25", "dc_pBTTS"]:
        assert (pred_df[col] >= 0).all() and (pred_df[col] <= 1).all()

    # 1X2 probabilities sum to 1
    sum_1x2 = pred_df["dc_pH"] + pred_df["dc_pD"] + pred_df["dc_pA"]
    np.testing.assert_allclose(sum_1x2, 1.0, rtol=1e-5)

    # Check summary metrics dataframe
    assert len(summary_df) == 3  # Overall, Season 2425, Season 2526
    for col in ["dc_1x2_log_loss", "dc_1x2_brier", "dc_1x2_rps", "dc_over25_log_loss", "dc_btts_log_loss"]:
        assert (summary_df[col] > 0).all()

    # Pinnacle benchmark is populated
    overall = summary_df.loc[summary_df["segment"] == "Overall (All Seasons)"].iloc[0]
    assert overall["pinnacle_matches"] > 500
    assert overall["pin_1x2_log_loss"] > 0
    assert overall["pin_1x2_rps"] > 0

    # Calibration table
    assert len(cal_df) == 50  # 5 markets * 10 bins
    assert set(cal_df["market"].unique()) == {"Home Win", "Draw", "Away Win", "Over 2.5 Goals", "BTTS"}
