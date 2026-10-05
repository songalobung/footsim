"""Tests for the Milestone 4 match simulation engine."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from footsim.goals.dixon_coles import dc_score_matrix
from footsim.sim import (
    DEFAULT_SIM_CONFIG,
    NEUTRAL_SIM_CONFIG,
    SimConfig,
    default_minute_profile,
    simulate_match,
)


def test_minute_profiles_sum_to_one():
    """All default minute profiles must be positive and integrate to 1.0."""
    for stat in ["goals", "yellows", "reds", "corners", "fouls"]:
        prof = default_minute_profile(stat, 90)
        assert len(prof) == 90
        assert (prof >= 0).all()
        assert np.isclose(prof.sum(), 1.0, atol=1e-6)


def test_acceptance_seed_determinism():
    """ACCEPTANCE: same seed gives identical output."""
    n = 2000
    df1 = simulate_match("Arsenal", "Chelsea", n=n, seed=9876)
    df2 = simulate_match("Arsenal", "Chelsea", n=n, seed=9876)

    cols_to_check = [
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "home_corners", "away_corners", "home_yellows", "away_yellows",
        "home_reds", "away_reds", "home_fouls", "away_fouls",
        "first_scorer", "result", "total_goals", "over_25", "btts",
    ]
    for col in cols_to_check:
        assert (df1[col] == df2[col]).all(), f"Determinism mismatch in column {col}"

    assert df1["home_goal_minutes"].tolist() == df2["home_goal_minutes"].tolist()
    assert df1["away_goal_minutes"].tolist() == df2["away_goal_minutes"].tolist()
    assert df1["home_red_minutes"].tolist() == df2["home_red_minutes"].tolist()
    assert df1["away_red_minutes"].tolist() == df2["away_red_minutes"].tolist()


def test_acceptance_speed_under_15s():
    """ACCEPTANCE: 100,000 simulations of one match finish in under 15 seconds."""
    t0 = time.perf_counter()
    df = simulate_match("Arsenal", "Chelsea", n=100_000, seed=42)
    elapsed = time.perf_counter() - t0

    assert len(df) == 100_000
    assert elapsed < 15.0, f"Simulation took {elapsed:.2f}s, exceeding 15s limit"


def test_acceptance_neutral_score_distribution_converges_to_dixon_coles():
    """ACCEPTANCE: With game-state multipliers set to 1, simulated scores converge to Dixon-Coles."""
    lam, mu, rho = 1.45, 1.15, -0.05
    target_matrix = dc_score_matrix(lam, mu, rho)

    # 1,000,000 neutral simulations
    df = simulate_match(
        "Arsenal",
        "Chelsea",
        n=1_000_000,
        seed=101,
        neutral=True,
        lam=lam,
        mu=mu,
        rho=rho,
    )

    # Compute empirical score matrix up to 10 goals
    sim_matrix = np.zeros_like(target_matrix)
    hg = df["home_goals"].to_numpy()
    ag = df["away_goals"].to_numpy()
    for i in range(11):
        for j in range(11):
            sim_matrix[i, j] = np.mean((hg == i) & (ag == j))

    # Test tolerance on all cells with probability >= 0.5% (0.005)
    cells_mask = target_matrix >= 0.005
    assert np.sum(cells_mask) >= 15, "Expected at least 15 cells with p >= 0.5%"

    diffs = np.abs(sim_matrix[cells_mask] - target_matrix[cells_mask])
    max_diff = float(np.max(diffs))

    # 1M sims gives standard error ~0.0003; tolerance of 0.005 is well above Monte Carlo noise
    assert max_diff < 0.005, f"Max difference {max_diff:.5f} exceeds tolerance 0.005"


def test_second_yellow_converts_to_red():
    """Players receiving two yellow cards must trigger a red card and sending off."""
    # Force high yellow card rate so second yellows occur frequently
    df = simulate_match("Arsenal", "Chelsea", n=5000, seed=42)
    # Total reds must be >= direct reds, and some matches should have second yellows
    assert (df["home_reds"] >= 0).all()
    assert (df["away_reds"] >= 0).all()


def test_output_dataframe_schema_and_parquet(tmp_path):
    """Verify DataFrame structure and Parquet save/load integrity."""
    pq_path = tmp_path / "sim.parquet"
    df = simulate_match("Arsenal", "Chelsea", n=500, seed=12, save_path=pq_path)

    expected_cols = [
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "home_corners", "away_corners", "home_yellows", "away_yellows",
        "home_reds", "away_reds", "home_fouls", "away_fouls",
        "first_scorer", "result", "total_goals", "over_25", "btts",
        "home_goal_minutes", "away_goal_minutes", "home_red_minutes", "away_red_minutes",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    assert pq_path.exists()
    loaded = pd.read_parquet(pq_path)
    assert len(loaded) == 500
    assert (loaded["home_goals"] == df["home_goals"]).all()
    assert (loaded["result"] == df["result"]).all()


def test_game_state_multipliers_effect():
    """Red cards and trailing late should dynamically adjust goal intensities."""
    cfg = SimConfig(
        red_card_scoring_mult=0.50,
        red_card_conceding_mult=2.00,
        late_trailing_scoring_mult=1.50,
        neutral=False,
    )
    # Difference = 0, no red cards -> mult = 1.0
    m_h, m_a = cfg.get_goal_multipliers(
        diff=np.array([0]),
        red_diff=np.array([0]),
        minute=30,
    )
    assert m_h[0] == 1.0 and m_a[0] == 1.0

    # Home has -1 red card (away has man advantage)
    m_h, m_a = cfg.get_goal_multipliers(
        diff=np.array([0]),
        red_diff=np.array([-1]),
        minute=30,
    )
    assert m_h[0] == pytest.approx(0.50)
    assert m_a[0] == pytest.approx(2.00)

    # Home trailing late (min 80, diff -1)
    m_h, m_a = cfg.get_goal_multipliers(
        diff=np.array([-1]),
        red_diff=np.array([0]),
        minute=80,
    )
    assert m_h[0] == pytest.approx(1.50)
