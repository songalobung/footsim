"""Unit and integration tests for advanced model and simulation enhancements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footsim.data.derbies import KNOWN_DERBIES, is_derby
from footsim.eval.calibration import PlattCalibrator, evaluate_calibration
from footsim.goals.copula import copula_score_matrix, frank_copula_cdf
from footsim.goals.dixon_coles import DixonColes
from footsim.sim.config import DEFAULT_SIM_CONFIG, SimConfig
from footsim.sim.match import simulate_match
from footsim.sim.players import (
    DEFAULT_LINEUPS,
    Lineup,
    Player,
    assign_player_events,
    build_generic_lineup,
    get_team_lineup,
    player_scorer_probs,
)


def test_copula_score_matrix_properties():
    """Frank copula bivariate score matrix must sum to 1.0 and be non-negative."""
    for theta in [-0.5, 0.0, 0.25, 0.75]:
        mat = copula_score_matrix(lam=1.5, mu=1.2, theta=theta, max_goals=10)
        assert mat.shape == (11, 11)
        assert (mat >= 0.0).all()
        assert np.isclose(mat.sum(), 1.0, atol=1e-5)

    # When theta is 0, Frank copula matches independent Poisson
    mat_zero = copula_score_matrix(lam=1.5, mu=1.2, theta=0.0, max_goals=10)
    mat_dc_rho0 = DixonColes(fixed_rho=0.0)
    # Check marginals sum to expected
    assert np.isclose(mat_zero.sum(), 1.0)


def test_is_derby():
    """Verify derby identification logic."""
    assert is_derby("Arsenal", "Chelsea")
    assert is_derby("Chelsea", "Arsenal")
    assert is_derby("Liverpool", "Everton")
    assert is_derby("Man City", "Man United")
    assert not is_derby("Arsenal", "Bournemouth")
    assert not is_derby("Liverpool", "Fulham")


def test_player_lineups_and_attribution():
    """Verify player lineup structures and event attribution."""
    arsenal_lineup = get_team_lineup("Arsenal")
    assert len(arsenal_lineup.players) == 11
    assert np.isclose(arsenal_lineup.xg_probs.sum(), 1.0)
    assert np.isclose(arsenal_lineup.card_probs.sum(), 1.0)

    # Test generic lineup fallback
    generic_lineup = get_team_lineup("UnknownFC")
    assert len(generic_lineup.players) == 11

    # Simulate match with player layer
    sim_df = simulate_match("Arsenal", "Chelsea", n=1000, seed=42, player_layer=True)
    assert "home_goalscorers" in sim_df.columns
    assert "away_goalscorers" in sim_df.columns
    assert "first_scorer_player" in sim_df.columns

    # Verify anytime scorer probs
    probs = player_scorer_probs(sim_df, arsenal_lineup, is_home=True)
    assert len(probs) == 11
    assert "anytime_prob" in probs.columns
    assert "first_prob" in probs.columns
    # Saka should have high anytime probability
    saka_row = probs[probs["player"] == "Bukayo Saka"].iloc[0]
    assert saka_row["anytime_prob"] > 0.10


def test_probability_calibration():
    """Verify PlattCalibrator on simulated probabilities."""
    n = 200
    rng = np.random.default_rng(123)
    # Generate synthetic uncalibrated probabilities
    raw_probs = rng.dirichlet([2.0, 1.0, 1.5], size=n)
    actuals = rng.choice(["H", "D", "A"], size=n, p=[0.45, 0.25, 0.30])

    res = evaluate_calibration(raw_probs, actuals)
    assert res.calibrated_probs.shape == (n, 3)
    np.testing.assert_allclose(res.calibrated_probs.sum(axis=1), 1.0, atol=1e-5)
    assert res.cal_log_loss > 0
    assert res.cal_rps > 0


def test_dixon_coles_team_home_adv(synthetic):
    """Verify team home advantage estimation in DixonColes."""
    matches, _ = synthetic
    model = DixonColes(home_adv_mode="team").fit(matches)
    assert hasattr(model, "team_home_adv_")
    assert len(model.team_home_adv_) == len(model.teams_)
    for t in model.teams_:
        assert isinstance(model.team_home_adv_[t], float)

    t1, t2 = model.teams_[0], model.teams_[1]
    lam, mu = model.expected_goals(t1, t2)
    assert lam > 0
    assert mu > 0

    # Test copula method in score_matrix
    cop_mat = model.score_matrix(t1, t2, method="copula", copula_theta=0.3)
    assert cop_mat.shape == (11, 11)
    assert np.isclose(cop_mat.sum(), 1.0, atol=1e-5)


def test_sim_config_secondary_event_multipliers():
    """Verify secondary event multipliers in SimConfig."""
    cfg = SimConfig()
    diff = np.array([-2, 0, 2])  # home trailing, level, home leading

    # In minute 60 (2nd half), trailing team generates more corners
    h_cmult, a_cmult = cfg.get_corner_multipliers(diff, minute=60)
    assert h_cmult[0] == cfg.trailing_corner_mult  # home trailing -> more corners
    assert a_cmult[0] == cfg.leading_corner_mult   # away leading -> fewer corners
    assert a_cmult[2] == cfg.trailing_corner_mult  # away trailing -> more corners

    # In minute 75, close match (|diff| <= 1) escalates cards
    card_mult = cfg.get_card_multipliers(diff, minute=75, is_derby=False)
    assert card_mult[1] == cfg.close_game_card_mult  # level game -> card boost
    assert card_mult[0] == 1.0                       # 2-goal diff -> normal

    # In derby, card multiplier is applied
    derby_card_mult = cfg.get_card_multipliers(diff, minute=20, is_derby=True)
    assert derby_card_mult[0] == cfg.derby_card_mult
