"""Tests for squad availability, key absences, VORP replacement factor, and live simulation."""

import numpy as np
import pandas as pd
import pytest

from footsim.sim.inplay import LiveState, simulate_live
from footsim.sim.match import simulate_match
from footsim.sim.players import Lineup, Player, get_team_lineup
from footsim.sim.squad import (
    adjust_squad_and_absences,
    calculate_absence_impact,
    calculate_h2h_edge,
    calculate_recent_form,
    get_squad_tier,
    reconstruct_lineup_with_absences,
)


def test_squad_depth_tiers():
    tier1 = get_squad_tier("Man City")
    tier4 = get_squad_tier("Ipswich")

    assert tier1.tier == 1
    assert tier1.replacement_factor > 0.70
    assert tier4.tier == 4
    assert tier4.replacement_factor < 0.35


def test_absence_impact_positional_asymmetry():
    # Forward absence drops attack
    fw_imp = calculate_absence_impact("Erling Haaland", "Man City")
    assert fw_imp.position == "FW"
    assert fw_imp.delta_attack < 0.0
    assert fw_imp.delta_defense == 0.0

    # Defensive midfielder (Rodri effect) increases concession significantly
    dm_imp = calculate_absence_impact("Rodri", "Man City")
    assert dm_imp.position == "DM"
    assert dm_imp.delta_defense > 0.0
    assert dm_imp.delta_attack < 0.0

    # Goalkeeper increases concession
    gk_imp = calculate_absence_impact("Alisson Becker", "Liverpool")
    assert gk_imp.position == "GK"
    assert gk_imp.delta_defense > 0.0


def test_adjust_squad_and_absences_lam_mu():
    base_lam = 2.0
    base_mu = 1.0

    # Test striker out
    adj_lam, adj_mu, impacts, notes = adjust_squad_and_absences(
        team="Arsenal",
        base_lam=base_lam,
        base_mu=base_mu,
        absent_players="Bukayo Saka",
    )
    assert adj_lam < base_lam
    assert len(impacts) == 1
    assert len(notes) == 1

    # Test short turnaround congestion (3 days rest)
    lam_rest, mu_rest, _, rest_notes = adjust_squad_and_absences(
        team="Arsenal",
        base_lam=base_lam,
        base_mu=base_mu,
        rest_days=3,
    )
    assert lam_rest < base_lam
    assert mu_rest > base_mu
    assert any("Congestion" in n for n in rest_notes)


def test_reconstruct_lineup():
    lineup = get_team_lineup("Arsenal")
    starter_names = [p.name for p in lineup.players]
    assert "Bukayo Saka" in starter_names

    adj_lineup, changes = reconstruct_lineup_with_absences(lineup, absent_players="Bukayo Saka")
    adj_names = [p.name for p in adj_lineup.players]
    assert "Bukayo Saka" not in adj_names
    assert any("Sub: Bench" in n for n in adj_names)
    assert len(changes) == 1


def test_simulate_live_inplay_minutes():
    # At minute 85 with 2-0 lead, home win prob should be > 95%
    state_late_lead = LiveState(
        home="Arsenal",
        away="Chelsea",
        minute=85,
        home_goals=2,
        away_goals=0,
    )
    res_late = simulate_live(state_late_lead, base_lam=1.8, base_mu=1.0, n=5_000, seed=42)
    assert res_late.ft_home_win_prob > 0.95
    assert res_late.rem_home_exp < 0.35  # only ~9 mins left

    # Red card hazard test: team with active red card scores less in remaining time
    state_11v11 = LiveState(home="Arsenal", away="Chelsea", minute=60, home_goals=0, away_goals=0)
    state_10v11 = LiveState(home="Arsenal", away="Chelsea", minute=60, home_goals=0, away_goals=0, home_reds=1)

    res_11v11 = simulate_live(state_11v11, base_lam=1.5, base_mu=1.0, n=5_000, seed=42)
    res_10v11 = simulate_live(state_10v11, base_lam=1.5, base_mu=1.0, n=5_000, seed=42)

    assert res_10v11.rem_home_exp < res_11v11.rem_home_exp
    assert res_10v11.rem_away_exp > res_11v11.rem_away_exp
    assert res_10v11.ft_home_win_prob < res_11v11.ft_home_win_prob


def test_match_simulation_with_squad_kwargs():
    sim_df = simulate_match(
        home="Arsenal",
        away="Chelsea",
        n=1_000,
        seed=42,
        home_absent="Saka",
        away_absent="Palmer",
        home_rest_days=3,
        player_layer=True,
    )
    assert len(sim_df) == 1_000
    assert "squad_notes" in sim_df.attrs
    assert len(sim_df.attrs["squad_notes"]) >= 2
    assert "first_scorer_player" in sim_df.columns
