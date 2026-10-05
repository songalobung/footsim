"""Vectorised minute-by-minute football match simulation engine (Milestone 4).

Simulates N matches simultaneously using NumPy across minutes 1..T (~95-100 min).
Every market query (result, correct score, totals, corners, cards, combinations)
is computed from this single set of simulated matches.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from footsim.data.derbies import is_derby as check_is_derby
from footsim.goals.dixon_coles import DixonColes
from footsim.events.rates import EventRates
from footsim.sim.config import (
    DEFAULT_SIM_CONFIG,
    NEUTRAL_SIM_CONFIG,
    SimConfig,
    default_minute_profile,
)
from footsim.sim.players import Lineup, assign_player_events, get_team_lineup
from footsim.sim.squad import adjust_squad_and_absences, reconstruct_lineup_with_absences

# League average defaults when event models are not supplied
DEFAULT_RATES = {
    "home_yellows": 1.70,
    "away_yellows": 1.93,
    "home_reds": 0.056,
    "away_reds": 0.060,
    "home_corners": 5.62,
    "away_corners": 4.69,
    "home_fouls": 10.68,
    "away_fouls": 10.88,
}


def simulate_match(
    home: str,
    away: str,
    n: int = 100_000,
    seed: int | None = None,
    referee: str | None = None,
    goals_model: DixonColes | None = None,
    events_model: EventRates | None = None,
    config: SimConfig | None = None,
    neutral: bool = False,
    save_path: str | Path | None = None,
    lam: float | None = None,
    mu: float | None = None,
    rho: float | None = None,
    is_derby: bool | None = None,
    player_layer: bool = False,
    home_lineup: Lineup | None = None,
    away_lineup: Lineup | None = None,
    home_absent: Sequence[str] | str | None = None,
    away_absent: Sequence[str] | str | None = None,
    home_rest_days: int | None = None,
    away_rest_days: int | None = None,
) -> pd.DataFrame:
    """Simulate N matches minute by minute.

    Args:
        home: home team name.
        away: away team name.
        n: number of simulations (default 100,000).
        seed: random seed for reproducibility.
        referee: match referee name.
        goals_model: fitted DixonColes model (optional if lam, mu are given).
        events_model: fitted EventRates model (optional; uses defaults if None).
        config: SimConfig instance (defaults to DEFAULT_SIM_CONFIG).
        neutral: if True, sets all game-state multipliers to 1.0 (analytical mode).
        save_path: optional filepath to save output as Parquet.
        lam, mu, rho: direct overrides for goal model parameters.
        is_derby: force derby rivalry multipliers.
        player_layer: attribute goals and cards to individual players.
        home_lineup, away_lineup: custom starting lineups.
        home_absent, away_absent: absent players to evaluate via VORP.
        home_rest_days, away_rest_days: days of rest since prior fixture.

    Returns:
        DataFrame with one row per simulated match.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if neutral:
        cfg = NEUTRAL_SIM_CONFIG
    else:
        cfg = config or DEFAULT_SIM_CONFIG

    actual_derby = check_is_derby(home, away) if is_derby is None else is_derby

    # 1. Goal parameters (from model or explicit arguments)
    if lam is None or mu is None:
        if goals_model is not None:
            lam, mu = goals_model.expected_goals(home, away)
            if rho is None:
                rho = getattr(goals_model, "rho_", 0.0)
        else:
            lam, mu, rho = 1.45, 1.15, -0.05
    if rho is None:
        rho = 0.0

    # Apply squad, key absence, and schedule congestion adjustments
    squad_notes: list[str] = []
    if home_absent or home_rest_days is not None:
        lam, mu, _, h_notes = adjust_squad_and_absences(
            team=home,
            base_lam=lam,
            base_mu=mu,
            absent_players=home_absent,
            rest_days=home_rest_days,
            lineup=home_lineup,
        )
        squad_notes.extend(h_notes)

    if away_absent or away_rest_days is not None:
        mu, lam, _, a_notes = adjust_squad_and_absences(
            team=away,
            base_lam=mu,
            base_mu=lam,
            absent_players=away_absent,
            rest_days=away_rest_days,
            lineup=away_lineup,
        )
        squad_notes.extend(a_notes)

    # 2. Event rates (cards, corners, fouls)
    if events_model is not None:
        ev_rates = events_model.expected(home, away, referee=referee)
    else:
        ev_rates = DEFAULT_RATES

    yh_rate = ev_rates.get("home_yellows", DEFAULT_RATES["home_yellows"])
    ya_rate = ev_rates.get("away_yellows", DEFAULT_RATES["away_yellows"])
    rh_rate = ev_rates.get("home_reds", DEFAULT_RATES["home_reds"])
    ra_rate = ev_rates.get("away_reds", DEFAULT_RATES["away_reds"])
    ch_rate = ev_rates.get("home_corners", DEFAULT_RATES["home_corners"])
    ca_rate = ev_rates.get("away_corners", DEFAULT_RATES["away_corners"])
    fh_rate = ev_rates.get("home_fouls", DEFAULT_RATES["home_fouls"])
    fa_rate = ev_rates.get("away_fouls", DEFAULT_RATES["away_fouls"])

    # 3. Minute profiles
    reg_mins = cfg.regulation_minutes
    p_goal = default_minute_profile("goals", reg_mins)
    p_yellow = default_minute_profile("yellows", reg_mins)
    p_red = default_minute_profile("reds", reg_mins)
    p_corner = default_minute_profile("corners", reg_mins)
    p_foul = default_minute_profile("fouls", reg_mins)

    # 4. Stoppage times
    if cfg.neutral:
        # In neutral analytical mode: exactly 45 mins in 1H, 45 mins in 2H = 90 mins
        ht_stop = np.zeros(n, dtype=np.int8)
        ft_stop = np.zeros(n, dtype=np.int8)
        total_steps = 90
    else:
        # Draw stoppage times: 1H typically 1-5 mins, 2H typically 2-8 mins
        ht_stop = rng.poisson(cfg.stoppage_half1_mean, size=n).astype(np.int8)
        ft_stop = rng.poisson(cfg.stoppage_half2_mean, size=n).astype(np.int8)
        np.clip(ht_stop, 0, 7, out=ht_stop)
        np.clip(ft_stop, 1, 10, out=ft_stop)
        total_steps = 90 + int(np.max(ht_stop)) + int(np.max(ft_stop))

    # 5. Initialize simulation state arrays (N,)
    home_goals = np.zeros(n, dtype=np.int16)
    away_goals = np.zeros(n, dtype=np.int16)
    ht_home_goals = np.zeros(n, dtype=np.int16)
    ht_away_goals = np.zeros(n, dtype=np.int16)

    home_corners = np.zeros(n, dtype=np.int16)
    away_corners = np.zeros(n, dtype=np.int16)
    home_yellows = np.zeros(n, dtype=np.int16)
    away_yellows = np.zeros(n, dtype=np.int16)
    home_reds = np.zeros(n, dtype=np.int16)
    away_reds = np.zeros(n, dtype=np.int16)
    home_fouls = np.zeros(n, dtype=np.int16)
    away_fouls = np.zeros(n, dtype=np.int16)

    # First scorer tracking: 0 = none, 1 = home, 2 = away
    first_scorer = np.zeros(n, dtype=np.int8)

    # Player yellow tracking for 2nd-yellow -> red conversion (11 slots per team)
    home_player_yellows = np.zeros((n, 11), dtype=np.int8)
    away_player_yellows = np.zeros((n, 11), dtype=np.int8)

    # Tracking lists for exact goal minutes and red card minutes
    home_goal_mins: list[list[int]] = [[] for _ in range(n)]
    away_goal_mins: list[list[int]] = [[] for _ in range(n)]
    home_red_mins: list[list[int]] = [[] for _ in range(n)]
    away_red_mins: list[list[int]] = [[] for _ in range(n)]

    max_ht_stop = int(np.max(ht_stop))
    ht_end_minute = 45 + max_ht_stop

    # --------------------------------------------------------------------------
    # Main Minute Loop (vectorised across N simulations)
    # --------------------------------------------------------------------------
    current_reg_min = 1
    in_second_half = False

    for step in range(1, total_steps + 1):
        if step <= 45:
            # First half regulation
            m_idx = step - 1
            active_mask = np.ones(n, dtype=bool)
        elif step <= ht_end_minute:
            # First half stoppage time
            extra = step - 45
            active_mask = extra <= ht_stop
            m_idx = 44  # use 45th minute profile intensity
        else:
            # Second half
            if not in_second_half:
                # Record half-time scores at transition
                ht_home_goals[:] = home_goals
                ht_away_goals[:] = away_goals
                in_second_half = True

            sh_step = step - ht_end_minute
            if sh_step <= 45:
                m_idx = 45 + sh_step - 1
                active_mask = np.ones(n, dtype=bool)
            else:
                extra = sh_step - 45
                active_mask = extra <= ft_stop
                m_idx = 89  # use 90th minute profile intensity

        if not np.any(active_mask):
            continue

        # Clamp m_idx to [0, 89]
        m_idx = min(max(m_idx, 0), reg_mins - 1)
        sim_minute = m_idx + 1

        # Current game state for active simulations
        score_diff = home_goals - away_goals
        # Net red difference: positive = home advantage (away has more reds)
        red_diff = away_reds - home_reds

        # Game state multipliers
        mult_h, mult_a = cfg.get_goal_multipliers(score_diff, red_diff, sim_minute)

        # Base minute intensities
        base_h = lam * p_goal[m_idx]
        base_a = mu * p_goal[m_idx]

        # 1. Goals (Poisson / Bernoulli hazard per minute)
        haz_gh = np.clip(base_h * mult_h, 0.0, 0.5)
        haz_ga = np.clip(base_a * mult_a, 0.0, 0.5)

        gh = (rng.random(n) < haz_gh) & active_mask
        ga = (rng.random(n) < haz_ga) & active_mask

        if np.any(gh):
            home_goals += gh
            gh_indices = np.flatnonzero(gh)
            for idx in gh_indices:
                home_goal_mins[idx].append(sim_minute)
            # Update first scorer where none yet
            new_gh = gh & (first_scorer == 0)
            first_scorer[new_gh] = 1

        if np.any(ga):
            away_goals += ga
            ga_indices = np.flatnonzero(ga)
            for idx in ga_indices:
                away_goal_mins[idx].append(sim_minute)
            # Update first scorer where none yet (if home also scored on same minute, home takes precedence)
            new_ga = ga & (first_scorer == 0)
            first_scorer[new_ga] = 2

        # 2. Corners
        cmult_h, cmult_a = cfg.get_corner_multipliers(score_diff, sim_minute)
        haz_ch = np.clip(ch_rate * p_corner[m_idx] * cmult_h, 0.0, 0.5)
        haz_ca = np.clip(ca_rate * p_corner[m_idx] * cmult_a, 0.0, 0.5)
        ch = (rng.random(n) < haz_ch) & active_mask
        ca = (rng.random(n) < haz_ca) & active_mask
        home_corners += ch
        away_corners += ca

        # 3. Fouls
        foul_mult = cfg.derby_foul_mult if (actual_derby and not cfg.neutral) else 1.0
        haz_fh = np.clip(fh_rate * p_foul[m_idx] * foul_mult, 0.0, 0.5)
        haz_fa = np.clip(fa_rate * p_foul[m_idx] * foul_mult, 0.0, 0.5)
        fh = (rng.random(n) < haz_fh) & active_mask
        fa = (rng.random(n) < haz_fa) & active_mask
        home_fouls += fh
        away_fouls += fa

        # 4. Yellow Cards & Second Yellows
        card_mult = cfg.get_card_multipliers(score_diff, sim_minute, is_derby=actual_derby)
        haz_yh = np.clip(yh_rate * p_yellow[m_idx] * card_mult, 0.0, 0.5)
        haz_ya = np.clip(ya_rate * p_yellow[m_idx] * card_mult, 0.0, 0.5)
        yh = (rng.random(n) < haz_yh) & active_mask
        ya = (rng.random(n) < haz_ya) & active_mask

        if np.any(yh):
            home_yellows += yh
            # Assign yellow to a player slot (0..10)
            yh_slots = rng.integers(0, 11, size=n)
            # Check for second yellow
            sim_idx = np.flatnonzero(yh)
            for s_idx in sim_idx:
                slot = yh_slots[s_idx]
                if home_player_yellows[s_idx, slot] == 1:
                    # Second yellow converts to red card
                    home_reds[s_idx] += 1
                    home_red_mins[s_idx].append(sim_minute)
                    home_player_yellows[s_idx, slot] = 2
                elif home_player_yellows[s_idx, slot] == 0:
                    home_player_yellows[s_idx, slot] = 1

        if np.any(ya):
            away_yellows += ya
            ya_slots = rng.integers(0, 11, size=n)
            sim_idx = np.flatnonzero(ya)
            for s_idx in sim_idx:
                slot = ya_slots[s_idx]
                if away_player_yellows[s_idx, slot] == 1:
                    # Second yellow converts to red card
                    away_reds[s_idx] += 1
                    away_red_mins[s_idx].append(sim_minute)
                    away_player_yellows[s_idx, slot] = 2
                elif away_player_yellows[s_idx, slot] == 0:
                    away_player_yellows[s_idx, slot] = 1

        # 5. Direct Red Cards
        haz_rh = np.clip(rh_rate * p_red[m_idx] * card_mult, 0.0, 0.5)
        haz_ra = np.clip(ra_rate * p_red[m_idx] * card_mult, 0.0, 0.5)
        rh = (rng.random(n) < haz_rh) & active_mask
        ra = (rng.random(n) < haz_ra) & active_mask

        if np.any(rh):
            home_reds += rh
            for idx in np.flatnonzero(rh):
                home_red_mins[idx].append(sim_minute)

        if np.any(ra):
            away_reds += ra
            for idx in np.flatnonzero(ra):
                away_red_mins[idx].append(sim_minute)

    # If match finished without half-time recorded (e.g. exactly 90 min neutral)
    if not in_second_half:
        ht_home_goals[:] = home_goals
        ht_away_goals[:] = away_goals

    # --------------------------------------------------------------------------
    # Dixon-Coles Tau Coupling for Low Scores (0-0, 1-0, 0-1, 1-1)
    # --------------------------------------------------------------------------
    if cfg.apply_tau_coupling and rho != 0.0:
        if rho < 0:
            # 1-0 -> 0-0 with probability |rho| * mu
            p10_to_00 = float(-rho * mu)
            mask10 = (home_goals == 1) & (away_goals == 0)
            move10 = mask10 & (rng.random(n) < p10_to_00)
            if np.any(move10):
                home_goals[move10] = 0
                for idx in np.flatnonzero(move10):
                    home_goal_mins[idx].clear()
                    first_scorer[idx] = 0

            # 0-1 -> 1-1 with probability |rho| * lam
            p01_to_11 = float(-rho * lam)
            mask01 = (home_goals == 0) & (away_goals == 1)
            move01 = mask01 & (rng.random(n) < p01_to_11)
            if np.any(move01):
                home_goals[move01] = 1
                for idx in np.flatnonzero(move01):
                    # Add late home goal
                    home_goal_mins[idx].append(80)

        elif rho > 0:
            # 0-0 -> 1-0 with probability rho * mu
            p00_to_10 = float(rho * mu)
            mask00 = (home_goals == 0) & (away_goals == 0)
            move00 = mask00 & (rng.random(n) < p00_to_10)
            if np.any(move00):
                home_goals[move00] = 1
                for idx in np.flatnonzero(move00):
                    home_goal_mins[idx].append(45)
                    first_scorer[idx] = 1

            # 1-1 -> 0-1 with probability rho * lam
            p11_to_01 = float(rho * lam)
            mask11 = (home_goals == 1) & (away_goals == 1)
            move11 = mask11 & (rng.random(n) < p11_to_01)
            if np.any(move11):
                home_goals[move11] = 0
                for idx in np.flatnonzero(move11):
                    home_goal_mins[idx].clear()
                    first_scorer[idx] = 2

    # Result string: 'H', 'D', 'A'
    res_arr = np.where(home_goals > away_goals, "H", np.where(home_goals == away_goals, "D", "A"))

    # HT Result string: 'H', 'D', 'A'
    ht_res_arr = np.where(ht_home_goals > ht_away_goals, "H", np.where(ht_home_goals == ht_away_goals, "D", "A"))
    ht_ft_arr = np.char.add(np.char.add(ht_res_arr, "/"), res_arr)

    # First scorer string: 'home', 'away', 'none'
    scorer_labels = np.where(first_scorer == 1, "home", np.where(first_scorer == 2, "away", "none"))

    total_g = home_goals + away_goals
    tot_corners = home_corners + away_corners
    tot_yellows = home_yellows + away_yellows
    tot_reds = home_reds + away_reds
    tot_cards = tot_yellows + tot_reds

    df = pd.DataFrame({
        "home_goals": home_goals,
        "away_goals": away_goals,
        "ht_home_goals": ht_home_goals,
        "ht_away_goals": ht_away_goals,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "total_corners": tot_corners,
        "home_yellows": home_yellows,
        "away_yellows": away_yellows,
        "total_yellows": tot_yellows,
        "home_reds": home_reds,
        "away_reds": away_reds,
        "total_reds": tot_reds,
        "total_cards": tot_cards,
        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "first_scorer": scorer_labels,
        "result": res_arr,
        "ht_result": ht_res_arr,
        "ht_ft": ht_ft_arr,
        "total_goals": total_g,
        "over_25": total_g > 2,
        "btts": (home_goals > 0) & (away_goals > 0),
        "clean_sheet_home": away_goals == 0,
        "clean_sheet_away": home_goals == 0,
        "home_goal_minutes": home_goal_mins,
        "away_goal_minutes": away_goal_mins,
        "home_red_minutes": home_red_mins,
        "away_red_minutes": away_red_mins,
    })

    if player_layer:
        h_lineup = home_lineup or get_team_lineup(home)
        a_lineup = away_lineup or get_team_lineup(away)
        if home_absent:
            h_lineup, _ = reconstruct_lineup_with_absences(h_lineup, home_absent)
        if away_absent:
            a_lineup, _ = reconstruct_lineup_with_absences(a_lineup, away_absent)
        df = assign_player_events(df, h_lineup, a_lineup, seed=seed)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, engine="pyarrow")

    elapsed = time.perf_counter() - t0
    # Store metadata on DataFrame
    df.attrs["elapsed_seconds"] = elapsed
    df.attrs["home"] = home
    df.attrs["away"] = away
    df.attrs["lam"] = lam
    df.attrs["mu"] = mu
    df.attrs["rho"] = rho
    df.attrs["squad_notes"] = squad_notes

    return df


# Alias per spec
simulate = simulate_match
