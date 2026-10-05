"""Live In-Play football match simulation engine.

Simulates remaining match events from an arbitrary in-play game state:
- Current minute (0-90+)
- Current scoreline (e.g. 1-0, 2-1)
- Active red cards per team
- Current corner and card counts

Dynamically adjusts remaining Poisson intensities for:
- Red card disadvantage (-32% scoring, +38% concession per active red)
- Late-game deficit desperation (trailing team corner urgency +35%, counter-attack vulnerability)
- Late-game close-match card escalation (+25% when within 1 goal after min 70)
- Computes in-play 1X2, Next Goal, Over/Under, and Asian Handicap markets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from footsim.goals.copula import simulate_bivariate_copula
from footsim.markets.query import MarketResult, wilson_score_interval


@dataclass
class LiveState:
    """Current live in-play match state.

    Attributes:
        home: home team name.
        away: away team name.
        minute: current match minute (0 to 90+).
        home_goals: current home goals scored.
        away_goals: current away goals scored.
        home_reds: active red cards for home team.
        away_reds: active red cards for away team.
        home_corners: current home corners won.
        away_corners: current away corners won.
        home_cards: current home yellow/red cards.
        away_cards: current away yellow/red cards.
    """

    home: str
    away: str
    minute: int
    home_goals: int = 0
    away_goals: int = 0
    home_reds: int = 0
    away_reds: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_cards: int = 0
    away_cards: int = 0

    def __post_init__(self) -> None:
        if self.minute < 0:
            raise ValueError(f"Minute cannot be negative, got {self.minute}")
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("Goals cannot be negative")
        if self.home_reds < 0 or self.away_reds < 0:
            raise ValueError("Red cards cannot be negative")


@dataclass
class LiveSimulationResult:
    """Results from in-play match simulation.

    Attributes:
        state: input live state.
        n_sims: number of simulations run.
        elapsed_seconds: runtime in seconds.
        rem_home_exp: expected remaining home goals.
        rem_away_exp: expected remaining away goals.
        ft_home_win_prob: probability home team wins at FT.
        ft_draw_prob: probability match ends in a draw at FT.
        ft_away_win_prob: probability away team wins at FT.
        next_goal_probs: dict mapping 'home', 'away', 'none' to probability.
        over_under_probs: dict of goal line -> {'over': prob, 'under': prob}.
        top_final_scores: list of tuple (score_str, prob, count).
        mean_ft_home_goals: projected mean full-time home goals.
        mean_ft_away_goals: projected mean full-time away goals.
        sim_df: simulation DataFrame containing final totals.
    """

    state: LiveState
    n_sims: int
    elapsed_seconds: float
    rem_home_exp: float
    rem_away_exp: float
    ft_home_win_prob: float
    ft_draw_prob: float
    ft_away_win_prob: float
    next_goal_probs: dict[str, float]
    over_under_probs: dict[float, dict[str, float]]
    top_final_scores: list[tuple[str, float, int]]
    mean_ft_home_goals: float
    mean_ft_away_goals: float
    sim_df: pd.DataFrame


def simulate_live(
    state: LiveState,
    base_lam: float,
    base_mu: float,
    n: int = 50_000,
    seed: int | None = None,
    base_home_corners: float = 5.60,
    base_away_corners: float = 4.70,
    base_home_cards: float = 1.80,
    base_away_cards: float = 2.00,
) -> LiveSimulationResult:
    """Simulate remaining match outcomes from the current live state.

    Args:
        state: LiveState instance.
        base_lam: pre-match home attacking expected goals (90 min).
        base_mu: pre-match away attacking expected goals (90 min).
        n: number of simulation runs.
        seed: random RNG seed.
        base_home_corners: 90-min home corner rate.
        base_away_corners: 90-min away corner rate.
        base_home_cards: 90-min home card rate.
        base_away_cards: 90-min away card rate.

    Returns:
        LiveSimulationResult with comprehensive in-play market pricing.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # 1. Remaining time fraction
    # Match duration is assumed to be 90 + ~4 min injury time
    total_expected_mins = 94.0
    rem_mins = max(1.0, total_expected_mins - float(state.minute))
    t_rem = rem_mins / 90.0

    # 2. Adjust remaining rates for active red cards
    # Each red card reduces scoring by ~32% and increases concession by ~38%
    home_red_mult_att = max(0.20, 1.0 - 0.32 * state.home_reds)
    home_red_mult_def = 1.0 + 0.38 * state.home_reds

    away_red_mult_att = max(0.20, 1.0 - 0.32 * state.away_reds)
    away_red_mult_def = 1.0 + 0.38 * state.away_reds

    rem_lam = base_lam * t_rem * home_red_mult_att * away_red_mult_def
    rem_mu = base_mu * t_rem * away_red_mult_att * home_red_mult_def

    rem_ch = base_home_corners * t_rem * home_red_mult_att
    rem_ca = base_away_corners * t_rem * away_red_mult_att
    rem_card_h = base_home_cards * t_rem
    rem_card_a = base_away_cards * t_rem

    # 3. Game-state urgency adjustments for remaining time
    curr_diff = state.home_goals - state.away_goals

    # Trailing team pushes for equalizer late in match (minutes 65+)
    if state.minute >= 65:
        if curr_diff == -1:
            # Home trailing by 1
            rem_ch *= 1.35
            rem_lam *= 1.10
            rem_mu *= 1.15  # Exposed on counter-attack
        elif curr_diff == 1:
            # Away trailing by 1
            rem_ca *= 1.35
            rem_mu *= 1.10
            rem_lam *= 1.15

        # Close match late card escalation (min 70+, margin <= 1)
        if abs(curr_diff) <= 1:
            rem_card_h *= 1.25
            rem_card_a *= 1.25

    # 4. Simulate remaining goals using Frank Copula bivariate distribution
    rem_hg, rem_ag = simulate_bivariate_copula(
        lam=rem_lam,
        mu=rem_mu,
        theta=-0.40,
        n=n,
        seed=seed,
    )

    # Simulate remaining corners and cards using Poisson
    rem_h_corners = rng.poisson(rem_ch, size=n)
    rem_a_corners = rng.poisson(rem_ca, size=n)
    rem_h_cards = rng.poisson(rem_card_h, size=n)
    rem_a_cards = rng.poisson(rem_card_a, size=n)

    # 5. Combine with current in-play state
    final_home_goals = state.home_goals + rem_hg
    final_away_goals = state.away_goals + rem_ag
    final_home_corners = state.home_corners + rem_h_corners
    final_away_corners = state.away_corners + rem_a_corners
    final_home_cards = state.home_cards + rem_h_cards
    final_away_cards = state.away_cards + rem_a_cards
    final_total_goals = final_home_goals + final_away_goals

    sim_df = pd.DataFrame({
        "ft_home_goals": final_home_goals,
        "ft_away_goals": final_away_goals,
        "ft_total_goals": final_total_goals,
        "rem_home_goals": rem_hg,
        "rem_away_goals": rem_ag,
        "ft_home_corners": final_home_corners,
        "ft_away_corners": final_away_corners,
        "ft_home_cards": final_home_cards,
        "ft_away_cards": final_away_cards,
    })

    # 6. Calculate full-time 1X2
    hw = np.sum(final_home_goals > final_away_goals)
    dr = np.sum(final_home_goals == final_away_goals)
    aw = np.sum(final_home_goals < final_away_goals)

    p_hw = hw / n
    p_dr = dr / n
    p_aw = aw / n

    # 7. Next goal probabilities
    # Analytical Poisson arrival:
    tot_rem_rate = rem_lam + rem_mu
    if tot_rem_rate > 1e-6:
        p_no_more = float(np.exp(-tot_rem_rate))
        p_any_goal = 1.0 - p_no_more
        p_home_next = float(p_any_goal * (rem_lam / tot_rem_rate))
        p_away_next = float(p_any_goal * (rem_mu / tot_rem_rate))
    else:
        p_no_more = 1.0
        p_home_next = 0.0
        p_away_next = 0.0

    next_goal_dict = {
        "home": p_home_next,
        "away": p_away_next,
        "none": p_no_more,
    }

    # 8. Over / Under goal markets based on current score
    curr_total = state.home_goals + state.away_goals
    thresholds = [curr_total + 0.5, curr_total + 1.5, curr_total + 2.5]
    ou_dict: dict[float, dict[str, float]] = {}
    for th in thresholds:
        over_cnt = np.sum(final_total_goals > th)
        ou_dict[th] = {
            "over": float(over_cnt / n),
            "under": float((n - over_cnt) / n),
        }

    # 9. Top final scorelines
    score_pairs = [f"{h}-{a}" for h, a in zip(final_home_goals, final_away_goals)]
    score_series = pd.Series(score_pairs).value_counts()
    top_scores = [
        (str(score), float(cnt / n), int(cnt))
        for score, cnt in score_series.head(8).items()
    ]

    elapsed = time.perf_counter() - t0

    return LiveSimulationResult(
        state=state,
        n_sims=n,
        elapsed_seconds=elapsed,
        rem_home_exp=float(rem_lam),
        rem_away_exp=float(rem_mu),
        ft_home_win_prob=float(p_hw),
        ft_draw_prob=float(p_dr),
        ft_away_win_prob=float(p_aw),
        next_goal_probs=next_goal_dict,
        over_under_probs=ou_dict,
        top_final_scores=top_scores,
        mean_ft_home_goals=float(np.mean(final_home_goals)),
        mean_ft_away_goals=float(np.mean(final_away_goals)),
        sim_df=sim_df,
    )
