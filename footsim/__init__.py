"""footsim: football match simulation engine."""

from footsim.events.rates import EventRates
from footsim.goals.dixon_coles import DixonColes
from footsim.markets.query import (
    MarketResult,
    market,
    market_1x2,
    market_asian_handicap,
    market_btts,
    market_clean_sheet,
    market_correct_score,
    market_correct_scores,
    market_first_scorer_team,
    market_ht_ft,
    market_over_under_goals,
    market_red_card,
    market_total_cards,
    market_total_corners,
    wilson_score_interval,
)
from footsim.sim.match import simulate, simulate_match

__version__ = "0.1.0"

__all__ = [
    "DixonColes",
    "EventRates",
    "MarketResult",
    "market",
    "market_1x2",
    "market_asian_handicap",
    "market_btts",
    "market_clean_sheet",
    "market_correct_score",
    "market_correct_scores",
    "market_first_scorer_team",
    "market_ht_ft",
    "market_over_under_goals",
    "market_red_card",
    "market_total_cards",
    "market_total_corners",
    "simulate",
    "simulate_match",
    "wilson_score_interval",
]
