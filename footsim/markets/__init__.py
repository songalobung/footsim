"""Market query package for footsim simulations (Milestone 5)."""

from footsim.markets.query import (
    AsianHandicapResult,
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

__all__ = [
    "AsianHandicapResult",
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
    "wilson_score_interval",
]
