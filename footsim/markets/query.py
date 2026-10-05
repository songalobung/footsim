"""Market query layer for footsim simulations (Milestone 5).

A market is a boolean filter over the simulation DataFrame.
Probability = matching sims / total sims.
Every query returns:
- probability
- number of matching simulations
- 95% Wilson confidence interval
- reliable flag (count >= 200)
- sample-size rule compliance (flags unreliable queries and recommends n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

RELIABILITY_THRESHOLD = 200
DEFAULT_CONFIDENCE = 0.95


def wilson_score_interval(
    count: int,
    total: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        count: Number of successes / matching simulations.
        total: Total number of trials / simulations.
        confidence: Confidence level (default 0.95).

    Returns:
        (ci_lower, ci_upper) tuple bounded in [0.0, 1.0].
    """
    if total <= 0:
        return (0.0, 0.0)

    p = count / total
    if confidence == 0.95:
        z = 1.959963984540054
    else:
        z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)

    lower = 0.0 if count <= 0 else max(0.0, centre - margin)
    upper = 1.0 if count >= total else min(1.0, centre + margin)
    return (float(lower), float(upper))


@dataclass(frozen=True)
class MarketResult:
    """Result of querying a betting/prediction market from simulated matches.

    Attributes:
        query: Filter expression or market description.
        prob: Point estimate of probability (count / total).
        count: Number of simulations matching the condition.
        total: Total number of simulations.
        ci_lower: Lower bound of the 95% Wilson score confidence interval.
        ci_upper: Upper bound of the 95% Wilson score confidence interval.
        reliable: True if count >= 200, False otherwise.
        suggested_n: Recommended simulation count to reach >= 200 matches if unreliable.
    """

    query: str
    prob: float
    count: int
    total: int
    ci_lower: float
    ci_upper: float
    reliable: bool
    suggested_n: int | None = None

    @property
    def probability(self) -> float:
        """Alias for prob per spec."""
        return self.prob

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval tuple."""
        return (self.ci_lower, self.ci_upper)

    def __float__(self) -> float:
        return self.prob

    def __repr__(self) -> str:
        pct = self.prob * 100.0
        ci_pct_low = self.ci_lower * 100.0
        ci_pct_high = self.ci_upper * 100.0
        if self.reliable:
            return (
                f"MarketResult(p={pct:.2f}%, CI=[{ci_pct_low:.2f}%, {ci_pct_high:.2f}%], "
                f"count={self.count:,}/{self.total:,}, reliable=True)"
            )
        else:
            return (
                f"MarketResult(UNRELIABLE: p={pct:.2f}%, CI=[{ci_pct_low:.2f}%, {ci_pct_high:.2f}%], "
                f"count={self.count:,}/{self.total:,} < 200, suggested_n={self.suggested_n:,})"
            )

    def summary(self) -> str:
        """Human-readable description compliant with the 200-sample size rule."""
        if self.reliable:
            return (
                f"{self.query}: {self.prob * 100.0:.2f}% "
                f"(95% CI: [{self.ci_lower * 100.0:.2f}%, {self.ci_upper * 100.0:.2f}%], "
                f"{self.count:,}/{self.total:,} sims, Reliable)"
            )
        else:
            rec = f"suggested n >= {self.suggested_n:,}" if self.suggested_n else "increase n"
            return (
                f"{self.query}: [UNRELIABLE - only {self.count:,} matching sims < 200] "
                f"95% CI: [{self.ci_lower * 100.0:.3f}%, {self.ci_upper * 100.0:.3f}%] ({rec})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return dict representation."""
        return {
            "query": self.query,
            "probability": self.prob,
            "count": self.count,
            "total": self.total,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_95": (self.ci_lower, self.ci_upper),
            "reliable": self.reliable,
            "suggested_n": self.suggested_n,
        }


def _calculate_result(query_name: str, mask: np.ndarray, total: int) -> MarketResult:
    """Helper to compute MarketResult from a boolean NumPy mask."""
    count = int(np.count_nonzero(mask))
    prob = count / total if total > 0 else 0.0
    ci_lower, ci_upper = wilson_score_interval(count, total, confidence=DEFAULT_CONFIDENCE)
    reliable = count >= RELIABILITY_THRESHOLD

    suggested_n = None
    if not reliable:
        if prob > 0:
            suggested_n = int(math.ceil(RELIABILITY_THRESHOLD / prob))
        else:
            # If 0 hits observed, rule of three gives upper bound 3/N, so need >= 200 * N / 3
            suggested_n = max(total * 10, 1_000_000)

    return MarketResult(
        query=query_name,
        prob=prob,
        count=count,
        total=total,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        reliable=reliable,
        suggested_n=suggested_n,
    )


def market(sims: pd.DataFrame, query_str: str) -> MarketResult:
    """Evaluate an arbitrary boolean filter expression over the simulation DataFrame.

    Example:
        p = market(sims, "home_goals == 2 and away_goals == 1 and home_reds >= 1")

    Args:
        sims: Simulation DataFrame (e.g. from simulate_match).
        query_str: Boolean expression evaluable on the DataFrame.

    Returns:
        MarketResult with probability, count, 95% Wilson CI, and reliability flag.
    """
    total = len(sims)
    if total == 0:
        return MarketResult(
            query=query_str,
            prob=0.0,
            count=0,
            total=0,
            ci_lower=0.0,
            ci_upper=0.0,
            reliable=False,
            suggested_n=100_000,
        )

    # Evaluate using pandas Python engine for maximum expressiveness (and, or, not, ==, etc.)
    try:
        mask = sims.eval(query_str, engine="python")
    except Exception:
        # Fallback to standard eval if needed
        try:
            mask = sims.eval(query_str)
        except Exception as err:
            raise ValueError(f"Invalid market query expression '{query_str}': {err}") from err

    # Ensure mask is a boolean array
    mask_arr = np.asarray(mask, dtype=bool)
    return _calculate_result(query_str, mask_arr, total)


# ---------------------------------------------------------------------------
# Ready-made Market Helpers
# ---------------------------------------------------------------------------


def market_1x2(sims: pd.DataFrame) -> dict[str, MarketResult]:
    """1X2 Match Result market.

    Returns:
        dict with keys:
            'H' (or '1'): Home win
            'D' (or 'X'): Draw
            'A' (or '2'): Away win
    """
    total = len(sims)
    hg = sims["home_goals"].to_numpy()
    ag = sims["away_goals"].to_numpy()

    h_mask = hg > ag
    d_mask = hg == ag
    a_mask = hg < ag

    res_h = _calculate_result("1X2: Home (H)", h_mask, total)
    res_d = _calculate_result("1X2: Draw (D)", d_mask, total)
    res_a = _calculate_result("1X2: Away (A)", a_mask, total)

    return {
        "H": res_h,
        "D": res_d,
        "A": res_a,
        "1": res_h,
        "X": res_d,
        "2": res_a,
    }


def market_correct_score(sims: pd.DataFrame, home_goals: int, away_goals: int) -> MarketResult:
    """Correct score market for a specific scoreline (e.g. 2-1)."""
    total = len(sims)
    hg = sims["home_goals"].to_numpy()
    ag = sims["away_goals"].to_numpy()
    mask = (hg == home_goals) & (ag == away_goals)
    return _calculate_result(f"Score {home_goals}-{away_goals}", mask, total)


def market_correct_scores(
    sims: pd.DataFrame,
    top_n: int = 10,
) -> list[tuple[str, MarketResult]]:
    """Return top N most likely correct scores ordered by probability descending."""
    total = len(sims)
    counts = sims.groupby(["home_goals", "away_goals"]).size().sort_values(ascending=False)
    results: list[tuple[str, MarketResult]] = []

    for (hg, ag), count in counts.head(top_n).items():
        score_str = f"{hg}-{ag}"
        res = market_correct_score(sims, int(hg), int(ag))
        results.append((score_str, res))

    return results


def market_over_under_goals(
    sims: pd.DataFrame,
    line: float = 2.5,
) -> dict[str, MarketResult]:
    """Over/Under total goals market."""
    total = len(sims)
    tg = sims["total_goals"].to_numpy()

    over_mask = tg > line
    under_mask = tg < line

    return {
        "over": _calculate_result(f"Over {line} Goals", over_mask, total),
        "under": _calculate_result(f"Under {line} Goals", under_mask, total),
    }


def market_btts(sims: pd.DataFrame) -> dict[str, MarketResult]:
    """Both Teams To Score (BTTS) market."""
    total = len(sims)
    hg = sims["home_goals"].to_numpy()
    ag = sims["away_goals"].to_numpy()

    yes_mask = (hg > 0) & (ag > 0)
    no_mask = ~yes_mask

    return {
        "yes": _calculate_result("BTTS: Yes", yes_mask, total),
        "no": _calculate_result("BTTS: No", no_mask, total),
    }


def market_clean_sheet(sims: pd.DataFrame, team: str = "home") -> MarketResult:
    """Clean sheet market for 'home' or 'away'."""
    team_clean = team.strip().lower()
    total = len(sims)
    if team_clean in ("home", "h", "1"):
        mask = sims["away_goals"].to_numpy() == 0
        return _calculate_result("Clean Sheet: Home", mask, total)
    elif team_clean in ("away", "a", "2"):
        mask = sims["home_goals"].to_numpy() == 0
        return _calculate_result("Clean Sheet: Away", mask, total)
    else:
        raise ValueError(f"Unknown team '{team}'. Expected 'home' or 'away'.")


@dataclass(frozen=True)
class AsianHandicapResult:
    """Asian handicap evaluation.

    Supports full, half, and quarter handicap lines for the Home side.
    """

    line: float
    home_win_prob: float
    away_win_prob: float
    push_prob: float
    home_cover_rate: float  # Expected fraction of stake returned positive (win + 0.5*half_win)
    away_cover_rate: float
    reliable: bool
    results: dict[str, MarketResult]


def market_asian_handicap(sims: pd.DataFrame, line: float) -> AsianHandicapResult:
    """Evaluate Asian Handicap line from the Home team's perspective.

    e.g. line = -0.5 is Home -0.5.
    line = 0.0 is Draw No Bet (Home DNB).
    line = -0.25 is Split 0.0 and -0.5.
    """
    total = len(sims)
    diff = (sims["home_goals"] - sims["away_goals"]).to_numpy()

    # Determine handicap type: full, half, quarter
    rem = abs(line * 4.0) % 4.0

    if rem == 0.0 or rem == 2.0:
        # Full (e.g. 0.0, -1.0) or Half (-0.5, +0.5)
        adjusted = diff + line
        win_mask = adjusted > 0
        push_mask = adjusted == 0
        loss_mask = adjusted < 0

        res_win = _calculate_result(f"AH {line:+.2f} Home Win", win_mask, total)
        res_push = _calculate_result(f"AH {line:+.2f} Push", push_mask, total)
        res_loss = _calculate_result(f"AH {line:+.2f} Away Win", loss_mask, total)

        cover = res_win.prob
        away_cover = res_loss.prob
        rel = res_win.reliable and res_loss.reliable

        return AsianHandicapResult(
            line=line,
            home_win_prob=res_win.prob,
            away_win_prob=res_loss.prob,
            push_prob=res_push.prob,
            home_cover_rate=cover,
            away_cover_rate=away_cover,
            reliable=rel,
            results={"home_win": res_win, "push": res_push, "away_win": res_loss},
        )
    else:
        # Quarter handicap: e.g. -0.25 (split 0 and -0.5) or -0.75 (split -0.5 and -1.0)
        l1 = line - 0.25
        l2 = line + 0.25
        adj1 = diff + l1
        adj2 = diff + l2

        # Home wins full if both win, half if one wins and one pushes
        full_win = (adj1 > 0) & (adj2 > 0)
        half_win = ((adj1 > 0) & (adj2 == 0)) | ((adj1 == 0) & (adj2 > 0))
        push = (adj1 == 0) & (adj2 == 0)
        half_loss = ((adj1 < 0) & (adj2 == 0)) | ((adj1 == 0) & (adj2 < 0))
        full_loss = (adj1 < 0) & (adj2 < 0)

        res_full_win = _calculate_result(f"AH {line:+.2f} Full Win", full_win, total)
        res_half_win = _calculate_result(f"AH {line:+.2f} Half Win", half_win, total)
        res_push = _calculate_result(f"AH {line:+.2f} Full Push", push, total)
        res_half_loss = _calculate_result(f"AH {line:+.2f} Half Loss", half_loss, total)
        res_full_loss = _calculate_result(f"AH {line:+.2f} Full Loss", full_loss, total)

        home_cover = res_full_win.prob + 0.5 * res_half_win.prob
        away_cover = res_full_loss.prob + 0.5 * res_half_loss.prob

        return AsianHandicapResult(
            line=line,
            home_win_prob=res_full_win.prob + res_half_win.prob,
            away_win_prob=res_full_loss.prob + res_half_loss.prob,
            push_prob=res_push.prob,
            home_cover_rate=home_cover,
            away_cover_rate=away_cover,
            reliable=res_full_win.reliable,
            results={
                "full_win": res_full_win,
                "half_win": res_half_win,
                "push": res_push,
                "half_loss": res_half_loss,
                "full_loss": res_full_loss,
            },
        )


def market_total_cards(
    sims: pd.DataFrame,
    line: float = 3.5,
) -> dict[str, MarketResult]:
    """Over/Under total cards (yellows + reds)."""
    total = len(sims)
    cards = sims["total_cards"].to_numpy()
    over_mask = cards > line
    under_mask = cards < line

    return {
        "over": _calculate_result(f"Over {line} Cards", over_mask, total),
        "under": _calculate_result(f"Under {line} Cards", under_mask, total),
    }


def market_total_corners(
    sims: pd.DataFrame,
    line: float = 9.5,
) -> dict[str, MarketResult]:
    """Over/Under total corners."""
    total = len(sims)
    corners = sims["total_corners"].to_numpy()
    over_mask = corners > line
    under_mask = corners < line

    return {
        "over": _calculate_result(f"Over {line} Corners", over_mask, total),
        "under": _calculate_result(f"Under {line} Corners", under_mask, total),
    }


def market_red_card(
    sims: pd.DataFrame,
    team: str | None = None,
) -> MarketResult:
    """Red card in match, or for a specific team ('home', 'away', or None)."""
    total = len(sims)
    if team is None or team.strip().lower() in ("match", "any", "all"):
        mask = sims["total_reds"].to_numpy() > 0
        return _calculate_result("Red Card in Match", mask, total)
    elif team.strip().lower() in ("home", "h", "1"):
        mask = sims["home_reds"].to_numpy() > 0
        return _calculate_result("Red Card: Home", mask, total)
    elif team.strip().lower() in ("away", "a", "2"):
        mask = sims["away_reds"].to_numpy() > 0
        return _calculate_result("Red Card: Away", mask, total)
    else:
        raise ValueError(f"Unknown team '{team}'. Expected 'home', 'away', or None.")


def market_first_scorer_team(sims: pd.DataFrame) -> dict[str, MarketResult]:
    """First goalscorer team ('home', 'away', or 'none')."""
    total = len(sims)
    fs = sims["first_scorer"].to_numpy()

    return {
        "home": _calculate_result("First Scorer: Home", fs == "home", total),
        "away": _calculate_result("First Scorer: Away", fs == "away", total),
        "none": _calculate_result("First Scorer: None (0-0)", fs == "none", total),
    }


def market_ht_ft(sims: pd.DataFrame) -> dict[str, MarketResult]:
    """Half-time / Full-time (HT/FT) double result market.

    Returns dict with keys: 'H/H', 'H/D', 'H/A', 'D/H', 'D/D', 'D/A', 'A/H', 'A/D', 'A/A'.
    """
    total = len(sims)
    ht_ft = sims["ht_ft"].to_numpy()
    outcomes = ["H/H", "H/D", "H/A", "D/H", "D/D", "D/A", "A/H", "A/D", "A/A"]

    results = {}
    for code in outcomes:
        mask = ht_ft == code
        results[code] = _calculate_result(f"HT/FT: {code}", mask, total)

    return results
