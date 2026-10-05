"""Opponent-adjusted Exponentially Weighted Moving Average (EWMA) ratings.

Phase C Implementation:
1. Strict Priority 0 no-lookahead invariant (ratings updated strictly using matches prior to kickoff).
2. Shot-level calibrated expected goals (xG proxy) from shots, shots on target, and corners.
3. Opponent adjustment: offensive output is scaled by opponent's pre-match defensive rating,
   and defensive concessions are scaled by opponent's pre-match offensive rating.
4. EWMA smoothing with configurable decay factor alpha (default 0.15, corresponding to ~10 match half-life).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from footsim.data.loader import load_matches, normalise_team_names


@dataclass
class TeamEWMARating:
    """Current rolling ratings for a football club."""

    team: str
    attack_rating: float  # > 1.0 = above league average offense
    defence_rating: float  # < 1.0 = above league average defense (fewer goals/xG conceded)
    xg_attack: float  # rolling opponent-adjusted xG generated
    xg_defence: float  # rolling opponent-adjusted xG conceded
    matches_played: int
    last_updated: pd.Timestamp


def compute_match_xg_proxy(
    shots: float,
    shots_on_target: float,
    corners: float = 0.0,
    goals: float = 0.0,
) -> float:
    """Calculate calibrated shot-quality expected goals (xG proxy).

    Based on empirical English Premier League shot conversion rates:
    - Shots on Target convert at ~31%
    - Off-target shots have ~4.5% baseline xG
    - Corners generate ~0.035 xG
    """
    if pd.isna(shots) or shots <= 0:
        return float(goals) if pd.notna(goals) else 1.2

    sot = max(0.0, min(float(shots_on_target) if pd.notna(shots_on_target) else 0.33 * shots, float(shots)))
    off_target = max(0.0, float(shots) - sot)
    corn = max(0.0, float(corners) if pd.notna(corners) else 0.0)

    xg = 0.31 * sot + 0.045 * off_target + 0.035 * corn
    # Clip to realistic match single-team xG bounds [0.05, 6.0]
    return float(np.clip(xg, 0.05, 6.0))


class OpponentAdjustedEWMA:
    """Engine computing opponent-adjusted EWMA ratings across historical matches.

    Maintains sequential team ratings strictly updated at match completion times,
    preventing any lookahead leakage into pre-match predictions.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        """Initialize EWMA engine.

        Args:
            alpha: smoothing factor in (0, 1]. alpha=0.15 yields ~10 match effective memory.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.team_ratings: dict[str, TeamEWMARating] = {}

    def get_rating(self, team: str) -> TeamEWMARating:
        """Get current rating for a team, defaulting to league-average baseline."""
        if team in self.team_ratings:
            return self.team_ratings[team]
        return TeamEWMARating(
            team=team,
            attack_rating=1.0,
            defence_rating=1.0,
            xg_attack=1.35,
            xg_defence=1.35,
            matches_played=0,
            last_updated=pd.Timestamp.min,
        )

    def compute_all_prematch_features(
        self,
        matches_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute pre-match EWMA ratings for every fixture in historical dataset.

        Enforces strict chronological sequencing: match k features depend ONLY
        on matches with Date < kickoff.

        Returns:
            DataFrame copy with pre-match feature columns added:
            - home_ewma_att, home_ewma_def, home_ewma_xg
            - away_ewma_att, away_ewma_def, away_ewma_xg
            - ewma_att_ratio, ewma_def_ratio
        """
        df = matches_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

        n = len(df)
        h_att = np.ones(n)
        h_def = np.ones(n)
        h_xg = np.full(n, 1.35)
        a_att = np.ones(n)
        a_def = np.ones(n)
        a_xg = np.full(n, 1.35)

        # Clear internal state
        self.team_ratings.clear()

        # Extract numpy arrays for ultra-fast nanosecond element access
        dates = df["Date"].to_numpy()
        h_teams = df["HomeTeam"].to_numpy()
        a_teams = df["AwayTeam"].to_numpy()
        fthgs = df["FTHG"].to_numpy(dtype=float)
        ftags = df["FTAG"].to_numpy(dtype=float)
        hss = df["HS"].to_numpy(dtype=float) if "HS" in df.columns else np.full(n, np.nan)
        hsts = df["HST"].to_numpy(dtype=float) if "HST" in df.columns else np.full(n, np.nan)
        hcs = df["HC"].to_numpy(dtype=float) if "HC" in df.columns else np.full(n, np.nan)
        ass = df["AS"].to_numpy(dtype=float) if "AS" in df.columns else np.full(n, np.nan)
        asts = df["AST"].to_numpy(dtype=float) if "AST" in df.columns else np.full(n, np.nan)
        acs = df["AC"].to_numpy(dtype=float) if "AC" in df.columns else np.full(n, np.nan)

        # Group indices by date
        date_groups: dict[Any, list[int]] = {}
        for i, d in enumerate(dates):
            date_groups.setdefault(d, []).append(i)

        base_goals = 1.35
        alpha = self.alpha
        one_minus_alpha = 1.0 - alpha

        for d, indices in date_groups.items():
            # 1. Record pre-match ratings for all fixtures kicking off on this date
            for idx in indices:
                h_team = h_teams[idx]
                a_team = a_teams[idx]

                r_h = self.get_rating(h_team)
                r_a = self.get_rating(a_team)

                h_att[idx] = r_h.attack_rating
                h_def[idx] = r_h.defence_rating
                h_xg[idx] = r_h.xg_attack
                a_att[idx] = r_a.attack_rating
                a_def[idx] = r_a.defence_rating
                a_xg[idx] = r_a.xg_attack

            # 2. Update ratings post-match with actual performance on this date
            for idx in indices:
                h_team = h_teams[idx]
                a_team = a_teams[idx]

                fthg = fthgs[idx]
                ftag = ftags[idx]

                xg_h = compute_match_xg_proxy(hss[idx], hsts[idx], hcs[idx], goals=fthg)
                xg_a = compute_match_xg_proxy(ass[idx], asts[idx], acs[idx], goals=ftag)

                # Prior pre-match ratings of opponents
                prev_h = self.get_rating(h_team)
                prev_a = self.get_rating(a_team)

                # Opponent adjustments
                h_att_perf = xg_h / max(prev_a.defence_rating * base_goals, 0.40)
                a_att_perf = xg_a / max(prev_h.defence_rating * base_goals, 0.40)

                h_def_perf = xg_a / max(prev_a.attack_rating * base_goals, 0.40)
                a_def_perf = xg_h / max(prev_h.attack_rating * base_goals, 0.40)

                # Smooth with EWMA
                new_h_att = float(np.clip(one_minus_alpha * prev_h.attack_rating + alpha * h_att_perf, 0.40, 2.50))
                new_h_def = float(np.clip(one_minus_alpha * prev_h.defence_rating + alpha * h_def_perf, 0.40, 2.50))
                new_h_xg = float(one_minus_alpha * prev_h.xg_attack + alpha * xg_h)

                new_a_att = float(np.clip(one_minus_alpha * prev_a.attack_rating + alpha * a_att_perf, 0.40, 2.50))
                new_a_def = float(np.clip(one_minus_alpha * prev_a.defence_rating + alpha * a_def_perf, 0.40, 2.50))
                new_a_xg = float(one_minus_alpha * prev_a.xg_attack + alpha * xg_a)

                self.team_ratings[h_team] = TeamEWMARating(
                    team=h_team,
                    attack_rating=round(new_h_att, 4),
                    defence_rating=round(new_h_def, 4),
                    xg_attack=round(new_h_xg, 3),
                    xg_defence=round(prev_h.xg_defence, 3),
                    matches_played=prev_h.matches_played + 1,
                    last_updated=d,
                )

                self.team_ratings[a_team] = TeamEWMARating(
                    team=a_team,
                    attack_rating=round(new_a_att, 4),
                    defence_rating=round(new_a_def, 4),
                    xg_attack=round(new_a_xg, 3),
                    xg_defence=round(prev_a.xg_defence, 3),
                    matches_played=prev_a.matches_played + 1,
                    last_updated=d,
                )

        df["home_ewma_att"] = h_att
        df["home_ewma_def"] = h_def
        df["home_ewma_xg"] = h_xg
        df["away_ewma_att"] = a_att
        df["away_ewma_def"] = a_def
        df["away_ewma_xg"] = a_xg
        df["ewma_att_ratio"] = h_att / np.maximum(a_def, 0.1)
        df["ewma_def_ratio"] = a_att / np.maximum(h_def, 0.1)

        return df



def get_current_ewma_ratings(
    as_of: pd.Timestamp | str | None = None,
    alpha: float = 0.15,
) -> dict[str, TeamEWMARating]:
    """Get opponent-adjusted EWMA ratings for all teams up to as_of timestamp.

    Guarantees no data leakage: only matches prior to as_of are consumed.
    """
    matches = load_matches()
    matches["Date"] = pd.to_datetime(matches["Date"])

    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        matches = matches[matches["Date"] < as_of_ts]

    engine = OpponentAdjustedEWMA(alpha=alpha)
    engine.compute_all_prematch_features(matches)
    return engine.team_ratings
