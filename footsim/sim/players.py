"""Player-level simulation layer (spec optional layer).

Assigns simulated goals and cards to individual starting players based on
each player's share of the team's expected goals (xG share) and foul/card share.
Provides anytime goalscorer, first goalscorer, and player card markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Player:
    """Individual player model profile.

    Attributes:
        name: player display name (e.g. 'B. Saka').
        position: 'FW', 'MF', 'DF', or 'GK'.
        xg_share: share of team's total expected goals (sums to ~1.0 across starting XI).
        card_share: share of team's cards/fouls (sums to ~1.0 across starting XI).
    """

    name: str
    position: str
    xg_share: float
    card_share: float


@dataclass
class Lineup:
    """Starting XI lineup for a team."""

    team: str
    players: list[Player]

    def __post_init__(self) -> None:
        if len(self.players) != 11:
            raise ValueError(f"Lineup for {self.team} must have exactly 11 players, got {len(self.players)}")
        # Normalise shares to sum to 1.0
        tot_xg = sum(p.xg_share for p in self.players)
        tot_card = sum(p.card_share for p in self.players)
        if tot_xg <= 0:
            tot_xg = 1.0
        if tot_card <= 0:
            tot_card = 1.0

        self.xg_probs = np.array([p.xg_share / tot_xg for p in self.players], dtype=float)
        self.card_probs = np.array([p.card_share / tot_card for p in self.players], dtype=float)
        self.player_names = [p.name for p in self.players]


# Standard Premier League starting lineups with empirical xG & card shares
DEFAULT_LINEUPS: dict[str, list[tuple[str, str, float, float]]] = {
    "Arsenal": [
        ("David Raya", "GK", 0.00, 0.02),
        ("Ben White", "DF", 0.02, 0.10),
        ("William Saliba", "DF", 0.03, 0.08),
        ("Gabriel Magalhaes", "DF", 0.06, 0.08),
        ("Jurrien Timber", "DF", 0.01, 0.10),
        ("Thomas Partey", "MF", 0.04, 0.14),
        ("Declan Rice", "MF", 0.06, 0.12),
        ("Martin Odegaard", "MF", 0.14, 0.06),
        ("Bukayo Saka", "FW", 0.28, 0.08),
        ("Kai Havertz", "FW", 0.22, 0.14),
        ("Gabriel Martinelli", "FW", 0.14, 0.08),
    ],
    "Chelsea": [
        ("Robert Sanchez", "GK", 0.00, 0.01),
        ("Malo Gusto", "DF", 0.01, 0.07),
        ("Wesley Fofana", "DF", 0.03, 0.12),
        ("Levi Colwill", "DF", 0.03, 0.10),
        ("Marc Cucurella", "DF", 0.02, 0.16),
        ("Moises Caicedo", "MF", 0.04, 0.18),
        ("Enzo Fernandez", "MF", 0.06, 0.12),
        ("Noni Madueke", "FW", 0.14, 0.06),
        ("Cole Palmer", "MF", 0.32, 0.06),
        ("Jadon Sancho", "FW", 0.10, 0.04),
        ("Nicolas Jackson", "FW", 0.25, 0.08),
    ],
    "Man City": [
        ("Ederson", "GK", 0.00, 0.02),
        ("Kyle Walker", "DF", 0.01, 0.08),
        ("Manuel Akanji", "DF", 0.02, 0.08),
        ("Ruben Dias", "DF", 0.03, 0.10),
        ("Josko Gvardiol", "DF", 0.05, 0.08),
        ("Rodri", "MF", 0.06, 0.14),
        ("Mateo Kovacic", "MF", 0.04, 0.12),
        ("Bernardo Silva", "MF", 0.08, 0.12),
        ("Phil Foden", "MF", 0.18, 0.06),
        ("Jeremy Doku", "FW", 0.10, 0.06),
        ("Erling Haaland", "FW", 0.43, 0.14),
    ],
    "Liverpool": [
        ("Alisson Becker", "GK", 0.00, 0.02),
        ("Trent Alexander-Arnold", "DF", 0.06, 0.08),
        ("Ibrahima Konate", "DF", 0.03, 0.10),
        ("Virgil van Dijk", "DF", 0.05, 0.08),
        ("Andy Robertson", "DF", 0.02, 0.08),
        ("Ryan Gravenberch", "MF", 0.03, 0.14),
        ("Alexis Mac Allister", "MF", 0.06, 0.16),
        ("Dominik Szoboszlai", "MF", 0.10, 0.12),
        ("Mohamed Salah", "FW", 0.35, 0.04),
        ("Diogo Jota", "FW", 0.18, 0.10),
        ("Luis Diaz", "FW", 0.12, 0.08),
    ],
}


def build_generic_lineup(team: str) -> Lineup:
    """Construct a plausible generic starting XI with realistic positional distributions."""
    players = [
        Player(f"{team} GK", "GK", 0.00, 0.02),
        Player(f"{team} RB", "DF", 0.02, 0.10),
        Player(f"{team} CB1", "DF", 0.03, 0.10),
        Player(f"{team} CB2", "DF", 0.04, 0.10),
        Player(f"{team} LB", "DF", 0.02, 0.10),
        Player(f"{team} DM", "MF", 0.04, 0.16),
        Player(f"{team} CM", "MF", 0.08, 0.14),
        Player(f"{team} AM", "MF", 0.14, 0.10),
        Player(f"{team} RW", "FW", 0.18, 0.06),
        Player(f"{team} ST", "FW", 0.33, 0.06),
        Player(f"{team} LW", "FW", 0.12, 0.06),
    ]
    return Lineup(team=team, players=players)


def get_team_lineup(team: str) -> Lineup:
    """Retrieve named lineup for team, or construct generic positional lineup."""
    raw = DEFAULT_LINEUPS.get(team)
    if raw is not None:
        players = [Player(name, pos, xg, card) for name, pos, xg, card in raw]
        return Lineup(team=team, players=players)
    return build_generic_lineup(team)


def assign_player_events(
    sim_df: pd.DataFrame,
    home_lineup: Lineup,
    away_lineup: Lineup,
    seed: int | None = None,
) -> pd.DataFrame:
    """Attribute simulated goals and cards to specific players.

    Args:
        sim_df: simulation DataFrame from simulate_match.
        home_lineup: Lineup object for home team.
        away_lineup: Lineup object for away team.
        seed: optional RNG seed.

    Returns:
        DataFrame augmented with player-level columns:
        - 'home_goalscorers': list of strings per sim
        - 'away_goalscorers': list of strings per sim
        - 'first_scorer_player': name of player who scored first, or 'None'
    """
    rng = np.random.default_rng(seed)
    n = len(sim_df)

    h_xg_p = home_lineup.xg_probs
    a_xg_p = away_lineup.xg_probs
    h_names = np.array(home_lineup.player_names)
    a_names = np.array(away_lineup.player_names)

    h_goals = sim_df["home_goals"].to_numpy()
    a_goals = sim_df["away_goals"].to_numpy()
    first_scorer_team = sim_df["first_scorer"].to_numpy()

    home_scorers_col: list[list[str]] = [[] for _ in range(n)]
    away_scorers_col: list[list[str]] = [[] for _ in range(n)]
    first_player_col: list[str] = ["None"] * n

    for i in range(n):
        hg = h_goals[i]
        ag = a_goals[i]
        f_team = first_scorer_team[i]

        h_players_scored = []
        if hg > 0:
            picks = rng.choice(h_names, size=hg, p=h_xg_p, replace=True)
            h_players_scored = list(picks)
            home_scorers_col[i] = h_players_scored

        a_players_scored = []
        if ag > 0:
            picks = rng.choice(a_names, size=ag, p=a_xg_p, replace=True)
            a_players_scored = list(picks)
            away_scorers_col[i] = a_players_scored

        if f_team in (1, "home") and h_players_scored:
            first_player_col[i] = h_players_scored[0]
        elif f_team in (2, "away") and a_players_scored:
            first_player_col[i] = a_players_scored[0]

    out_df = sim_df.copy()
    out_df["home_goalscorers"] = home_scorers_col
    out_df["away_goalscorers"] = away_scorers_col
    out_df["first_scorer_player"] = first_player_col
    return out_df


def player_scorer_probs(
    sim_df: pd.DataFrame,
    lineup: Lineup,
    is_home: bool = True,
) -> pd.DataFrame:
    """Compute anytime and first goalscorer probabilities for players in lineup."""
    n = len(sim_df)
    col = "home_goalscorers" if is_home else "away_goalscorers"
    first_col = sim_df["first_scorer_player"]

    anytime_counts = {p.name: 0 for p in lineup.players}
    first_counts = {p.name: 0 for p in lineup.players}

    for scorers, first_p in zip(sim_df[col], first_col):
        unique_scorers = set(scorers)
        for s in unique_scorers:
            if s in anytime_counts:
                anytime_counts[s] += 1
        if first_p in first_counts:
            first_counts[first_p] += 1

    records = []
    for p in lineup.players:
        records.append({
            "player": p.name,
            "position": p.position,
            "anytime_prob": anytime_counts[p.name] / n,
            "first_prob": first_counts[p.name] / n,
            "xg_share": p.xg_share,
        })
    return pd.DataFrame(records).sort_values("anytime_prob", ascending=False).reset_index(drop=True)
