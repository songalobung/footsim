"""Squad availability, key absences, squad depth, and congestion modeling.

Quantifies the impact of:
1. Absent players (star forwards, creative playmakers, defensive anchors, GKs)
   using Value Over Replacement Player (VORP) and team squad depth tiers.
2. Rest days and schedule congestion (e.g. 3-day turnaround fatigue).
3. Lineup reconstruction with bench replacement quality.
4. Recent 24-month Head-to-Head (H2H) and rolling 5-match form momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from footsim.sim.players import Lineup, Player, get_team_lineup


@dataclass(frozen=True)
class SquadTier:
    """Squad depth and replacement quality tier profile.

    Attributes:
        tier: integer tier (1=elite deep squad, 4=shallow squad).
        depth_index: relative bench quality score [0.0, 1.0].
        replacement_factor: expected fraction of starter output delivered by backup [0.0, 1.0].
    """

    tier: int
    depth_index: float
    replacement_factor: float


# Squad depth profiles based on bench market value, wage bill, and squad depth
SQUAD_DEPTH_PROFILES: dict[str, SquadTier] = {
    # Tier 1: World-class depth (bench could contend for top 4)
    "Man City": SquadTier(tier=1, depth_index=0.86, replacement_factor=0.76),
    "Chelsea": SquadTier(tier=1, depth_index=0.83, replacement_factor=0.73),
    # Tier 2: Strong European depth
    "Arsenal": SquadTier(tier=2, depth_index=0.78, replacement_factor=0.68),
    "Liverpool": SquadTier(tier=2, depth_index=0.78, replacement_factor=0.68),
    "Man United": SquadTier(tier=2, depth_index=0.74, replacement_factor=0.64),
    "Tottenham": SquadTier(tier=2, depth_index=0.73, replacement_factor=0.63),
    "Newcastle": SquadTier(tier=2, depth_index=0.72, replacement_factor=0.62),
    "Aston Villa": SquadTier(tier=2, depth_index=0.72, replacement_factor=0.62),
    # Tier 3: Established Premier League mid-table
    "Brighton": SquadTier(tier=3, depth_index=0.60, replacement_factor=0.50),
    "West Ham": SquadTier(tier=3, depth_index=0.58, replacement_factor=0.48),
    "Fulham": SquadTier(tier=3, depth_index=0.56, replacement_factor=0.46),
    "Crystal Palace": SquadTier(tier=3, depth_index=0.55, replacement_factor=0.45),
    "Bournemouth": SquadTier(tier=3, depth_index=0.54, replacement_factor=0.44),
    "Brentford": SquadTier(tier=3, depth_index=0.53, replacement_factor=0.43),
    "Everton": SquadTier(tier=3, depth_index=0.52, replacement_factor=0.42),
    "Nott'm Forest": SquadTier(tier=3, depth_index=0.52, replacement_factor=0.42),
    # Tier 4: Shallow squads / relegation battle
    "Wolves": SquadTier(tier=4, depth_index=0.45, replacement_factor=0.35),
    "Leicester": SquadTier(tier=4, depth_index=0.42, replacement_factor=0.32),
    "Ipswich": SquadTier(tier=4, depth_index=0.38, replacement_factor=0.28),
    "Southampton": SquadTier(tier=4, depth_index=0.38, replacement_factor=0.28),
    "Leeds": SquadTier(tier=4, depth_index=0.40, replacement_factor=0.30),
    "Sunderland": SquadTier(tier=4, depth_index=0.38, replacement_factor=0.28),
    "Coventry": SquadTier(tier=4, depth_index=0.36, replacement_factor=0.26),
    "Hull": SquadTier(tier=4, depth_index=0.36, replacement_factor=0.26),
}

DEFAULT_SQUAD_TIER = SquadTier(tier=3, depth_index=0.50, replacement_factor=0.40)


def get_squad_tier(team: str) -> SquadTier:
    """Retrieve squad depth tier for a team."""
    return SQUAD_DEPTH_PROFILES.get(team, DEFAULT_SQUAD_TIER)


@dataclass
class AbsenceImpact:
    """Quantified impact of an absent player.

    Attributes:
        player_name: name of absent player.
        team: team name.
        position: 'FW', 'MF', 'DM', 'DF', 'GK'.
        xg_share: player's baseline share of team goals.
        delta_attack: percentage multiplier on team attacking lambda.
        delta_defense: percentage multiplier on team concession mu.
        rationale: human-readable explanation of impact.
    """

    player_name: str
    team: str
    position: str
    xg_share: float
    delta_attack: float
    delta_defense: float
    rationale: str


# Well-known positional key anchors across top clubs
KNOWN_ROLES: dict[str, tuple[str, float]] = {
    # Key Defensive Anchors / CDMs (Rodri effect)
    "rodri": ("DM", 0.06),
    "thomas partey": ("DM", 0.04),
    "declan rice": ("MF", 0.06),
    "moises caicedo": ("DM", 0.04),
    "ryan gravenberch": ("DM", 0.03),
    "casemiro": ("DM", 0.04),
    "bruno guimaraes": ("DM", 0.07),
    # Key Center Backs
    "william saliba": ("CB", 0.03),
    "gabriel magalhaes": ("CB", 0.06),
    "virgil van dijk": ("CB", 0.05),
    "ruben dias": ("CB", 0.03),
    "cristian romero": ("CB", 0.04),
    "micky van de ven": ("CB", 0.03),
    # Key Goalkeepers
    "alisson becker": ("GK", 0.00),
    "alisson": ("GK", 0.00),
    "david raya": ("GK", 0.00),
    "ederson": ("GK", 0.00),
    "emiliano martinez": ("GK", 0.00),
    # Elite Attackers / Strikers / Playmakers
    "erling haaland": ("FW", 0.43),
    "haaland": ("FW", 0.43),
    "mohamed salah": ("FW", 0.35),
    "salah": ("FW", 0.35),
    "bukayo saka": ("FW", 0.28),
    "saka": ("FW", 0.28),
    "cole palmer": ("MF", 0.32),
    "palmer": ("MF", 0.32),
    "heung-min son": ("FW", 0.26),
    "son": ("FW", 0.26),
    "alexander isak": ("FW", 0.36),
    "isak": ("FW", 0.36),
    "ollie watkins": ("FW", 0.34),
    "watkins": ("FW", 0.34),
    "martin odegaard": ("MF", 0.16),
    "odegaard": ("MF", 0.16),
    "kevin de bruyne": ("MF", 0.15),
    "de bruyne": ("MF", 0.15),
    "phil foden": ("MF", 0.18),
    "foden": ("MF", 0.18),
    "bruno fernandes": ("MF", 0.22),
}


def _match_player_role(player_name: str, lineup: Lineup | None = None) -> tuple[str, float]:
    """Find position and xG share for player name."""
    clean_p = player_name.strip().lower()
    # 1. Exact match in KNOWN_ROLES
    if clean_p in KNOWN_ROLES:
        return KNOWN_ROLES[clean_p]

    # 2. Check in lineup
    if lineup is not None:
        for p in lineup.players:
            if clean_p == p.name.lower() or clean_p in p.name.lower():
                return p.position, p.xg_share

    # 3. Substring match in KNOWN_ROLES
    for k, v in KNOWN_ROLES.items():
        if k in clean_p or clean_p in k:
            return v

    # 4. Default generic starter assumption
    return "MF", 0.12


def calculate_absence_impact(
    player_name: str,
    team: str,
    lineup: Lineup | None = None,
) -> AbsenceImpact:
    """Calculate VORP attacking and defensive delta for an absent player."""
    pos, xg_share = _match_player_role(player_name, lineup)
    tier = get_squad_tier(team)
    unreplaced = 1.0 - tier.replacement_factor

    pos_upper = pos.upper()
    if pos_upper in ("FW", "ST", "LW", "RW"):
        # Primary attacking drop
        delta_att = -xg_share * unreplaced * 1.15
        delta_def = 0.0
        rationale = f"Primary finisher out; -{abs(delta_att)*100:.1f}% attack (bench replacement factor {tier.replacement_factor:.2f})"
    elif pos_upper in ("AM", "MF", "CM"):
        # Playmaker/creator affects both creation and slight pressing
        delta_att = -max(xg_share, 0.10) * unreplaced * 0.90
        delta_def = +0.05 * unreplaced
        rationale = f"Midfield creator out; -{abs(delta_att)*100:.1f}% attack, +{delta_def*100:.1f}% opponent chances"
    elif pos_upper in ("DM", "CDM"):
        # Defensive anchor (The Rodri Effect)
        delta_def = +0.22 * unreplaced
        delta_att = -0.06 * unreplaced
        rationale = f"Defensive anchor out; +{delta_def*100:.1f}% concession hazard, -{abs(delta_att)*100:.1f}% transition control"
    elif pos_upper in ("CB", "DF"):
        # Central defender
        delta_def = +0.18 * unreplaced
        delta_att = -0.02 * unreplaced
        rationale = f"Central defensive pillar out; +{delta_def*100:.1f}% concession hazard"
    elif pos_upper in ("FB", "LB", "RB"):
        # Fullback
        delta_def = +0.09 * unreplaced
        delta_att = -0.04 * unreplaced
        rationale = f"Fullback out; +{delta_def*100:.1f}% flank exposure, -{abs(delta_att)*100:.1f}% overlap width"
    elif pos_upper == "GK":
        # Goalkeeper
        delta_def = +0.24 * unreplaced
        delta_att = 0.0
        rationale = f"Starting goalkeeper out; backup keeper PSxG concession hazard +{delta_def*100:.1f}%"
    else:
        delta_att = -0.05 * unreplaced
        delta_def = +0.05 * unreplaced
        rationale = f"Starter out; generic squad degradation"

    return AbsenceImpact(
        player_name=player_name.strip(),
        team=team,
        position=pos,
        xg_share=xg_share,
        delta_attack=float(delta_att),
        delta_defense=float(delta_def),
        rationale=rationale,
    )


def adjust_squad_and_absences(
    team: str,
    base_lam: float,
    base_mu: float,
    absent_players: Sequence[str] | str | None = None,
    rest_days: int | None = None,
    lineup: Lineup | None = None,
) -> tuple[float, float, list[AbsenceImpact], list[str]]:
    """Adjust team's attacking lambda and defensive mu for squad factors.

    Args:
        team: team name.
        base_lam: unadjusted attacking expected goals.
        base_mu: unadjusted defensive concession expected goals.
        absent_players: list or comma-separated string of absent players.
        rest_days: days of rest since previous match.
        lineup: optional starting XI Lineup.

    Returns:
        tuple of (adj_lam, adj_mu, list of AbsenceImpact, list of notes).
    """
    lam = base_lam
    mu = base_mu
    impacts: list[AbsenceImpact] = []
    notes: list[str] = []

    # 1. Parse absent players
    if isinstance(absent_players, str):
        player_list = [p.strip() for p in absent_players.split(",") if p.strip()]
    elif absent_players is not None:
        player_list = [p.strip() for p in absent_players if p.strip()]
    else:
        player_list = []

    for p_name in player_list:
        imp = calculate_absence_impact(p_name, team, lineup)
        impacts.append(imp)
        lam *= (1.0 + imp.delta_attack)
        mu *= (1.0 + imp.delta_defense)
        notes.append(f"Absence [{imp.player_name} ({imp.position})]: {imp.rationale}")

    # 2. Schedule congestion & rest days
    if rest_days is not None:
        tier = get_squad_tier(team)
        if rest_days <= 3:
            # Short turnaround: 2 or 3 days
            days_penalty = (4 - rest_days) / 4.0  # 0.25 for 3 days, 0.50 for 2 days
            fatigue_vuln = 1.0 - (0.5 * tier.depth_index)
            att_drop = 0.08 * days_penalty * fatigue_vuln
            def_drop = 0.09 * days_penalty * fatigue_vuln
            lam *= (1.0 - att_drop)
            mu *= (1.0 + def_drop)
            notes.append(
                f"Congestion ({rest_days}d rest): -{att_drop*100:.1f}% attack, "
                f"+{def_drop*100:.1f}% concession (Squad depth index {tier.depth_index:.2f})"
            )
        elif rest_days >= 7:
            notes.append(f"Optimal recovery ({rest_days}d rest): full physical conditioning")

    # Safety clipping to prevent non-physical outputs
    lam = max(0.20, min(5.50, lam))
    mu = max(0.20, min(5.50, mu))

    return lam, mu, impacts, notes


def reconstruct_lineup_with_absences(
    lineup: Lineup,
    absent_players: Sequence[str] | str | None,
) -> tuple[Lineup, list[str]]:
    """Reconstruct starting XI replacing absent players with bench substitutes."""
    if isinstance(absent_players, str):
        player_list = [p.strip().lower() for p in absent_players.split(",") if p.strip()]
    elif absent_players is not None:
        player_list = [p.strip().lower() for p in absent_players if p.strip()]
    else:
        player_list = []

    if not player_list:
        return lineup, []

    tier = get_squad_tier(lineup.team)
    new_players: list[Player] = []
    changes: list[str] = []

    for p in lineup.players:
        p_clean = p.name.lower()
        matched = False
        for abs_p in player_list:
            if abs_p == p_clean or abs_p in p_clean:
                # Replace with bench substitute
                sub_name = f"{p.name} (Sub: Bench {p.position})"
                sub_xg = p.xg_share * tier.replacement_factor
                sub_card = p.card_share * 1.10  # Bench subs often pick up fouls when chasing
                new_players.append(Player(name=sub_name, position=p.position, xg_share=sub_xg, card_share=sub_card))
                changes.append(f"Replaced {p.name} -> {sub_name} (xG share scaled by {tier.replacement_factor:.2f})")
                matched = True
                break
        if not matched:
            new_players.append(p)

    return Lineup(team=lineup.team, players=new_players), changes


def calculate_recent_form(
    team: str,
    matches_df: pd.DataFrame,
    as_of: str | None = None,
    n_matches: int = 5,
) -> tuple[float, dict[str, float]]:
    """Calculate rolling form momentum factor from last N matches.

    Returns:
        tuple of (form_factor, summary_metrics).
        form_factor > 1.0 indicates outperforming expectation (hot streak),
        form_factor < 1.0 indicates underperforming (slump).
    """
    if "Date" not in matches_df.columns:
        return 1.0, {}

    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if as_of is not None:
        df = df[df["Date"] < pd.to_datetime(as_of)]

    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date").tail(n_matches)
    if len(team_matches) < 3:
        return 1.0, {"matches_found": len(team_matches)}

    pts = 0
    gf = 0
    ga = 0
    for _, row in team_matches.iterrows():
        is_home = row["HomeTeam"] == team
        hg = int(row.get("FTHG", 0))
        ag = int(row.get("FTAG", 0))
        team_goals = hg if is_home else ag
        opp_goals = ag if is_home else hg
        gf += team_goals
        ga += opp_goals
        if team_goals > opp_goals:
            pts += 3
        elif team_goals == opp_goals:
            pts += 1

    ppg = pts / len(team_matches)
    # Expected points per game baseline ~ 1.35
    # Empirical Bayes shrinkage: limit swing to +- 8%
    raw_delta = (ppg - 1.35) / 3.0
    shrunken_delta = float(np.clip(raw_delta * 0.15, -0.08, 0.08))
    form_factor = 1.0 + shrunken_delta

    return form_factor, {
        "matches_evaluated": len(team_matches),
        "points_won": pts,
        "ppg": round(ppg, 2),
        "goals_for": gf,
        "goals_against": ga,
        "form_factor": round(form_factor, 3),
    }


def calculate_h2h_edge(
    home: str,
    away: str,
    matches_df: pd.DataFrame,
    as_of: str | None = None,
    max_days: int = 730,  # 24 months
) -> tuple[float, dict[str, float]]:
    """Calculate tactical head-to-head edge strictly within the last 24 months.

    Returns:
        tuple of (h2h_edge, summary_dict).
        h2h_edge is a goal adjustment to home lambda (-away mu).
    """
    if "Date" not in matches_df.columns:
        return 0.0, {}

    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    cutoff = pd.to_datetime(as_of) if as_of is not None else df["Date"].max()
    min_date = cutoff - pd.Timedelta(days=max_days)

    h2h_games = df[
        (df["Date"] <= cutoff)
        & (df["Date"] >= min_date)
        & (
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
            | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        )
    ]

    if len(h2h_games) < 2:
        return 0.0, {"h2h_matches": len(h2h_games), "note": "Insufficient recent H2H samples (need >= 2 in last 24m)"}

    home_net_gd = 0
    for _, row in h2h_games.iterrows():
        is_h = row["HomeTeam"] == home
        hg = int(row.get("FTHG", 0))
        ag = int(row.get("FTAG", 0))
        gd = (hg - ag) if is_h else (ag - hg)
        home_net_gd += gd

    avg_gd = home_net_gd / len(h2h_games)
    # Shrunk tactical edge: max +- 0.12 goals
    edge = float(np.clip(avg_gd * 0.08, -0.12, 0.12))

    return edge, {
        "h2h_matches": len(h2h_games),
        "home_net_gd": home_net_gd,
        "avg_gd_per_game": round(avg_gd, 2),
        "tactical_edge_goals": round(edge, 3),
    }
