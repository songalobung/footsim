"""Premier League rivalries and derby fixture definitions.

Derby fixtures exhibit historically higher card counts and foul rates, as well
as heightened late-match intensity.
"""

from __future__ import annotations

#: Canonical pairwise Premier League derbies and fierce rivalries
KNOWN_DERBIES: set[frozenset[str]] = {
    frozenset({"Arsenal", "Tottenham"}),          # North London Derby
    frozenset({"Arsenal", "Chelsea"}),            # London Derby
    frozenset({"Chelsea", "Tottenham"}),          # London Derby
    frozenset({"Liverpool", "Everton"}),          # Merseyside Derby
    frozenset({"Man City", "Man United"}),        # Manchester Derby
    frozenset({"Liverpool", "Man United"}),       # North-West Derby
    frozenset({"Aston Villa", "Wolves"}),         # West Midlands Derby
    frozenset({"Crystal Palace", "Brighton"}),    # M23 Derby
    frozenset({"Chelsea", "Fulham"}),             # West London Derby
    frozenset({"Brentford", "Fulham"}),           # West London Derby
    frozenset({"Chelsea", "Brentford"}),          # West London Derby
    frozenset({"Newcastle", "Sunderland"}),       # Tyne-Wear Derby
    frozenset({"Leeds", "Man United"}),           # Roses Rivalry
}


def is_derby(team1: str, team2: str) -> bool:
    """Return True if the fixture between team1 and team2 is a recognised derby/rivalry.

    Case and leading/trailing whitespace are stripped, but canonical names from
    team_names.csv are expected.
    """
    pair = frozenset({team1.strip(), team2.strip()})
    return pair in KNOWN_DERBIES
