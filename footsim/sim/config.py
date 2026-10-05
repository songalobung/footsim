"""Simulation configuration: minute profiles, stoppage time, and game-state multipliers.

All default values here are empirical placeholders until granular minute-level
tracking data is integrated. See docs/decisions.md for full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def default_minute_profile(stat: str, n_minutes: int = 90) -> np.ndarray:
    """Return a normalised 1D minute profile (sums to 1.0 over n_minutes).

    Args:
        stat: 'goals', 'yellows', 'reds', 'corners', or 'fouls'.
        n_minutes: regulation minutes (default 90).

    Returns:
        1D float array of length n_minutes summing to 1.0.
    """
    minutes = np.arange(1, n_minutes + 1, dtype=float)
    if stat == "goals":
        # Goals are slightly higher in the 2nd half (approx 44% in 1H, 56% in 2H).
        # Smooth linear trend w(m) = 1 + 0.3 * (m / 90)
        w = 1.0 + 0.3 * (minutes / n_minutes)
    elif stat == "yellows":
        # Yellow cards are much rarer early; referees show leniency in opening minutes.
        # Quadratic rise: ~32% in 1H, 68% in 2H.
        w = 0.3 + 1.4 * (minutes / n_minutes) ** 1.5
    elif stat == "reds":
        # Red cards: rare in early stages, steady increase toward late game.
        w = 0.4 + 1.2 * (minutes / n_minutes)
    elif stat == "corners":
        # Corners: relatively uniform with a slight uptick in the final 15 minutes.
        w = np.ones(n_minutes, dtype=float)
        w[75:] += 0.25
    elif stat == "fouls":
        # Fouls: uniform across the match.
        w = np.ones(n_minutes, dtype=float)
    else:
        w = np.ones(n_minutes, dtype=float)

    return w / np.sum(w)


@dataclass
class SimConfig:
    """Configuration for match simulation.

    Attributes:
        regulation_minutes: regulation match length (default 90).
        stoppage_half1_mean: mean extra time minutes in 1st half.
        stoppage_half2_mean: mean extra time minutes in 2nd half.
        red_card_scoring_mult: scoring multiplier per man disadvantage (e.g. 0.70 for 10 vs 11).
        red_card_conceding_mult: conceding multiplier per man disadvantage (e.g. 1.35 for 10 vs 11).
        late_trailing_minute: minute at which trailing urgency begins (default 70).
        late_trailing_scoring_mult: scoring multiplier for trailing team late in match.
        late_trailing_conceding_mult: conceding multiplier for trailing team (counter-attack risk).
        late_leading_scoring_mult: scoring multiplier for leading team late in match.
        late_leading_conceding_mult: conceding multiplier for leading team late in match.
        apply_tau_coupling: if True, applies Dixon-Coles tau correlation for low scores.
        neutral: if True, all game-state multipliers are fixed to 1.0 (analytical benchmark mode).
    """

    regulation_minutes: int = 90
    stoppage_half1_mean: float = 2.0
    stoppage_half2_mean: float = 4.5

    # Red card multipliers (per net man advantage/disadvantage):
    red_card_scoring_mult: float = 0.70       # team with -1 red scores 30% less
    red_card_conceding_mult: float = 1.35     # team with -1 red concedes 35% more

    # Game-state score difference multipliers:
    late_trailing_minute: int = 70
    late_trailing_scoring_mult: float = 1.20    # trailing late pushes forward
    late_trailing_conceding_mult: float = 1.25  # vulnerable to counter-attack
    late_leading_scoring_mult: float = 0.85     # protects lead
    late_leading_conceding_mult: float = 0.90

    # Secondary event game-state multipliers (empirically calibrated):
    trailing_corner_mult: float = 1.30          # trailing team generates 30% more corners in 2H
    leading_corner_mult: float = 0.85           # leading team generates 15% fewer corners in 2H
    close_game_card_mult: float = 1.25          # card hazard +25% when goal diff <= 1 after min 70
    derby_card_mult: float = 1.25               # card hazard +25% in derby matches
    derby_foul_mult: float = 1.15               # foul hazard +15% in derby matches

    apply_tau_coupling: bool = True
    neutral: bool = False

    def get_goal_multipliers(
        self,
        diff: np.ndarray,
        red_diff: np.ndarray,
        minute: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute home and away goal intensity multipliers given current game state.

        Args:
            diff: (N,) array of score differences (home_goals - away_goals).
            red_diff: (N,) array of on-pitch player difference (home_players - away_players).
                Positive means home has man advantage; negative means away has advantage.
            minute: current simulation minute (1-indexed).

        Returns:
            (mult_home, mult_away): tuple of (N,) arrays of multipliers.
        """
        n = len(diff)
        if self.neutral:
            return np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)

        # 1. Red card effect:
        # red_diff > 0: home has more players (+1 -> 11 vs 10)
        # red_diff < 0: home has fewer players (-1 -> 10 vs 11)
        mult_h = np.ones(n, dtype=np.float32)
        mult_a = np.ones(n, dtype=np.float32)

        # Home man down
        mask_h_down = red_diff < 0
        if np.any(mask_h_down):
            down_cnt = -red_diff[mask_h_down]
            mult_h[mask_h_down] *= (self.red_card_scoring_mult ** down_cnt)
            mult_a[mask_h_down] *= (self.red_card_conceding_mult ** down_cnt)

        # Away man down (home man up)
        mask_a_down = red_diff > 0
        if np.any(mask_a_down):
            down_cnt = red_diff[mask_a_down]
            mult_a[mask_a_down] *= (self.red_card_scoring_mult ** down_cnt)
            mult_h[mask_a_down] *= (self.red_card_conceding_mult ** down_cnt)

        # 2. Score urgency in late minutes:
        if minute >= self.late_trailing_minute:
            # Home trailing (diff < 0): home pushes, away has counter opportunities
            mask_h_trail = diff < 0
            if np.any(mask_h_trail):
                mult_h[mask_h_trail] *= self.late_trailing_scoring_mult
                mult_a[mask_h_trail] *= self.late_trailing_conceding_mult

            # Away trailing (diff > 0): away pushes, home has counter opportunities
            mask_a_trail = diff > 0
            if np.any(mask_a_trail):
                mult_a[mask_a_trail] *= self.late_trailing_scoring_mult
                mult_h[mask_a_trail] *= self.late_trailing_conceding_mult

        return mult_h, mult_a

    def get_corner_multipliers(
        self,
        diff: np.ndarray,
        minute: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute corner intensity multipliers based on match state (trailing teams cross more)."""
        n = len(diff)
        if self.neutral or minute < 45:
            return np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)

        mult_h = np.ones(n, dtype=np.float32)
        mult_a = np.ones(n, dtype=np.float32)

        mask_h_trail = diff < 0
        if np.any(mask_h_trail):
            mult_h[mask_h_trail] *= self.trailing_corner_mult
            mult_a[mask_h_trail] *= self.leading_corner_mult

        mask_a_trail = diff > 0
        if np.any(mask_a_trail):
            mult_a[mask_a_trail] *= self.trailing_corner_mult
            mult_h[mask_a_trail] *= self.leading_corner_mult

        return mult_h, mult_a

    def get_card_multipliers(
        self,
        diff: np.ndarray,
        minute: int,
        is_derby: bool = False,
    ) -> np.ndarray:
        """Compute card intensity multiplier based on match score difference and derby context."""
        n = len(diff)
        if self.neutral:
            return np.ones(n, dtype=np.float32)

        mult = np.ones(n, dtype=np.float32)
        if is_derby:
            mult *= self.derby_card_mult

        if minute >= 70:
            close_mask = np.abs(diff) <= 1
            if np.any(close_mask):
                mult[close_mask] *= self.close_game_card_mult

        return mult


DEFAULT_SIM_CONFIG = SimConfig()
NEUTRAL_SIM_CONFIG = SimConfig(neutral=True, apply_tau_coupling=True)
