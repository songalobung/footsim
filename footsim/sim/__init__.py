"""Match simulation package (Milestone 4)."""

from footsim.sim.config import (
    DEFAULT_SIM_CONFIG,
    NEUTRAL_SIM_CONFIG,
    SimConfig,
    default_minute_profile,
)
from footsim.sim.inplay import LiveSimulationResult, LiveState, simulate_live
from footsim.sim.match import simulate, simulate_match
from footsim.sim.squad import (
    AbsenceImpact,
    SquadTier,
    adjust_squad_and_absences,
    calculate_absence_impact,
    calculate_h2h_edge,
    calculate_recent_form,
    get_squad_tier,
    reconstruct_lineup_with_absences,
)

__all__ = [
    "DEFAULT_SIM_CONFIG",
    "NEUTRAL_SIM_CONFIG",
    "SimConfig",
    "default_minute_profile",
    "simulate",
    "simulate_match",
    "LiveState",
    "LiveSimulationResult",
    "simulate_live",
    "SquadTier",
    "AbsenceImpact",
    "get_squad_tier",
    "calculate_absence_impact",
    "adjust_squad_and_absences",
    "reconstruct_lineup_with_absences",
    "calculate_recent_form",
    "calculate_h2h_edge",
]

