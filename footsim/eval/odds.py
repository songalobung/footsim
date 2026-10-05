"""Bookmaker margin removal methods (Shin and Proportional).

Converts raw decimal odds (e.g. from Pinnacle) to fair implied probabilities
by stripping out the bookmaker's overround.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def proportional_probs(odds: np.ndarray | list[float]) -> np.ndarray:
    """Convert decimal odds to probabilities using the proportional method.

    Normalized so that sum(p) == 1.0.

    Args:
        odds: 1D array of decimal odds for mutually exclusive outcomes.

    Returns:
        1D array of probabilities.
    """
    arr = np.asarray(odds, dtype=float)
    if np.any(np.isnan(arr)) or np.any(arr <= 1.0):
        return np.full_like(arr, np.nan)
    inv = 1.0 / arr
    total = np.sum(inv)
    if total <= 0:
        return np.full_like(arr, np.nan)
    return inv / total


def shin_probs(odds: np.ndarray | list[float], tol: float = 1e-8) -> np.ndarray:
    """Convert decimal odds to fair probabilities using Shin's method.

    Shin (1992, 1993) models the betting market with a fraction z of informed
    traders (insiders) possessing private information, naturally explaining the
    favorite-longshot bias.

    Formulation (Jullien & Salanie 1994, Strumbelj 2014):
    Let pi_i = 1 / O_i, and total margin beta = sum(pi_i).
    The fair probability p_i is given by:
        p_i = (sqrt(z^2 + 4 * (1 - z) * (pi_i^2 / beta)) - z) / (2 * (1 - z))
    where z is the unique root in [0, 1) satisfying sum(p_i) == 1.

    If odds sum to 1 or numerical root-finding does not converge, falls back
    to proportional probabilities.

    Args:
        odds: 1D array of decimal odds for mutually exclusive outcomes.
        tol: solver tolerance for root finding.

    Returns:
        1D array of probabilities summing to 1.0.
    """
    arr = np.asarray(odds, dtype=float)
    if np.any(np.isnan(arr)) or np.any(arr <= 1.0):
        return np.full_like(arr, np.nan)

    inv = 1.0 / arr
    beta = np.sum(inv)

    # If market has zero or negative margin, return proportional
    if abs(beta - 1.0) < 1e-6 or beta <= 1.0:
        return inv / beta

    def objective(z: float) -> float:
        rad = z * z + 4.0 * (1.0 - z) * (inv * inv) / beta
        p = (np.sqrt(np.maximum(rad, 0.0)) - z) / (2.0 * (1.0 - z))
        return float(np.sum(p) - 1.0)

    try:
        # At z=0, objective > 0; at z=1, limit is negative if beta > 1
        f0 = objective(0.0)
        f1 = objective(0.9999)
        if f0 * f1 < 0:
            z_star = brentq(objective, 0.0, 0.9999, xtol=tol)
            rad = z_star * z_star + 4.0 * (1.0 - z_star) * (inv * inv) / beta
            p = (np.sqrt(np.maximum(rad, 0.0)) - z_star) / (2.0 * (1.0 - z_star))
            p = p / np.sum(p)
            return p
    except Exception:
        pass

    return inv / beta


def remove_margin(
    odds: np.ndarray,
    method: str = "shin",
) -> np.ndarray:
    """Vectorised or 2D margin removal across multiple matches.

    Args:
        odds: (N, K) array of decimal odds (e.g. Home, Draw, Away).
        method: "shin" or "proportional".

    Returns:
        (N, K) array of fair probabilities summing to 1 across columns.
    """
    arr = np.asarray(odds, dtype=float)
    if arr.ndim == 1:
        if method == "shin":
            return shin_probs(arr)
        elif method == "proportional":
            return proportional_probs(arr)
        else:
            raise ValueError(f"Unknown margin method {method!r}")

    out = np.empty_like(arr)
    fn = shin_probs if method == "shin" else proportional_probs
    for i in range(len(arr)):
        row = arr[i]
        if np.any(np.isnan(row)):
            out[i] = np.nan
        else:
            out[i] = fn(row)
    return out
