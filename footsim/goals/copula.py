"""Copula-based bivariate scoreline distributions.

Dixon-Coles adjusts only the four lowest-scoring scorelines (0-0, 1-0, 0-1, 1-1)
via parameter rho. The Frank Copula bivariate discrete distribution provides
full-range scoreline dependence across all scores (0-10+ goals), modeling
correlated open shootouts (e.g. 2-2, 3-3) as well as defensive stalemates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def frank_copula_cdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate bivariate Frank copula CDF C(u, v; theta).

    Args:
        u, v: arrays of marginal probabilities in [0, 1].
        theta: copula dependence parameter (theta > 0 = positive dependence).
    """
    if abs(theta) < 1e-6:
        return u * v
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)
    num = (np.expm1(-theta * u)) * (np.expm1(-theta * v))
    den = np.expm1(-theta)
    return - (1.0 / theta) * np.log1p(num / den)


def copula_score_matrix(
    lam: float,
    mu: float,
    theta: float = 0.25,
    max_goals: int = 10,
) -> np.ndarray:
    """Generate joint scoreline probability matrix via Frank Copula.

    Args:
        lam: expected home goals.
        mu: expected away goals.
        theta: copula correlation parameter (default 0.25 based on EPL empirical correlation).
        max_goals: maximum goals per team (default 10).

    Returns:
        (max_goals+1, max_goals+1) probability matrix summing to exactly 1.0.
    """
    if abs(theta) < 1e-6:
        goals = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        return m / m.sum()

    # Pre-compute marginal CDFs F(x) for x = -1, 0, 1, ..., max_goals
    # F(-1) = 0.0
    k_range = np.arange(max_goals + 1)
    cdf_x = np.zeros(max_goals + 2)
    cdf_y = np.zeros(max_goals + 2)
    cdf_x[1:] = poisson.cdf(k_range, lam)
    cdf_y[1:] = poisson.cdf(k_range, mu)

    # Compute C(F_X(i), F_Y(j)) on grid (max_goals+2, max_goals+2)
    U, V = np.meshgrid(cdf_x, cdf_y, indexing="ij")
    C = frank_copula_cdf(U, V, theta)

    # 2D discrete difference: P(X=i, Y=j) = C[i+1, j+1] - C[i, j+1] - C[i+1, j] + C[i, j]
    m = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    m = np.maximum(m, 0.0)
    return m / m.sum()
