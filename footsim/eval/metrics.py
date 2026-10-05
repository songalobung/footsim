"""Evaluation metrics for match predictions and odds benchmarks.

Includes:
- Multiclass log loss, Brier score, Ranked Probability Score (RPS) for 1X2.
- Binary log loss for Over/Under 2.5 and BTTS.
- Calibration tables binning predicted probabilities against observed frequencies.
- Mean Poisson deviance and NB log loss for count predictions (event rates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def multiclass_log_loss(
    probs: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute multiclass log loss (negative log-likelihood of observed outcomes).

    Args:
        probs: (N, K) array of predicted probabilities.
        outcomes: (N,) array of integer class labels in 0..K-1.
        eps: clipping parameter to avoid log(0).

    Returns:
        Mean negative log loss.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(outcomes)
    if n == 0:
        return float("nan")
    p_obs = np.clip(probs[np.arange(n), outcomes], eps, 1.0)
    return float(-np.mean(np.log(p_obs)))


def brier_score(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute the multi-category Brier score (mean squared error across classes).

    BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    Args:
        probs: (N, K) array of predicted probabilities.
        outcomes: (N,) array of integer class labels in 0..K-1.

    Returns:
        Mean Brier score.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n, k = probs.shape
    if n == 0:
        return float("nan")
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute the Ranked Probability Score (RPS) for ordered outcomes (1X2).

    For K categories:
        RPS = (1/N) * sum_i [ 1/(K-1) * sum_{m=1}^{K-1} (P_im - Y_im)^2 ]
    where P_im and Y_im are cumulative probabilities and cumulative outcomes.

    For 1X2 (K=3):
        RPS = (1/2) * [ (p_H - y_H)^2 + ((p_H + p_D) - (y_H + y_D))^2 ]

    Args:
        probs: (N, K) array of probabilities ordered H, D, A.
        outcomes: (N,) integer array with 0=Home, 1=Draw, 2=Away.

    Returns:
        Mean RPS across matches.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n, k = probs.shape
    if n == 0:
        return float("nan")
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), outcomes] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    diff_sq = (cum_p - cum_o) ** 2
    return float(np.mean(np.sum(diff_sq, axis=1) / (k - 1)))


def binary_log_loss(
    probs: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute binary cross-entropy log loss.

    Args:
        probs: (N,) array of predicted probabilities for outcome == 1.
        outcomes: (N,) array of binary 0/1 outcomes.
        eps: clipping parameter.

    Returns:
        Mean binary log loss.
    """
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance: (2/N) * sum[y log(y/mu) - (y - mu)].

    Lower is better. Uses only the predicted mean, so it compares Poisson and
    negative binomial predictions on equal terms.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    if len(y) == 0:
        return float("nan")
    term = np.zeros_like(y)
    pos = y > 0
    term[pos] = y[pos] * np.log(y[pos] / mu[pos])
    return float(2.0 * np.mean(term - (y - mu)))


def count_log_loss(y: np.ndarray, mu: np.ndarray, alpha: float | np.ndarray = 0.0) -> float:
    """Mean negative log-likelihood of counts under NB2 (Poisson if alpha == 0)."""
    from footsim.events.rates import nb_loglik  # local import avoids a cycle

    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(nb_loglik(y, np.asarray(mu, dtype=float), alpha)))


def calibration_table(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration bins comparing predicted probability vs observed frequency.

    Args:
        probs: (N,) 1D array of predicted probabilities.
        outcomes: (N,) 1D array of binary indicators (0 or 1).
        n_bins: number of equal-width bins in [0, 1].

    Returns:
        DataFrame with columns:
        - bin_lower, bin_upper: interval bounds
        - count: number of predictions falling into bin
        - pred_mean: average predicted probability in bin
        - obs_frequency: observed proportion of 1s in bin
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    valid = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[valid], y[valid]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            p_mean = float(np.mean(p[mask]))
            obs_rate = float(np.mean(y[mask]))
        else:
            p_mean = float((lo + hi) / 2.0)
            obs_rate = np.nan
        rows.append({
            "bin_lower": round(lo, 2),
            "bin_upper": round(hi, 2),
            "count": cnt,
            "pred_mean": round(p_mean, 4) if not np.isnan(p_mean) else np.nan,
            "obs_frequency": round(obs_rate, 4) if not np.isnan(obs_rate) else np.nan,
        })
    return pd.DataFrame(rows)
