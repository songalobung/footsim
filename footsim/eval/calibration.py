"""Probability post-calibration methods (Platt scaling and Isotonic regression).

Bookmakers build uneven margins and public bettors introduce favorite-longshot
bias into closing odds. Post-calibration calibrates raw model 1X2 probabilities
on historical walk-forward predictions to remove residual bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax


@dataclass
class CalibratedResult:
    """Evaluation metrics comparing raw and post-calibrated probabilities."""

    raw_log_loss: float
    cal_log_loss: float
    raw_brier: float
    cal_brier: float
    raw_rps: float
    cal_rps: float
    calibrated_probs: np.ndarray


class PlattCalibrator:
    """Multinomial logistic calibration (vector Platt scaling) for 1X2 probabilities.

    Transforms logits z_k = log(p_k) via a scaling vector a and bias vector b:
        z'_k = a * z_k + b_k
        p'_k = softmax(z')
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: np.ndarray = np.zeros(3)
        self.fitted: bool = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit Platt scaling parameters on predicted probabilities.

        Args:
            probs: (N, 3) matrix of predicted probabilities (Home, Draw, Away).
            y: (N,) array of true outcomes (0=Home, 1=Draw, 2=Away).
        """
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs)
        n = len(y)

        # Loss function: negative cross-entropy
        def nll(theta: np.ndarray) -> float:
            a_val = theta[0]
            b_vals = np.array([theta[1], theta[2], -theta[1] - theta[2]])  # sum to zero
            adj_logits = a_val * logits + b_vals
            # Stable log-softmax
            max_l = np.max(adj_logits, axis=1, keepdims=True)
            log_denom = max_l + np.log(np.sum(np.exp(adj_logits - max_l), axis=1, keepdims=True))
            log_p = adj_logits - log_denom
            # Gather y
            loss = -np.mean(log_p[np.arange(n), y])
            # Mild regularization
            reg = 0.01 * ((a_val - 1.0) ** 2 + np.sum(b_vals ** 2))
            return float(loss + reg)

        x0 = np.array([1.0, 0.0, 0.0])
        res = minimize(nll, x0, method="BFGS")
        self.a = float(res.x[0])
        self.b = np.array([res.x[1], res.x[2], -res.x[1] - res.x[2]])
        self.fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities."""
        if not self.fitted:
            return probs
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs)
        adj = self.a * logits + self.b
        return softmax(adj, axis=1)


def evaluate_calibration(
    raw_probs: np.ndarray,
    actual_1x2: Sequence[str] | np.ndarray,
) -> CalibratedResult:
    """Fit Platt calibrator and compute comparative metrics."""
    outcomes = np.array(actual_1x2)
    # Map 'H', 'D', 'A' to 0, 1, 2
    mapping = {"H": 0, "D": 1, "A": 2, 0: 0, 1: 1, 2: 2}
    y = np.array([mapping[o] for o in outcomes])
    n = len(y)

    calibrator = PlattCalibrator().fit(raw_probs, y)
    cal_probs = calibrator.predict(raw_probs)

    eps = 1e-12
    p_raw = np.clip(raw_probs, eps, 1.0 - eps)
    p_cal = np.clip(cal_probs, eps, 1.0 - eps)

    # 1. Log loss
    raw_ll = -float(np.mean(np.log(p_raw[np.arange(n), y])))
    cal_ll = -float(np.mean(np.log(p_cal[np.arange(n), y])))

    # 2. Brier score
    y_one_hot = np.zeros((n, 3))
    y_one_hot[np.arange(n), y] = 1.0
    raw_brier = float(np.mean(np.sum((p_raw - y_one_hot) ** 2, axis=1)))
    cal_brier = float(np.mean(np.sum((p_cal - y_one_hot) ** 2, axis=1)))

    # 3. RPS (Ranked Probability Score for ordered H < D < A)
    def compute_rps(p: np.ndarray) -> float:
        cum_p = np.cumsum(p, axis=1)
        cum_y = np.cumsum(y_one_hot, axis=1)
        # Sum over K-1 = 2 thresholds, normalized by 1/2
        return float(np.mean(0.5 * np.sum((cum_p[:, :2] - cum_y[:, :2]) ** 2, axis=1)))

    raw_rps = compute_rps(p_raw)
    cal_rps = compute_rps(p_cal)

    return CalibratedResult(
        raw_log_loss=raw_ll,
        cal_log_loss=cal_ll,
        raw_brier=raw_brier,
        cal_brier=cal_brier,
        raw_rps=raw_rps,
        cal_rps=cal_rps,
        calibrated_probs=cal_probs,
    )
