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


@dataclass
class SegmentMetrics:
    """Calibration and predictive accuracy metrics for an individual segment."""

    segment_name: str
    n_matches: int
    brier_score: float
    log_loss: float
    rps: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    mean_pred_prob: float
    observed_frequency: float


@dataclass
class SegmentedCalibrationReport:
    """Comprehensive evaluation across disparate fixture segments."""

    overall_brier: float
    overall_rps: float
    overall_ece: float
    segments: dict[str, SegmentMetrics]

    def format_report(self) -> str:
        lines = []
        hdr = "SEGMENTED CALIBRATION & RELIABILITY REPORT"
        div = "=" * 76
        lines.append(div)
        lines.append(hdr)
        lines.append(div)
        lines.append(
            f"  Overall: Brier: {self.overall_brier:.5f} | RPS: {self.overall_rps:.5f} | "
            f"ECE: {self.overall_ece * 100:5.2f}%"
        )
        lines.append("-" * 76)
        lines.append(f"{'Segment':<28} {'N':>5} {'Brier':>8} {'RPS':>8} {'ECE (%)':>9} {'MCE (%)':>9} {'Mean P':>8} {'Actual':>8}")
        lines.append("-" * 76)
        for name, m in self.segments.items():
            lines.append(
                f"{name:<28} {m.n_matches:>5} {m.brier_score:>8.4f} {m.rps:>8.4f} "
                f"{m.ece * 100:>8.2f}% {m.mce * 100:>8.2f}% {m.mean_pred_prob:>8.3f} {m.observed_frequency:>8.3f}"
            )
        lines.append(div)
        return "\n".join(lines)


def calculate_ece_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 5,
) -> tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: 1D array of predicted probabilities in [0, 1].
        labels: 1D binary outcome indicators (0 or 1).
        n_bins: number of equal-width calibration bins.

    Returns:
        (ece, mce).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return 0.0, 0.0

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (probs >= low) & (probs <= high if i == n_bins - 1 else probs < high)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(labels[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (n_b / n) * diff
            mce = max(mce, diff)

    return float(ece), float(mce)


def evaluate_segmented_calibration(
    probs: np.ndarray,
    actual_1x2: Sequence[str | int] | np.ndarray,
    n_bins: int = 5,
) -> SegmentedCalibrationReport:
    """Evaluate reliability and calibration across market regimes (Favorites, Balanced, Longshots).

    Args:
        probs: (N, 3) matrix of Home, Draw, Away probabilities.
        actual_1x2: (N,) array of true outcomes ('H'/'D'/'A' or 0/1/2).
        n_bins: number of probability bins for ECE/MCE.
    """
    mapping = {"H": 0, "D": 1, "A": 2, 0: 0, 1: 1, 2: 2}
    y = np.array([mapping[o] for o in actual_1x2])
    n = len(y)
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)

    y_one_hot = np.zeros((n, 3))
    y_one_hot[np.arange(n), y] = 1.0

    # Overall metrics
    overall_brier = float(np.mean(np.sum((p - y_one_hot) ** 2, axis=1)))
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(y_one_hot, axis=1)
    overall_rps = float(np.mean(0.5 * np.sum((cum_p[:, :2] - cum_y[:, :2]) ** 2, axis=1)))

    # Compute overall multiclass ECE as average of 3 one-vs-rest ECEs
    eces = [calculate_ece_mce(p[:, c], (y == c).astype(int), n_bins)[0] for c in range(3)]
    overall_ece = float(np.mean(eces))

    # Segment definitions
    segments: dict[str, np.ndarray] = {
        "Home Favorites (pH >= 0.50)": p[:, 0] >= 0.50,
        "Away Favorites (pA >= 0.38)": p[:, 2] >= 0.38,
        "Draw / Balanced (Contested)": (p[:, 0] < 0.50) & (p[:, 2] < 0.38),
        "Extreme Longshots (p < 0.15)": np.min(p[:, [0, 2]], axis=1) < 0.15,
    }

    results: dict[str, SegmentMetrics] = {}

    for name, mask in segments.items():
        n_seg = int(np.sum(mask))
        if n_seg == 0:
            continue

        p_sub = p[mask]
        y_sub = y[mask]
        y_oh_sub = y_one_hot[mask]

        brier_sub = float(np.mean(np.sum((p_sub - y_oh_sub) ** 2, axis=1)))
        ll_sub = -float(np.mean(np.log(p_sub[np.arange(n_seg), y_sub])))

        c_p_sub = np.cumsum(p_sub, axis=1)
        c_y_sub = np.cumsum(y_oh_sub, axis=1)
        rps_sub = float(np.mean(0.5 * np.sum((c_p_sub[:, :2] - c_y_sub[:, :2]) ** 2, axis=1)))

        # Relevant target class for calibration error
        if "Home" in name:
            target_cls = 0
        elif "Away" in name:
            target_cls = 2
        elif "Draw" in name:
            target_cls = 1
        else:
            target_cls = 0

        seg_ece, seg_mce = calculate_ece_mce(
            p_sub[:, target_cls],
            (y_sub == target_cls).astype(int),
            n_bins=n_bins,
        )

        results[name] = SegmentMetrics(
            segment_name=name,
            n_matches=n_seg,
            brier_score=round(brier_sub, 5),
            log_loss=round(ll_sub, 5),
            rps=round(rps_sub, 5),
            ece=round(seg_ece, 5),
            mce=round(seg_mce, 5),
            mean_pred_prob=round(float(np.mean(p_sub[:, target_cls])), 4),
            observed_frequency=round(float(np.mean(y_sub == target_cls)), 4),
        )

    return SegmentedCalibrationReport(
        overall_brier=round(overall_brier, 5),
        overall_rps=round(overall_rps, 5),
        overall_ece=round(overall_ece, 5),
        segments=results,
    )


class SegmentedCalibrator:
    """Calibrate probabilities using segment-specific Platt models."""

    def __init__(self) -> None:
        self.home_fav_cal = PlattCalibrator()
        self.away_fav_cal = PlattCalibrator()
        self.general_cal = PlattCalibrator()
        self.fitted = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "SegmentedCalibrator":
        """Fit segment-specific calibrators."""
        n = len(y)
        h_mask = probs[:, 0] >= 0.50
        a_mask = probs[:, 2] >= 0.38
        other_mask = ~h_mask & ~a_mask

        # Fit each if sufficient samples (at least 20)
        if np.sum(h_mask) >= 20:
            self.home_fav_cal.fit(probs[h_mask], y[h_mask])
        else:
            self.home_fav_cal.fit(probs, y)

        if np.sum(a_mask) >= 20:
            self.away_fav_cal.fit(probs[a_mask], y[a_mask])
        else:
            self.away_fav_cal.fit(probs, y)

        if np.sum(other_mask) >= 20:
            self.general_cal.fit(probs[other_mask], y[other_mask])
        else:
            self.general_cal.fit(probs, y)

        self.fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Apply segment-tailored calibration."""
        if not self.fitted:
            return probs

        n = len(probs)
        out = np.zeros_like(probs)

        for i in range(n):
            row = probs[i : i + 1]
            if row[0, 0] >= 0.50:
                out[i] = self.home_fav_cal.predict(row)[0]
            elif row[0, 2] >= 0.38:
                out[i] = self.away_fav_cal.predict(row)[0]
            else:
                out[i] = self.general_cal.predict(row)[0]

        return out

