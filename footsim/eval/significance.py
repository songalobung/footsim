"""Statistical significance testing and benchmark validation engine.

Phase B Implementation:
1. Paired bootstrap hypothesis tests for RPS, Brier score, and Log Loss.
2. Diebold-Mariano (1995) forecast comparison tests with Newey-West variance.
3. Copula vs Independent Poisson out-of-sample A/B benchmark.
4. Paired model vs market closing line statistical comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from footsim.eval.metrics import binary_log_loss, brier_score, ranked_probability_score
from footsim.goals.copula import copula_score_matrix


@dataclass
class SignificanceResult:
    """Hypothesis test result comparing two predictive models."""

    metric_name: str
    model_mean: float
    baseline_mean: float
    mean_diff: float  # model_mean - baseline_mean (negative indicates model is lower/better)
    p_value: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    test_type: str
    is_significant: bool  # True if p_value < alpha

    def format_summary(self) -> str:
        sig_str = "SIGNIFICANT (p < 0.05)" if self.is_significant else "NOT SIGNIFICANT"
        better_str = "OUTPERFORMING" if self.mean_diff < 0 else "UNDERPERFORMING"
        return (
            f"[{self.metric_name.upper()}] Model: {self.model_mean:.5f} | "
            f"Baseline: {self.baseline_mean:.5f} | Diff: {self.mean_diff:+.5f} "
            f"(95% CI: [{self.ci_lower:+.5f}, {self.ci_upper:+.5f}]) | "
            f"p={self.p_value:.4f} -> {better_str} ({sig_str})"
        )


@dataclass
class CopulaABResult:
    """Out-of-sample A/B test results comparing Copula vs Independent Poisson."""

    copula_rps: float
    poisson_rps: float
    rps_diff: float
    rps_p_value: float
    copula_brier: float
    poisson_brier: float
    brier_p_value: float
    copula_log_loss: float
    poisson_log_loss: float
    log_loss_p_value: float
    n_matches: int
    theta: float

    def format_report(self) -> str:
        lines = []
        hdr = f"COPULA VS INDEPENDENT POISSON OUT-OF-SAMPLE A/B BENCHMARK (N={self.n_matches:,})"
        div = "=" * len(hdr)
        lines.append(div)
        lines.append(hdr)
        lines.append(div)
        lines.append(f"  Dependence Parameter: Frank Copula theta = {self.theta:.3f} vs theta = 0.000")
        lines.append(
            f"  Ranked Prob Score (RPS): Copula {self.copula_rps:.5f} vs "
            f"Poisson {self.poisson_rps:.5f} | Diff: {self.rps_diff:+.5f} "
            f"(p={self.rps_p_value:.4f}) {'*' if self.rps_p_value < 0.05 else ''}"
        )
        lines.append(
            f"  Brier Score:             Copula {self.copula_brier:.5f} vs "
            f"Poisson {self.poisson_brier:.5f} | Diff: {self.copula_brier - self.poisson_brier:+.5f} "
            f"(p={self.brier_p_value:.4f}) {'*' if self.brier_p_value < 0.05 else ''}"
        )
        lines.append(
            f"  Multi-class Log Loss:    Copula {self.copula_log_loss:.5f} vs "
            f"Poisson {self.poisson_log_loss:.5f} | Diff: {self.copula_log_loss - self.poisson_log_loss:+.5f} "
            f"(p={self.log_loss_p_value:.4f}) {'*' if self.log_loss_p_value < 0.05 else ''}"
        )
        winner = "Frank Copula" if self.rps_diff < 0 else "Independent Poisson"
        lines.append(f"  Conclusion: {winner} achieves superior out-of-sample calibration.")
        lines.append(div)
        return "\n".join(lines)


def paired_bootstrap_test(
    model_losses: np.ndarray | Sequence[float],
    baseline_losses: np.ndarray | Sequence[float],
    metric_name: str = "metric",
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    alternative: str = "less",
) -> SignificanceResult:
    """Compute paired non-parametric bootstrap test comparing two forecast models.

    Args:
        model_losses: array of individual observation losses (e.g. RPS per match) for candidate model.
        baseline_losses: array of individual observation losses for reference baseline.
        metric_name: label for metric (e.g. 'rps', 'brier', 'log_loss').
        n_bootstraps: number of Monte Carlo resamples (default 2000).
        alpha: significance level (default 0.05 for 95% CI).
        seed: RNG seed for reproducible bootstrapping.
        alternative: 'less' (H1: model < baseline), 'greater', or 'two_sided'.

    Returns:
        SignificanceResult with empirical p-value and percentile confidence interval.
    """
    m = np.asarray(model_losses, dtype=float)
    b = np.asarray(baseline_losses, dtype=float)

    if len(m) != len(b):
        raise ValueError(f"Length mismatch: model has {len(m)} scores, baseline has {len(b)}.")
    n = len(m)
    if n == 0:
        raise ValueError("Cannot perform bootstrap test on empty array.")

    diffs = m - b
    mean_diff = float(np.mean(diffs))
    model_mean = float(np.mean(m))
    baseline_mean = float(np.mean(b))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_diff_means = np.mean(diffs[boot_indices], axis=1)

    # Percentile confidence intervals
    ci_lower = float(np.percentile(boot_diff_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diff_means, 100 * (1.0 - alpha / 2.0)))

    # Empirical p-value
    if alternative == "less":
        # H0: mean_diff >= 0 (model is not better) vs H1: mean_diff < 0
        p_val = float(np.mean(boot_diff_means >= 0.0))
    elif alternative == "greater":
        p_val = float(np.mean(boot_diff_means <= 0.0))
    else:  # two-sided
        p_one = float(np.mean(boot_diff_means >= 0.0))
        p_val = float(2.0 * min(p_one, 1.0 - p_one))

    # Bound p-value safely within [1/n_bootstraps, 1.0]
    p_val = min(max(p_val, 1.0 / n_bootstraps), 1.0)

    return SignificanceResult(
        metric_name=metric_name,
        model_mean=round(model_mean, 5),
        baseline_mean=round(baseline_mean, 5),
        mean_diff=round(mean_diff, 5),
        p_value=round(p_val, 4),
        ci_lower=round(ci_lower, 5),
        ci_upper=round(ci_upper, 5),
        n_samples=n,
        test_type="paired_bootstrap",
        is_significant=(p_val < alpha),
    )


def diebold_mariano_test(
    model_losses: np.ndarray | Sequence[float],
    baseline_losses: np.ndarray | Sequence[float],
    metric_name: str = "metric",
    max_lag: int = 1,
    alternative: str = "less",
) -> SignificanceResult:
    """Diebold-Mariano (1995) forecast accuracy test with Newey-West variance.

    Accounts for autocorrelation in sequential matchday predictions.
    """
    m = np.asarray(model_losses, dtype=float)
    b = np.asarray(baseline_losses, dtype=float)
    n = len(m)
    if n != len(b):
        raise ValueError("Model and baseline loss lengths must match.")
    if n <= max_lag + 1:
        raise ValueError("Insufficient observations for Diebold-Mariano test.")

    d = m - b
    mean_d = float(np.mean(d))
    model_mean = float(np.mean(m))
    baseline_mean = float(np.mean(b))

    # Autocovariance gamma_0, gamma_1, ..., gamma_h
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0

    for k in range(1, max_lag + 1):
        gamma_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        weight = 1.0 - (k / (max_lag + 1))  # Bartlett kernel
        var_d += 2.0 * weight * gamma_k

    var_d = max(var_d, 1e-12)
    dm_stat = mean_d / math.sqrt(var_d / n)

    if alternative == "less":
        p_val = float(stats.norm.cdf(dm_stat))
    elif alternative == "greater":
        p_val = float(1.0 - stats.norm.cdf(dm_stat))
    else:
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    se = math.sqrt(var_d / n)
    ci_lower = mean_d - 1.96 * se
    ci_upper = mean_d + 1.96 * se

    return SignificanceResult(
        metric_name=metric_name,
        model_mean=round(model_mean, 5),
        baseline_mean=round(baseline_mean, 5),
        mean_diff=round(mean_d, 5),
        p_value=round(p_val, 4),
        ci_lower=round(ci_lower, 5),
        ci_upper=round(ci_upper, 5),
        n_samples=n,
        test_type="diebold_mariano",
        is_significant=(p_val < 0.05),
    )


def compute_paired_match_losses(
    probs: np.ndarray,
    actual_1x2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate individual per-match loss arrays (RPS, Brier, Log Loss).

    Args:
        probs: (N, 3) matrix of Home, Draw, Away probabilities.
        actual_1x2: (N,) array of integer outcomes (0=Home, 1=Draw, 2=Away).

    Returns:
        (rps_array, brier_array, log_loss_array).
    """
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)
    n = len(actual_1x2)

    y_one_hot = np.zeros((n, 3))
    y_one_hot[np.arange(n), actual_1x2] = 1.0

    # 1. Log loss per match
    ll_arr = -np.log(p[np.arange(n), actual_1x2])

    # 2. Brier score per match (sum of squared errors across 3 outcomes)
    brier_arr = np.sum((p - y_one_hot) ** 2, axis=1)

    # 3. RPS per match (Ranked Probability Score for ordered H, D, A)
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(y_one_hot, axis=1)
    rps_arr = 0.5 * np.sum((cum_p[:, :2] - cum_y[:, :2]) ** 2, axis=1)

    return rps_arr, brier_arr, ll_arr


def compare_model_against_market(
    backtest_df: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, SignificanceResult]:
    """Compare FootSim model predictions against Pinnacle closing line on matched sample.

    Args:
        backtest_df: DataFrame from walk_forward_backtest containing dc_p* and pin_p*.
        n_bootstraps: bootstrap resamples.
        seed: RNG seed.

    Returns:
        dict mapping metric name ('rps', 'brier', 'log_loss') to SignificanceResult.
    """
    valid = backtest_df[
        backtest_df["pin_pH"].notna()
        & backtest_df["pin_pD"].notna()
        & backtest_df["pin_pA"].notna()
    ].copy()

    if len(valid) == 0:
        raise ValueError("No matches with valid Pinnacle closing probabilities found in backtest.")

    actual = valid["outcome_1x2"].to_numpy()
    m_probs = valid[["dc_pH", "dc_pD", "dc_pA"]].to_numpy()
    pin_probs = valid[["pin_pH", "pin_pD", "pin_pA"]].to_numpy()

    m_rps, m_brier, m_ll = compute_paired_match_losses(m_probs, actual)
    pin_rps, pin_brier, pin_ll = compute_paired_match_losses(pin_probs, actual)

    res_rps = paired_bootstrap_test(
        m_rps, pin_rps, metric_name="RPS (vs Pinnacle)", n_bootstraps=n_bootstraps, seed=seed
    )
    res_brier = paired_bootstrap_test(
        m_brier, pin_brier, metric_name="Brier (vs Pinnacle)", n_bootstraps=n_bootstraps, seed=seed
    )
    res_ll = paired_bootstrap_test(
        m_ll, pin_ll, metric_name="Log Loss (vs Pinnacle)", n_bootstraps=n_bootstraps, seed=seed
    )

    return {
        "rps": res_rps,
        "brier": res_brier,
        "log_loss": res_ll,
    }


def run_copula_ab_benchmark(
    lambdas: np.ndarray,
    mus: np.ndarray,
    actual_1x2: np.ndarray,
    theta: float = 0.25,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> CopulaABResult:
    """Run out-of-sample A/B test of Frank Copula vs Independent Poisson.

    Args:
        lambdas: (N,) array of home goal expected rates.
        mus: (N,) array of away goal expected rates.
        actual_1x2: (N,) array of true outcomes (0=H, 1=D, 2=A).
        theta: Frank copula dependence parameter (default 0.25).
        n_bootstraps: bootstrap resamples for paired significance.
        seed: RNG seed.

    Returns:
        CopulaABResult comparing out-of-sample performance.
    """
    n = len(actual_1x2)
    copula_probs = np.zeros((n, 3))
    poisson_probs = np.zeros((n, 3))

    for i in range(n):
        lam = float(lambdas[i])
        mu = float(mus[i])

        # 1. Frank Copula distribution
        mat_cop = copula_score_matrix(lam, mu, theta=theta)
        # Sum 1X2 outcomes: H = tril(-1), D = diag, A = triu(1)
        copula_probs[i, 0] = np.sum(np.tril(mat_cop, -1))
        copula_probs[i, 1] = np.sum(np.diag(mat_cop))
        copula_probs[i, 2] = np.sum(np.triu(mat_cop, 1))

        # 2. Independent Poisson distribution (theta=0.0)
        mat_poi = copula_score_matrix(lam, mu, theta=0.0)
        poisson_probs[i, 0] = np.sum(np.tril(mat_poi, -1))
        poisson_probs[i, 1] = np.sum(np.diag(mat_poi))
        poisson_probs[i, 2] = np.sum(np.triu(mat_poi, 1))

    c_rps, c_brier, c_ll = compute_paired_match_losses(copula_probs, actual_1x2)
    p_rps, p_brier, p_ll = compute_paired_match_losses(poisson_probs, actual_1x2)

    rps_sig = paired_bootstrap_test(c_rps, p_rps, metric_name="RPS", n_bootstraps=n_bootstraps, seed=seed)
    brier_sig = paired_bootstrap_test(c_brier, p_brier, metric_name="Brier", n_bootstraps=n_bootstraps, seed=seed)
    ll_sig = paired_bootstrap_test(c_ll, p_ll, metric_name="Log Loss", n_bootstraps=n_bootstraps, seed=seed)

    return CopulaABResult(
        copula_rps=rps_sig.model_mean,
        poisson_rps=rps_sig.baseline_mean,
        rps_diff=rps_sig.mean_diff,
        rps_p_value=rps_sig.p_value,
        copula_brier=brier_sig.model_mean,
        poisson_brier=brier_sig.baseline_mean,
        brier_p_value=brier_sig.p_value,
        copula_log_loss=ll_sig.model_mean,
        poisson_log_loss=ll_sig.baseline_mean,
        log_loss_p_value=ll_sig.p_value,
        n_matches=n,
        theta=theta,
    )
