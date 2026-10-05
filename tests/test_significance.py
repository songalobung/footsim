"""Unit tests for statistical significance testing, Diebold-Mariano, and segmented calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footsim.eval.calibration import (
    SegmentedCalibrator,
    calculate_ece_mce,
    evaluate_segmented_calibration,
)
from footsim.eval.significance import (
    compare_model_against_market,
    diebold_mariano_test,
    paired_bootstrap_test,
    run_copula_ab_benchmark,
)


def test_paired_bootstrap_test() -> None:
    """Test paired bootstrap with distinct model losses."""
    rng = np.random.default_rng(42)
    baseline_losses = rng.uniform(0.18, 0.22, size=100)
    # Model is systematically better (losses lower by ~0.03)
    model_losses = baseline_losses - 0.03 + rng.normal(0, 0.005, size=100)

    res = paired_bootstrap_test(
        model_losses=model_losses,
        baseline_losses=baseline_losses,
        metric_name="rps",
        n_bootstraps=500,
        seed=42,
    )

    assert res.mean_diff < -0.02
    assert res.is_significant is True
    assert res.p_value < 0.01
    assert res.ci_lower < res.ci_upper
    assert "OUTPERFORMING" in res.format_summary()


def test_diebold_mariano_test() -> None:
    """Test Diebold-Mariano test on time-series losses."""
    rng = np.random.default_rng(42)
    b_loss = rng.uniform(0.5, 0.7, size=80)
    m_loss = b_loss - 0.05  # model has lower loss

    dm_res = diebold_mariano_test(
        model_losses=m_loss,
        baseline_losses=b_loss,
        metric_name="brier",
        max_lag=2,
    )

    assert dm_res.mean_diff < 0
    assert dm_res.is_significant is True
    assert dm_res.p_value < 0.05


def test_copula_ab_benchmark_synthetic() -> None:
    """Test out-of-sample A/B benchmark comparing Frank Copula against Independent Poisson."""
    n = 60
    rng = np.random.default_rng(42)
    lambdas = rng.uniform(1.2, 2.2, size=n)
    mus = rng.uniform(0.8, 1.8, size=n)
    actuals = rng.choice([0, 1, 2], size=n, p=[0.45, 0.25, 0.30])

    ab_res = run_copula_ab_benchmark(
        lambdas=lambdas,
        mus=mus,
        actual_1x2=actuals,
        theta=0.25,
        n_bootstraps=300,
        seed=42,
    )

    assert ab_res.n_matches == n
    assert ab_res.copula_rps > 0
    assert ab_res.poisson_rps > 0
    assert ab_res.theta == 0.25
    rep = ab_res.format_report()
    assert "COPULA VS INDEPENDENT POISSON" in rep
    assert "Ranked Prob Score (RPS)" in rep


def test_calculate_ece_mce() -> None:
    """Test expected and maximum calibration error calculations."""
    # Perfectly calibrated case
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece, mce = calculate_ece_mce(probs, labels, n_bins=2)
    assert pytest.approx(ece, abs=0.15) == 0.1
    assert mce <= 0.2


def test_evaluate_segmented_calibration() -> None:
    """Verify segment partitioning and segmented metrics calculation."""
    n = 100
    rng = np.random.default_rng(42)
    # Generate random probabilities summing to 1
    raw_p = rng.dirichlet(alpha=[2, 1, 1.5], size=n)
    actual_y = rng.choice([0, 1, 2], size=n)

    report = evaluate_segmented_calibration(raw_p, actual_y)

    assert report.overall_brier > 0
    assert report.overall_rps > 0
    assert len(report.segments) > 0
    txt = report.format_report()
    assert "SEGMENTED CALIBRATION" in txt
    assert "Home Favorites" in txt or "Draw" in txt


def test_segmented_calibrator_fit_predict() -> None:
    """Test fitting and predicting with SegmentedCalibrator."""
    n = 120
    rng = np.random.default_rng(42)
    p = rng.dirichlet(alpha=[3, 1, 2], size=n)
    y = rng.choice([0, 1, 2], size=n)

    cal = SegmentedCalibrator()
    cal.fit(p, y)
    assert cal.fitted is True

    p_cal = cal.predict(p)
    assert p_cal.shape == (n, 3)
    np.testing.assert_allclose(np.sum(p_cal, axis=1), 1.0, rtol=1e-5)
