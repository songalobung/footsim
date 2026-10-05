"""Walk-forward evaluation and benchmarking layer."""

from footsim.eval.backtest import (
    DEFAULT_EVAL_SEASONS,
    compute_all_calibrations,
    compute_summary_metrics,
    matchday_blocks,
    plot_calibration,
    run_backtest_report,
    walk_forward_backtest,
)
from footsim.eval.betting import (
    BettingSimulationResult,
    EdgeResult,
    assert_feature_timestamp,
    calculate_edge_and_ev,
    calculate_stake,
    run_economic_backtest,
)
from footsim.eval.calibration import (
    CalibratedResult,
    PlattCalibrator,
    SegmentMetrics,
    SegmentedCalibrationReport,
    SegmentedCalibrator,
    calculate_ece_mce,
    evaluate_calibration,
    evaluate_segmented_calibration,
)
from footsim.eval.inspection import (
    MatchInspectionReport,
    RefereeDisciplineSummary,
    TeamFormSummary,
    inspect_fixture,
)
from footsim.eval.metrics import (
    binary_log_loss,
    brier_score,
    calibration_table,
    multiclass_log_loss,
    ranked_probability_score,
)
from footsim.eval.odds import (
    proportional_probs,
    remove_margin,
    shin_probs,
)
from footsim.eval.significance import (
    CopulaABResult,
    SignificanceResult,
    compare_model_against_market,
    diebold_mariano_test,
    paired_bootstrap_test,
    run_copula_ab_benchmark,
)

__all__ = [
    "DEFAULT_EVAL_SEASONS",
    "compute_all_calibrations",
    "compute_summary_metrics",
    "matchday_blocks",
    "plot_calibration",
    "run_backtest_report",
    "walk_forward_backtest",
    "binary_log_loss",
    "brier_score",
    "calibration_table",
    "multiclass_log_loss",
    "ranked_probability_score",
    "proportional_probs",
    "remove_margin",
    "shin_probs",
    "inspect_fixture",
    "MatchInspectionReport",
    "TeamFormSummary",
    "RefereeDisciplineSummary",
    "assert_feature_timestamp",
    "calculate_edge_and_ev",
    "calculate_stake",
    "run_economic_backtest",
    "BettingSimulationResult",
    "EdgeResult",
    "SignificanceResult",
    "CopulaABResult",
    "paired_bootstrap_test",
    "diebold_mariano_test",
    "compare_model_against_market",
    "run_copula_ab_benchmark",
    "SegmentMetrics",
    "SegmentedCalibrationReport",
    "SegmentedCalibrator",
    "calculate_ece_mce",
    "evaluate_calibration",
    "evaluate_segmented_calibration",
]


