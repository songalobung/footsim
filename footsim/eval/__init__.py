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
]
