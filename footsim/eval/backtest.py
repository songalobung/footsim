"""Walk-forward evaluation and benchmarking engine.

Implements Milestone 2:
- Matchday block walk-forward validation (refitting at most once per matchday block).
- Computes 1X2 log loss, Brier score, and Ranked Probability Score (RPS).
- Computes Over/Under 2.5 and BTTS log loss.
- Benchmarks against Pinnacle closing odds stripped of margin via Shin (or Proportional).
- Generates calibration tables and calibration plots.
- Exports reports to CSV and PNG.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from footsim.data.loader import DEFAULT_SEASONS, load_matches
from footsim.eval.calibration import evaluate_calibration
from footsim.eval.metrics import (
    binary_log_loss,
    brier_score,
    calibration_table,
    multiclass_log_loss,
    ranked_probability_score,
)
from footsim.eval.odds import remove_margin
from footsim.goals.dixon_coles import DEFAULT_XI, DixonColes

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
DEFAULT_EVAL_SEASONS: tuple[str, ...] = ("2425", "2526")


def matchday_blocks(dates: pd.Series, max_gap_days: int = 2) -> pd.Series:
    """Group dates into matchday blocks separated by gaps > max_gap_days.

    In the EPL, a weekend round spans Friday to Monday (gaps <= 1 day),
    while midweek rounds span Tuesday to Thursday (gaps <= 1 day).
    Any gap of 3+ days marks a new matchday fixture block.
    """
    s = pd.to_datetime(dates)
    unique_dates = pd.Series(sorted(s.unique()))
    gaps = (unique_dates.diff().dt.days > max_gap_days).cumsum()
    date_to_blk = dict(zip(unique_dates, gaps))
    return s.map(date_to_blk)


def walk_forward_backtest(
    matches: pd.DataFrame,
    eval_seasons: Sequence[str] = DEFAULT_EVAL_SEASONS,
    xi: float = DEFAULT_XI,
    margin_method: str = "shin",
    **model_kwargs,
) -> pd.DataFrame:
    """Run walk-forward simulation across evaluation seasons.

    Refits the Dixon-Coles model once per matchday block using strictly earlier
    matches (as_of = block start date).

    Args:
        matches: full historical DataFrame of matches.
        eval_seasons: seasons to evaluate (e.g. ("2425", "2526")).
        xi: time decay parameter.
        margin_method: "shin" or "proportional" for Pinnacle odds margin removal.
        **model_kwargs: passed to DixonColes constructor.

    Returns:
        DataFrame with predictions, fair Pinnacle probabilities, and observed outcomes.
    """
    eval_df = matches[matches["season"].isin(eval_seasons)].copy()
    if eval_df.empty:
        raise ValueError(f"No matches found for seasons: {eval_seasons}")

    eval_df = eval_df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    eval_df["block"] = matchday_blocks(eval_df["Date"])

    rows = []
    n_blocks = eval_df["block"].nunique()
    print(f"Starting walk-forward backtest over {len(eval_df)} matches in {n_blocks} matchday blocks...")

    for blk_id, blk in eval_df.groupby("block", sort=True):
        as_of = blk["Date"].min()
        fixture_teams = set(blk["HomeTeam"]) | set(blk["AwayTeam"])

        # Refit once per matchday block strictly as of the first kickoff date
        model = DixonColes(xi=xi, **model_kwargs).fit(matches, as_of=as_of, new_teams=fixture_teams)

        for match in blk.itertuples(index=False):
            score_mat = model.score_matrix(match.HomeTeam, match.AwayTeam)

            # 1X2 Probabilities
            p_home = float(np.tril(score_mat, -1).sum())
            p_draw = float(np.trace(score_mat))
            p_away = float(np.triu(score_mat, 1).sum())

            # Over/Under 2.5: sum of cells where x + y >= 3
            grid_i, grid_j = np.indices(score_mat.shape)
            p_over25 = float(score_mat[grid_i + grid_j >= 3].sum())
            p_under25 = 1.0 - p_over25

            # BTTS: sum of cells where x >= 1 and y >= 1
            p_btts = float(score_mat[1:, 1:].sum())
            p_btts_no = 1.0 - p_btts

            # Outcomes
            hg, ag = int(match.FTHG), int(match.FTAG)
            outcome_1x2 = 0 if hg > ag else (1 if hg == ag else 2)
            outcome_over25 = 1 if (hg + ag) > 2.5 else 0
            outcome_btts = 1 if (hg > 0 and ag > 0) else 0

            # Pinnacle Odds: prefer closing (PSCH, PSCD, PSCA), fallback to opening (PSH, PSD, PSA)
            pin_h = match.PSCH if pd.notna(match.PSCH) else getattr(match, "PSH", np.nan)
            pin_d = match.PSCD if pd.notna(match.PSCD) else getattr(match, "PSD", np.nan)
            pin_a = match.PSCA if pd.notna(match.PSCA) else getattr(match, "PSA", np.nan)

            pin_probs = [np.nan, np.nan, np.nan]
            if pd.notna(pin_h) and pd.notna(pin_d) and pd.notna(pin_a):
                pin_probs = remove_margin(np.array([pin_h, pin_d, pin_a]), method=margin_method)

            rows.append({
                "Date": match.Date,
                "season": match.season,
                "block": blk_id,
                "HomeTeam": match.HomeTeam,
                "AwayTeam": match.AwayTeam,
                "FTHG": hg,
                "FTAG": ag,
                "outcome_1x2": outcome_1x2,
                "outcome_over25": outcome_over25,
                "outcome_btts": outcome_btts,
                "dc_pH": p_home,
                "dc_pD": p_draw,
                "dc_pA": p_away,
                "dc_pOver25": p_over25,
                "dc_pUnder25": p_under25,
                "dc_pBTTS": p_btts,
                "dc_pBTTS_No": p_btts_no,
                "pin_H_odds": pin_h,
                "pin_D_odds": pin_d,
                "pin_A_odds": pin_a,
                "pin_pH": pin_probs[0],
                "pin_pD": pin_probs[1],
                "pin_pA": pin_probs[2],
            })

    out = pd.DataFrame(rows)
    print(f"Completed walk-forward backtest: {len(out)} matches processed.")
    return out


def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall and per-season comparative evaluation metrics."""
    records = []

    def _calc_metrics(subset: pd.DataFrame, label: str):
        n_total = len(subset)
        if n_total == 0:
            return

        # DC 1X2 metrics
        dc_probs = subset[["dc_pH", "dc_pD", "dc_pA"]].to_numpy()
        y_1x2 = subset["outcome_1x2"].to_numpy()
        dc_ll = multiclass_log_loss(dc_probs, y_1x2)
        dc_bs = brier_score(dc_probs, y_1x2)
        dc_rps = ranked_probability_score(dc_probs, y_1x2)

        # DC Over 2.5 and BTTS metrics
        dc_over25_ll = binary_log_loss(subset["dc_pOver25"].to_numpy(), subset["outcome_over25"].to_numpy())
        dc_btts_ll = binary_log_loss(subset["dc_pBTTS"].to_numpy(), subset["outcome_btts"].to_numpy())

        # Pinnacle metrics (on available matches)
        pin_valid = subset["pin_pH"].notna()
        n_pin = int(pin_valid.sum())
        pin_subset = subset[pin_valid]

        pin_ll = pin_bs = pin_rps = np.nan
        dc_ll_on_pin = dc_bs_on_pin = dc_rps_on_pin = np.nan

        if n_pin > 0:
            pin_probs = pin_subset[["pin_pH", "pin_pD", "pin_pA"]].to_numpy()
            pin_y = pin_subset["outcome_1x2"].to_numpy()
            pin_ll = multiclass_log_loss(pin_probs, pin_y)
            pin_bs = brier_score(pin_probs, pin_y)
            pin_rps = ranked_probability_score(pin_probs, pin_y)

            # Compare DC on identical sample
            dc_probs_pin = pin_subset[["dc_pH", "dc_pD", "dc_pA"]].to_numpy()
            dc_ll_on_pin = multiclass_log_loss(dc_probs_pin, pin_y)
            dc_bs_on_pin = brier_score(dc_probs_pin, pin_y)
            dc_rps_on_pin = ranked_probability_score(dc_probs_pin, pin_y)

        records.append({
            "segment": label,
            "matches": n_total,
            "dc_1x2_log_loss": round(dc_ll, 5),
            "dc_1x2_brier": round(dc_bs, 5),
            "dc_1x2_rps": round(dc_rps, 5),
            "dc_over25_log_loss": round(dc_over25_ll, 5),
            "dc_btts_log_loss": round(dc_btts_ll, 5),
            "pinnacle_matches": n_pin,
            "pin_1x2_log_loss": round(pin_ll, 5) if not np.isnan(pin_ll) else np.nan,
            "pin_1x2_brier": round(pin_bs, 5) if not np.isnan(pin_bs) else np.nan,
            "pin_1x2_rps": round(pin_rps, 5) if not np.isnan(pin_rps) else np.nan,
            "dc_1x2_log_loss_on_pin": round(dc_ll_on_pin, 5) if not np.isnan(dc_ll_on_pin) else np.nan,
            "dc_1x2_rps_on_pin": round(dc_rps_on_pin, 5) if not np.isnan(dc_rps_on_pin) else np.nan,
        })

    # Overall
    _calc_metrics(df, "Overall (All Seasons)")

    # Per-season
    for s, grp in df.groupby("season", sort=True):
        _calc_metrics(grp, f"Season {s}")

    return pd.DataFrame(records)


def compute_all_calibrations(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration tables for Home, Draw, Away, Over 2.5, and BTTS."""
    tables = []

    # Home Win
    t_h = calibration_table(df["dc_pH"].to_numpy(), (df["outcome_1x2"] == 0).astype(int).to_numpy(), n_bins)
    t_h["market"] = "Home Win"
    tables.append(t_h)

    # Draw
    t_d = calibration_table(df["dc_pD"].to_numpy(), (df["outcome_1x2"] == 1).astype(int).to_numpy(), n_bins)
    t_d["market"] = "Draw"
    tables.append(t_d)

    # Away Win
    t_a = calibration_table(df["dc_pA"].to_numpy(), (df["outcome_1x2"] == 2).astype(int).to_numpy(), n_bins)
    t_a["market"] = "Away Win"
    tables.append(t_a)

    # Over 2.5 Goals
    t_o = calibration_table(df["dc_pOver25"].to_numpy(), df["outcome_over25"].to_numpy(), n_bins)
    t_o["market"] = "Over 2.5 Goals"
    tables.append(t_o)

    # BTTS
    t_b = calibration_table(df["dc_pBTTS"].to_numpy(), df["outcome_btts"].to_numpy(), n_bins)
    t_b["market"] = "BTTS"
    tables.append(t_b)

    out = pd.concat(tables, ignore_index=True)
    cols = ["market", "bin_lower", "bin_upper", "count", "pred_mean", "obs_frequency"]
    return out[cols]


def plot_calibration(cal_df: pd.DataFrame, save_path: Path | str) -> None:
    """Generate and save calibration curves across markets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left plot: 1X2 markets
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    colors = {"Home Win": "tab:blue", "Draw": "tab:orange", "Away Win": "tab:green"}

    for market in ["Home Win", "Draw", "Away Win"]:
        sub = cal_df[(cal_df["market"] == market) & cal_df["obs_frequency"].notna() & (cal_df["count"] > 0)]
        ax1.plot(sub["pred_mean"], sub["obs_frequency"], "o-", color=colors[market], label=f"{market} (DC)")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Observed Frequency")
    ax1.set_title("1X2 Outcome Calibration")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right plot: Goals markets (Over 2.5 and BTTS)
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    g_colors = {"Over 2.5 Goals": "tab:purple", "BTTS": "tab:red"}

    for market in ["Over 2.5 Goals", "BTTS"]:
        sub = cal_df[(cal_df["market"] == market) & cal_df["obs_frequency"].notna() & (cal_df["count"] > 0)]
        ax2.plot(sub["pred_mean"], sub["obs_frequency"], "s-", color=g_colors[market], label=f"{market} (DC)")

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Observed Frequency")
    ax2.set_title("Goals Markets Calibration (Over 2.5 & BTTS)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_backtest_report(
    seasons: Sequence[str] = DEFAULT_EVAL_SEASONS,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    xi: float = DEFAULT_XI,
    margin_method: str = "shin",
    xg_blend: float = 0.0,
    home_adv_mode: str = "league",
    calibrate: bool = False,
    **model_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete backtest, save all reports, and return result frames.

    Outputs:
    - {reports_dir}/backtest_predictions.csv
    - {reports_dir}/backtest_summary.csv
    - {reports_dir}/calibration_table.csv
    - {reports_dir}/calibration_plot.png

    Returns:
        (predictions_df, summary_df, calibration_df)
    """
    t0 = time.perf_counter()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    pred_df = walk_forward_backtest(
        matches=matches,
        eval_seasons=seasons,
        xi=xi,
        margin_method=margin_method,
        xg_blend=xg_blend,
        home_adv_mode=home_adv_mode,
        **model_kwargs,
    )

    if calibrate:
        probs = pred_df[["dc_pH", "dc_pD", "dc_pA"]].to_numpy()
        cal_res = evaluate_calibration(probs, pred_df["outcome_1x2"])
        pred_df["cal_pH"] = cal_res.calibrated_probs[:, 0]
        pred_df["cal_pD"] = cal_res.calibrated_probs[:, 1]
        pred_df["cal_pA"] = cal_res.calibrated_probs[:, 2]
        print("\n--- Post-Hoc Probability Calibration ---")
        print(f"  Raw 1X2 Log Loss: {cal_res.raw_log_loss:.5f} -> Calibrated: {cal_res.cal_log_loss:.5f}")
        print(f"  Raw 1X2 RPS:      {cal_res.raw_rps:.5f} -> Calibrated: {cal_res.cal_rps:.5f}")
        print(f"  Raw Brier:        {cal_res.raw_brier:.5f} -> Calibrated: {cal_res.cal_brier:.5f}")

    summary_df = compute_summary_metrics(pred_df)
    cal_df = compute_all_calibrations(pred_df)

    pred_df.to_csv(reports_dir / "backtest_predictions.csv", index=False)
    summary_df.to_csv(reports_dir / "backtest_summary.csv", index=False)
    cal_df.to_csv(reports_dir / "calibration_table.csv", index=False)
    plot_calibration(cal_df, reports_dir / "calibration_plot.png")

    elapsed = time.perf_counter() - t0
    print(f"\n--- Backtest Finished in {elapsed:.2f}s ---")
    print(f"Reports saved to {reports_dir.resolve()}/")
    print("\nSummary Metrics Table:")
    print(summary_df.to_string(index=False))

    return pred_df, summary_df, cal_df


if __name__ == "__main__":
    run_backtest_report()
