"""Walk-forward evaluation of the event rate models (Milestone 3).

For each matchday block in the evaluation seasons, every statistic's model
and its league-average baseline are fitted on earlier matches only, and the
block's team-match counts are predicted. The acceptance metric is mean
Poisson deviance (model must be lower than baseline); NB/Poisson log loss is
reported alongside.

Baseline: intercept-only model on all earlier matches, unweighted (xi = 0),
same count family as the model. That is the "league-average" rate per
team-match, with no home, team or referee information.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from footsim.data.loader import load_matches
from footsim.eval.backtest import DEFAULT_EVAL_SEASONS, DEFAULT_REPORTS_DIR, matchday_blocks
from footsim.eval.metrics import count_log_loss, poisson_deviance
from footsim.events.rates import (
    DEFAULT_EVENT_XI,
    DEFAULT_SPECS,
    CountRateModel,
    RateSpec,
    baseline_spec,
)

#: Seasons used to choose prior sds (kept apart from the evaluation seasons).
DEFAULT_TUNE_SEASONS: tuple[str, ...] = ("2223", "2324")


def walk_forward_events(
    matches: pd.DataFrame,
    eval_seasons: Sequence[str] = DEFAULT_EVAL_SEASONS,
    xi: float = DEFAULT_EVENT_XI,
    specs: Mapping[str, RateSpec] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk forward by matchday block and predict every event statistic.

    Args:
        matches: full match history (as from :func:`load_matches`).
        eval_seasons: seasons whose matches are predicted.
        xi: time-decay rate for the models (the baseline uses xi = 0).
        specs: statistics to evaluate; defaults to DEFAULT_SPECS.
        verbose: print progress.

    Returns:
        One row per (match, side, statistic) with the observed count ``y``,
        model mean/alpha and baseline mean/alpha.
    """
    specs = dict(DEFAULT_SPECS if specs is None else specs)
    ev = matches[matches["season"].isin(eval_seasons)].copy()
    if ev.empty:
        raise ValueError(f"No matches found for seasons: {list(eval_seasons)}")
    ev = ev.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    ev["block"] = matchday_blocks(ev["Date"])
    if verbose:
        print(f"Event walk-forward: {len(ev)} matches, {ev['block'].nunique()} blocks, "
              f"stats={list(specs)}")

    out = []
    for blk_id, blk in ev.groupby("block", sort=True):
        as_of = blk["Date"].min()
        fixture_teams = set(blk["HomeTeam"]) | set(blk["AwayTeam"])
        for name, spec in specs.items():
            model = CountRateModel(spec, xi=xi).fit(matches, as_of=as_of, new_teams=fixture_teams)
            base = CountRateModel(baseline_spec(spec), xi=0.0).fit(
                matches, as_of=as_of, new_teams=fixture_teams
            )
            mu_m_h = np.empty(len(blk))
            mu_m_a = np.empty(len(blk))
            mu_b_h = np.empty(len(blk))
            mu_b_a = np.empty(len(blk))
            for k, row in enumerate(blk.itertuples(index=False)):
                ref = row.Referee if pd.notna(row.Referee) else None
                mh, ma = model.expected(row.HomeTeam, row.AwayTeam, ref)
                bh, ba = base.expected(row.HomeTeam, row.AwayTeam, ref)
                mu_m_h[k], mu_m_a[k] = mh, ma
                mu_b_h[k], mu_b_a[k] = bh, ba

            for side, col, mu_m, mu_b in (
                ("home", spec.home_col, mu_m_h, mu_b_h),
                ("away", spec.away_col, mu_m_a, mu_b_a),
            ):
                y = pd.to_numeric(blk[col], errors="coerce").to_numpy()
                out.append(pd.DataFrame({
                    "Date": blk["Date"].to_numpy(),
                    "season": blk["season"].to_numpy(),
                    "block": blk_id,
                    "HomeTeam": blk["HomeTeam"].to_numpy(),
                    "AwayTeam": blk["AwayTeam"].to_numpy(),
                    "Referee": blk["Referee"].to_numpy(),
                    "stat": name,
                    "side": side,
                    "y": y,
                    "mu_model": mu_m,
                    "alpha_model": model.alpha_,
                    "mu_base": mu_b,
                    "alpha_base": base.alpha_,
                }))
    res = pd.concat(out, ignore_index=True)
    return res.dropna(subset=["y"]).reset_index(drop=True)


def summarise_events(pred: pd.DataFrame) -> pd.DataFrame:
    """Deviance and log loss of model vs baseline, per statistic, overall and per season."""
    records = []

    def _row(sub: pd.DataFrame, stat: str, segment: str) -> None:
        y = sub["y"].to_numpy()
        dev_m = poisson_deviance(y, sub["mu_model"].to_numpy())
        dev_b = poisson_deviance(y, sub["mu_base"].to_numpy())
        ll_m = count_log_loss(y, sub["mu_model"].to_numpy(), sub["alpha_model"].to_numpy())
        ll_b = count_log_loss(y, sub["mu_base"].to_numpy(), sub["alpha_base"].to_numpy())
        records.append({
            "stat": stat,
            "segment": segment,
            "team_matches": len(sub),
            "mean_observed": round(float(y.mean()), 4),
            "mean_pred_model": round(float(sub["mu_model"].mean()), 4),
            "model_poisson_deviance": round(dev_m, 5),
            "baseline_poisson_deviance": round(dev_b, 5),
            "deviance_improvement_pct": round(100.0 * (dev_b - dev_m) / dev_b, 2),
            "model_log_loss": round(ll_m, 5),
            "baseline_log_loss": round(ll_b, 5),
            "mean_alpha_model": round(float(sub["alpha_model"].mean()), 4),
            "beats_baseline": bool(dev_m < dev_b),
        })

    for stat, grp in pred.groupby("stat", sort=False):
        _row(grp, stat, "Overall")
        for s, g in grp.groupby("season", sort=True):
            _row(g, stat, f"Season {s}")
    return pd.DataFrame(records)


def tune_event_priors(
    matches: pd.DataFrame,
    tune_seasons: Sequence[str] = DEFAULT_TUNE_SEASONS,
    team_grid: Sequence[float] = (0.05, 0.1, 0.15, 0.25),
    ref_grid: Sequence[float] = (0.03, 0.05, 0.1, 0.2),
    xi: float = DEFAULT_EVENT_XI,
    specs: Mapping[str, RateSpec] | None = None,
) -> pd.DataFrame:
    """Grid-search team / referee prior sds by walk-forward Poisson deviance.

    Uses ``tune_seasons`` only, so the evaluation seasons stay untouched.
    Returns one row per (stat, prior_sd_team, prior_sd_ref).
    """
    specs = dict(DEFAULT_SPECS if specs is None else specs)
    rows = []
    for name, spec in specs.items():
        for sd_t, sd_r in itertools.product(team_grid, ref_grid):
            s = replace(spec, prior_sd_team=sd_t, prior_sd_ref=sd_r)
            pred = walk_forward_events(matches, tune_seasons, xi=xi, specs={name: s}, verbose=False)
            rows.append({
                "stat": name,
                "prior_sd_team": sd_t,
                "prior_sd_ref": sd_r,
                "model_poisson_deviance": poisson_deviance(pred["y"], pred["mu_model"]),
                "baseline_poisson_deviance": poisson_deviance(pred["y"], pred["mu_base"]),
            })
    return pd.DataFrame(rows)


def run_event_report(
    seasons: Sequence[str] = DEFAULT_EVAL_SEASONS,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    xi: float = DEFAULT_EVENT_XI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the event walk-forward evaluation and save the reports.

    Outputs:
    - {reports_dir}/event_rates_predictions.csv
    - {reports_dir}/event_rates_summary.csv

    Returns:
        (predictions_df, summary_df)
    """
    t0 = time.perf_counter()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred = walk_forward_events(load_matches(), seasons, xi=xi)
    summary = summarise_events(pred)
    pred.to_csv(reports_dir / "event_rates_predictions.csv", index=False)
    summary.to_csv(reports_dir / "event_rates_summary.csv", index=False)
    print(f"\n--- Event backtest finished in {time.perf_counter() - t0:.2f}s ---")
    print(f"Reports saved to {reports_dir.resolve()}/")
    print(summary.to_string(index=False))
    return pred, summary


if __name__ == "__main__":
    run_event_report()
