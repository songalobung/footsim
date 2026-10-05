"""Tune the time-decay rate xi by walk-forward 1X2 log loss.

Walk-forward: evaluation matches are grouped into weekly blocks (Tuesday to
Monday). For each block the model is fitted with ``as_of`` = the first match
date in the block, so only earlier matches are used, then every match in the
block is predicted. Teams in the block without history are passed as
``new_teams`` (fixture information only, no results).

Run as a script to produce ``reports/xi_tuning.csv`` and ``reports/xi_tuning.png``::

    python -m footsim.goals.tune
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from footsim.data.loader import load_matches
from footsim.goals.dixon_coles import DixonColes

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

#: Seasons used to tune xi. Deliberately NOT the last two seasons, which are
#: kept for the Milestone 2 backtest (avoids tuning on the test set).
TUNE_SEASONS: tuple[str, ...] = ("2223", "2324")

#: Grid of xi values per day.
XI_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.0, 0.00651, 0.0005), 4))


def onex2_metrics(probs: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """1X2 metrics over the full (home, draw, away) distribution.

    Args:
        probs: (n, 3) array of P(H), P(D), P(A).
        outcome: (n,) int array, 0 = home win, 1 = draw, 2 = away win.

    Returns:
        dict with ``log_loss`` (mean -log p of observed outcome, the
        multiclass log loss), ``brier`` (sum over the 3 classes) and ``rps``.
    """
    n = len(outcome)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), outcome] = 1.0
    p_obs = np.clip(probs[np.arange(n), outcome], 1e-15, 1.0)
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_o = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "log_loss": float(-np.log(p_obs).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "rps": float((((cum_p - cum_o) ** 2).sum(axis=1) / 2.0).mean()),
    }


def weekly_blocks(dates: pd.Series) -> pd.Series:
    """Label each date with its Tuesday-to-Monday week."""
    return pd.to_datetime(dates).dt.to_period("W-MON")


def walk_forward_1x2(
    matches: pd.DataFrame,
    xi: float,
    eval_seasons: Sequence[str],
    **model_kwargs,
) -> pd.DataFrame:
    """Walk-forward 1X2 predictions for every match in ``eval_seasons``.

    Returns a frame with Date, HomeTeam, AwayTeam, pH, pD, pA, outcome.
    """
    ev = matches[matches["season"].isin(eval_seasons)].copy()
    ev["block"] = weekly_blocks(ev["Date"])
    rows = []
    for _, blk in ev.groupby("block", sort=True):
        as_of = blk["Date"].min()
        teams = set(blk["HomeTeam"]) | set(blk["AwayTeam"])
        model = DixonColes(xi=xi, **model_kwargs).fit(matches, as_of=as_of, new_teams=teams)
        for r in blk.itertuples(index=False):
            ph, pd_, pa = model.predict_1x2(r.HomeTeam, r.AwayTeam)
            rows.append((r.Date, r.HomeTeam, r.AwayTeam, ph, pd_, pa, r.FTHG, r.FTAG))
    out = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA", "FTHG", "FTAG"])
    out["outcome"] = np.select([out.FTHG > out.FTAG, out.FTHG == out.FTAG], [0, 1], 2)
    return out


def xi_curve(
    matches: pd.DataFrame,
    xis: Sequence[float] = XI_GRID,
    eval_seasons: Sequence[str] = TUNE_SEASONS,
    **model_kwargs,
) -> pd.DataFrame:
    """Walk-forward 1X2 metrics for each xi in the grid."""
    rows = []
    for xi in xis:
        t0 = time.perf_counter()
        pred = walk_forward_1x2(matches, xi, eval_seasons, **model_kwargs)
        met = onex2_metrics(pred[["pH", "pD", "pA"]].to_numpy(), pred["outcome"].to_numpy())
        rows.append({"xi": xi, "n_matches": len(pred), **met, "seconds": time.perf_counter() - t0})
        print(f"xi={xi:.4f}  log_loss={met['log_loss']:.5f}  rps={met['rps']:.5f}  "
              f"({rows[-1]['seconds']:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def plot_curve(curve: pd.DataFrame, path: Path) -> None:
    """Save the xi vs log loss (and RPS) curve as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(curve["xi"], curve["log_loss"], "o-", color="tab:blue", label="1X2 log loss")
    best = curve.loc[curve["log_loss"].idxmin()]
    ax1.axvline(best["xi"], color="tab:blue", ls=":", lw=1)
    ax1.set_xlabel("xi (per day)")
    ax1.set_ylabel("walk-forward 1X2 log loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(curve["xi"], curve["rps"], "s--", color="tab:orange", label="RPS")
    ax2.set_ylabel("RPS", color="tab:orange")
    ax1.set_title(f"Dixon-Coles xi tuning (best xi={best['xi']:.4f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the xi grid search and write the report files."""
    matches = load_matches()
    curve = xi_curve(matches)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(REPORTS_DIR / "xi_tuning.csv", index=False)
    plot_curve(curve, REPORTS_DIR / "xi_tuning.png")
    best = curve.loc[curve["log_loss"].idxmin()]
    print(f"best xi = {best['xi']:.4f} (log loss {best['log_loss']:.5f})")


if __name__ == "__main__":
    main()
