"""Economic betting backtest, Edge & EV engine, and Closing Line Value (CLV) tracking.

Phase A Implementation:
1. Strict data leakage assertions (available_at < kickoff).
2. Edge calculation: (model_prob - market_prob) and Expected Value (EV).
3. Closing Line Value (CLV): beating the closing line as statistical proof of edge.
4. Capital staking strategies: Flat, Fractional Kelly (0.10, 0.25, 0.50), and Edge-Weighted.
5. Portfolio risk metrics: ROI / Yield, Win Rate, Max Drawdown (MDD), Sharpe ratio, and Equity Curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
import numpy as np
import pandas as pd

from footsim.eval.odds import remove_margin, shin_probs


@dataclass
class EdgeResult:
    """Calculated market edge and expected value for an individual outcome.

    Attributes:
        selection: outcome name (e.g. 'Home Win', 'Draw', 'Away Win').
        model_prob: model probability p in [0, 1].
        fair_odds: 1 / p.
        market_odds: bookmaker decimal odds O.
        devigged_prob: bookmaker fair probability with margin removed.
        edge: model_prob - devigged_prob.
        ev: expected value (model_prob * market_odds - 1.0).
        clv: closing line value (if opening vs closing odds are provided).
        is_value: True if edge > min_edge and ev > 0.
    """

    selection: str
    model_prob: float
    fair_odds: float
    market_odds: float
    devigged_prob: float
    edge: float
    ev: float
    clv: float | None = None
    is_value: bool = False


@dataclass
class BetOutcome:
    """Individual wager placed in economic simulation."""

    date: pd.Timestamp
    season: str
    home_team: str
    away_team: str
    selection: str
    model_prob: float
    bet_odds: float
    closing_odds: float | None
    devigged_prob: float
    edge: float
    ev: float
    clv: float | None
    stake: float
    won: bool
    pnl: float
    bankroll_after: float


@dataclass
class BettingSimulationResult:
    """Comprehensive economic backtest report and metrics.

    Attributes:
        initial_bankroll: starting capital.
        final_bankroll: capital at end of evaluation period.
        total_bets: number of wagers executed.
        turnover: total cumulative currency wagered.
        net_profit: final_bankroll - initial_bankroll.
        roi_pct: (net_profit / turnover) * 100%.
        win_rate_pct: fraction of winning bets * 100%.
        avg_edge_pct: mean model edge on placed bets * 100%.
        avg_ev_pct: mean expected value on placed bets * 100%.
        avg_clv_pct: mean closing line value * 100%.
        max_drawdown_pct: maximum peak-to-trough equity drop * 100%.
        max_drawdown_bets: longest consecutive losing bets streak.
        sharpe_ratio: annualized return-to-volatility ratio.
        staking_mode: staking method applied.
        bets_df: detailed DataFrame with all individual wagers.
        equity_curve: Series of bankroll balance after each wager.
    """

    initial_bankroll: float
    final_bankroll: float
    total_bets: int
    turnover: float
    net_profit: float
    roi_pct: float
    win_rate_pct: float
    avg_edge_pct: float
    avg_ev_pct: float
    avg_clv_pct: float | None
    max_drawdown_pct: float
    max_drawdown_bets: int
    sharpe_ratio: float
    staking_mode: str
    bets_df: pd.DataFrame
    equity_curve: pd.Series

    def format_report(self) -> str:
        """Format terminal report of economic performance."""
        lines = []
        hdr = f"ECONOMIC BETTING BACKTEST & CLV REPORT ({self.staking_mode.upper()})"
        div = "=" * max(len(hdr), 72)
        lines.append(div)
        lines.append(hdr)
        lines.append(div)
        lines.append(
            f"  Capital Performance: Initial ${self.initial_bankroll:,.2f} -> "
            f"Final ${self.final_bankroll:,.2f} (Net P&L: {self.net_profit:+,.2f})"
        )
        lines.append(
            f"  Portfolio Return:    ROI / Yield: {self.roi_pct:+6.2f}% | "
            f"Annualized Sharpe: {self.sharpe_ratio:5.2f}"
        )
        lines.append(
            f"  Wagers & Turnover:   Total Bets: {self.total_bets:,} | "
            f"Turnover: ${self.turnover:,.2f} | Win Rate: {self.win_rate_pct:5.2f}%"
        )
        lines.append(
            f"  Edge & Market Alpha: Avg Model Edge: {self.avg_edge_pct:+5.2f}% | "
            f"Avg EV: {self.avg_ev_pct:+5.2f}%"
        )
        if self.avg_clv_pct is not None:
            lines.append(
                f"  Closing Line Value:  Avg CLV: {self.avg_clv_pct:+5.2f}% "
                f"({'OUTPERFORMING' if self.avg_clv_pct > 0 else 'UNDERPERFORMING'} Closing Market)"
            )
        lines.append(
            f"  Risk & Drawdown:     Max Drawdown: -{self.max_drawdown_pct:5.2f}% | "
            f"Max Losing Streak: {self.max_drawdown_bets} bets"
        )
        lines.append(div)
        return "\n".join(lines)


def assert_feature_timestamp(
    feature_name: str,
    feature_timestamp: pd.Timestamp | str,
    kickoff_timestamp: pd.Timestamp | str,
) -> None:
    """Enforce strict Priority 0 timestamp assertion to prevent data leakage."""
    f_ts = pd.to_datetime(feature_timestamp)
    k_ts = pd.to_datetime(kickoff_timestamp)
    if f_ts >= k_ts:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Feature '{feature_name}' timestamp ({f_ts}) "
            f"is not strictly before kickoff ({k_ts})!"
        )


def calculate_edge_and_ev(
    model_prob: float,
    market_odds: float,
    devigged_prob: float | None = None,
    closing_odds: float | None = None,
    min_edge: float = 0.02,
    min_ev: float = 0.02,
) -> EdgeResult:
    """Calculate mathematical edge, EV, fair odds, and CLV for an outcome."""
    if model_prob <= 0.0 or model_prob >= 1.0 or market_odds <= 1.0:
        return EdgeResult(
            selection="Unknown",
            model_prob=model_prob,
            fair_odds=math.inf,
            market_odds=market_odds,
            devigged_prob=0.0,
            edge=0.0,
            ev=-1.0,
            clv=None,
            is_value=False,
        )

    fair_odds = 1.0 / model_prob
    dev_p = devigged_prob if devigged_prob is not None else (1.0 / market_odds)
    edge = model_prob - dev_p
    ev = (model_prob * market_odds) - 1.0

    clv = None
    if closing_odds is not None and closing_odds > 1.0:
        # Closing Line Value: bet_odds / closing_odds - 1.0
        clv = (market_odds / closing_odds) - 1.0

    is_val = (edge >= min_edge) and (ev >= min_ev)

    return EdgeResult(
        selection="",
        model_prob=model_prob,
        fair_odds=round(fair_odds, 3),
        market_odds=market_odds,
        devigged_prob=round(dev_p, 4),
        edge=round(edge, 4),
        ev=round(ev, 4),
        clv=round(clv, 4) if clv is not None else None,
        is_value=is_val,
    )


def calculate_stake(
    bankroll: float,
    model_prob: float,
    market_odds: float,
    edge: float,
    staking_mode: Literal["flat", "fractional_kelly", "edge_weighted"] = "fractional_kelly",
    kelly_fraction: float = 0.25,
    flat_stake_amount: float = 10.0,
    max_stake_pct: float = 0.03,
) -> float:
    """Calculate wager stake size adhering to bankroll risk controls."""
    if bankroll <= 0.0 or market_odds <= 1.0 or model_prob <= 0.0:
        return 0.0

    max_stake = bankroll * max_stake_pct

    if staking_mode == "flat":
        return min(flat_stake_amount, max_stake)

    elif staking_mode == "fractional_kelly":
        b = market_odds - 1.0
        q = 1.0 - model_prob
        # Full Kelly formula: f* = (b*p - q) / b
        f_star = (b * model_prob - q) / b
        if f_star <= 0.0:
            return 0.0
        stake = bankroll * f_star * kelly_fraction
        return float(np.clip(stake, 0.0, max_stake))

    elif staking_mode == "edge_weighted":
        # Stake scales proportionally with edge (0.5% to 3.0% of bankroll)
        scale = np.clip(edge * 2.0, 0.005, max_stake_pct)
        return float(bankroll * scale)

    return 0.0


def run_economic_backtest(
    backtest_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    min_edge: float = 0.02,
    min_ev: float = 0.02,
    staking_mode: Literal["flat", "fractional_kelly", "edge_weighted"] = "fractional_kelly",
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.03,
    prefer_opening_odds: bool = True,
) -> BettingSimulationResult:
    """Simulate complete portfolio execution across walk-forward match predictions.

    Args:
        backtest_df: output DataFrame from walk_forward_backtest.
        initial_bankroll: initial capital (default $1,000).
        min_edge: minimum probability edge required to place bet (e.g. 0.02 = +2.0%).
        min_ev: minimum expected value required to place bet (e.g. 0.02 = +2.0%).
        staking_mode: 'flat', 'fractional_kelly', or 'edge_weighted'.
        kelly_fraction: multiplier applied to full Kelly stake (0.10, 0.25, 0.50).
        max_stake_pct: maximum bankroll percentage permitted on a single match (default 3%).
        prefer_opening_odds: if True, bet at Opening Odds (PSH/PSD/PSA) and evaluate CLV
            against Closing Odds (PSCH/PSCD/PSCA).

    Returns:
        BettingSimulationResult containing equity curve, drawdown, and CLV distributions.
    """
    date_col = "Date" if "Date" in backtest_df.columns else ("date" if "date" in backtest_df.columns else None)
    if date_col is not None:
        df = backtest_df.sort_values(date_col).reset_index(drop=True)
    else:
        df = backtest_df.copy().reset_index(drop=True)

    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    current_losing_streak = 0
    max_losing_streak = 0

    bet_records: list[BetOutcome] = []
    equity_curve: list[float] = [bankroll]

    # Map indices 0, 1, 2 to 1X2 outcomes
    outcomes_map = {0: "H", 1: "D", 2: "A"}

    for row in df.itertuples(index=False):
        # Determine odds:
        # Opening odds: PSH, PSD, PSA (if present in dataset)
        # Closing odds: pin_H_odds, pin_D_odds, pin_A_odds
        h_open = getattr(row, "PSH", getattr(row, "pin_H_odds", np.nan))
        d_open = getattr(row, "PSD", getattr(row, "pin_D_odds", np.nan))
        a_open = getattr(row, "PSA", getattr(row, "pin_A_odds", np.nan))

        h_close = getattr(row, "pin_H_odds", getattr(row, "PSCH", np.nan))
        d_close = getattr(row, "pin_D_odds", getattr(row, "PSCD", np.nan))
        a_close = getattr(row, "pin_A_odds", getattr(row, "PSCA", np.nan))

        h_bet = h_open if prefer_opening_odds and pd.notna(h_open) else h_close
        d_bet = d_open if prefer_opening_odds and pd.notna(d_open) else d_close
        a_bet = a_open if prefer_opening_odds and pd.notna(a_open) else a_close

        cand_odds = [h_bet, d_bet, a_bet]
        cand_close = [h_close, d_close, a_close]

        p_h = getattr(row, "cal_pH", getattr(row, "dc_pH", getattr(row, "model_H", np.nan)))
        p_d = getattr(row, "cal_pD", getattr(row, "dc_pD", getattr(row, "model_D", np.nan)))
        p_a = getattr(row, "cal_pA", getattr(row, "dc_pA", getattr(row, "model_A", np.nan)))
        cand_probs = [p_h, p_d, p_a]


        d_h = getattr(row, "pin_pH", getattr(row, "market_pH", np.nan))
        d_d = getattr(row, "pin_pD", getattr(row, "market_pD", np.nan))
        d_a = getattr(row, "pin_pA", getattr(row, "market_pA", np.nan))
        cand_devig = [d_h, d_d, d_a]

        actual_outcome = getattr(row, "outcome_1x2", None)
        if actual_outcome is None:
            ftr = getattr(row, "actual_ftr", getattr(row, "FTR", None))
            if ftr == "H":
                actual_outcome = 0
            elif ftr == "D":
                actual_outcome = 1
            elif ftr == "A":
                actual_outcome = 2

        # Search for highest EV bet among H, D, A
        best_edge_res: EdgeResult | None = None
        best_idx: int | None = None

        for idx in range(3):
            p = cand_probs[idx]
            o = cand_odds[idx]
            dev_p = cand_devig[idx]
            c_o = cand_close[idx]

            if pd.isna(o) or o <= 1.0 or pd.isna(p):
                continue

            edge_res = calculate_edge_and_ev(
                model_prob=p,
                market_odds=o,
                devigged_prob=dev_p if pd.notna(dev_p) else None,
                closing_odds=c_o if pd.notna(c_o) else None,
                min_edge=min_edge,
                min_ev=min_ev,
            )

            if edge_res.is_value:
                if best_edge_res is None or edge_res.ev > best_edge_res.ev:
                    best_edge_res = edge_res
                    best_idx = idx

        # Place wager if value criterion satisfied
        if best_edge_res is not None and best_idx is not None and bankroll > 10.0:
            stake = calculate_stake(
                bankroll=bankroll,
                model_prob=best_edge_res.model_prob,
                market_odds=best_edge_res.market_odds,
                edge=best_edge_res.edge,
                staking_mode=staking_mode,
                kelly_fraction=kelly_fraction,
                max_stake_pct=max_stake_pct,
            )

            if stake >= 1.0:
                won = (actual_outcome == best_idx)
                if won:
                    pnl = stake * (best_edge_res.market_odds - 1.0)
                    current_losing_streak = 0
                else:
                    pnl = -stake
                    current_losing_streak += 1
                    max_losing_streak = max(max_losing_streak, current_losing_streak)

                bankroll += pnl
                peak_bankroll = max(peak_bankroll, bankroll)
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                max_drawdown = max(max_drawdown, drawdown)
                equity_curve.append(bankroll)

                row_date = getattr(row, "Date", getattr(row, "date", pd.Timestamp.now()))
                row_season = str(getattr(row, "season", getattr(row, "Season", "")))
                row_home = str(getattr(row, "HomeTeam", getattr(row, "home_team", "")))
                row_away = str(getattr(row, "AwayTeam", getattr(row, "away_team", "")))

                bet_records.append(
                    BetOutcome(
                        date=row_date,
                        season=row_season,
                        home_team=row_home,
                        away_team=row_away,
                        selection=outcomes_map[best_idx],
                        model_prob=best_edge_res.model_prob,
                        bet_odds=best_edge_res.market_odds,
                        closing_odds=best_edge_res.market_odds if cand_close[best_idx] is None else cand_close[best_idx],
                        devigged_prob=best_edge_res.devigged_prob,
                        edge=best_edge_res.edge,
                        ev=best_edge_res.ev,
                        clv=best_edge_res.clv,
                        stake=round(stake, 2),
                        won=won,
                        pnl=round(pnl, 2),
                        bankroll_after=round(bankroll, 2),
                    )
                )

    bets_df = pd.DataFrame([b.__dict__ for b in bet_records])
    n_bets = len(bets_df)

    if n_bets > 0:
        turnover = float(bets_df["stake"].sum())
        net_profit = bankroll - initial_bankroll
        roi = (net_profit / turnover) * 100.0 if turnover > 0 else 0.0
        win_rate = (bets_df["won"].mean()) * 100.0
        avg_edge = float(bets_df["edge"].mean()) * 100.0
        avg_ev = float(bets_df["ev"].mean()) * 100.0
        valid_clv = bets_df["clv"].dropna()
        avg_clv = float(valid_clv.mean()) * 100.0 if len(valid_clv) > 0 else None

        # Annualized Sharpe ratio on returns per wager
        returns = bets_df["pnl"] / bets_df["stake"]
        std_ret = float(returns.std())
        mean_ret = float(returns.mean())
        # Scaling by sqrt(380) matches ~380 bets per season
        sharpe = (mean_ret / std_ret * np.sqrt(380)) if std_ret > 1e-6 else 0.0
    else:
        turnover = 0.0
        net_profit = 0.0
        roi = 0.0
        win_rate = 0.0
        avg_edge = 0.0
        avg_ev = 0.0
        avg_clv = None
        sharpe = 0.0

    return BettingSimulationResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=round(bankroll, 2),
        total_bets=n_bets,
        turnover=round(turnover, 2),
        net_profit=round(net_profit, 2),
        roi_pct=round(roi, 2),
        win_rate_pct=round(win_rate, 2),
        avg_edge_pct=round(avg_edge, 2),
        avg_ev_pct=round(avg_ev, 2),
        avg_clv_pct=round(avg_clv, 2) if avg_clv is not None else None,
        max_drawdown_pct=round(max_drawdown * 100.0, 2),
        max_drawdown_bets=max_losing_streak,
        sharpe_ratio=round(sharpe, 2),
        staking_mode=staking_mode,
        bets_df=bets_df,
        equity_curve=pd.Series(equity_curve),
    )
