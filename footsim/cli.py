"""Command-line interface for footsim (Milestone 6).

Commands:
- footsim fit --league E0 --as-of 2025-01-11
- footsim predict --home Arsenal --away Chelsea --referee "M Oliver" --sims 200000
- footsim market "home_goals == 2 and away_goals == 1" --home Arsenal --away Chelsea
- footsim backtest --seasons 2023,2024
- footsim backtest-events --seasons 2425,2526
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from footsim.data.loader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEASONS,
    load_matches,
    normalise_team_names,
)
from footsim.eval.backtest import DEFAULT_EVAL_SEASONS, DEFAULT_REPORTS_DIR, run_backtest_report
from footsim.eval.events import run_event_report
from footsim.events.rates import EventRates
from footsim.goals.dixon_coles import DEFAULT_XI, DixonColes
from footsim.markets.query import (
    market,
    market_1x2,
    market_btts,
    market_correct_scores,
    market_over_under_goals,
    market_red_card,
    market_total_cards,
    market_total_corners,
)
from footsim.sim.match import simulate_match
from footsim.sim.players import get_team_lineup, player_scorer_probs

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "cache" / "models"

app = typer.Typer(
    name="footsim",
    help="Football match simulation engine and analytics.",
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Football match simulation engine and analytics."""
    pass


def _parse_seasons(raw: str) -> list[str]:
    """Parse comma-separated season codes or start years.

    Supports:
    - 4-digit start year: '2024' -> '2425'
    - 4-digit code: '2425' -> '2425'
    """
    out = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        if len(s) == 4 and s.startswith("20") and int(s) >= 2000:
            y1, y2 = int(s[:2]), int(s[2:])
            if y2 == (y1 + 1) % 100:
                out.append(s)
            else:
                yr = int(s)
                out.append(f"{yr % 100:02d}{(yr + 1) % 100:02d}")
        else:
            out.append(s)
    return out


def _load_and_fit_models(
    league: str = "E0",
    as_of: str | None = None,
    seasons: list[str] | None = None,
    xi: float = DEFAULT_XI,
    xg_blend: float = 0.0,
    home_adv_mode: str = "league",
) -> tuple[DixonColes, EventRates]:
    """Load matches and fit DixonColes and EventRates models."""
    use_seasons = seasons or list(DEFAULT_SEASONS)
    matches = load_matches(seasons=use_seasons, league=league)

    dc = DixonColes(xi=xi, xg_blend=xg_blend, home_adv_mode=home_adv_mode).fit(matches, as_of=as_of)
    ev = EventRates().fit(matches, as_of=as_of)
    return dc, ev


@app.command()
def fit(
    league: str = typer.Option("E0", "--league", "-l", help="League code (e.g. 'E0')"),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Training cutoff date (YYYY-MM-DD)"),
    seasons: Optional[str] = typer.Option(
        None,
        "--seasons",
        "-s",
        help="Comma-separated seasons (e.g. '2023,2024' or '2324,2425')",
    ),
    xi: float = typer.Option(DEFAULT_XI, "--xi", help="Time-decay parameter per day"),
    xg_blend: float = typer.Option(0.0, "--xg-blend", help="Weight given to shot-conversion expected goals proxy [0.0, 1.0]"),
    home_adv_mode: str = typer.Option("league", "--home-adv-mode", help="Home advantage mode: 'league' or 'team'"),
    out_dir: Path = typer.Option(DEFAULT_MODEL_DIR, "--out-dir", "-o", help="Directory to save fitted models"),
) -> None:
    """Fit Dixon-Coles goal model and event rate models and save to disk."""
    season_list = _parse_seasons(seasons) if seasons else None
    as_of_str = as_of or "latest available"
    typer.echo(f"Fitting models for league {league} up to {as_of_str} (xi={xi}, xg_blend={xg_blend}, home_adv_mode={home_adv_mode})...")

    t0 = time.perf_counter()
    dc, ev = _load_and_fit_models(
        league=league,
        as_of=as_of,
        seasons=season_list,
        xi=xi,
        xg_blend=xg_blend,
        home_adv_mode=home_adv_mode,
    )
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{league}_{as_of or 'latest'}"
    dc_file = out_dir / f"dixon_coles_{slug}.pkl"
    ev_file = out_dir / f"event_rates_{slug}.pkl"

    with open(dc_file, "wb") as f:
        pickle.dump(dc, f)
    with open(ev_file, "wb") as f:
        pickle.dump(ev, f)

    typer.echo(f"Successfully fitted in {elapsed:.2f}s:")
    typer.echo(f"  - Dixon-Coles: intercept={dc.intercept_:.3f}, home_adv={dc.home_adv_:.3f}, rho={dc.rho_:.4f}")
    typer.echo(f"    Teams tracked: {len(dc.teams_)}")
    typer.echo(f"  - Event rates: {len(ev.models_)} models fitted (yellows, reds, corners, fouls)")
    typer.echo(f"Models saved to: {out_dir}")


@app.command()
def predict(
    home: str = typer.Option(..., "--home", "-H", help="Home team canonical or alias name"),
    away: str = typer.Option(..., "--away", "-A", help="Away team canonical or alias name"),
    referee: Optional[str] = typer.Option(None, "--referee", "-r", help="Match referee name"),
    sims: int = typer.Option(200_000, "--sims", "-n", help="Number of simulations"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Historical cutoff date (YYYY-MM-DD)"),
    league: str = typer.Option("E0", "--league", "-l", help="League code"),
    neutral: bool = typer.Option(False, "--neutral", help="Disable game-state multipliers"),
    xg_blend: float = typer.Option(0.0, "--xg-blend", help="Weight given to shot-conversion expected goals proxy [0.0, 1.0]"),
    derby: Optional[bool] = typer.Option(None, "--derby/--no-derby", help="Force derby rivalry mode (auto-detected if None)"),
    player_layer: bool = typer.Option(False, "--player-layer", help="Simulate individual player scorers and cards"),
    save_sims: Optional[Path] = typer.Option(None, "--save-sims", help="Path to save simulation parquet"),
) -> None:
    """Simulate a match and print a compact prediction table."""
    # Check if saved model exists or fit on the fly (<0.6s)
    dc, ev = _load_and_fit_models(league=league, as_of=as_of, xg_blend=xg_blend)

    # Validate teams
    home_norm = str(normalise_team_names(pd.Series([home])).iloc[0])
    away_norm = str(normalise_team_names(pd.Series([away])).iloc[0])

    lam, mu = dc.expected_goals(home_norm, away_norm)
    rho = getattr(dc, "rho_", -0.05)
    ev_rates = ev.expected(home_norm, away_norm, referee=referee)

    sim_df = simulate_match(
        home=home_norm,
        away=away_norm,
        n=sims,
        seed=seed,
        referee=referee,
        goals_model=dc,
        events_model=ev,
        neutral=neutral,
        is_derby=derby,
        player_layer=player_layer,
        save_path=save_sims,
    )

    p_1x2 = market_1x2(sim_df)
    ou_15 = market_over_under_goals(sim_df, 1.5)
    ou_25 = market_over_under_goals(sim_df, 2.5)
    ou_35 = market_over_under_goals(sim_df, 3.5)
    btts_res = market_btts(sim_df)
    top10_scores = market_correct_scores(sim_df, top_n=10)
    cards_res = market_total_cards(sim_df, 3.5)
    corners_res = market_total_corners(sim_df, 9.5)
    red_match = market_red_card(sim_df)
    red_h = market_red_card(sim_df, team="home")
    red_a = market_red_card(sim_df, team="away")

    exp_yh = float(ev_rates.get("home_yellows", 0.0))
    exp_ya = float(ev_rates.get("away_yellows", 0.0))
    exp_ch = float(ev_rates.get("home_corners", 0.0))
    exp_ca = float(ev_rates.get("away_corners", 0.0))

    sim_time = sim_df.attrs.get("elapsed_seconds", 0.0)

    # Format compact table
    ref_str = f" | Referee: {referee}" if referee else ""
    hdr = f"MATCH SIMULATION: {home_norm} vs {away_norm} (N = {sims:,} sims{ref_str})"
    line = "=" * max(len(hdr), 72)

    typer.echo(line)
    typer.echo(hdr)
    typer.echo(
        f"Goal intensities: Home lambda = {lam:.2f}, Away mu = {mu:.2f} (Total: {lam+mu:.2f}) | "
        f"Sim time: {sim_time:.2f}s"
    )
    typer.echo(line)

    typer.echo("\n--- 1X2 MATCH RESULT ---")
    typer.echo(
        f"  Home Win (H):    {p_1x2['H'].prob*100:6.2f}%  "
        f"[95% CI: {p_1x2['H'].ci_lower*100:5.2f}% - {p_1x2['H'].ci_upper*100:5.2f}%]"
    )
    typer.echo(
        f"  Draw     (D):    {p_1x2['D'].prob*100:6.2f}%  "
        f"[95% CI: {p_1x2['D'].ci_lower*100:5.2f}% - {p_1x2['D'].ci_upper*100:5.2f}%]"
    )
    typer.echo(
        f"  Away Win (A):    {p_1x2['A'].prob*100:6.2f}%  "
        f"[95% CI: {p_1x2['A'].ci_lower*100:5.2f}% - {p_1x2['A'].ci_upper*100:5.2f}%]"
    )

    typer.echo("\n--- GOALS: OVER / UNDER & BTTS ---")
    typer.echo(
        f"  Over 1.5 Goals:  {ou_15['over'].prob*100:6.2f}%  |  "
        f"Under 1.5 Goals: {ou_15['under'].prob*100:6.2f}%"
    )
    typer.echo(
        f"  Over 2.5 Goals:  {ou_25['over'].prob*100:6.2f}%  |  "
        f"Under 2.5 Goals: {ou_25['under'].prob*100:6.2f}%"
    )
    typer.echo(
        f"  Over 3.5 Goals:  {ou_35['over'].prob*100:6.2f}%  |  "
        f"Under 3.5 Goals: {ou_35['under'].prob*100:6.2f}%"
    )
    typer.echo(
        f"  BTTS Yes:        {btts_res['yes'].prob*100:6.2f}%  |  "
        f"BTTS No:         {btts_res['no'].prob*100:6.2f}%"
    )

    typer.echo("\n--- TOP 10 CORRECT SCORES ---")
    for idx, (score, res) in enumerate(top10_scores, 1):
        typer.echo(
            f"  {idx:2d}.  Score {score:5s}:  {res.prob*100:6.2f}%  "
            f"[95% CI: {res.ci_lower*100:5.2f}% - {res.ci_upper*100:5.2f}%]  "
            f"({res.count:,} sims)"
        )

    typer.echo("\n--- CARDS & RED CARD HAZARDS ---")
    typer.echo(
        f"  Expected Yellows: Home {exp_yh:.2f}, Away {exp_ya:.2f}, Total {exp_yh+exp_ya:.2f}"
    )
    typer.echo(
        f"  Red Card Prob:    Match {red_match.prob*100:5.2f}%  "
        f"(Home: {red_h.prob*100:5.2f}%, Away: {red_a.prob*100:5.2f}%)"
    )
    typer.echo(
        f"  Over 3.5 Cards:   {cards_res['over'].prob*100:5.2f}%  |  "
        f"Under 3.5 Cards:  {cards_res['under'].prob*100:5.2f}%"
    )

    typer.echo("\n--- CORNERS ---")
    typer.echo(
        f"  Expected Corners: Home {exp_ch:.2f}, Away {exp_ca:.2f}, Total {exp_ch+exp_ca:.2f}"
    )
    typer.echo(
        f"  Over 9.5 Corners: {corners_res['over'].prob*100:5.2f}%  |  "
        f"Under 9.5 Corners: {corners_res['under'].prob*100:5.2f}%"
    )

    if player_layer:
        h_lineup = get_team_lineup(home_norm)
        a_lineup = get_team_lineup(away_norm)
        h_probs = player_scorer_probs(sim_df, h_lineup, is_home=True).head(5)
        a_probs = player_scorer_probs(sim_df, a_lineup, is_home=False).head(5)

        typer.echo("\n--- TOP ANYTIME & FIRST GOALSCORERS ---")
        typer.echo(f"  {home_norm}:")
        for _, row in h_probs.iterrows():
            typer.echo(f"    {row['player']:22s} ({row['position']}): Anytime {row['anytime_prob']*100:5.2f}% | First Goal {row['first_prob']*100:5.2f}%")
        typer.echo(f"  {away_norm}:")
        for _, row in a_probs.iterrows():
            typer.echo(f"    {row['player']:22s} ({row['position']}): Anytime {row['anytime_prob']*100:5.2f}% | First Goal {row['first_prob']*100:5.2f}%")

    typer.echo(line)

    if save_sims:
        typer.echo(f"Simulation details saved to {save_sims}")


@app.command("market")
def market_cli(
    expression: str = typer.Argument(..., help="Boolean filter expression on simulations"),
    home: str = typer.Option(..., "--home", "-H", help="Home team canonical or alias name"),
    away: str = typer.Option(..., "--away", "-A", help="Away team canonical or alias name"),
    referee: Optional[str] = typer.Option(None, "--referee", "-r", help="Match referee name"),
    sims: int = typer.Option(200_000, "--sims", "-n", help="Number of simulations"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Historical cutoff date (YYYY-MM-DD)"),
    league: str = typer.Option("E0", "--league", "-l", help="League code"),
    neutral: bool = typer.Option(False, "--neutral", help="Disable game-state multipliers"),
) -> None:
    """Evaluate an arbitrary market query over simulated matches."""
    dc, ev = _load_and_fit_models(league=league, as_of=as_of)
    home_norm = str(normalise_team_names(pd.Series([home])).iloc[0])
    away_norm = str(normalise_team_names(pd.Series([away])).iloc[0])

    sim_df = simulate_match(
        home=home_norm,
        away=away_norm,
        n=sims,
        seed=seed,
        referee=referee,
        goals_model=dc,
        events_model=ev,
        neutral=neutral,
    )

    res = market(sim_df, expression)

    typer.echo("=" * 64)
    typer.echo(f"MARKET QUERY: {home_norm} vs {away_norm}")
    typer.echo(f"Expression: {res.query}")
    typer.echo("=" * 64)
    typer.echo(f"Matching Sims:   {res.count:,} / {res.total:,}")
    typer.echo(f"Probability:     {res.prob * 100:.3f}%")
    typer.echo(f"95% Wilson CI:   [{res.ci_lower * 100:.3f}%, {res.ci_upper * 100:.3f}%]")

    if res.reliable:
        typer.echo("Reliability:     RELIABLE (count >= 200)")
    else:
        typer.echo(
            f"Reliability:     UNRELIABLE (count {res.count} < 200 threshold!)\n"
            f"                 Sample-size rule warning: do not use as point estimate.\n"
            f"                 Recommendation: increase n to at least {res.suggested_n:,}."
        )
    typer.echo("=" * 64)


@app.command()
def backtest(
    seasons: str = typer.Option(
        ",".join(DEFAULT_EVAL_SEASONS),
        "--seasons",
        "-s",
        help="Comma-separated season codes (e.g. '2425,2526' or '2024,2025')",
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--reports-dir",
        "-o",
        help="Directory to save CSV and PNG reports",
    ),
    xi: float = typer.Option(
        DEFAULT_XI,
        "--xi",
        help="Time-decay parameter per day",
    ),
    margin_method: str = typer.Option(
        "shin",
        "--margin-method",
        "-m",
        help="Odds margin removal method: 'shin' or 'proportional'",
    ),
    xg_blend: float = typer.Option(
        0.0,
        "--xg-blend",
        help="Weight for shot-conversion expected goals proxy [0.0, 1.0]",
    ),
    home_adv_mode: str = typer.Option(
        "league",
        "--home-adv-mode",
        help="Home advantage mode: 'league' or 'team'",
    ),
    calibrate: bool = typer.Option(
        False,
        "--calibrate/--no-calibrate",
        help="Perform post-hoc probability calibration on walk-forward outputs",
    ),
) -> None:
    """Run walk-forward backtest comparing model predictions with Pinnacle closing odds."""
    season_list = _parse_seasons(seasons)
    typer.echo(f"Running walk-forward backtest for seasons: {season_list} (xg_blend={xg_blend}, home_adv_mode={home_adv_mode}, calibrate={calibrate})")
    run_backtest_report(
        seasons=season_list,
        reports_dir=reports_dir,
        xi=xi,
        margin_method=margin_method,
        xg_blend=xg_blend,
        home_adv_mode=home_adv_mode,
        calibrate=calibrate,
    )


@app.command("backtest-events")
def backtest_events(
    seasons: str = typer.Option(
        ",".join(DEFAULT_EVAL_SEASONS),
        "--seasons",
        "-s",
        help="Comma-separated season codes (e.g. '2425,2526' or '2024,2025')",
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--reports-dir",
        "-o",
        help="Directory to save CSV reports",
    ),
) -> None:
    """Run walk-forward evaluation on event rate models (cards, corners, fouls)."""
    season_list = _parse_seasons(seasons)
    typer.echo(f"Running walk-forward events backtest for seasons: {season_list}")
    run_event_report(
        seasons=season_list,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    app()
