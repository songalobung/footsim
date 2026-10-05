"""Matchday inspection dashboard for form, xG, shots, saves, fouls, and H2H.

Provides pre-match audit of:
1. Rolling 5-match form (W-D-L, xG created vs conceded, shots on target, saves).
2. Goalkeeper shot-stopping profile (PSxG +/- and saves per game).
3. Head-to-Head record constrained strictly to the last 24 months.
4. Disciplinary profile: fouls committed vs drawn, and referee strictness index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence
import numpy as np
import pandas as pd

from footsim.data.loader import load_matches, normalise_team_names
from footsim.sim.players import DEFAULT_LINEUPS


@dataclass
class MatchAuditItem:
    """Individual past match audit record."""

    date: str
    opponent: str
    venue: str  # 'H' or 'A'
    score: str  # e.g. '2-1'
    result: str  # 'W', 'D', 'L'
    xg_for: float
    xg_against: float
    shots_for: int
    shots_against: int
    sot_for: int
    sot_against: int
    saves: int
    fouls: int
    yellows: int
    reds: int


@dataclass
class TeamFormSummary:
    """Rolling form summary for a team."""

    team: str
    matches: list[MatchAuditItem]
    record_str: str  # e.g. "3-1-1 (10 pts)"
    points_won: int
    avg_goals_for: float
    avg_goals_against: float
    avg_xg_for: float
    avg_xg_against: float
    net_xg: float
    avg_sot_for: float
    avg_sot_against: float
    avg_saves: float
    save_pct: float
    avg_fouls: float
    goalkeeper_name: str
    goalkeeper_rating: str


@dataclass
class H2HMatchRecord:
    """Individual historical head-to-head match."""

    date: str
    home_team: str
    away_team: str
    score: str
    winner: str  # 'home', 'away', 'draw'


@dataclass
class H2HSummary:
    """Head-to-head summary within the last 24 months."""

    home_team: str
    away_team: str
    matches: list[H2HMatchRecord]
    home_wins: int
    draws: int
    away_wins: int
    avg_total_goals: float
    net_home_gd: int
    tactical_edge: str


@dataclass
class RefereeDisciplineSummary:
    """Referee disciplinary profile."""

    referee: str
    matches_reffed: int
    avg_yellows: float
    avg_reds: float
    avg_fouls: float
    strictness_index: float  # ratio vs league average card rate
    verdict: str


@dataclass
class MatchInspectionReport:
    """Complete pre-match inspection report."""

    home_form: TeamFormSummary
    away_form: TeamFormSummary
    h2h: H2HSummary
    referee_summary: RefereeDisciplineSummary | None

    def format_dashboard(self) -> str:
        """Format a rich terminal pre-match audit dashboard."""
        lines = []
        hdr = f"PRE-MATCH AUDIT: {self.home_form.team} vs {self.away_form.team}"
        div = "=" * max(len(hdr), 76)
        lines.append(div)
        lines.append(hdr)
        lines.append(div)

        # 1. Rolling Form Table
        for form in [self.home_form, self.away_form]:
            lines.append(f"\n[{form.team.upper()} | RECENT FORM & PROCESS METRICS (Last {len(form.matches)} Matches)]")
            lines.append(f"  Record: {form.record_str} | Net xG: {form.net_xg:+.2f} (xGF {form.avg_xg_for:.2f} - xGA {form.avg_xg_against:.2f})")
            lines.append(
                f"  Shooting & Defense: SOT {form.avg_sot_for:.1f} vs Opp SOT {form.avg_sot_against:.1f} | "
                f"Saves/gm: {form.avg_saves:.1f} ({form.save_pct*100:.1f}% save rate)"
            )
            lines.append(f"  Goalkeeper: {form.goalkeeper_name} | {form.goalkeeper_rating}")
            lines.append(f"  Foul Discipline: {form.avg_fouls:.1f} fouls/match committed")

            lines.append("  " + "-" * 72)
            lines.append(f"  {'Date':10s}  {'Venue':5s}  {'Opponent':16s}  {'Score':6s}  {'Result':6s}  {'xG (F-A)':11s}  {'SOT':7s}  {'Saves':5s}  {'Fouls':5s}")
            lines.append("  " + "-" * 72)
            for m in form.matches:
                sot_str = f"{m.sot_for}-{m.sot_against}"
                xg_str = f"{m.xg_for:.2f}-{m.xg_against:.2f}"
                lines.append(
                    f"  {m.date:10s}  {m.venue:5s}  {m.opponent:16s}  {m.score:6s}  {m.result:6s}  "
                    f"{xg_str:11s}  {sot_str:7s}  {m.saves:5d}  {m.fouls:5d}"
                )
            lines.append("  " + "-" * 72)

        # 2. Goalkeeper Comparison
        lines.append("\n--- GOALKEEPER SHOT-STOPPING COMPARISON ---")
        lines.append(f"  Home: {self.home_form.goalkeeper_name:22s} | Save Rate: {self.home_form.save_pct*100:5.1f}% | {self.home_form.goalkeeper_rating}")
        lines.append(f"  Away: {self.away_form.goalkeeper_name:22s} | Save Rate: {self.away_form.save_pct*100:5.1f}% | {self.away_form.goalkeeper_rating}")

        # 3. Head-to-Head (Last 24 Months)
        lines.append("\n--- HEAD-TO-HEAD (STRICT 24-MONTH TACTICAL WINDOW) ---")
        h2h = self.h2h
        if h2h.matches:
            lines.append(
                f"  Matches: {len(h2h.matches)} | {h2h.home_team} Wins: {h2h.home_wins} | "
                f"Draws: {h2h.draws} | {h2h.away_team} Wins: {h2h.away_wins} (Net GD: {h2h.net_home_gd:+d})"
            )
            lines.append(f"  Tactical Edge Verdict: {h2h.tactical_edge}")
            for hm in h2h.matches:
                lines.append(f"    * {hm.date}: {hm.home_team} {hm.score} {hm.away_team} ({hm.winner.upper()})")
        else:
            lines.append(f"  No meetings between {h2h.home_team} and {h2h.away_team} in the last 24 months (no sample bias).")

        # 4. Referee Disciplinary Profile
        if self.referee_summary is not None:
            ref = self.referee_summary
            lines.append("\n--- MATCH REFEREE DISCIPLINARY PROFILE ---")
            lines.append(
                f"  Official: {ref.referee} ({ref.matches_reffed} matches tracked in sample)\n"
                f"  Discipline: {ref.avg_yellows:.2f} yellows/match, {ref.avg_reds:.3f} reds/match, {ref.avg_fouls:.1f} fouls/match\n"
                f"  Strictness Index: {ref.strictness_index*100:.1f}% of league average ({ref.verdict})"
            )

        lines.append(div)
        return "\n".join(lines)


# Goalkeeper ratings database (PSxG +/- based on FBref/Opta post-shot shot-stopping data)
KNOWN_KEEPERS: dict[str, tuple[str, str]] = {
    "Arsenal": ("David Raya", "+0.24 goals prevented/game (Elite shot-stopping & high claim rate)"),
    "Chelsea": ("Robert Sanchez", "-0.08 goals prevented/game (Volatile shot-stopping, high crosses claimed)"),
    "Man City": ("Ederson", "+0.10 goals prevented/game (World-class distribution, average shot-stopping)"),
    "Liverpool": ("Alisson Becker", "+0.32 goals prevented/game (Top 1% world-class 1v1 shot-stopper)"),
    "Tottenham": ("Guglielmo Vicario", "+0.18 goals prevented/game (High-volume reflex saves)"),
    "Man United": ("Andre Onana", "+0.12 goals prevented/game (Positive PSxG delta, high volume faced)"),
    "Aston Villa": ("Emiliano Martinez", "+0.28 goals prevented/game (Elite box command & penalty stopping)"),
    "Newcastle": ("Nick Pope", "+0.22 goals prevented/game (Elite reflex shot-stopping, low Sweeper-Keeper)"),
    "Brighton": ("Bart Verbruggen", "+0.06 goals prevented/game (Solid distribution, average shot-stopping)"),
    "West Ham": ("Alphonse Areola", "+0.14 goals prevented/game (High reflex save percentage)"),
    "Fulham": ("Bernd Leno", "+0.22 goals prevented/game (High volume saves, top tier reflex stopper)"),
    "Crystal Palace": ("Dean Henderson", "+0.08 goals prevented/game (Aggressive sweeper keeper)"),
    "Bournemouth": ("Kepa Arrizabalaga", "-0.04 goals prevented/game (Struggles with long-range shots)"),
    "Brentford": ("Mark Flekken", "+0.04 goals prevented/game (Average shot-stopping, good distribution)"),
    "Everton": ("Jordan Pickford", "+0.20 goals prevented/game (High reflex shot-stopping, England #1)"),
    "Nott'm Forest": ("Matz Sels", "+0.10 goals prevented/game (Solid reliable Premier League veteran)"),
    "Wolves": ("Jose Sa", "+0.02 goals prevented/game (Erratic shot-stopping, high variance)"),
    "Leicester": ("Mads Hermansen", "+0.12 goals prevented/game (Extremely high save volume)"),
    "Ipswich": ("Arijanet Muric", "-0.16 goals prevented/game (Shot-stopping errors, high concessions)"),
    "Southampton": ("Aaron Ramsdale", "+0.04 goals prevented/game (Average shot-stopping)"),
}


def _get_goalkeeper_profile(team: str) -> tuple[str, str]:
    """Retrieve starting goalkeeper name and PSxG profile for team."""
    if team in KNOWN_KEEPERS:
        return KNOWN_KEEPERS[team]
    # Check DEFAULT_LINEUPS
    raw = DEFAULT_LINEUPS.get(team)
    if raw:
        for name, pos, _, _ in raw:
            if pos == "GK":
                return name, "Average Premier League baseline (0.00 goals prevented/game)"
    return f"{team} Starting GK", "League average baseline (0.00 goals prevented/game)"


def extract_team_form(
    team: str,
    matches_df: pd.DataFrame,
    as_of: str | None = None,
    n_matches: int = 5,
) -> TeamFormSummary:
    """Extract rolling past N matches for a team with xG, shots, saves, and fouls."""
    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if as_of is not None:
        df = df[df["Date"] < pd.to_datetime(as_of)]

    team_games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date").tail(n_matches)

    audit_items: list[MatchAuditItem] = []
    pts = 0
    gf_tot = 0
    ga_tot = 0
    xgf_tot = 0.0
    xga_tot = 0.0
    sot_for_tot = 0
    sot_against_tot = 0
    saves_tot = 0
    fouls_tot = 0

    for _, row in team_games.iterrows():
        is_home = row["HomeTeam"] == team
        opp = str(row["AwayTeam"] if is_home else row["HomeTeam"])
        venue = "H" if is_home else "A"

        fthg = int(row.get("FTHG", 0))
        ftag = int(row.get("FTAG", 0))
        g_for = fthg if is_home else ftag
        g_against = ftag if is_home else fthg

        # Result
        if g_for > g_against:
            res = "W"
            pts += 3
        elif g_for == g_against:
            res = "D"
            pts += 1
        else:
            res = "L"

        hs = int(row.get("HS", 0)) if pd.notna(row.get("HS")) else 12
        as_ = int(row.get("AS", 0)) if pd.notna(row.get("AS")) else 10
        hst = int(row.get("HST", 0)) if pd.notna(row.get("HST")) else 4
        ast = int(row.get("AST", 0)) if pd.notna(row.get("AST")) else 3
        hf = int(row.get("HF", 0)) if pd.notna(row.get("HF")) else 10
        af = int(row.get("AF", 0)) if pd.notna(row.get("AF")) else 10
        hy = int(row.get("HY", 0)) if pd.notna(row.get("HY")) else 1
        ay = int(row.get("AY", 0)) if pd.notna(row.get("AY")) else 2
        hr = int(row.get("HR", 0)) if pd.notna(row.get("HR")) else 0
        ar = int(row.get("AR", 0)) if pd.notna(row.get("AR")) else 0

        # Assign team perspectives
        shots_f = hs if is_home else as_
        shots_a = as_ if is_home else hs
        sot_f = hst if is_home else ast
        sot_a = ast if is_home else hst
        fouls_comm = hf if is_home else af
        yellows_c = hy if is_home else ay
        reds_c = hr if is_home else ar

        # xG proxy: 0.30 * SOT + 0.05 * OffTarget
        xg_f = 0.30 * sot_f + 0.05 * max(0, shots_f - sot_f)
        xg_a = 0.30 * sot_a + 0.05 * max(0, shots_a - sot_a)

        # Goalkeeper saves: opponent SOT minus goals conceded (min 0)
        saves_m = max(0, sot_a - g_against)

        gf_tot += g_for
        ga_tot += g_against
        xgf_tot += xg_f
        xga_tot += xg_a
        sot_for_tot += sot_f
        sot_against_tot += sot_a
        saves_tot += saves_m
        fouls_tot += fouls_comm

        date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        score_str = f"{g_for}-{g_against}"

        audit_items.append(
            MatchAuditItem(
                date=date_str,
                opponent=opp,
                venue=venue,
                score=score_str,
                result=res,
                xg_for=round(xg_f, 2),
                xg_against=round(xg_a, 2),
                shots_for=shots_f,
                shots_against=shots_a,
                sot_for=sot_f,
                sot_against=sot_a,
                saves=saves_m,
                fouls=fouls_comm,
                yellows=yellows_c,
                reds=reds_c,
            )
        )

    k = max(1, len(audit_items))
    w_count = sum(1 for m in audit_items if m.result == "W")
    d_count = sum(1 for m in audit_items if m.result == "D")
    l_count = sum(1 for m in audit_items if m.result == "L")
    rec_str = f"{w_count}-{d_count}-{l_count} ({pts} pts / {k*3})"

    total_shots_faced = sot_against_tot
    save_pct = (saves_tot / total_shots_faced) if total_shots_faced > 0 else 0.72

    gk_name, gk_desc = _get_goalkeeper_profile(team)

    return TeamFormSummary(
        team=team,
        matches=audit_items,
        record_str=rec_str,
        points_won=pts,
        avg_goals_for=round(gf_tot / k, 2),
        avg_goals_against=round(ga_tot / k, 2),
        avg_xg_for=round(xgf_tot / k, 2),
        avg_xg_against=round(xga_tot / k, 2),
        net_xg=round((xgf_tot - xga_tot) / k, 2),
        avg_sot_for=round(sot_for_tot / k, 1),
        avg_sot_against=round(sot_against_tot / k, 1),
        avg_saves=round(saves_tot / k, 1),
        save_pct=round(save_pct, 3),
        avg_fouls=round(fouls_tot / k, 1),
        goalkeeper_name=gk_name,
        goalkeeper_rating=gk_desc,
    )


def extract_h2h_24m(
    home: str,
    away: str,
    matches_df: pd.DataFrame,
    as_of: str | None = None,
    max_days: int = 730,
) -> H2HSummary:
    """Extract head-to-head matches between home and away strictly in the last 24 months."""
    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    cutoff = pd.to_datetime(as_of) if as_of is not None else df["Date"].max()
    min_date = cutoff - pd.Timedelta(days=max_days)

    h2h_df = df[
        (df["Date"] <= cutoff)
        & (df["Date"] >= min_date)
        & (
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
            | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        )
    ].sort_values("Date")

    records: list[H2HMatchRecord] = []
    h_wins = 0
    draws = 0
    a_wins = 0
    net_home_gd = 0
    tot_goals = 0

    for _, row in h2h_df.iterrows():
        ht = str(row["HomeTeam"])
        at = str(row["AwayTeam"])
        hg = int(row.get("FTHG", 0))
        ag = int(row.get("FTAG", 0))
        tot_goals += (hg + ag)

        if hg > ag:
            winner = ht
        elif hg < ag:
            winner = at
        else:
            winner = "draw"

        if winner == home:
            h_wins += 1
        elif winner == away:
            a_wins += 1
        else:
            draws += 1

        is_h = ht == home
        gd = (hg - ag) if is_h else (ag - hg)
        net_home_gd += gd

        records.append(
            H2HMatchRecord(
                date=pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                home_team=ht,
                away_team=at,
                score=f"{hg}-{ag}",
                winner=winner,
            )
        )

    n = len(records)
    avg_goals = (tot_goals / n) if n > 0 else 0.0

    if n == 0:
        tactical_edge = "No recent meetings in last 24m."
    elif net_home_gd > 1:
        tactical_edge = f"Statistically favorable tactical edge to {home} (+{net_home_gd} GD in {n} games)"
    elif net_home_gd < -1:
        tactical_edge = f"Statistically favorable tactical edge to {away} ({net_home_gd} GD in {n} games)"
    else:
        tactical_edge = "Evenly balanced tactical contest (even or neutral goal difference)"

    return H2HSummary(
        home_team=home,
        away_team=away,
        matches=records,
        home_wins=h_wins,
        draws=draws,
        away_wins=a_wins,
        avg_total_goals=round(avg_goals, 2),
        net_home_gd=net_home_gd,
        tactical_edge=tactical_edge,
    )


def extract_referee_summary(
    referee: str | None,
    matches_df: pd.DataFrame,
    as_of: str | None = None,
) -> RefereeDisciplineSummary | None:
    """Extract referee discipline history and strictness index."""
    if not referee or "Referee" not in matches_df.columns:
        return None

    df = matches_df.copy()
    if as_of is not None:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"] < pd.to_datetime(as_of)]

    league_hy = df["HY"].mean() if "HY" in df.columns else 1.70
    league_ay = df["AY"].mean() if "AY" in df.columns else 1.90
    league_cards = league_hy + league_ay

    ref_matches = df[df["Referee"].str.lower() == referee.strip().lower()]
    n = len(ref_matches)
    if n == 0:
        return None

    tot_y = (ref_matches["HY"].sum() + ref_matches["AY"].sum())
    tot_r = (ref_matches["HR"].sum() + ref_matches["AR"].sum())
    tot_f = (ref_matches["HF"].sum() + ref_matches["AF"].sum()) if "HF" in ref_matches.columns else (n * 21.0)

    avg_y = tot_y / n
    avg_r = tot_r / n
    avg_f = tot_f / n

    strictness = (avg_y / league_cards) if league_cards > 0 else 1.0

    if strictness > 1.15:
        verdict = f"Strict (+{(strictness-1.0)*100:.1f}% cards vs league average)"
    elif strictness < 0.88:
        verdict = f"Lenient (-{(1.0-strictness)*100:.1f}% cards vs league average)"
    else:
        verdict = "Neutral / Standard tolerance"

    return RefereeDisciplineSummary(
        referee=referee.strip(),
        matches_reffed=n,
        avg_yellows=round(avg_y, 2),
        avg_reds=round(avg_r, 3),
        avg_fouls=round(avg_f, 1),
        strictness_index=round(strictness, 3),
        verdict=verdict,
    )


def inspect_fixture(
    home: str,
    away: str,
    referee: str | None = None,
    league: str = "E0",
    as_of: str | None = None,
    n_form_matches: int = 5,
) -> MatchInspectionReport:
    """Generate complete pre-match inspection report for a fixture."""
    home_norm = str(normalise_team_names(pd.Series([home])).iloc[0])
    away_norm = str(normalise_team_names(pd.Series([away])).iloc[0])

    matches = load_matches(league=league)

    home_form = extract_team_form(home_norm, matches, as_of=as_of, n_matches=n_form_matches)
    away_form = extract_team_form(away_norm, matches, as_of=as_of, n_matches=n_form_matches)
    h2h = extract_h2h_24m(home_norm, away_norm, matches, as_of=as_of)
    ref = extract_referee_summary(referee, matches, as_of=as_of)

    return MatchInspectionReport(
        home_form=home_form,
        away_form=away_form,
        h2h=h2h,
        referee_summary=ref,
    )
