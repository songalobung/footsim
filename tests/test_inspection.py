"""Tests for pre-match inspection dashboard, rolling xG, saves, and H2H."""

import pytest
from typer.testing import CliRunner

from footsim.cli import app
from footsim.data.loader import load_matches
from footsim.eval.inspection import (
    extract_h2h_24m,
    extract_referee_summary,
    extract_team_form,
    inspect_fixture,
)

runner = CliRunner()


def test_extract_team_form():
    matches = load_matches()
    form = extract_team_form("Arsenal", matches, n_matches=5)

    assert form.team == "Arsenal"
    assert len(form.matches) == 5
    assert form.points_won >= 0
    assert form.avg_sot_for > 0.0
    assert form.avg_saves >= 0.0
    assert 0.0 <= form.save_pct <= 1.0
    assert form.goalkeeper_name == "David Raya"

    # Verify individual match records
    m0 = form.matches[0]
    assert m0.shots_for >= m0.sot_for
    assert m0.saves >= 0
    assert m0.fouls >= 0
    assert m0.result in ("W", "D", "L")


def test_extract_h2h_24m():
    matches = load_matches()
    h2h = extract_h2h_24m("Arsenal", "Chelsea", matches, max_days=730)

    assert h2h.home_team == "Arsenal"
    assert h2h.away_team == "Chelsea"
    assert len(h2h.matches) >= 1
    assert (h2h.home_wins + h2h.draws + h2h.away_wins) == len(h2h.matches)
    assert isinstance(h2h.tactical_edge, str)


def test_extract_referee_summary():
    matches = load_matches()
    ref = extract_referee_summary("M Oliver", matches)

    assert ref is not None
    assert ref.referee == "M Oliver"
    assert ref.matches_reffed > 10
    assert ref.avg_yellows > 0.0
    assert ref.strictness_index > 0.0
    assert ref.verdict != ""


def test_inspect_fixture_dashboard():
    report = inspect_fixture("Arsenal", "Chelsea", referee="M Oliver")
    dashboard = report.format_dashboard()

    assert "PRE-MATCH AUDIT: Arsenal vs Chelsea" in dashboard
    assert "RECENT FORM & PROCESS METRICS" in dashboard
    assert "GOALKEEPER SHOT-STOPPING COMPARISON" in dashboard
    assert "HEAD-TO-HEAD (STRICT 24-MONTH TACTICAL WINDOW)" in dashboard
    assert "MATCH REFEREE DISCIPLINARY PROFILE" in dashboard
    assert "M Oliver" in dashboard


def test_cli_inspect_command():
    res = runner.invoke(app, ["inspect", "--home", "Arsenal", "--away", "Chelsea", "--referee", "M Oliver"])
    assert res.exit_code == 0
    assert "PRE-MATCH AUDIT: Arsenal vs Chelsea" in res.stdout
    assert "David Raya" in res.stdout
