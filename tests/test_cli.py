"""Tests for the footsim CLI (Milestone 6)."""

from typer.testing import CliRunner
import pytest

from footsim.cli import app, _parse_seasons

runner = CliRunner()


def test_parse_seasons():
    """Test season string parsing."""
    assert _parse_seasons("2425,2526") == ["2425", "2526"]
    assert _parse_seasons("2023,2024") == ["2324", "2425"]
    assert _parse_seasons("2223") == ["2223"]


def test_cli_help():
    """Test footsim --help lists all commands."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "fit" in res.stdout
    assert "predict" in res.stdout
    assert "market" in res.stdout
    assert "backtest" in res.stdout
    assert "backtest-events" in res.stdout


def test_cli_fit(tmp_path):
    """Test footsim fit command."""
    res = runner.invoke(
        app,
        [
            "fit",
            "--league", "E0",
            "--as-of", "2025-01-11",
            "--seasons", "2324,2425",
            "--out-dir", str(tmp_path),
        ],
    )
    assert res.exit_code == 0
    assert "Successfully fitted" in res.stdout
    assert "Dixon-Coles:" in res.stdout
    assert (tmp_path / "dixon_coles_E0_2025-01-11.pkl").exists()
    assert (tmp_path / "event_rates_E0_2025-01-11.pkl").exists()


def test_cli_predict():
    """Test footsim predict command."""
    res = runner.invoke(
        app,
        [
            "predict",
            "--home", "Arsenal",
            "--away", "Chelsea",
            "--sims", "2000",
            "--seed", "42",
            "--neutral",
        ],
    )
    assert res.exit_code == 0
    assert "MATCH SIMULATION: Arsenal vs Chelsea" in res.stdout
    assert "1X2 MATCH RESULT" in res.stdout
    assert "TOP 10 CORRECT SCORES" in res.stdout
    assert "CARDS & RED CARD HAZARDS" in res.stdout
    assert "CORNERS" in res.stdout


def test_cli_market():
    """Test footsim market command."""
    # Reliable query
    res = runner.invoke(
        app,
        [
            "market",
            "home_goals > away_goals",
            "--home", "Arsenal",
            "--away", "Chelsea",
            "--sims", "2000",
            "--seed", "42",
        ],
    )
    assert res.exit_code == 0
    assert "MARKET QUERY" in res.stdout
    assert "Probability:" in res.stdout
    assert "RELIABLE" in res.stdout

    # Unreliable query (<200 matches)
    res_rare = runner.invoke(
        app,
        [
            "market",
            "home_goals == 7 and away_goals == 6",
            "--home", "Arsenal",
            "--away", "Chelsea",
            "--sims", "2000",
            "--seed", "42",
        ],
    )
    assert res_rare.exit_code == 0
    assert "UNRELIABLE" in res_rare.stdout
    assert "Sample-size rule warning" in res_rare.stdout
