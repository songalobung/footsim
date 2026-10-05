"""Tests for the Milestone 3 event rate models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import nbinom, poisson

from footsim.eval.metrics import count_log_loss, poisson_deviance
from footsim.events import (
    DEFAULT_SPECS,
    CountRateModel,
    EventRates,
    RateSpec,
    baseline_spec,
    nb_loglik,
    to_team_rows,
)

SEED = 7


def make_event_league(
    seed: int = SEED,
    n_teams: int = 20,
    n_seasons: int = 4,
    alpha: float = 0.08,
    home_adv: float = 0.15,
    intercept: float = 1.5,
    team_sd: float = 0.2,
    n_refs: int = 15,
    ref_sd: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """Seeded synthetic league with NB2 counts in columns HX/AX."""
    rng = np.random.default_rng(seed)
    teams = [f"T{k:02d}" for k in range(n_teams)]
    refs = [f"R{k:02d}" for k in range(n_refs)]
    f = rng.normal(0, team_sd, n_teams)
    g = rng.normal(0, team_sd, n_teams)
    r = rng.normal(0, ref_sd, n_refs)
    rows = []
    start = pd.Timestamp("2018-08-10")
    for s in range(n_seasons):
        fixtures = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        for k, idx in enumerate(rng.permutation(len(fixtures))):
            i, j = fixtures[idx]
            q = rng.integers(n_refs)
            mh = np.exp(intercept + home_adv + f[i] + g[j] + r[q])
            ma = np.exp(intercept + f[j] + g[i] + r[q])
            size = 1.0 / alpha
            yh = rng.negative_binomial(size, size / (size + mh))
            ya = rng.negative_binomial(size, size / (size + ma))
            date = start + pd.Timedelta(days=365 * s + k // 10)
            rows.append((date, teams[i], teams[j], refs[q], yh, ya))
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "Referee", "HX", "AX"])
    truth = {"teams": teams, "refs": refs, "for": f, "against": g, "ref": r,
             "alpha": alpha, "home_adv": home_adv, "intercept": intercept}
    return df.sort_values("Date", kind="mergesort").reset_index(drop=True), truth


@pytest.fixture(scope="module")
def league():
    return make_event_league()


SPEC = RateSpec("x", "HX", "AX", "nb", prior_sd_team=1.0, prior_sd_ref=1.0)


# ---------------------------------------------------------------- basics
def test_nb_loglik_is_a_pmf_and_matches_scipy():
    y = np.arange(0, 200)
    for m, a in [(0.06, 0.0), (1.8, 0.2), (5.5, 0.08), (11.0, 0.01)]:
        ll = nb_loglik(y, np.full(len(y), m), a)
        assert np.exp(ll).sum() == pytest.approx(1.0, abs=1e-9)
        ref = poisson.logpmf(y, m) if a == 0 else nbinom.logpmf(y, 1 / a, 1 / (1 + a * m))
        np.testing.assert_allclose(ll, ref, rtol=1e-9, atol=1e-9)


def test_poisson_deviance_basic():
    y = np.array([0, 1, 3, 7])
    assert poisson_deviance(y, y + 0.0) == pytest.approx(0.0, abs=1e-12)
    assert poisson_deviance(y, np.full(4, y.mean())) > 0
    assert count_log_loss(y, np.full(4, 2.0), 0.0) == pytest.approx(-poisson.logpmf(y, 2.0).mean())


def test_to_team_rows_two_rows_per_match_and_drops_missing():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "HomeTeam": ["A", "B"], "AwayTeam": ["B", "A"], "Referee": ["R", None],
        "HX": [3, np.nan], "AX": [1, 2],
    })
    rows = to_team_rows(df, "HX", "AX")
    assert len(rows) == 3
    assert rows["is_home"].sum() == 1
    assert set(rows.columns) >= {"team", "opp", "is_home", "referee", "y"}


# ------------------------------------------------------------- fitting
@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_analytic_gradient_matches_numerical(league, family):
    df, _ = league
    spec = RateSpec("x", "HX", "AX", family, prior_sd_team=0.3, prior_sd_ref=0.2)
    m = CountRateModel(spec, xi=0.002).fit(df)
    rng = np.random.default_rng(1)
    x = m.opt_result_.x + rng.normal(0, 0.05, m.opt_result_.x.size)
    err = check_grad(lambda t: m._objective(t)[0], lambda t: m._objective(t)[1], x)
    assert err < 1e-5


def test_recovers_synthetic_parameters(league):
    df, truth = league
    m = CountRateModel(SPEC, xi=0.0).fit(df)
    assert m.converged_
    assert m.home_adv_ == pytest.approx(truth["home_adv"], abs=0.04)
    assert m.alpha_ == pytest.approx(truth["alpha"], abs=0.03)
    # Effects are identified up to a shift absorbed by the intercept.
    f_hat = m.for_.reindex(truth["teams"]).to_numpy()
    g_hat = m.against_.reindex(truth["teams"]).to_numpy()
    r_hat = m.referee_.reindex(truth["refs"]).to_numpy()
    assert np.corrcoef(f_hat, truth["for"])[0, 1] > 0.9
    assert np.corrcoef(g_hat, truth["against"])[0, 1] > 0.9
    assert np.corrcoef(r_hat, truth["ref"])[0, 1] > 0.8


def test_fit_is_deterministic(league):
    df, _ = league
    a = CountRateModel(SPEC).fit(df, as_of="2020-06-01")
    b = CountRateModel(SPEC).fit(df, as_of="2020-06-01")
    np.testing.assert_array_equal(a.opt_result_.x, b.opt_result_.x)


def test_no_lookahead(league):
    """Corrupting matches on/after as_of must not change anything."""
    df, _ = league
    as_of = pd.Timestamp("2020-03-01")
    base = CountRateModel(SPEC).fit(df, as_of=as_of)
    bad = df.copy()
    future = bad["Date"] >= as_of
    assert future.any() and (bad["Date"] == as_of).any() or future.sum() > 0
    bad.loc[future, ["HX", "AX"]] = 99
    bad.loc[future, "Referee"] = "NEW REF"
    other = CountRateModel(SPEC).fit(bad, as_of=as_of)
    np.testing.assert_array_equal(base.opt_result_.x, other.opt_result_.x)
    assert "NEW REF" not in other.referees_


def test_referee_with_few_matches_is_shrunk_more():
    """Two referees with the same true effect: the rare one is pulled toward 0."""
    df, _ = make_event_league(ref_sd=0.0, n_seasons=3)
    rng = np.random.default_rng(3)
    busy = rng.choice(len(df), 200, replace=False)
    rare = np.setdiff1d(np.arange(len(df)), busy)[:5]
    df.loc[busy, "Referee"] = "Busy"
    df.loc[rare, "Referee"] = "Rare"
    for idx in (busy, rare):  # same true +0.4 log effect for both
        for col in ("HX", "AX"):
            df.loc[idx, col] = rng.poisson(df.loc[idx, col] * np.exp(0.4))
    m = CountRateModel(RateSpec("x", "HX", "AX", "nb", 0.3, 0.1), xi=0.0).fit(df)
    assert m.referee_["Busy"] > 0.2
    assert 0 < m.referee_["Rare"] < m.referee_["Busy"]
    assert m.referee_matches_["Rare"] == 5
    assert m.referee_effect("Never Seen") == 0.0
    assert m.referee_effect(None) == 0.0


def test_baseline_is_league_average(league):
    df, _ = league
    spec = baseline_spec(RateSpec("x", "HX", "AX", "poisson"))
    m = CountRateModel(spec, xi=0.0).fit(df)
    rows = to_team_rows(df, "HX", "AX")
    mh, ma = m.expected("T00", "T01", referee="R00")
    assert mh == pytest.approx(rows["y"].mean(), rel=1e-6)
    assert ma == pytest.approx(mh)


def test_poisson_family_has_zero_alpha(league):
    df, _ = league
    m = CountRateModel(RateSpec("x", "HX", "AX", "poisson"), xi=0.0).fit(df)
    assert m.alpha_ == 0.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        CountRateModel(RateSpec("x", "HX", "AX", "gamma"))
    with pytest.raises(ValueError):
        CountRateModel(SPEC, xi=-1)
    with pytest.raises(RuntimeError):
        CountRateModel(SPEC).expected("A", "B")


# ------------------------------------------------------ real-data API
@pytest.mark.network
def test_event_rates_public_api(epl):
    ev = EventRates().fit(epl, as_of="2025-01-11", new_teams=["Newcomers FC"])
    out = ev.expected("Arsenal", "Chelsea", referee="M Oliver")
    expected_keys = {f"{side}_{s}" for s in DEFAULT_SPECS for side in ("home", "away")}
    assert set(out) == expected_keys
    assert all(v > 0 for v in out.values())
    assert 0.5 < out["home_yellows"] < 5 and 0.01 < out["home_reds"] < 0.3
    assert 2 < out["home_corners"] < 10 and 6 < out["home_fouls"] < 16
    assert set(ev.dispersion()) == set(DEFAULT_SPECS)
    assert ev.dispersion()["reds"] == 0.0
    assert all(m.converged_ for m in ev.models_.values())
    # Newly promoted team without history -> league-average team effects.
    assert ev.expected("Newcomers FC", "Chelsea")["home_corners"] > 0
    with pytest.raises(KeyError):
        ev.expected("Arsenl", "Chelsea")


@pytest.mark.network
def test_event_rates_no_lookahead_on_real_data(epl):
    as_of = pd.Timestamp("2024-02-01")
    a = EventRates().fit(epl, as_of=as_of)
    bad = epl.copy()
    bad.loc[bad["Date"] >= as_of, ["HY", "AY", "HR", "AR", "HC", "AC", "HF", "AF"]] = 50
    b = EventRates().fit(bad, as_of=as_of)
    assert a.expected("Arsenal", "Chelsea", "M Oliver") == b.expected("Arsenal", "Chelsea", "M Oliver")


@pytest.mark.network
def test_acceptance_each_model_beats_league_average_baseline():
    """ACCEPTANCE: each model beats a league-average baseline on walk-forward Poisson deviance."""
    from pathlib import Path
    summary_path = Path(__file__).resolve().parents[1] / "reports" / "event_rates_summary.csv"
    assert summary_path.exists(), "Event rates summary report does not exist"
    summary = pd.read_csv(summary_path)

    overall = summary[summary["segment"] == "Overall"]
    assert len(overall) == 4
    for stat in ["yellows", "reds", "corners", "fouls"]:
        row = overall[overall["stat"] == stat]
        assert not row.empty, f"Missing stat {stat}"
        m_dev = float(row["model_poisson_deviance"].iloc[0])
        b_dev = float(row["baseline_poisson_deviance"].iloc[0])
        beats = bool(row["beats_baseline"].iloc[0])
        assert beats, f"{stat} model failed to beat baseline: model={m_dev}, baseline={b_dev}"
        assert m_dev < b_dev
