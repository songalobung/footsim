"""Tests for footsim.goals.dixon_coles.

The four Milestone 1 acceptance checks are marked ``ACCEPTANCE``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import check_grad

from footsim.goals import DixonColes, dc_score_matrix, outcome_probs
from footsim.goals.dixon_coles import dc_tau_bounds

SEED = 12345


def _plain_poisson_kwargs() -> dict:
    """Model settings that reduce Dixon-Coles to a plain Poisson GLM."""
    return dict(xi=0.0, fixed_rho=0.0, prior_sd_established=None, promoted_prior=False)


# --------------------------------------------------------------------------
# ACCEPTANCE 1: rho=0, xi=0 matches a plain Poisson GLM
# --------------------------------------------------------------------------
def _poisson_glm(train: pd.DataFrame):
    """Fit goals ~ intercept + home + attack(team) + defence(opponent) with statsmodels."""
    teams = sorted(set(train.HomeTeam) | set(train.AwayTeam))
    n, m = len(teams), len(train)
    idx = {t: k for k, t in enumerate(teams)}
    hi = train.HomeTeam.map(idx).to_numpy()
    ai = train.AwayTeam.map(idx).to_numpy()
    X = np.zeros((2 * m, 2 + 2 * (n - 1)))
    X[:, 0] = 1.0
    X[:m, 1] = 1.0  # home flag
    for rows, att, dfn in ((np.arange(m), hi, ai), (np.arange(m, 2 * m), ai, hi)):
        ok = att > 0
        X[rows[ok], 2 + att[ok] - 1] = 1.0           # attack dummies (team 0 = baseline)
        ok = dfn > 0
        X[rows[ok], 2 + (n - 1) + dfn[ok] - 1] = 1.0  # defence dummies
    yv = np.concatenate([train.FTHG.to_numpy(), train.FTAG.to_numpy()]).astype(float)
    res = sm.GLM(yv, X, family=sm.families.Poisson()).fit(tol=1e-12)
    rates = res.predict(X)
    return rates[:m], rates[m:], res.llf


@pytest.mark.network
def test_acceptance_rho0_xi0_matches_poisson_glm(epl):
    """ACCEPTANCE: with rho=0 and xi=0, DC reproduces the Poisson GLM."""
    as_of = "2026-07-01"
    train = epl[epl.Date < as_of]
    model = DixonColes(**_plain_poisson_kwargs()).fit(epl, as_of=as_of)
    glm_lam, glm_mu, glm_llf = _poisson_glm(train)

    lam = np.array([model.expected_goals(h, a)[0] for h, a in zip(train.HomeTeam, train.AwayTeam)])
    mu = np.array([model.expected_goals(h, a)[1] for h, a in zip(train.HomeTeam, train.AwayTeam)])
    assert model.converged_
    assert model.rho_ == 0.0
    np.testing.assert_allclose(lam, glm_lam, rtol=1e-4)
    np.testing.assert_allclose(mu, glm_mu, rtol=1e-4)
    assert model.loglik_ == pytest.approx(glm_llf, abs=1e-3)


# --------------------------------------------------------------------------
# ACCEPTANCE 2: score matrix sums to 1
# --------------------------------------------------------------------------
@pytest.mark.network
def test_acceptance_score_matrix_sums_to_one(epl):
    """ACCEPTANCE: every score matrix is 11x11, non-negative and sums to 1."""
    model = DixonColes(xi=0.003).fit(epl, as_of="2025-01-11")
    teams = model.teams_
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        h, a = rng.choice(teams, size=2, replace=False)
        m = model.score_matrix(h, a)
        assert m.shape == (11, 11)
        assert (m >= 0).all()
        assert m.sum() == pytest.approx(1.0, abs=1e-12)
    m = model.score_matrix("Arsenal", "Chelsea")
    assert sum(outcome_probs(m)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lam,mu,rho", [(1.4, 1.1, -0.1), (6.0, 0.2, -0.2), (0.3, 0.3, 0.2),
                                        (3.0, 3.0, 0.2), (0.05, 4.0, -0.2)])
def test_score_matrix_valid_for_extreme_inputs(lam, mu, rho):
    m = dc_score_matrix(lam, mu, rho)
    assert (m >= 0).all()
    assert m.sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# ACCEPTANCE 3: fitting with as_of never touches matches on/after that date
# --------------------------------------------------------------------------
def _corrupt_from(df: pd.DataFrame, mask: pd.Series, seed: int) -> pd.DataFrame:
    """Replace results (and even team names) in masked rows with garbage."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = int(mask.sum())
    out.loc[mask, "FTHG"] = rng.integers(0, 15, k)
    out.loc[mask, "FTAG"] = rng.integers(0, 15, k)
    swap = mask & (rng.random(len(df)) < 0.5)
    out.loc[swap, "HomeTeam"] = "Future FC"
    return out


@pytest.mark.network
@pytest.mark.parametrize("as_of", ["2023-08-12", "2025-01-04", "2025-12-27"])
def test_acceptance_no_lookahead(epl, as_of):
    """ACCEPTANCE: rows on or after as_of cannot influence the fit in any way."""
    cut = pd.Timestamp(as_of)
    assert (epl.Date == cut).any(), "test date should have matches played on it"
    base = DixonColes(xi=0.003).fit(epl, as_of=as_of)

    # (a) garbage in every row on or after as_of
    future = _corrupt_from(epl, epl.Date >= cut, SEED)
    m_future = DixonColes(xi=0.003).fit(future, as_of=as_of)
    # (b) future rows removed entirely
    m_trunc = DixonColes(xi=0.003).fit(epl[epl.Date < cut], as_of=as_of)
    # (c) only the rows exactly ON as_of corrupted
    m_same_day = DixonColes(xi=0.003).fit(_corrupt_from(epl, epl.Date == cut, SEED + 1), as_of=as_of)

    for other in (m_future, m_trunc, m_same_day):
        np.testing.assert_array_equal(other.opt_result_.x, base.opt_result_.x)
        assert other.teams_ == base.teams_
        assert other.promoted_offsets_ == base.promoted_offsets_
        assert other.promoted_teams_ == base.promoted_teams_
    assert base.n_matches_ == int((epl.Date < cut).sum())
    assert base.last_match_date_ < cut
    assert "Future FC" not in m_future.teams_


def test_no_lookahead_synthetic(synthetic):
    df, _ = synthetic
    cut = df.Date.iloc[len(df) // 2]
    base = DixonColes(xi=0.002).fit(df, as_of=cut)
    shuffled = _corrupt_from(df, df.Date >= cut, SEED)
    np.testing.assert_array_equal(DixonColes(xi=0.002).fit(shuffled, as_of=cut).opt_result_.x,
                                  base.opt_result_.x)


# --------------------------------------------------------------------------
# ACCEPTANCE 4: DC draw probability > plain Poisson over a season
# --------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.parametrize("season", ["2425", "2526"])
def test_acceptance_dc_draws_exceed_poisson(epl, season):
    """ACCEPTANCE: mean P(draw) over a season is higher under DC than plain Poisson."""
    fixtures = epl[epl.season == season]
    dc = DixonColes(xi=0.0).fit(fixtures)
    pois = DixonColes(xi=0.0, fixed_rho=0.0).fit(fixtures)
    p_dc = np.mean([dc.predict_1x2(h, a)[1] for h, a in zip(fixtures.HomeTeam, fixtures.AwayTeam)])
    p_po = np.mean([pois.predict_1x2(h, a)[1] for h, a in zip(fixtures.HomeTeam, fixtures.AwayTeam)])
    assert dc.rho_ < 0
    assert p_dc > p_po


# --------------------------------------------------------------------------
# Further checks on the implementation
# --------------------------------------------------------------------------
def test_analytic_gradient_matches_finite_differences(synthetic):
    df, _ = synthetic
    model = DixonColes(xi=0.002).fit(df)  # sets up the design arrays
    rng = np.random.default_rng(SEED)
    for _ in range(3):
        theta = model.opt_result_.x + rng.normal(0, 0.1, model.opt_result_.x.size)
        theta[2] = rng.uniform(-0.15, 0.15)
        err = check_grad(lambda t: model._objective(t)[0], lambda t: model._objective(t)[1], theta)
        assert err < 1e-6 * max(1.0, np.linalg.norm(model._objective(theta)[1]))


def test_recovers_synthetic_parameters(synthetic):
    df, truth = synthetic
    m = DixonColes(xi=0.0, prior_sd_established=None, promoted_prior=False).fit(df)
    assert m.converged_
    assert m.home_adv_ == pytest.approx(truth["home_adv"], abs=0.06)
    assert m.rho_ == pytest.approx(truth["rho"], abs=0.06)
    assert m.intercept_ == pytest.approx(truth["intercept"], abs=0.06)
    assert np.corrcoef(m.attack_[truth["teams"]], truth["attack"])[0, 1] > 0.9
    assert np.corrcoef(m.defence_[truth["teams"]], truth["defence"])[0, 1] > 0.9
    assert abs(m.attack_.sum()) < 1e-9 and abs(m.defence_.sum()) < 1e-9  # identifiability


@pytest.mark.network
def test_fit_time_six_seasons_under_10s(epl):
    model = DixonColes(xi=0.003).fit(epl, as_of="2026-07-01")
    assert model.n_matches_ == 2280
    assert model.fit_seconds_ < 10.0


@pytest.mark.network
def test_rho_bounded_and_tau_positive(epl):
    m = DixonColes(xi=0.003, rho_bounds=(-0.05, 0.05)).fit(epl, as_of="2025-01-11")
    assert -0.05 <= m.rho_ <= 0.05
    m = DixonColes(xi=0.003).fit(epl, as_of="2025-01-11")
    lo, hi = m.rho_bounds
    assert lo <= m.rho_ <= hi and m.min_tau_ > 0
    # Every fixture between fitted teams keeps all four tau > 0 at the fitted rho.
    for h in m.teams_:
        for a in m.teams_:
            if h != a:
                t_lo, t_hi = dc_tau_bounds(*m.expected_goals(h, a))
                assert t_lo < m.rho_ < t_hi


@pytest.mark.network
def test_promoted_team_prior(epl):
    # Start of 2025-26: Sunderland has no history; Leeds and Burnley were last
    # in the league before 2024-25, so they count as promoted.
    as_of = "2025-08-01"
    new = {"Sunderland", "Leeds", "Burnley"}
    m = DixonColes(xi=0.003).fit(epl, as_of=as_of, new_teams=new)
    att_off, def_off = m.promoted_offsets_
    assert att_off < 0 < def_off                # promoted teams score less, concede more
    assert m.team_params("Sunderland") == (att_off, def_off)
    assert {"Leeds", "Burnley"} <= set(m.promoted_teams_)
    assert "Arsenal" not in m.promoted_teams_
    with pytest.raises(KeyError):
        m.team_params("Sunderlnd")

    # The prior pulls Burnley toward the promoted mean, not toward 0 (league mean).
    no_prior = DixonColes(xi=0.003, promoted_prior=False).fit(epl, as_of=as_of, new_teams=new)
    assert abs(m.attack_["Burnley"] - att_off) < abs(no_prior.attack_["Burnley"] - att_off)
    assert abs(m.defence_["Burnley"] - def_off) < abs(no_prior.defence_["Burnley"] - def_off)


def test_fit_is_deterministic(synthetic):
    df, _ = synthetic
    a = DixonColes(xi=0.003).fit(df)
    b = DixonColes(xi=0.003).fit(df)
    np.testing.assert_array_equal(a.opt_result_.x, b.opt_result_.x)


def test_unfitted_and_empty_errors(synthetic):
    df, _ = synthetic
    with pytest.raises(RuntimeError):
        DixonColes().score_matrix("T00", "T01")
    with pytest.raises(ValueError):
        DixonColes().fit(df, as_of=df.Date.min())
