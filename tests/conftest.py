"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footsim.data import load_matches
from footsim.goals.dixon_coles import dc_score_matrix

SEED = 20240811


@pytest.fixture(scope="session")
def epl() -> pd.DataFrame:
    """The 6 cached EPL seasons (downloaded once on first use)."""
    return load_matches()


def make_synthetic_league(
    seed: int = SEED,
    n_teams: int = 20,
    n_seasons: int = 3,
    home_adv: float = 0.25,
    rho: float = -0.10,
    intercept: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a double round-robin league from a known Dixon-Coles model.

    Fully determined by ``seed``. Returns the matches frame and the true params.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{k:02d}" for k in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    att -= att.mean()
    dfn = rng.normal(0, 0.3, n_teams)
    dfn -= dfn.mean()
    rows = []
    start = pd.Timestamp("2018-08-10")
    cells = (n_goals := 11) * n_goals
    for s in range(n_seasons):
        fixtures = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        order = rng.permutation(len(fixtures))
        for k, f in enumerate(order):
            i, j = fixtures[f]
            lam = np.exp(intercept + att[i] + dfn[j] + home_adv)
            mu = np.exp(intercept + att[j] + dfn[i])
            p = dc_score_matrix(lam, mu, rho).ravel()
            cell = rng.choice(cells, p=p)
            date = start + pd.Timedelta(days=365 * s + (k // 10) * 7 // 2)
            rows.append((date, teams[i], teams[j], cell // n_goals, cell % n_goals))
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    truth = {"teams": teams, "attack": att, "defence": dfn, "home_adv": home_adv,
             "rho": rho, "intercept": intercept}
    return df, truth


@pytest.fixture(scope="session")
def synthetic() -> tuple[pd.DataFrame, dict]:
    """A seeded synthetic league with known parameters."""
    return make_synthetic_league()
