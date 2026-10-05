"""Dixon-Coles goals model with time decay and a promoted-team prior.

Model (match k, home team i, away team j)::

    log lambda_k = c + attack_i + defence_j + home_adv
    log mu_k     = c + attack_j + defence_i
    P(x, y) = tau(x, y; lambda, mu, rho) * Pois(x; lambda) * Pois(y; mu)

Identifiability: attack and defence each sum to zero in log space, and a free
intercept ``c`` carries the overall scoring level. This is the same model as
the spec's "attack sums to zero" version: the spec's defence_j equals
``c + defence_j`` here. See docs/decisions.md.

Each match's log-likelihood is weighted by ``exp(-xi * days_before_as_of)``.
A Gaussian prior (ridge penalty) shrinks each team's attack/defence toward 0
(established teams, weak) or toward an empirical promoted-team mean
(promoted teams, stronger).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from footsim.data.ewma import OpponentAdjustedEWMA
from footsim.data.loader import season_start_year
from footsim.goals.copula import copula_score_matrix

#: Default time-decay rate per day (tuned by walk-forward 1X2 log loss, see
#: footsim.goals.tune and docs/decisions.md).
DEFAULT_XI = 0.003

#: Fallback promoted-team offsets (log-rate vs league average) used when the
#: training data contains no completed promotion to estimate them from.
FALLBACK_PROMOTED_ATTACK = -0.30
FALLBACK_PROMOTED_DEFENCE = 0.25

#: Minimum promoted team-matches needed to trust the empirical offsets.
MIN_PROMOTED_TEAM_MATCHES = 38


def dc_tau_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Return the open interval of rho for which all four tau values are positive."""
    lo = max(-1.0 / lam, -1.0 / mu)
    hi = min(1.0 / (lam * mu), 1.0)
    return lo, hi


def dc_score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Dixon-Coles scoreline probability matrix.

    Entry ``[x, y]`` is P(home scores x, away scores y), for 0..max_goals each.
    rho is clipped into the fixture's tau-positive interval (only matters for
    extreme lambda/mu), and the matrix is renormalised to sum to 1 to remove
    the truncation above ``max_goals``.
    """
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    lo, hi = dc_tau_bounds(lam, mu)
    r = float(np.clip(rho, 0.999 * lo, 0.999 * hi))
    m[0, 0] *= 1.0 - lam * mu * r
    m[0, 1] *= 1.0 + lam * r
    m[1, 0] *= 1.0 + mu * r
    m[1, 1] *= 1.0 - r
    return m / m.sum()


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (P(home win), P(draw), P(away win)) from a score matrix."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


@dataclass
class _Design:
    """Pre-computed arrays for the vectorised likelihood."""

    hi: np.ndarray
    ai: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    const: np.ndarray  # -log x! - log y!
    i00: np.ndarray
    i01: np.ndarray
    i10: np.ndarray
    i11: np.ndarray
    n_teams: int
    prior_mean_att: np.ndarray
    prior_mean_def: np.ndarray
    prior_prec: np.ndarray  # 1 / sd^2 per team (0 = no prior)


class DixonColes:
    """Dixon-Coles model with time decay.

    Args:
        xi: time-decay rate per day; match weight is ``exp(-xi * days)``.
        max_goals: score matrix covers 0..max_goals goals per team.
        rho_bounds: box bounds for rho (see docs/decisions.md for why they
            keep tau positive).
        fixed_rho: if given, rho is held at this value instead of fitted.
        prior_sd_established: sd of the Gaussian prior toward 0 for teams that
            played in the previous season. ``None`` disables it.
        prior_sd_promoted: sd of the prior toward the promoted-team mean.
        promoted_prior: if False, every team is treated as established.
        ftol, gtol, maxiter: L-BFGS-B settings.
    """

    def __init__(
        self,
        xi: float = DEFAULT_XI,
        max_goals: int = 10,
        rho_bounds: tuple[float, float] = (-0.2, 0.2),
        fixed_rho: float | None = None,
        prior_sd_established: float | None = 1.0,
        prior_sd_promoted: float | None = 0.25,
        promoted_prior: bool = True,
        ftol: float = 1e-12,
        gtol: float = 1e-7,
        maxiter: int = 2000,
        xg_blend: float = 0.0,
        home_adv_mode: str = "league",
        ewma_weight: float = 0.0,
        ewma_alpha: float = 0.15,
    ) -> None:
        if xi < 0:
            raise ValueError("xi must be >= 0")
        if not (0.0 <= xg_blend <= 1.0):
            raise ValueError("xg_blend must be in [0.0, 1.0]")
        if not (0.0 <= ewma_weight <= 1.0):
            raise ValueError("ewma_weight must be in [0.0, 1.0]")
        if home_adv_mode not in ("league", "team"):
            raise ValueError(f"Unknown home_adv_mode {home_adv_mode!r}")
        self.xi = xi
        self.max_goals = max_goals
        self.rho_bounds = rho_bounds
        self.fixed_rho = fixed_rho
        self.prior_sd_established = prior_sd_established
        self.prior_sd_promoted = prior_sd_promoted
        self.promoted_prior = promoted_prior
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter
        self.xg_blend = xg_blend
        self.home_adv_mode = home_adv_mode
        self.ewma_weight = ewma_weight
        self.ewma_alpha = ewma_alpha


    # ------------------------------------------------------------------ fit
    def fit(
        self,
        matches: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
        new_teams: Iterable[str] | None = None,
    ) -> "DixonColes":
        """Fit on matches played strictly before ``as_of``.

        Args:
            matches: frame with ``Date, HomeTeam, AwayTeam, FTHG, FTAG``.
            as_of: cut-off date. Matches on or after it are discarded before
                anything else happens. ``None`` means "use every match" and
                sets as_of to the day after the last match.
            new_teams: teams known to be in the current season (fixture
                information, not results). Only teams WITHOUT any training
                history are affected: they get the promoted-team prior mean.
                Teams with history are classified by the automatic rule in
                :meth:`_promoted_teams`, so passing a full fixture list is safe.

        Returns:
            self
        """
        t0 = time.perf_counter()
        dates = pd.to_datetime(matches["Date"])
        if as_of is None:
            as_of_ts = dates.max().normalize() + pd.Timedelta(days=1)
        else:
            as_of_ts = pd.Timestamp(as_of).normalize()

        # --- No lookahead: drop every match on/after as_of FIRST, and only
        # --- keep the columns the model needs.
        keep = (dates < as_of_ts).to_numpy()
        train = matches.loc[keep, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        train["Date"] = dates[keep].to_numpy()
        train = train.dropna(subset=["FTHG", "FTAG"])
        if train.empty:
            raise ValueError(f"No matches before as_of={as_of_ts.date()}")

        teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)

        new_teams_set = set(new_teams or [])
        promoted = self._promoted_teams(train, as_of_ts)
        att_off, def_off, n_prom = self._promoted_offsets(train)

        # Prior means and precisions.
        is_prom = np.array([t in promoted for t in teams]) & self.promoted_prior
        mean_att = np.where(is_prom, att_off, 0.0)
        mean_def = np.where(is_prom, def_off, 0.0)
        prec_est = 0.0 if self.prior_sd_established is None else self.prior_sd_established**-2
        prec_prom = 0.0 if self.prior_sd_promoted is None else self.prior_sd_promoted**-2
        prec = np.where(is_prom, prec_prom, prec_est)

        days = (as_of_ts - train["Date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-self.xi * days)
        x = train["FTHG"].to_numpy().astype(float)
        y = train["FTAG"].to_numpy().astype(float)
        if self.xg_blend > 0 and {"HST", "AST", "HS", "AS"}.issubset(matches.columns):
            hst = pd.to_numeric(matches.loc[keep, "HST"], errors="coerce").fillna(train["FTHG"]).to_numpy()
            ast = pd.to_numeric(matches.loc[keep, "AST"], errors="coerce").fillna(train["FTAG"]).to_numpy()
            hs = pd.to_numeric(matches.loc[keep, "HS"], errors="coerce").fillna(train["FTHG"] * 3).to_numpy()
            as_shots = pd.to_numeric(matches.loc[keep, "AS"], errors="coerce").fillna(train["FTAG"] * 3).to_numpy()
            xg_h = 0.30 * hst + 0.05 * np.maximum(hs - hst, 0.0)
            xg_a = 0.30 * ast + 0.05 * np.maximum(as_shots - ast, 0.0)
            x = (1.0 - self.xg_blend) * x + self.xg_blend * xg_h
            y = (1.0 - self.xg_blend) * y + self.xg_blend * xg_a
        des = _Design(
            hi=train["HomeTeam"].map(idx).to_numpy(),
            ai=train["AwayTeam"].map(idx).to_numpy(),
            x=x,
            y=y,
            w=w,
            const=-gammaln(x + 1) - gammaln(y + 1),
            i00=np.flatnonzero((x == 0) & (y == 0)),
            i01=np.flatnonzero((x == 0) & (y == 1)),
            i10=np.flatnonzero((x == 1) & (y == 0)),
            i11=np.flatnonzero((x == 1) & (y == 1)),
            n_teams=n,
            prior_mean_att=mean_att,
            prior_mean_def=mean_def,
            prior_prec=prec,
        )
        self._des = des
        self._wsum = float(w.sum())

        rho0 = 0.0 if self.fixed_rho is None else float(self.fixed_rho)
        x0 = np.zeros(3 + 2 * (n - 1))
        x0[0] = np.log(max((x.mean() + y.mean()) / 2.0, 1e-3))
        x0[1] = 0.2
        x0[2] = rho0
        rho_b = (rho0, rho0) if self.fixed_rho is not None else self.rho_bounds
        bounds = [(None, None), (None, None), rho_b] + [(None, None)] * (2 * (n - 1))

        res = minimize(
            self._objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": self.ftol, "gtol": self.gtol, "maxiter": self.maxiter, "maxcor": 20},
        )
        c, h, rho, a, d = self._unpack(res.x, n)

        self.teams_ = teams
        self.team_index_ = idx
        self.intercept_ = float(c)
        self.home_adv_ = float(h)
        if self.home_adv_mode == "team":
            tau_shrink = 19.0
            team_h_adv = {}
            for t in teams:
                t_home = train[train["HomeTeam"] == t]
                t_away = train[train["AwayTeam"] == t]
                n_h = len(t_home)
                if n_h > 0 and len(t_away) > 0:
                    diff_rate = float(np.log(max(t_home["FTHG"].mean() / max(t_away["FTAG"].mean(), 0.1), 0.1)))
                    shrink_w = n_h / (n_h + tau_shrink)
                    team_h_adv[t] = float(shrink_w * diff_rate + (1.0 - shrink_w) * h)
                else:
                    team_h_adv[t] = float(h)
            self.team_home_adv_ = pd.Series(team_h_adv, name="home_adv")
        else:
            self.team_home_adv_ = pd.Series({t: float(h) for t in teams}, name="home_adv")
        self.rho_ = float(rho)
        self.attack_ = pd.Series(a, index=teams, name="attack")
        self.defence_ = pd.Series(d, index=teams, name="defence")
        self.promoted_teams_ = sorted(promoted | (new_teams_set - set(teams)))
        self.new_teams_ = sorted(new_teams_set - set(teams))
        self.promoted_offsets_ = (att_off, def_off)
        self.n_promoted_team_matches_ = n_prom
        self.as_of_ = as_of_ts
        self.n_matches_ = len(train)
        self.last_match_date_ = train["Date"].max()
        self.weights_ = w
        self.loglik_ = float(self._loglik(res.x))
        self.min_tau_ = float(self._min_tau(res.x))
        self.converged_ = bool(res.success)
        self.opt_result_ = res
        self.fit_seconds_ = time.perf_counter() - t0
        if self.ewma_weight > 0:
            ewma_eng = OpponentAdjustedEWMA(alpha=self.ewma_alpha)
            ewma_eng.compute_all_prematch_features(matches.loc[keep].copy())
            self.ewma_ratings_ = ewma_eng.team_ratings
        return self



    # ------------------------------------------------- promoted-team prior
    @staticmethod
    def _promoted_teams(train: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
        """Teams in the training data that did not play in the previous season.

        "Previous season" is the season before the one containing ``as_of``.
        Uses training rows only. If the previous season is absent from the
        data, no team is flagged.
        """
        season = train["Date"].map(season_start_year)
        prev = season_start_year(as_of) - 1
        prev_rows = train[season == prev]
        if prev_rows.empty:
            return set()
        prev_teams = set(prev_rows["HomeTeam"]) | set(prev_rows["AwayTeam"])
        all_teams = set(train["HomeTeam"]) | set(train["AwayTeam"])
        return all_teams - prev_teams

    @staticmethod
    def _promoted_offsets(train: pd.DataFrame) -> tuple[float, float, int]:
        """Empirical promoted-team attack/defence offsets (log scale).

        For each season S in the training data whose previous season S-1 is
        also present, promoted teams are those in S but not in S-1. Their
        goals for/against per match in S (training rows only) are pooled and
        compared with the league goals per team-match in the same season:

            attack_offset  = log(sum GF_promoted / sum expected_GF)
            defence_offset = log(sum GA_promoted / sum expected_GA)

        Falls back to fixed defaults if fewer than MIN_PROMOTED_TEAM_MATCHES
        promoted team-matches are available.
        """
        season = train["Date"].map(season_start_year).to_numpy()
        home = pd.DataFrame(
            {"season": season, "team": train["HomeTeam"].to_numpy(),
             "gf": train["FTHG"].to_numpy(), "ga": train["FTAG"].to_numpy()}
        )
        away = pd.DataFrame(
            {"season": season, "team": train["AwayTeam"].to_numpy(),
             "gf": train["FTAG"].to_numpy(), "ga": train["FTHG"].to_numpy()}
        )
        tm = pd.concat([home, away], ignore_index=True)
        league_avg = tm.groupby("season")["gf"].mean()  # goals per team-match
        teams_by_season = tm.groupby("season")["team"].agg(set)

        gf = ga = expected = 0.0
        n = 0
        for s, teams in teams_by_season.items():
            if s - 1 not in teams_by_season.index:
                continue
            prom = teams - teams_by_season[s - 1]
            rows = tm[(tm["season"] == s) & tm["team"].isin(prom)]
            gf += rows["gf"].sum()
            ga += rows["ga"].sum()
            expected += len(rows) * league_avg[s]
            n += len(rows)
        if n < MIN_PROMOTED_TEAM_MATCHES or gf <= 0 or ga <= 0:
            return FALLBACK_PROMOTED_ATTACK, FALLBACK_PROMOTED_DEFENCE, n
        return float(np.log(gf / expected)), float(np.log(ga / expected)), n

    # -------------------------------------------------------- likelihood
    @staticmethod
    def _unpack(theta: np.ndarray, n: int):
        c, h, rho = theta[0], theta[1], theta[2]
        a = np.empty(n)
        a[:-1] = theta[3 : 3 + n - 1]
        a[-1] = -a[:-1].sum()
        d = np.empty(n)
        d[:-1] = theta[3 + n - 1 :]
        d[-1] = -d[:-1].sum()
        return c, h, rho, a, d

    def _rates(self, theta: np.ndarray):
        des = self._des
        c, h, rho, a, d = self._unpack(theta, des.n_teams)
        loglam = c + a[des.hi] + d[des.ai] + h
        logmu = c + a[des.ai] + d[des.hi]
        return c, h, rho, a, d, loglam, logmu

    def _tau_terms(self, lam, mu, rho):
        """log tau and its derivatives w.r.t. log lambda, log mu and rho."""
        des = self._des
        m = len(lam)
        log_tau = np.zeros(m)
        d_ll = np.zeros(m)
        d_lm = np.zeros(m)
        d_r = np.zeros(m)
        floor = 1e-12

        i = des.i00
        t = np.maximum(1.0 - lam[i] * mu[i] * rho, floor)
        log_tau[i] = np.log(t)
        d_ll[i] = -lam[i] * mu[i] * rho / t
        d_lm[i] = d_ll[i]
        d_r[i] = -lam[i] * mu[i] / t

        i = des.i01  # x=0, y=1: tau = 1 + lam*rho
        t = np.maximum(1.0 + lam[i] * rho, floor)
        log_tau[i] = np.log(t)
        d_ll[i] = lam[i] * rho / t
        d_r[i] = lam[i] / t

        i = des.i10  # x=1, y=0: tau = 1 + mu*rho
        t = np.maximum(1.0 + mu[i] * rho, floor)
        log_tau[i] = np.log(t)
        d_lm[i] = mu[i] * rho / t
        d_r[i] = mu[i] / t

        i = des.i11
        t = max(1.0 - rho, floor)
        log_tau[i] = np.log(t)
        d_r[i] = -1.0 / t
        return log_tau, d_ll, d_lm, d_r

    def _loglik(self, theta: np.ndarray) -> float:
        """Weighted log-likelihood (no prior), including the -log x! terms."""
        des = self._des
        _, _, rho, _, _, loglam, logmu = self._rates(theta)
        lam, mu = np.exp(loglam), np.exp(logmu)
        log_tau, *_ = self._tau_terms(lam, mu, rho)
        ll = log_tau + des.x * loglam - lam + des.y * logmu - mu + des.const
        return float(np.dot(des.w, ll))

    def _min_tau(self, theta: np.ndarray) -> float:
        _, _, rho, _, _, loglam, logmu = self._rates(theta)
        lam, mu = np.exp(loglam), np.exp(logmu)
        taus = np.concatenate([1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, [1 - rho]])
        return float(taus.min())

    def _objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Penalised weighted NLL divided by the total weight, and its gradient."""
        des = self._des
        n = des.n_teams
        c, h, rho, a, d, loglam, logmu = self._rates(theta)
        lam, mu = np.exp(loglam), np.exp(logmu)
        log_tau, d_ll, d_lm, d_r = self._tau_terms(lam, mu, rho)

        w = des.w
        ll = log_tau + des.x * loglam - lam + des.y * logmu - mu + des.const
        gl = w * (des.x - lam + d_ll)  # d ll / d log lambda
        gm = w * (des.y - mu + d_lm)  # d ll / d log mu

        ra = a - des.prior_mean_att
        rd = d - des.prior_mean_def
        penalty = 0.5 * np.dot(des.prior_prec, ra * ra + rd * rd)
        f = -np.dot(w, ll) + penalty

        g_a = -(np.bincount(des.hi, gl, n) + np.bincount(des.ai, gm, n)) + des.prior_prec * ra
        g_d = -(np.bincount(des.ai, gl, n) + np.bincount(des.hi, gm, n)) + des.prior_prec * rd
        grad = np.empty_like(theta)
        grad[0] = -(gl.sum() + gm.sum())
        grad[1] = -gl.sum()
        grad[2] = -np.dot(w, d_r)
        grad[3 : 3 + n - 1] = g_a[:-1] - g_a[-1]  # chain rule through sum-to-zero
        grad[3 + n - 1 :] = g_d[:-1] - g_d[-1]
        return f / self._wsum, grad / self._wsum

    # ---------------------------------------------------------- predict
    def team_params(self, team: str) -> tuple[float, float]:
        """Return (attack, defence) for a team.

        Teams passed in ``new_teams`` without any history get the promoted-team
        prior mean. Unknown teams raise ``KeyError`` (catches typos).
        """
        self._check_fitted()
        if team in self.team_index_:
            return float(self.attack_[team]), float(self.defence_[team])
        if team in self.new_teams_:
            return self.promoted_offsets_
        raise KeyError(
            f"Unknown team {team!r}. Pass it in fit(new_teams=[...]) if it is newly promoted."
        )

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Return (lambda, mu): Poisson means for home and away goals."""
        ah, dh = self.team_params(home)
        aa, da = self.team_params(away)
        h_adv = self.team_home_adv_.get(home, self.home_adv_) if hasattr(self, "team_home_adv_") else self.home_adv_
        lam = np.exp(self.intercept_ + ah + da + h_adv)
        mu = np.exp(self.intercept_ + aa + dh)

        if getattr(self, "ewma_weight", 0.0) > 0 and hasattr(self, "ewma_ratings_"):
            r_h = self.ewma_ratings_.get(home)
            r_a = self.ewma_ratings_.get(away)
            if r_h is not None and r_a is not None:
                ewma_lam = self.ewma_weight * (np.log(r_h.attack_rating) + np.log(r_a.defence_rating))
                ewma_mu = self.ewma_weight * (np.log(r_a.attack_rating) + np.log(r_h.defence_rating))
                lam *= np.exp(ewma_lam)
                mu *= np.exp(ewma_mu)

        return float(lam), float(mu)


    def score_matrix(
        self,
        home: str,
        away: str,
        method: str = "dixon_coles",
        copula_theta: float | None = None,
    ) -> np.ndarray:
        """Scoreline probability matrix, shape (max_goals+1, max_goals+1), sums to 1.

        Args:
            home: home team name.
            away: away team name.
            method: 'dixon_coles' (default) or 'copula' (Frank copula bivariate distribution).
            copula_theta: dependence parameter when method='copula' (default 0.25).
        """
        lam, mu = self.expected_goals(home, away)
        if method == "copula":
            theta = 0.25 if copula_theta is None else copula_theta
            return copula_score_matrix(lam, mu, theta=theta, max_goals=self.max_goals)
        return dc_score_matrix(lam, mu, self.rho_, self.max_goals)

    def predict_1x2(self, home: str, away: str) -> tuple[float, float, float]:
        """Return (P(home win), P(draw), P(away win))."""
        return outcome_probs(self.score_matrix(home, away))

    def params(self) -> pd.DataFrame:
        """Per-team attack/defence table (log scale, each summing to zero)."""
        self._check_fitted()
        df = pd.concat([self.attack_, self.defence_], axis=1)
        df["promoted"] = df.index.isin(self.promoted_teams_)
        return df

    def _check_fitted(self) -> None:
        if not hasattr(self, "teams_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
