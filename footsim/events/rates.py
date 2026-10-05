"""Event rate models: yellow cards, red cards, corners and fouls (Milestone 3).

Each statistic is modelled per team-match (two rows per match: the home side
and the away side) as a count with log-linear mean::

    log m = c + home_adv * is_home + for_team + against_opponent + referee

- ``for_team``: the team's own tendency (e.g. how many corners it wins).
- ``against_opponent``: what the opponent concedes / provokes.
- ``referee``: referee effect, absent for missing or unknown referees.

The count distribution is negative binomial (NB2, Var = m + alpha m^2) or
Poisson (alpha = 0). Each team-match log-likelihood is weighted by
``exp(-xi * days_before_as_of)`` as in the goals model.

Shrinkage: team and referee effects get zero-mean Gaussian priors (a ridge
penalty). A referee with few matches therefore stays close to the average,
and red cards use much tighter priors than the other stats. Intercept,
home advantage and the dispersion are unpenalised. See docs/decisions.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

#: Default time-decay rate per day for the event models (same as goals).
DEFAULT_EVENT_XI = 0.003

#: Bounds on log(r), where r = 1/alpha is the NB size parameter. At the upper
#: bound (r ~ 3000, alpha ~ 3e-4) the NB is Poisson for practical purposes;
#: statistics that are not overdispersed (or are underdispersed, which NB
#: cannot represent) end up there. A higher bound causes gammaln cancellation
#: that stalls the optimizer.
LOG_R_BOUNDS = (-3.0, 8.0)


@dataclass(frozen=True)
class RateSpec:
    """Configuration of one event-rate model.

    Attributes:
        name: statistic name, used as the key suffix in outputs.
        home_col, away_col: football-data columns for home / away counts.
        family: ``"nb"`` (negative binomial) or ``"poisson"``.
        prior_sd_team: sd of the Gaussian prior on team for/against effects.
        prior_sd_ref: sd of the Gaussian prior on referee effects.
        use_teams, use_home, use_referee: switch feature groups on or off
            (all off gives the league-average baseline).
        xi: optional statistic-specific time-decay rate (None uses DEFAULT_EVENT_XI).
    """

    name: str
    home_col: str
    away_col: str
    family: str = "nb"
    prior_sd_team: float = 0.2
    prior_sd_ref: float = 0.1
    use_teams: bool = True
    use_home: bool = True
    use_referee: bool = True
    xi: float | None = None


#: Default specs. Prior sds were chosen by walk-forward deviance on the
#: 2022-23 and 2023-24 seasons (see footsim.eval.events and decisions.md).
DEFAULT_SPECS: dict[str, RateSpec] = {
    "yellows": RateSpec("yellows", "HY", "AY", "nb", prior_sd_team=0.20, prior_sd_ref=0.08),
    "reds": RateSpec("reds", "HR", "AR", "poisson", prior_sd_team=0.15, prior_sd_ref=0.01, use_home=False, xi=0.0),
    "corners": RateSpec("corners", "HC", "AC", "nb", prior_sd_team=0.20, prior_sd_ref=0.03),
    "fouls": RateSpec("fouls", "HF", "AF", "nb", prior_sd_team=0.10, prior_sd_ref=0.08),
}


def baseline_spec(spec: RateSpec) -> RateSpec:
    """Return the league-average baseline version of a spec (intercept only)."""
    return replace(spec, use_teams=False, use_home=False, use_referee=False)


def to_team_rows(matches: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    """Reshape matches to two rows per match (home side, away side).

    Columns: ``Date, team, opp, is_home, referee, y``. Rows whose count is
    missing are dropped (they are flagged ``events_missing`` in the loader).
    """
    ref = matches["Referee"] if "Referee" in matches else pd.Series(pd.NA, index=matches.index)
    home = pd.DataFrame({
        "Date": pd.to_datetime(matches["Date"]).to_numpy(),
        "team": matches["HomeTeam"].to_numpy(),
        "opp": matches["AwayTeam"].to_numpy(),
        "is_home": 1.0,
        "referee": ref.to_numpy(),
        "y": pd.to_numeric(matches[home_col], errors="coerce").to_numpy(),
    })
    away = pd.DataFrame({
        "Date": pd.to_datetime(matches["Date"]).to_numpy(),
        "team": matches["AwayTeam"].to_numpy(),
        "opp": matches["HomeTeam"].to_numpy(),
        "is_home": 0.0,
        "referee": ref.to_numpy(),
        "y": pd.to_numeric(matches[away_col], errors="coerce").to_numpy(),
    })
    out = pd.concat([home, away], ignore_index=True)
    return out.dropna(subset=["y"]).reset_index(drop=True)


def nb_loglik(y: np.ndarray, m: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Per-observation NB2 log-likelihood (Poisson where ``alpha == 0``).

    ``alpha`` may be a scalar or an array broadcastable to ``y``.
    """
    y = np.asarray(y, dtype=float)
    m = np.clip(np.asarray(m, dtype=float), 1e-12, None)
    a = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape)
    out = y * np.log(m) - m - gammaln(y + 1)
    nb = a > 0
    if nb.any():
        r, yy, mm = 1.0 / a[nb], y[nb], m[nb]
        out[nb] = (gammaln(yy + r) - gammaln(r) - gammaln(yy + 1)
                   + r * np.log(r / (r + mm)) + yy * np.log(mm / (r + mm)))
    return out


class CountRateModel:
    """Weighted, shrunk Poisson / negative binomial model for one statistic.

    Args:
        spec: model configuration (columns, family, priors, features).
        xi: time-decay rate per day.
    """

    def __init__(self, spec: RateSpec, xi: float = DEFAULT_EVENT_XI) -> None:
        if spec.family not in ("nb", "poisson"):
            raise ValueError(f"family must be 'nb' or 'poisson', got {spec.family!r}")
        effective_xi = spec.xi if spec.xi is not None else xi
        if effective_xi < 0:
            raise ValueError("xi must be >= 0")
        self.spec = spec
        self.xi = effective_xi

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        matches: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
        new_teams: Iterable[str] | None = None,
    ) -> "CountRateModel":
        """Fit on matches played strictly before ``as_of``.

        Args:
            matches: frame with ``Date, HomeTeam, AwayTeam, Referee`` and the
                spec's count columns.
            as_of: cut-off date; matches on or after it are discarded first.
                ``None`` uses every match.
            new_teams: teams with no history that may be predicted later
                (they get zero team effects, i.e. the league average).

        Returns:
            self
        """
        t0 = time.perf_counter()
        spec = self.spec
        dates = pd.to_datetime(matches["Date"])
        if as_of is None:
            as_of_ts = dates.max().normalize() + pd.Timedelta(days=1)
        else:
            as_of_ts = pd.Timestamp(as_of).normalize()

        # No lookahead: filter before anything else is computed.
        keep = (dates < as_of_ts).to_numpy()
        cols = ["Date", "HomeTeam", "AwayTeam", spec.home_col, spec.away_col]
        if "Referee" in matches:
            cols.append("Referee")
        rows = to_team_rows(matches.loc[keep, cols], spec.home_col, spec.away_col)
        if rows.empty:
            raise ValueError(f"No {spec.name} data before as_of={as_of_ts.date()}")

        teams = sorted(set(rows["team"]) | set(rows["opp"]))
        tidx = {t: k for k, t in enumerate(teams)}
        refs = sorted(rows["referee"].dropna().unique().tolist())
        ridx = {r: k for k, r in enumerate(refs)}
        n, m = len(teams), len(refs)

        days = (as_of_ts - rows["Date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-self.xi * days)
        y = rows["y"].to_numpy().astype(float)
        self._ti = rows["team"].map(tidx).to_numpy()
        self._oi = rows["opp"].map(tidx).to_numpy()
        # Missing referee -> index m, a slot whose effect is fixed at 0.
        self._ri = rows["referee"].map(lambda r: ridx.get(r, m) if pd.notna(r) else m).to_numpy()
        self._home = rows["is_home"].to_numpy()
        self._y, self._w, self._n, self._m = y, w, n, m
        self._wsum = float(w.sum())
        self._const = -gammaln(y + 1)
        self._prec_team = spec.prior_sd_team ** -2
        self._prec_ref = spec.prior_sd_ref ** -2

        nb = spec.family == "nb"
        k = 2 + int(nb) + 2 * n + m
        x0 = np.zeros(k)
        x0[0] = np.log(max(np.average(y, weights=w), 1e-6))
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * k
        if not spec.use_home:
            bounds[1] = (0.0, 0.0)
        if nb:
            x0[2] = np.log(10.0)
            bounds[2] = LOG_R_BOUNDS
        off = 2 + int(nb)
        if not spec.use_teams:
            bounds[off : off + 2 * n] = [(0.0, 0.0)] * (2 * n)
        if not spec.use_referee:
            bounds[off + 2 * n :] = [(0.0, 0.0)] * m

        res = minimize(
            self._objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 5000, "maxcor": 20},
        )
        c, h, log_r, f, g, r = self._unpack(res.x)

        self.teams_ = teams
        self.team_index_ = tidx
        self.referees_ = refs
        self.intercept_ = float(c)
        self.home_adv_ = float(h)
        self.alpha_ = float(np.exp(-log_r)) if nb else 0.0
        self.for_ = pd.Series(f, index=teams, name="for")
        self.against_ = pd.Series(g, index=teams, name="against")
        self.referee_ = pd.Series(r, index=refs, name="referee", dtype=float)
        self.referee_matches_ = rows.groupby("referee").size().reindex(refs).fillna(0).astype(int) // 2
        self.new_teams_ = sorted(set(new_teams or []) - set(teams))
        self._for_dict = dict(zip(teams, f.astype(float)))
        self._against_dict = dict(zip(teams, g.astype(float)))
        self._referee_dict = dict(zip(refs, r.astype(float)))
        self.as_of_ = as_of_ts
        self.n_obs_ = len(rows)
        self.converged_ = bool(res.success)
        self.opt_result_ = res
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    # -------------------------------------------------------- likelihood
    def _unpack(self, theta: np.ndarray):
        n, m = self._n, self._m
        nb = self.spec.family == "nb"
        c, h = theta[0], theta[1]
        log_r = theta[2] if nb else np.inf
        off = 2 + int(nb)
        f = theta[off : off + n]
        g = theta[off + n : off + 2 * n]
        r = theta[off + 2 * n : off + 2 * n + m]
        return c, h, log_r, f, g, r

    def _eta(self, theta: np.ndarray) -> np.ndarray:
        c, h, _, f, g, r = self._unpack(theta)
        r_full = np.append(r, 0.0)
        return c + h * self._home + f[self._ti] + g[self._oi] + r_full[self._ri]

    def _objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Penalised weighted NLL (divided by total weight) and its gradient."""
        n, m = self._n, self._m
        nb = self.spec.family == "nb"
        _, _, log_r, f, g, r = self._unpack(theta)
        eta = self._eta(theta)
        mu = np.exp(eta)
        y, w = self._y, self._w

        if nb:
            rr = np.exp(log_r)
            ll = (gammaln(y + rr) - gammaln(rr) + self._const
                  + rr * (log_r - np.log(rr + mu)) + y * (eta - np.log(rr + mu)))
            ge = w * rr * (y - mu) / (rr + mu)  # d ll / d eta
            g_logr = np.dot(w, rr * (digamma(y + rr) - digamma(rr) + log_r
                                     - np.log(rr + mu) + 1.0 - (rr + y) / (rr + mu)))
        else:
            ll = y * eta - mu + self._const
            ge = w * (y - mu)
            g_logr = 0.0

        penalty = 0.5 * (self._prec_team * (f @ f + g @ g) + self._prec_ref * (r @ r))
        obj = -np.dot(w, ll) + penalty

        grad = np.empty_like(theta)
        grad[0] = -ge.sum()
        grad[1] = -np.dot(ge, self._home)
        off = 2
        if nb:
            grad[2] = -g_logr
            off = 3
        grad[off : off + n] = -np.bincount(self._ti, ge, n) + self._prec_team * f
        grad[off + n : off + 2 * n] = -np.bincount(self._oi, ge, n) + self._prec_team * g
        grad[off + 2 * n :] = -np.bincount(self._ri, ge, m + 1)[:m] + self._prec_ref * r
        return obj / self._wsum, grad / self._wsum

    # ---------------------------------------------------------- predict
    def _team_effects(self, team: str) -> tuple[float, float]:
        self._check_fitted()
        if team in self._for_dict:
            return self._for_dict[team], self._against_dict[team]
        if team in self.new_teams_:
            return 0.0, 0.0
        raise KeyError(
            f"Unknown team {team!r}. Pass it in fit(new_teams=[...]) if it is newly promoted."
        )

    def referee_effect(self, referee: str | None) -> float:
        """Referee log-rate effect; 0 for ``None`` or referees never seen."""
        self._check_fitted()
        if referee is None or (isinstance(referee, float) and np.isnan(referee)) or pd.isna(referee):
            return 0.0
        return self._referee_dict.get(str(referee).strip(), 0.0)

    def expected(self, home: str, away: str, referee: str | None = None) -> tuple[float, float]:
        """Return (expected count for the home team, expected count for the away team)."""
        fh, gh = self._team_effects(home)
        fa, ga = self._team_effects(away)
        ref = self.referee_effect(referee)
        mh = np.exp(self.intercept_ + self.home_adv_ + fh + ga + ref)
        ma = np.exp(self.intercept_ + fa + gh + ref)
        return float(mh), float(ma)

    def _check_fitted(self) -> None:
        if not hasattr(self, "teams_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")


class EventRates:
    """Expected yellow cards, red cards, corners and fouls for a fixture.

    One :class:`CountRateModel` per statistic, all fitted on the same data
    window. Example::

        ev = EventRates().fit(matches_df, as_of="2025-01-11")
        ev.expected("Arsenal", "Chelsea", referee="M Oliver")
        # {"home_yellows": ..., "away_yellows": ..., "home_reds": ..., ...}

    Args:
        xi: time-decay rate per day shared by all statistics.
        specs: mapping name -> :class:`RateSpec`; defaults to DEFAULT_SPECS.
    """

    def __init__(self, xi: float = DEFAULT_EVENT_XI, specs: Mapping[str, RateSpec] | None = None) -> None:
        self.xi = xi
        self.specs = dict(DEFAULT_SPECS if specs is None else specs)

    def fit(
        self,
        matches: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
        new_teams: Iterable[str] | None = None,
    ) -> "EventRates":
        """Fit every statistic on matches strictly before ``as_of``."""
        nt = list(new_teams or [])
        self.models_ = {
            name: CountRateModel(spec, xi=self.xi).fit(matches, as_of=as_of, new_teams=nt)
            for name, spec in self.specs.items()
        }
        return self

    def expected(self, home: str, away: str, referee: str | None = None) -> dict[str, float]:
        """Expected counts per team, keyed ``home_<stat>`` and ``away_<stat>``."""
        self._check_fitted()
        out: dict[str, float] = {}
        for name, model in self.models_.items():
            mh, ma = model.expected(home, away, referee)
            out[f"home_{name}"] = mh
            out[f"away_{name}"] = ma
        return out

    def dispersion(self) -> dict[str, float]:
        """NB overdispersion alpha per statistic (Var = m + alpha m^2; 0 = Poisson)."""
        self._check_fitted()
        return {name: model.alpha_ for name, model in self.models_.items()}

    def _check_fitted(self) -> None:
        if not hasattr(self, "models_"):
            raise RuntimeError("EventRates is not fitted. Call fit() first.")
