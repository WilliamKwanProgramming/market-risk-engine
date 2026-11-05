"""Backtests of VaR and ES forecasts.

Sign conventions as in :mod:`mre.measures`: realised P&L ``X_t`` (gain positive), ``VaR_t`` and
``ES_t`` positive loss numbers. An **exception** on day t is ``X_t < -VaR_t`` (the loss strictly
exceeds VaR).

Contents
--------
* :func:`kupiec_pof` - unconditional coverage (Kupiec 1995).
* :func:`christoffersen` - independence and conditional coverage (Christoffersen 1998).
* :func:`traffic_light`, :func:`zone_occupancy`, :func:`binomial_zone_probabilities` - Basel
  traffic light for 99% VaR over 250 days (Basel Committee 1996, Table 2).
* :func:`acerbi_szekely` - ES tests Z1 and Z2 (Acerbi and Szekely 2014) with Monte Carlo
  p-values under the model's own predictive distributions.
* :func:`fz0_loss` and :func:`diebold_mariano` - joint VaR/ES scoring (Fissler and Ziegel 2016,
  FZ0 form of Patton, Ziegel and Chen 2019) and the Diebold-Mariano (1995) comparison.
* :func:`wilson_interval` (Wilson 1927) and :func:`holm` (Holm 1979) - an interval for an
  exception rate and the step-down multiple-testing adjustment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.special import chdtrc, ndtr, xlogy

from mre.measures import Predictive

__all__ = [
    "TestResult",
    "exceptions",
    "kupiec_pof",
    "christoffersen",
    "TrafficLight",
    "traffic_light",
    "zone_occupancy",
    "binomial_zone_probabilities",
    "BASEL_PLUS_FACTOR",
    "AcerbiSzekelySimulation",
    "acerbi_szekely",
    "fz0_loss",
    "diebold_mariano",
    "newey_west_lag",
    "wilson_interval",
    "holm",
]


@dataclass(frozen=True)
class TestResult:
    """A likelihood-ratio or similar test: statistic, p-value, degrees of freedom, details."""

    statistic: float
    p_value: float
    df: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def reject(self, size: float = 0.05) -> bool:
        """True if the p-value is below ``size`` (False when the test is undefined)."""
        return bool(np.isfinite(self.p_value) and self.p_value < size)


def exceptions(realized: ArrayLike, var: ArrayLike) -> NDArray[np.bool_]:
    """Exception indicator ``X_t < -VaR_t``."""
    return np.asarray(realized, dtype=float) < -np.asarray(var, dtype=float)


def _binom_loglik(x: float, n: float, p: float) -> float:
    """``x log p + (n - x) log(1 - p)`` with the convention ``0 log 0 = 0``."""
    return float(xlogy(x, p) + xlogy(n - x, 1.0 - p))


def kupiec_pof(exc: ArrayLike, level: float = 0.99) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    With ``n`` days, ``x`` exceptions and ``p = 1 - level``::

        LR_uc = -2 [ x log p + (n-x) log(1-p) - x log(x/n) - (n-x) log(1 - x/n) ]  ~ chi2(1)

    using ``0 log 0 = 0``, so ``x = 0`` gives ``LR_uc = -2 n log(1 - p)``.
    """
    e = np.asarray(exc, dtype=bool)
    n, x = e.size, int(e.sum())
    if n == 0:
        return TestResult(np.nan, np.nan, 1, {"n": 0, "x": 0})
    p = 1.0 - level
    lr = -2.0 * (_binom_loglik(x, n, p) - _binom_loglik(x, n, x / n))
    lr = max(lr, 0.0)
    return TestResult(lr, float(chdtrc(1, lr)), 1,
                      {"n": n, "x": x, "rate": x / n, "expected": n * p})


def christoffersen(exc: ArrayLike, level: float = 0.99) -> dict[str, TestResult]:
    """Christoffersen (1998) independence and conditional-coverage LR tests.

    Transition counts over the ``n - 1`` consecutive pairs ``(I_{t-1}, I_t)``: ``n_ij`` is the
    number of days with ``I_{t-1} = i`` and ``I_t = j``. With ``pi_01 = n01/(n00+n01)``,
    ``pi_11 = n11/(n10+n11)``, ``pi = (n01+n11)/(n-1)`` and ``p = 1 - level``::

        log L(Pi) = n00 log(1-pi_01) + n01 log pi_01 + n10 log(1-pi_11) + n11 log pi_11
        LR_ind = -2 [ (n00+n10) log(1-pi) + (n01+n11) log pi - log L(Pi) ]        ~ chi2(1)
        LR_cc  = -2 [ (n00+n10) log(1-p)  + (n01+n11) log p  - log L(Pi) ]        ~ chi2(2)

    so ``LR_cc = LR_uc + LR_ind`` exactly when ``LR_uc`` is computed on the same ``n - 1``
    transitions (the first observation is conditioned on, as in the paper).

    Degenerate cases (``0 log 0 = 0`` throughout): with fewer than two observations both tests
    are undefined (NaN). If no exception occurs among the first ``n - 1`` days, ``pi_11`` is not
    identified, ``pi_01 = pi`` and ``LR_ind = 0`` (p-value 1); the flag ``ind_testable`` is then
    False. Zero exceptions overall give ``LR_ind = 0`` and ``LR_cc = -2 (n-1) log(1-p)``.
    """
    e = np.asarray(exc, dtype=bool).astype(int)
    if e.size < 2:
        nan = TestResult(np.nan, np.nan, None, {"reason": "fewer than two observations"})
        return {"ind": nan, "cc": nan}
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    m = n00 + n01 + n10 + n11
    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi = (n01 + n11) / m
    ll_markov = _binom_loglik(n01, n00 + n01, pi01) + _binom_loglik(n11, n10 + n11, pi11)
    ll_iid = _binom_loglik(n01 + n11, m, pi)
    ll_null = _binom_loglik(n01 + n11, m, 1.0 - level)
    lr_ind = max(-2.0 * (ll_iid - ll_markov), 0.0)
    lr_cc = max(-2.0 * (ll_null - ll_markov), 0.0)
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11,
              "pi": pi, "ind_testable": n10 + n11 > 0}
    return {
        "ind": TestResult(lr_ind, float(chdtrc(1, lr_ind)), 1, counts),
        "cc": TestResult(lr_cc, float(chdtrc(2, lr_cc)), 2, counts),
    }


BASEL_PLUS_FACTOR: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.40, 6: 0.50,
                                       7: 0.65, 8: 0.75, 9: 0.85}
"""Increase in the scaling factor by number of exceptions in 250 days (Basel Committee 1996,
Table 2); 10 or more exceptions give 1.00."""


@dataclass(frozen=True)
class TrafficLight:
    """Basel zone for a count of 99% VaR exceptions in 250 days and its plus factor."""

    exceptions: int
    zone: str
    plus_factor: float


def traffic_light(n_exceptions: int) -> TrafficLight:
    """Green for 0-4 exceptions, yellow for 5-9, red for 10 or more (Basel Committee 1996).

    The zones and plus factors are defined for 250 daily observations of 99% VaR.
    """
    k = int(n_exceptions)
    if k < 0:
        raise ValueError("number of exceptions must be non-negative")
    if k <= 4:
        zone = "green"
    elif k <= 9:
        zone = "yellow"
    else:
        zone = "red"
    return TrafficLight(k, zone, BASEL_PLUS_FACTOR.get(k, 1.0))


def zone_occupancy(exc: ArrayLike, window: int = 250) -> tuple[pd.Series, dict[str, float]]:
    """Rolling Basel zone: exception count over each window of ``window`` days ending on day t.

    Returns the count series (one value per complete window, i.e. from day ``window - 1`` on) and
    the fraction of those windows that are green, yellow and red.
    """
    e = np.asarray(exc, dtype=int)
    if e.size < window:
        raise ValueError("series shorter than one window")
    c = np.convolve(e, np.ones(window, dtype=int), mode="valid")
    zones = np.array([traffic_light(k).zone for k in c])
    frac = {z: float(np.mean(zones == z)) for z in ("green", "yellow", "red")}
    return pd.Series(c, index=np.arange(window - 1, e.size)), frac


def binomial_zone_probabilities(n: int = 250, p: float = 0.01) -> dict[str, float]:
    """Exact probabilities of the green, yellow and red zones when exceptions are iid
    Bernoulli(``p``) over ``n`` days: ``P(X <= 4)``, ``P(5 <= X <= 9)``, ``P(X >= 10)``."""
    cdf4 = float(stats.binom.cdf(4, n, p))
    cdf9 = float(stats.binom.cdf(9, n, p))
    return {"green": cdf4, "yellow": cdf9 - cdf4, "red": 1.0 - cdf9}


class AcerbiSzekelySimulation:
    """Acerbi-Szekely (2014) Z1 and Z2 with Monte Carlo p-values, accumulated day by day.

    Feed one day at a time with :meth:`add`; for each day ``n_sim`` P&L values are drawn from the
    model's predictive distribution and only the simulated tail events ``X + VaR < 0`` are kept
    (day, simulation number and ``X / ES``). Memory is proportional to ``n_sim`` plus the number
    of simulated tail events (about ``T n_sim p``), never ``T x n_sim``. :meth:`result` computes
    the statistics on all days or on a subset of them, so a common evaluation sample can be
    imposed after the forecasts are made. Draws are taken from one generator seeded with
    ``seed`` in the order the days are added, so :func:`acerbi_szekely` gives the same numbers as
    a version that draws the whole ``(T, n_sim)`` array at once.
    """

    def __init__(self, level: float = 0.975, n_sim: int = 2000, seed: int = 0):
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        self.level, self.n_sim = level, int(n_sim)
        self._rng = np.random.default_rng(seed)
        self._obs: list[tuple[int, float, float, float]] = []
        self._sim_day: list[NDArray[np.int64]] = []
        self._sim_idx: list[NDArray[np.int64]] = []
        self._sim_ratio: list[NDArray[np.float64]] = []

    def add(self, day: int, realized: float, predictive: Predictive) -> None:
        """Record day ``day`` (any integer label) with realised P&L and its predictive."""
        var, es = predictive.var(self.level), predictive.es(self.level)
        x = predictive.sample(self._rng, self.n_sim)
        hit = np.flatnonzero(x + var < 0.0)
        self._obs.append((int(day), float(realized), float(var), float(es)))
        self._sim_day.append(np.full(hit.size, int(day)))
        self._sim_idx.append(hit)
        self._sim_ratio.append(x[hit] / es)

    @property
    def days(self) -> NDArray[np.int64]:
        """Labels of the days added so far, in order."""
        return np.array([o[0] for o in self._obs], dtype=np.int64)

    def result(self, days: ArrayLike | None = None) -> dict[str, TestResult]:
        """Z1 and Z2 on the given day labels (all days if ``None``); formulas in
        :func:`acerbi_szekely`."""
        obs = np.array(self._obs, dtype=float).reshape(-1, 4)
        keep = np.ones(len(obs), bool) if days is None else np.isin(obs[:, 0], days)
        x, var, es = obs[keep, 1], obs[keep, 2], obs[keep, 3]
        T, p, M = int(keep.sum()), 1.0 - self.level, self.n_sim
        if T == 0:
            raise ValueError("no days selected")
        hit = x + var < 0.0
        n_exc = int(hit.sum())
        ratio_obs = float(np.sum(x[hit] / es[hit]))
        z1_obs = ratio_obs / n_exc + 1.0 if n_exc > 0 else 0.0
        z2_obs = ratio_obs / (T * p) + 1.0
        sd = np.concatenate(self._sim_day) if self._sim_day else np.empty(0, np.int64)
        si = np.concatenate(self._sim_idx) if self._sim_idx else np.empty(0, np.int64)
        sr = np.concatenate(self._sim_ratio) if self._sim_ratio else np.empty(0)
        if days is not None:
            sel = np.isin(sd, obs[keep, 0])
            si, sr = si[sel], sr[sel]
        tot = np.bincount(si, weights=sr, minlength=M)
        cnt = np.bincount(si, minlength=M)
        z1_sim = np.where(cnt > 0, tot / np.maximum(cnt, 1) + 1.0, 0.0)
        z2_sim = tot / (T * p) + 1.0
        p1 = (1.0 + np.sum(z1_sim <= z1_obs)) / (M + 1.0)
        p2 = (1.0 + np.sum(z2_sim <= z2_obs)) / (M + 1.0)
        common = {"n_exceptions": n_exc, "n_sim": M, "level": self.level, "n_days": T}
        return {"Z1": TestResult(z1_obs, float(p1), None, common),
                "Z2": TestResult(z2_obs, float(p2), None, common)}


def acerbi_szekely(realized: ArrayLike, predictives: Sequence[Predictive], level: float = 0.975,
                   n_sim: int = 2000, seed: int = 0) -> dict[str, TestResult]:
    """Acerbi-Szekely (2014) ES backtests Z1 (conditional) and Z2 (unconditional).

    With ``p = 1 - level``, ``VaR_t`` and ``ES_t`` the model's VaR and ES *at the same level*
    (both read from the predictive distribution ``P_t``) and ``I_t = 1{X_t + VaR_t < 0}``::

        Z1 = (1/N) sum_t X_t I_t / ES_t + 1,    N = sum_t I_t
        Z2 = sum_t X_t I_t / (T p ES_t) + 1

    Under the null the expected value of each is 0 (Z1 conditionally on ``N > 0``); realised
    losses beyond what the model predicts make them negative. Convention: ``Z1 = 0`` when
    ``N = 0``.

    p-values are one-sided Monte Carlo p-values under the model's own predictive distributions:
    ``n_sim`` paths are simulated with ``X_t^(m)`` drawn independently from ``P_t`` for every
    day t (``Predictive.sample``), the statistics are recomputed with the same ``VaR_t`` and
    ``ES_t``, and ``p = (1 + #{Z^(m) <= Z_obs}) / (n_sim + 1)``. The simulation is accumulated
    day by day (:class:`AcerbiSzekelySimulation`), so memory does not grow with ``T x n_sim``.
    """
    x = np.asarray(realized, dtype=float)
    if len(predictives) != x.size:
        raise ValueError("one predictive distribution per realised value is needed")
    sim = AcerbiSzekelySimulation(level, n_sim, seed)
    for t, (xt, d) in enumerate(zip(x, predictives, strict=True)):
        sim.add(t, float(xt), d)
    return sim.result()


def fz0_loss(realized: ArrayLike, var: ArrayLike, es: ArrayLike, level: float) -> NDArray:
    """FZ0 joint loss for (VaR, ES) at the same confidence ``level``.

    Patton, Ziegel and Chen (2019), eq. (6), a member of the Fissler and Ziegel (2016) class. In
    their notation ``v = -VaR`` and ``e = -ES`` are the (negative) lower-tail quantile and ES of
    the return ``Y`` at tail probability ``alpha = 1 - level``::

        L(Y, v, e) = -(1/(alpha e)) 1{Y <= v} (v - Y) + v/e + log(-e) - 1

    Lower is better; requires ``ES > 0``. The expected loss is minimised by the true (VaR, ES).
    """
    y = np.asarray(realized, dtype=float)
    v = -np.asarray(var, dtype=float)
    e = -np.asarray(es, dtype=float)
    if np.any(e >= 0.0):
        raise ValueError("FZ0 needs strictly positive ES")
    a = 1.0 - level
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (a * e) + v / e + np.log(-e) - 1.0


def newey_west_lag(n: int) -> int:
    """Lag truncation ``floor(4 (n/100)^(2/9))`` used by :func:`diebold_mariano` by default."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(loss_a: ArrayLike, loss_b: ArrayLike, lag: int | None = None) -> TestResult:
    """Diebold-Mariano (1995) test of equal expected loss, ``d_t = loss_a - loss_b``.

    ``DM = mean(d) / sqrt(S / n)`` with ``S`` the Newey-West (1987) long-run variance of ``d``
    with Bartlett weights ``1 - j/(L+1)``, ``j = 1..L``; ``L`` defaults to
    :func:`newey_west_lag`. Two-sided p-value from the standard normal. A negative statistic
    means model A has the lower average loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    L = newey_west_lag(n) if lag is None else int(lag)
    u = d - d.mean()
    s = float(u @ u) / n
    for j in range(1, L + 1):
        s += 2.0 * (1.0 - j / (L + 1.0)) * float(u[j:] @ u[:-j]) / n
    if not s > 0.0:
        return TestResult(np.nan, np.nan, None, {"lag": L, "mean_diff": float(d.mean())})
    stat = float(d.mean() / np.sqrt(s / n))
    return TestResult(stat, float(2.0 * ndtr(-abs(stat))), None,
                      {"lag": L, "mean_diff": float(d.mean()), "n": n})


def wilson_interval(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``x / n`` (Wilson 1927)::

        (phat + z^2/(2n) -/+ z sqrt(phat (1 - phat)/n + z^2/(4 n^2))) / (1 + z^2/n)

    with ``z`` the ``(1 + conf)/2`` standard normal quantile.
    """
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("need 0 <= x <= n and n > 0")
    z = float(stats.norm.ppf(0.5 + conf / 2.0))
    ph = x / n
    centre = ph + z * z / (2.0 * n)
    half = z * np.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n))
    den = 1.0 + z * z / n
    return float((centre - half) / den), float((centre + half) / den)


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm (1979) step-down adjusted p-values, in the input order.

    With the ``m`` finite p-values sorted ``p_(1) <= ... <= p_(m)``:
    ``adj_(k) = max_{j <= k} min(1, (m - j + 1) p_(j))``. NaN entries (undefined tests) are
    not counted in ``m`` and stay NaN. Rejecting where ``adj < a`` controls the family-wise error
    rate at ``a``.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out
