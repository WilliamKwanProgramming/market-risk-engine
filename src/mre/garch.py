"""GARCH(1,1) with standardised Student t innovations: simulation, filtering and maximum likelihood.

Model (Bollerslev 1986 for GARCH; Bollerslev 1987 for the Student t innovations)::

    r_t = mu + eps_t,   eps_t = sigma_t z_t,   z_t iid Student t(nu) scaled to unit variance,
    sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2,

with ``omega > 0``, ``alpha, beta >= 0``, ``alpha + beta < 1`` and ``nu > 2``.

**Start of the recursion (backcast).** The variance recursion needs a value for
``eps_0^2`` and ``sigma_0^2``. Both are set to the backcast ``b = sum_i w_i eps_i^2`` over the first
``tau = min(75, n)`` residuals with weights ``w_i proportional to 0.94^i`` (the rule used by the
``arch`` package, so that the two estimators maximise the same likelihood). Hence
``sigma_1^2 = omega + (alpha + beta) b``.

**Log-likelihood** of one observation (standardised t density)::

    l_t = log G((nu+1)/2) - log G(nu/2) - 0.5 log(pi (nu-2)) - 0.5 log sigma_t^2
          - (nu+1)/2 * log(1 + eps_t^2 / (sigma_t^2 (nu-2)))

Estimation maximises ``sum_t l_t`` with SLSQP (scipy) under the bounds and the stationarity
constraint ``alpha + beta <= 1 - 1e-6``, on returns divided by their sample standard deviation
(rescaled afterwards). The optimiser's success flag is kept in :class:`GarchFit` and never
silently replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal
from scipy.special import gammaln

__all__ = ["GarchParams", "GarchFit", "backcast", "garch_filter", "garch_loglik", "fit_garch_t",
           "simulate_garch_t"]

_BACKCAST_TAU = 75
_BACKCAST_DECAY = 0.94
_STATIONARITY_MARGIN = 1e-6


@dataclass(frozen=True)
class GarchParams:
    """Parameters of the GARCH(1,1)-t model in the units of the returns."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


@dataclass(frozen=True)
class GarchFit:
    """Result of :func:`fit_garch_t`. ``converged`` is the optimiser's own success flag."""

    params: GarchParams
    loglik: float
    converged: bool
    message: str
    n_iter: int
    n_obs: int


def backcast(resid: ArrayLike) -> float:
    """Exponentially weighted mean of the first ``min(75, n)`` squared residuals (decay 0.94)."""
    e = np.asarray(resid, dtype=float)
    tau = min(_BACKCAST_TAU, e.size)
    w = _BACKCAST_DECAY ** np.arange(tau)
    return float(np.sum(w * e[:tau] ** 2) / w.sum())


def garch_filter(r: ArrayLike, params: GarchParams) -> tuple[NDArray[np.float64], float]:
    """Conditional variances ``sigma_t^2`` for t = 1..n and the one-step forecast ``sigma_{n+1}^2``.

    ``sigma_t^2`` uses only ``r_1..r_{t-1}`` (and the backcast, which is computed from the first
    residuals of the same sample). The recursion is evaluated as the linear filter
    ``sigma_t^2 - beta sigma_{t-1}^2 = omega + alpha eps_{t-1}^2``.
    """
    eps = np.asarray(r, dtype=float) - params.mu
    b = backcast(eps)
    # forcing term x_t = omega + alpha eps_{t-1}^2, with eps_0^2 = b; one extra entry gives the
    # forecast for n + 1
    x = params.omega + params.alpha * np.concatenate(([b], eps**2))
    s2 = signal.lfilter([1.0], [1.0, -params.beta], x, zi=[params.beta * b])[0]
    return s2[:-1], float(s2[-1])


def _loglik_terms(eps: NDArray[np.float64], s2: NDArray[np.float64], nu: float):
    const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    return const - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(eps**2 / (s2 * (nu - 2.0)))


def garch_loglik(r: ArrayLike, params: GarchParams) -> float:
    """Total log-likelihood of ``r`` under ``params`` (formula in the module docstring)."""
    r = np.asarray(r, dtype=float)
    s2, _ = garch_filter(r, params)
    return float(np.sum(_loglik_terms(r - params.mu, s2, params.nu)))


def fit_garch_t(r: ArrayLike, max_iter: int = 400) -> GarchFit:
    """Maximum-likelihood estimate of GARCH(1,1)-t by SLSQP (see module docstring).

    Starting values: the best of a small grid of (alpha, beta) pairs with ``nu = 8``, ``mu`` the
    sample mean and ``omega`` set so that the unconditional variance equals the sample variance.
    Needs at least 50 observations.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if n < 50:
        raise ValueError("GARCH estimation needs at least 50 observations")
    scale = float(np.std(r))
    if not scale > 0.0:
        raise ValueError("returns have zero variance")
    y = r / scale

    def unpack(theta):
        return GarchParams(mu=theta[0], omega=theta[1], alpha=theta[2], beta=theta[3],
                           nu=theta[4])

    def nll(theta):
        s2, _ = garch_filter(y, unpack(theta))
        if np.any(s2 <= 0.0) or not np.all(np.isfinite(s2)):
            return 1e10
        return -float(np.sum(_loglik_terms(y - theta[0], s2, theta[4]))) / n

    ybar = float(np.mean(y))
    starts = []
    for a, b in ((0.05, 0.90), (0.10, 0.85), (0.05, 0.94), (0.15, 0.80), (0.03, 0.96)):
        theta0 = np.array([ybar, max(1.0 - a - b, 1e-3), a, b, 8.0])
        starts.append((nll(theta0), theta0))
    theta0 = min(starts, key=lambda s: s[0])[1]
    bounds = [(-10.0, 10.0), (1e-8, 10.0), (0.0, 1.0), (0.0, 1.0), (2.05, 500.0)]
    cons = [{"type": "ineq", "fun": lambda th: 1.0 - _STATIONARITY_MARGIN - th[2] - th[3]}]
    res = optimize.minimize(nll, theta0, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": max_iter, "ftol": 1e-10})
    th = res.x
    params = GarchParams(mu=th[0] * scale, omega=th[1] * scale**2, alpha=th[2], beta=th[3],
                         nu=th[4])
    return GarchFit(params=params, loglik=garch_loglik(r, params), converged=bool(res.success),
                    message=str(res.message), n_iter=int(res.nit), n_obs=n)


def simulate_garch_t(n: int, params: GarchParams, rng: np.random.Generator,
                     burn: int = 500) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate ``n`` returns and their true conditional variances from GARCH(1,1)-t.

    The recursion starts at the unconditional variance and the first ``burn`` values are
    discarded. Returns ``(r, sigma2)`` where ``sigma2[t]`` is the variance of ``r[t]`` given the
    past, i.e. the true one-step-ahead forecast.
    """
    total = n + burn
    nu = params.nu
    z = rng.standard_t(nu, total) * np.sqrt((nu - 2.0) / nu)
    s2 = np.empty(total)
    eps = np.empty(total)
    prev_s2 = params.unconditional_variance
    prev_e2 = prev_s2
    for t in range(total):
        s2[t] = params.omega + params.alpha * prev_e2 + params.beta * prev_s2
        eps[t] = np.sqrt(s2[t]) * z[t]
        prev_s2, prev_e2 = s2[t], eps[t] ** 2
    return params.mu + eps[burn:], s2[burn:]
