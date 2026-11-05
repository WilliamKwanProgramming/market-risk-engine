"""One-day VaR and ES models and the rolling driver that enforces the information convention.

**Information convention.** Rows of the factor-move matrix ``X`` are trading days; row ``t`` holds
the moves from the close of day ``t-1`` to the close of day ``t``. The forecast *for* day ``t``
(the distribution of the P&L realised over row ``t``) is made at the close of day ``t-1`` and uses
rows ``0..t-1`` only. :func:`rolling_forecasts` is the only place that slices the history
(``X[:t]``); models receive the slice and never see row ``t``. A refit scheduled for day ``t``
also uses ``X[:t]`` only.

**Output.** Every model returns a predictive P&L distribution (:mod:`mre.measures`); VaR (default
99%) and ES (default 97.5%) are read off it as positive loss numbers.

Models
------
* :class:`HistoricalSimulation` - the last ``window`` moves applied to today's portfolio.
* :class:`EWMAParametric` - RiskMetrics EWMA covariance, delta-normal (or delta-t) P&L.
* :class:`FilteredHistoricalSimulation` - GARCH(1,1)-t per factor, standardised residuals
  rescaled by the one-step volatility forecast (Barone-Adesi, Giannopoulos and Vosper 1999).
* :class:`MonteCarlo` - multivariate normal or Student t moves with the EWMA covariance,
  antithetic pairs, full revaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from mre.garch import GarchFit, fit_garch_t, garch_filter
from mre.measures import EmpiricalPnL, LocationScalePnL, Predictive
from mre.portfolio import LinearPnL

__all__ = [
    "PnLFunction",
    "RiskForecast",
    "RiskModel",
    "HistoricalSimulation",
    "EWMAParametric",
    "FilteredHistoricalSimulation",
    "MonteCarlo",
    "ewma_covariance",
    "ForecastPath",
    "rolling_forecasts",
]


class PnLFunction(Protocol):
    """Maps an ``(n, k)`` array of factor moves to ``n`` P&L values; knows its linear exposure."""

    def __call__(self, moves: ArrayLike) -> NDArray[np.float64]: ...

    def linear_exposure(self) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class RiskForecast:
    """A one-day risk forecast. ``var``, ``es`` and ``var_at_es_level`` are positive loss numbers.

    ``predictive`` is ``None`` when :func:`rolling_forecasts` was asked not to keep it, or when
    the forecast failed (then ``var`` and ``es`` are NaN and ``diagnostics["error"]`` says why).
    """

    date: Any
    method: str
    level_var: float
    level_es: float
    var: float
    es: float
    diagnostics: Mapping[str, Any]
    predictive: Predictive | None
    var_at_es_level: float = float("nan")


class RiskModel(ABC):
    """Common interface: optional ``fit`` on a schedule, then ``predict`` every day.

    ``window`` is the number of most recent rows the model uses; ``refit_every`` is the number of
    days between refits (``None``: the model has nothing to fit).
    """

    name: str = "model"
    window: int = 0
    refit_every: int | None = None

    def fit(self, history: NDArray[np.float64]) -> Any:
        """Estimate parameters from ``history`` (rows up to t-1). Default: nothing to fit."""
        return None

    @abstractmethod
    def predict(self, history: NDArray[np.float64], fitted: Any, pnl: PnLFunction,
                index: int) -> tuple[Predictive, dict[str, Any]]:
        """Predictive P&L distribution for the day after the last row of ``history``.

        ``index`` is the row number of the forecast day (used only to seed random draws).
        """


class HistoricalSimulation(RiskModel):
    """Historical simulation: the empirical distribution of P&L under the last ``window`` moves.

    Each of the last ``window`` daily factor-move vectors is applied to the current portfolio
    (full revaluation through ``pnl``). VaR and ES are those of the empirical distribution of the
    ``window`` scenarios, with the finite-sample definitions of :mod:`mre.measures` (for 250 days:
    99% VaR = 3rd worst loss; 97.5% ES = (sum of 6 worst + 0.25 x 7th worst) / 6.25).
    """

    def __init__(self, window: int = 500):
        if window < 40:
            raise ValueError("window must be at least 40 days for a 97.5% ES")
        self.window = window
        self.name = f"HS{window}"

    def predict(self, history, fitted, pnl, index):
        scen = history[-self.window:]
        if scen.shape[0] < self.window:
            raise ValueError(f"{self.name} needs {self.window} rows, got {scen.shape[0]}")
        return EmpiricalPnL(pnl(scen)), {"n_scenarios": scen.shape[0]}


def ewma_covariance(history: ArrayLike, lam: float = 0.94,
                    window: int | None = None) -> NDArray[np.float64]:
    """RiskMetrics exponentially weighted covariance with zero mean (J.P. Morgan/Reuters 1996).

    ``Sigma = sum_{i=0}^{n-1} w_i x_{t-1-i} x_{t-1-i}'`` with ``w_i = (1 - lam) lam^i / (1 -
    lam^n)``, i.e. the infinite RiskMetrics sum truncated to the last ``n`` rows (``n = window``
    or all rows) and renormalised so the weights sum to 1. Row ``x_{t-1}`` (the most recent)
    gets the largest weight ``(1 - lam) / (1 - lam^n)``.
    """
    x = np.atleast_2d(np.asarray(history, dtype=float))
    if x.shape[0] == 1 and np.ndim(history) == 1:
        x = x.T
    if window is not None:
        x = x[-window:]
    n = x.shape[0]
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    i = np.arange(n)[::-1]  # 0 for the most recent row
    w = (1.0 - lam) * lam**i / (1.0 - lam**n)
    return (x * w[:, None]).T @ x


class EWMAParametric(RiskModel):
    """EWMA (RiskMetrics) covariance with a normal or Student t distribution of P&L.

    The P&L is approximated linearly in the factor moves with the exposure ``e`` of the
    portfolio (delta-normal; for linear positions in returns this is exact):
    ``sigma_P^2 = e' Sigma e`` with ``Sigma`` from :func:`ewma_covariance` and zero mean. With
    ``dist = 'normal'``: ``VaR = z_a sigma_P`` and ``ES = phi(z_a)/(1-a) sigma_P``. With
    ``dist = 't'``: the unit-variance Student t(``nu``) quantile and ES replace the normal ones
    (``nu`` is fixed by the user, not estimated).
    """

    def __init__(self, lam: float = 0.94, dist: str = "normal", nu: float = 6.0,
                 window: int = 500):
        self.lam, self.dist, self.nu, self.window = lam, dist, nu, window
        self.name = "EWMA-N" if dist == "normal" else f"EWMA-t{nu:g}"

    def predict(self, history, fitted, pnl, index):
        cov = ewma_covariance(history, self.lam, self.window)
        e = pnl.linear_exposure()
        sd = float(np.sqrt(e @ cov @ e))
        return (LocationScalePnL(0.0, sd, self.dist, self.nu),
                {"sigma": sd, "n_obs": min(self.window, history.shape[0])})


class FilteredHistoricalSimulation(RiskModel):
    """Filtered historical simulation with GARCH(1,1)-t volatility per factor.

    For each factor ``j``, GARCH(1,1)-t is estimated by :func:`mre.garch.fit_garch_t` on the last
    ``window`` rows, every ``refit_every`` days. Each day the variance filter is run over the
    last ``window`` rows with the current parameters, giving standardised residuals
    ``z_{s,j} = (x_{s,j} - mu_j) / sigma_{s,j}`` and the one-step forecast ``sigma_{t,j}``. The
    scenarios are ``x*_{s,j} = mu_j + sigma_{t,j} z_{s,j}``, one per residual date ``s`` (rows
    kept together, so cross-sectional dependence of the residuals is preserved).

    With ``n_boot = None`` (default) the scenarios are all ``window`` residual dates with equal
    weights: at a one-day horizon the bootstrap distribution of one resampled residual vector is
    the empirical distribution of the residuals itself, so it is enumerated exactly. With an
    integer ``n_boot`` the scenarios are ``n_boot`` residual dates drawn with replacement
    (residual bootstrap) by ``numpy.random.default_rng([seed, t])`` for forecast row ``t``, so a
    forecast does not depend on which other days were computed.

    Non-convergence of an estimate is reported in the diagnostics (``converged``) and the
    estimate is used as returned by the optimiser.
    """

    def __init__(self, window: int = 1000, refit_every: int = 250, n_boot: int | None = None,
                 seed: int = 0):
        self.window, self.refit_every = window, refit_every
        self.n_boot, self.seed = n_boot, seed
        self.name = f"FHS{window}"

    def fit(self, history):
        h = history[-self.window:]
        return [fit_garch_t(h[:, j]) for j in range(h.shape[1])]

    def predict(self, history, fitted: list[GarchFit], pnl, index):
        h = history[-self.window:]
        scen = np.empty_like(h)
        sig_next = np.empty(h.shape[1])
        for j, fit in enumerate(fitted):
            s2, s2_next = garch_filter(h[:, j], fit.params)
            z = (h[:, j] - fit.params.mu) / np.sqrt(s2)
            sig_next[j] = np.sqrt(s2_next)
            scen[:, j] = fit.params.mu + sig_next[j] * z
        if self.n_boot is not None:
            rng = np.random.default_rng([self.seed, index])
            scen = scen[rng.integers(0, scen.shape[0], size=self.n_boot)]
        diag = {"converged": all(f.converged for f in fitted), "sigma_next": sig_next,
                "params": [f.params for f in fitted]}
        return EmpiricalPnL(pnl(scen)), diag


class MonteCarlo(RiskModel):
    """Monte Carlo with EWMA covariance, normal or Student t moves, antithetic pairs.

    ``Sigma = L L'`` (Cholesky). Draws: ``Z`` of shape ``(n/2, k)`` iid standard normal; for
    ``dist = 'normal'`` the moves are ``X = Z L'``; for ``dist = 't'`` they are
    ``X = sqrt((nu - 2)/W) Z L'`` with ``W ~ chi2(nu)`` per draw (multivariate t with covariance
    ``Sigma``). The antithetic partner of each draw is ``-X`` (same ``W``). Every scenario is
    fully revalued through ``pnl``. The random generator for forecast day ``t`` is seeded with
    ``(seed, t)``, so a forecast does not depend on which other days were computed.
    """

    def __init__(self, lam: float = 0.94, dist: str = "normal", nu: float = 6.0,
                 n_draws: int = 10_000, seed: int = 0, window: int = 500):
        if n_draws % 2:
            raise ValueError("n_draws must be even (antithetic pairs)")
        self.lam, self.dist, self.nu = lam, dist, nu
        self.n_draws, self.seed, self.window = n_draws, seed, window
        self.name = "MC-N" if dist == "normal" else f"MC-t{nu:g}"

    def draw_moves(self, cov: NDArray[np.float64], index: int) -> NDArray[np.float64]:
        """The ``(n_draws, k)`` antithetic factor moves for forecast day ``index``."""
        rng = np.random.default_rng([self.seed, index])
        L = np.linalg.cholesky(cov)
        half = self.n_draws // 2
        x = rng.standard_normal((half, cov.shape[0])) @ L.T
        if self.dist == "t":
            w = rng.chisquare(self.nu, half)
            x = x * np.sqrt((self.nu - 2.0) / w)[:, None]
        return np.vstack([x, -x])

    def predict(self, history, fitted, pnl, index):
        cov = ewma_covariance(history, self.lam, self.window)
        return EmpiricalPnL(pnl(self.draw_moves(cov, index))), {"n_draws": self.n_draws}


@dataclass
class ForecastPath:
    """Forecasts and realised P&L for consecutive days (output of :func:`rolling_forecasts`)."""

    dates: pd.Index
    forecasts: list[RiskForecast]
    realized: NDArray[np.float64]
    fits: list[Any] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        """One row per day: var, es, var at the ES level, realised P&L and the exception flag."""
        f = self.forecasts
        return pd.DataFrame({
            "var": [x.var for x in f],
            "es": [x.es for x in f],
            "var_at_es_level": [x.var_at_es_level for x in f],
            "realized": self.realized,
            "exception": self.realized < -np.array([x.var for x in f]),
        }, index=self.dates)


def rolling_forecasts(model: RiskModel, moves: pd.DataFrame | pd.Series | ArrayLike,
                      start: int, end: int | None = None,
                      pnl: PnLFunction | Callable[[int], PnLFunction] | None = None,
                      level_var: float = 0.99, level_es: float = 0.975,
                      on_forecast: Callable[[int, RiskForecast, float], None] | None = None,
                      keep_predictive: bool = True, on_error: str = "raise") -> ForecastPath:
    """Forecasts for rows ``start..end-1`` of ``moves``, each using rows before it only.

    ``pnl`` is either one :class:`PnLFunction` used every day, or a callable mapping a row number
    to the P&L function valid for that day (a portfolio whose state changes), or ``None`` for a
    single factor whose move is the P&L (``LinearPnL([1.0])``). The realised P&L of day ``t``
    is ``pnl(moves[t])``. Refits happen at ``start`` and then every ``model.refit_every`` days.

    ``on_forecast(t, forecast, realized)`` is called after each successful forecast (row number
    ``t``, with the predictive distribution attached) so that a caller can use the predictive
    (for example to simulate an ES backtest) without storing it; ``keep_predictive=False`` then
    drops it from the returned path to save memory. ``on_error="record"`` turns an exception or a
    non-finite VaR/ES into a failed forecast (NaN numbers, ``diagnostics["error"]``) instead of
    raising; failed forecasts are not passed to ``on_forecast``.
    """
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if isinstance(moves, (pd.DataFrame, pd.Series)):
        index = moves.index
        x = moves.to_numpy(dtype=float)
    else:
        x = np.asarray(moves, dtype=float)
        index = pd.RangeIndex(x.shape[0])
    if x.ndim == 1:
        x = x[:, None]
    end = x.shape[0] if end is None else end
    if not 0 < start < end <= x.shape[0]:
        raise ValueError("need 0 < start < end <= number of rows")
    if start < model.window:
        raise ValueError(f"{model.name} needs {model.window} rows of history before start")
    if pnl is None:
        pnl = LinearPnL(np.ones(1))

    def pnl_for(t: int) -> PnLFunction:
        return pnl if hasattr(pnl, "linear_exposure") else pnl(t)

    forecasts: list[RiskForecast] = []
    realized = np.empty(end - start)
    fits: list[Any] = []
    fitted = None
    for t in range(start, end):
        history = x[:t]
        if model.refit_every is not None and (t - start) % model.refit_every == 0:
            fitted = model.fit(history)
            fits.append(fitted)
        f_t = pnl_for(t)
        realized[t - start] = float(f_t(x[t:t + 1])[0])
        try:
            pred, diag = model.predict(history, fitted, f_t, t)
            nums = (pred.var(level_var), pred.es(level_es), pred.var(level_es))
            if not np.all(np.isfinite(nums)):
                raise FloatingPointError(f"non-finite forecast {nums}")
        except Exception as exc:  # recorded in the forecast, not swallowed
            if on_error == "raise":
                raise
            nan = float("nan")
            forecasts.append(RiskForecast(index[t], model.name, level_var, level_es, nan, nan,
                                          {"error": f"{type(exc).__name__}: {exc}"}, None, nan))
            continue
        fc = RiskForecast(date=index[t], method=model.name, level_var=level_var,
                          level_es=level_es, var=nums[0], es=nums[1], diagnostics=diag,
                          predictive=pred, var_at_es_level=nums[2])
        if on_forecast is not None:
            on_forecast(t, fc, realized[t - start])
        if not keep_predictive:
            fc = RiskForecast(fc.date, fc.method, level_var, level_es, fc.var, fc.es, diag,
                              None, fc.var_at_es_level)
        forecasts.append(fc)
    return ForecastPath(dates=index[start:end], forecasts=forecasts, realized=realized,
                        fits=fits)
