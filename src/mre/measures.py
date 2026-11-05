"""VaR and expected shortfall of P&L distributions, and the predictive-distribution classes.

Sign convention used in the whole package: ``X`` is profit and loss (P&L, positive = gain) and
VaR and ES are reported as **positive loss numbers**. At confidence level ``a`` (for example 0.99)
with tail probability ``p = 1 - a``:

* ``VaR_a(X) = -q_p(X)``, where ``q_p`` is the lower ``p``-quantile
  ``inf{x : P(X <= x) >= p}``;
* ``ES_a(X) = -(1/p) * integral_0^p q_u(X) du`` (Acerbi and Tasche 2002), which for a continuous
  distribution equals ``-E[X | X <= -VaR_a]``.

Finite samples (historical simulation, filtered historical simulation and Monte Carlo) use the
same definitions applied to the empirical distribution of the ``n`` scenarios. With the sorted
P&L ``x_(1) <= ... <= x_(n)``, ``k = n p`` and ``m = floor(k)``::

    VaR = -x_(ceil(k))                                  (the ceil(k)-th worst scenario)
    ES  = -(1/k) * [ x_(1) + ... + x_(m) + (k - m) x_(m+1) ]

so for ``n = 250, a = 0.99`` the VaR is the 3rd worst scenario, and for ``n = 250, a = 0.975``
(k = 6.25) the ES averages the 6 worst scenarios plus a quarter of the 7th. When ``k`` is an
integer the ES is the plain mean of the ``k`` worst scenarios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, ndtri, stdtrit

__all__ = [
    "empirical_var_es",
    "std_t_quantile",
    "std_t_es",
    "normal_es",
    "EmpiricalPnL",
    "LocationScalePnL",
    "Predictive",
]


def _tail_count(n: int, level: float) -> float:
    """``n (1 - level)`` rounded to 9 decimals so that 250 * 0.025 is exactly 6.25."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return round(n * (1.0 - level), 9)


def _var_es_sorted(x: NDArray[np.float64], level_var: float,
                   level_es: float) -> tuple[float, float]:
    """VaR and ES from an ascending-sorted sample (definitions in the module docstring)."""
    n = x.size
    kv = _tail_count(n, level_var)
    ke = _tail_count(n, level_es)
    if kv <= 0.0 or ke <= 0.0:
        raise ValueError(f"sample of {n} scenarios is too small for these levels")
    var = -x[int(np.ceil(kv)) - 1]
    m = int(np.floor(ke))
    tail = x[:m].sum() + (ke - m) * (x[m] if m < n else 0.0)
    return float(var), float(-tail / ke)


def empirical_var_es(pnl: ArrayLike, level_var: float, level_es: float) -> tuple[float, float]:
    """VaR and ES of the empirical distribution of a P&L sample (definitions in module docstring).

    Raises ValueError for an empty sample.
    """
    return _var_es_sorted(np.sort(np.asarray(pnl, dtype=float).ravel()), level_var, level_es)


def std_t_quantile(p: ArrayLike, nu: float) -> NDArray[np.float64]:
    """Quantile of the Student t distribution with ``nu > 2`` rescaled to unit variance."""
    return stdtrit(nu, p) * np.sqrt((nu - 2.0) / nu)


def _t_pdf(x: float, nu: float) -> float:
    """Density of the standard Student t distribution."""
    logc = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    return float(np.exp(logc - 0.5 * (nu + 1.0) * np.log1p(x * x / nu)))


def std_t_es(level: float, nu: float) -> float:
    """ES (positive number) of a unit-variance Student t variable at confidence ``level``.

    For a standard t variable with density ``g_nu`` and lower ``p``-quantile ``t_p`` (``p = 1 -
    level``) the lower-tail expectation is ``E[T | T <= t_p] = -g_nu(t_p) (nu + t_p^2) /
    ((nu - 1) p)``; the result is multiplied by ``sqrt((nu - 2)/nu)`` for unit variance (formula
    as in McNeil, Frey and Embrechts, Quantitative Risk Management, 2015).
    """
    p = 1.0 - level
    tq = float(stdtrit(nu, p))
    es_std = _t_pdf(tq, nu) * (nu + tq * tq) / ((nu - 1.0) * p)
    return float(es_std * np.sqrt((nu - 2.0) / nu))


def normal_es(level: float) -> float:
    """ES (positive number) of a standard normal variable: ``phi(z_p) / p``, ``p = 1 - level``."""
    p = 1.0 - level
    z = float(ndtri(p))
    return float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) / p)


class Predictive(ABC):
    """A forecast distribution of next-period P&L: VaR, ES and sampling."""

    @abstractmethod
    def var(self, level: float) -> float:
        """VaR (positive loss number) at confidence ``level``."""

    @abstractmethod
    def es(self, level: float) -> float:
        """ES (positive loss number) at confidence ``level``."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        """Draw ``size`` independent P&L values from the distribution."""


@dataclass(frozen=True)
class EmpiricalPnL(Predictive):
    """Equally weighted P&L scenarios (historical, filtered historical or Monte Carlo)."""

    scenarios: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_sorted", np.sort(np.asarray(self.scenarios, float).ravel()))

    def var(self, level: float) -> float:
        return _var_es_sorted(self._sorted, level, level)[0]

    def es(self, level: float) -> float:
        return _var_es_sorted(self._sorted, level, level)[1]

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        """Draw ``size`` scenarios with replacement (the empirical distribution itself)."""
        return self.scenarios[rng.integers(0, self.scenarios.size, size=size)]


@dataclass(frozen=True)
class LocationScalePnL(Predictive):
    """P&L ``= loc + scale * Z`` with ``Z`` standard normal or unit-variance Student t(nu)."""

    loc: float
    scale: float
    dist: str = "normal"
    nu: float = np.inf

    def __post_init__(self) -> None:
        if self.dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        if self.dist == "t" and not self.nu > 2.0:
            raise ValueError("Student t needs nu > 2 for a finite variance")

    def _z_quantile(self, p: float) -> float:
        if self.dist == "normal":
            return float(ndtri(p))
        return float(std_t_quantile(p, self.nu))

    def var(self, level: float) -> float:
        return -(self.loc + self.scale * self._z_quantile(1.0 - level))

    def es(self, level: float) -> float:
        z_es = normal_es(level) if self.dist == "normal" else std_t_es(level, self.nu)
        return -self.loc + self.scale * z_es

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        if self.dist == "normal":
            z = rng.standard_normal(size)
        else:
            z = rng.standard_t(self.nu, size) * np.sqrt((self.nu - 2.0) / self.nu)
        return self.loc + self.scale * z
