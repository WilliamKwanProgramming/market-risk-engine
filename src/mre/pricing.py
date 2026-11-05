"""Black-Scholes-Merton pricing of European options with a continuous dividend yield.

All functions broadcast over numpy arrays, so one call prices an option under many scenarios
(shocked spot, shocked volatility, shortened maturity) at once.

Conventions (used everywhere in the package):

* ``S`` spot, ``K`` strike, ``T`` time to expiry in years, ``r`` continuously compounded risk-free
  rate, ``q`` continuous dividend yield, ``sigma`` annualised volatility (0.2 means 20%).
* ``is_call`` is a boolean (array): True for a call, False for a put.
* ``vega`` is per unit of volatility (per 1.00, not per volatility point), ``rho`` per unit of
  rate, ``theta`` is the change in value per year of calendar time passing, i.e.
  ``theta = -dV/dT``.
* ``T <= 0`` means expired: the value is the intrinsic value ``max(S - K, 0)`` (call) or
  ``max(K - S, 0)`` (put), and gamma, vega, theta and rho are 0.
* ``sigma = 0`` with ``T > 0`` is the deterministic limit: value
  ``max(S e^{-qT} - K e^{-rT}, 0)`` for a call.

Formulas (Black and Scholes 1973, Merton 1973), with
``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))`` and ``d2 = d1 - sigma sqrt(T)``::

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

__all__ = ["Greeks", "bsm_price", "bsm_greeks", "implied_vol"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S, K, T, r, q, sigma):
    """Return (d1, d2, sd, degenerate mask, T clipped at 0) with safe limits.

    When ``sigma * sqrt(T) == 0`` the option value is a deterministic function of the forward,
    and d1 = d2 = +inf (in the money), -inf (out of the money) or 0 (exactly at the money),
    which makes the general formulas return the correct limits for value and delta. When
    ``sigma = +inf`` with ``T > 0`` (a scenario whose volatility move overflows), d1 = +inf and
    d2 = -inf, the limits of the formulas: a call is worth ``S e^{-qT}`` and a put ``K e^{-rT}``.
    """
    S, K, T, r, q, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma)))
    Tc = np.maximum(T, 0.0)
    with np.errstate(invalid="ignore"):
        sd = np.where(Tc > 0.0, np.maximum(sigma, 0.0) * np.sqrt(Tc), 0.0)
    deg = sd <= 0.0
    big = np.isinf(sd)
    # log-moneyness of the discounted spot against the discounted strike
    m = np.log(S / K) + (r - q) * Tc
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = np.where(deg, np.sign(m) * np.inf, (m + 0.5 * sd * sd) / np.where(deg, 1.0, sd))
        d1 = np.where(deg & (m == 0.0), 0.0, d1)
        d1 = np.where(big, np.inf, d1)
        d2 = np.where(big, -np.inf, d1 - sd)
    return d1, d2, sd, deg, Tc, S, K, r, q


def bsm_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
              sigma: ArrayLike, is_call: ArrayLike) -> NDArray[np.float64]:
    """Black-Scholes-Merton price of a European call or put (formulas in the module docstring).

    Expired options (``T <= 0``) are worth their intrinsic value on the spot.
    """
    d1, d2, _, _, Tc, S, K, r, q = _d1_d2(S, K, T, r, q, sigma)
    dq = np.exp(-q * Tc)
    dr = np.exp(-r * Tc)
    flag = np.asarray(is_call, dtype=bool)
    # Guard against tiny negative values from cancellation deep out of the money.
    if flag.ndim == 0:  # one option type: evaluate only its formula (the scenario hot path)
        if flag:
            return np.maximum(S * dq * ndtr(d1) - K * dr * ndtr(d2), 0.0)
        return np.maximum(K * dr * ndtr(-d2) - S * dq * ndtr(-d1), 0.0)
    flag = np.broadcast_to(flag, d1.shape)
    call = S * dq * ndtr(d1) - K * dr * ndtr(d2)
    put = K * dr * ndtr(-d2) - S * dq * ndtr(-d1)
    return np.maximum(np.where(flag, call, put), 0.0)


@dataclass(frozen=True)
class Greeks:
    """Analytic BSM sensitivities (same shape as the broadcast inputs).

    ``delta = dV/dS``, ``gamma = d2V/dS2``, ``vega = dV/dsigma``, ``theta = -dV/dT`` (per year),
    ``rho = dV/dr``, ``vanna = d2V/(dS dsigma)``, ``volga = d2V/dsigma2``.
    """

    price: NDArray[np.float64]
    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]
    rho: NDArray[np.float64]
    vanna: NDArray[np.float64]
    volga: NDArray[np.float64]


def bsm_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
               sigma: ArrayLike, is_call: ArrayLike) -> Greeks:
    """Price and analytic Greeks of a European option under BSM with dividend yield ``q``.

    With ``phi`` the standard normal density and ``N`` its distribution function::

        delta_call = e^{-qT} N(d1)                 delta_put = -e^{-qT} N(-d1)
        gamma      = e^{-qT} phi(d1) / (S sigma sqrt(T))
        vega       = S e^{-qT} phi(d1) sqrt(T)
        theta_call = -S e^{-qT} phi(d1) sigma/(2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
        theta_put  = -S e^{-qT} phi(d1) sigma/(2 sqrt T) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
        rho_call   = K T e^{-rT} N(d2)             rho_put = -K T e^{-rT} N(-d2)
        vanna      = -e^{-qT} phi(d1) d2 / sigma   (same for calls and puts)
        volga      = vega d1 d2 / sigma            (same for calls and puts)

    For ``sigma sqrt(T) = 0`` the limits are used: gamma = vega = vanna = volga = 0; for
    ``T > 0`` theta and rho are those of the deterministic payoff; for ``T <= 0`` theta = rho = 0.
    """
    d1, d2, sd, deg, Tc, S, K, r, q = _d1_d2(S, K, T, r, q, sigma)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), d1.shape)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), d1.shape)
    dq = np.exp(-q * Tc)
    dr = np.exp(-r * Tc)
    price = bsm_price(S, K, Tc, r, q, sigma, is_call)
    pdf1 = np.where(deg, 0.0, _pdf(np.where(deg, 0.0, d1)))
    sqrtT = np.sqrt(Tc)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(deg, 0.0, dq * pdf1 / (S * np.where(deg, 1.0, sd)))
        decay = np.where(deg, 0.0, -S * dq * pdf1 * sigma / (2.0 * np.where(deg, 1.0, sqrtT)))
    vega = S * dq * pdf1 * sqrtT
    Nd1, Nd2, Nmd1, Nmd2 = ndtr(d1), ndtr(d2), ndtr(-d1), ndtr(-d2)
    delta = np.where(is_call, dq * Nd1, -dq * Nmd1)
    theta_c = decay - r * K * dr * Nd2 + q * S * dq * Nd1
    theta_p = decay + r * K * dr * Nmd2 - q * S * dq * Nmd1
    theta = np.where(is_call, theta_c, theta_p)
    rho = np.where(is_call, K * Tc * dr * Nd2, -K * Tc * dr * Nmd2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_safe = np.where(deg, 1.0, sigma)
        d1s, d2s = np.where(deg, 0.0, d1), np.where(deg, 0.0, d2)
        vanna = np.where(deg, 0.0, -dq * pdf1 * d2s / sig_safe)
        volga = np.where(deg, 0.0, vega * d1s * d2s / sig_safe)
    expired = Tc <= 0.0
    theta = np.where(expired, 0.0, theta)
    rho = np.where(expired, 0.0, rho)
    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
                  vanna=vanna, volga=volga)


def implied_vol(price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
                q: ArrayLike, is_call: ArrayLike, lo: float = 1e-6, hi: float = 5.0,
                tol: float = 1e-12, max_iter: int = 200) -> NDArray[np.float64]:
    """Implied volatility by vectorised bisection on the bracket ``[lo, hi]``.

    The BSM price is strictly increasing in sigma for ``T > 0`` (vega > 0), so bisection on a
    bracket that contains the root converges. Inputs whose price lies outside
    ``(price(lo), price(hi))``, i.e. below the no-arbitrage lower bound or above what any
    volatility up to ``hi`` can produce, return NaN, as do expired options. Iteration stops when
    the bracket is narrower than ``tol``.
    """
    price, S, K, T, r, q, is_call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)),
        np.asarray(is_call, dtype=bool))
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    pa = bsm_price(S, K, T, r, q, a, is_call) - price
    pb = bsm_price(S, K, T, r, q, b, is_call) - price
    ok = (pa < 0.0) & (pb > 0.0) & (T > 0.0)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        pm = bsm_price(S, K, T, r, q, m, is_call) - price
        below = pm < 0.0
        a = np.where(below, m, a)
        b = np.where(below, b, m)
        if np.all((b - a)[ok] < tol):
            break
    return np.where(ok, 0.5 * (a + b), np.nan)
