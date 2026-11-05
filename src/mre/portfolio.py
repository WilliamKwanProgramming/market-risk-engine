"""Positions, risk factors and scenario P&L by full revaluation or by a Greek approximation.

**Risk factors.** For every underlying ``u`` the factor ``logS:u`` is the log spot price. For every
underlying that has at least one option position, the factor ``logIV:u`` is the log of a single,
flat implied-volatility level used for all options on ``u`` (no smile or term structure). A
scenario is a vector of factor *moves* (log changes): the shocked spot is ``S e^{x}`` and the
shocked volatility ``sigma e^{y}``. The rate and dividend yields are held fixed.

**Full revaluation.** Over a horizon of ``h`` years (default one trading day, ``1/252``) every
option is repriced with BSM at the shocked spot, the shocked volatility and the remaining time
``T - h``; an option with ``T - h <= 0`` is worth its intrinsic value. The P&L is the change in
value. Stock P&L is ``quantity * (S' - S)``; dividends paid during the horizon are ignored.

**Delta-gamma(-vega)(-theta) approximation**, for comparison with full revaluation::

    dV ~ delta dS + 0.5 gamma dS^2 [+ vega dsigma] [+ theta h]
         [+ vanna dS dsigma + 0.5 volga dsigma^2],
    dS = S (e^x - 1),  dsigma = sigma (e^y - 1)

with the Greeks of :func:`mre.pricing.bsm_greeks` at the current state. The vanna and volga terms
(``second_order_vol=True``) complete the second-order expansion in spot and volatility; they are
off by default.

This is model P&L (hypothetical P&L from revaluing positions with a pricing model), not P&L from
traded option prices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mre.pricing import bsm_greeks, bsm_price

__all__ = ["Stock", "EuropeanOption", "MarketState", "Portfolio", "PortfolioPnL", "LinearPnL",
           "TRADING_DAYS"]

TRADING_DAYS = 252
"""Trading days per year: the default horizon is ``1 / TRADING_DAYS`` years."""


@dataclass(frozen=True)
class Stock:
    """A linear position of ``quantity`` units of ``underlying``."""

    underlying: str
    quantity: float


@dataclass(frozen=True)
class EuropeanOption:
    """``quantity`` European options (times ``multiplier`` units each) on ``underlying``.

    ``expiry`` is the time to expiry in years at the valuation date of the
    :class:`MarketState` it is valued with.
    """

    underlying: str
    strike: float
    expiry: float
    is_call: bool
    quantity: float
    multiplier: float = 1.0


Position = Stock | EuropeanOption


@dataclass(frozen=True)
class MarketState:
    """Spot prices, flat implied volatilities, the risk-free rate and dividend yields."""

    spot: Mapping[str, float]
    vol: Mapping[str, float] = field(default_factory=dict)
    rate: float = 0.0
    div_yield: Mapping[str, float] = field(default_factory=dict)

    def q(self, underlying: str) -> float:
        return float(self.div_yield.get(underlying, 0.0))


@dataclass(frozen=True)
class Portfolio:
    """An ordered collection of positions; factor order is fixed by :meth:`factor_names`."""

    positions: tuple[Position, ...]

    def __init__(self, positions: Sequence[Position]):
        object.__setattr__(self, "positions", tuple(positions))
        if not self.positions:
            raise ValueError("a portfolio needs at least one position")

    def underlyings(self) -> list[str]:
        out: list[str] = []
        for p in self.positions:
            if p.underlying not in out:
                out.append(p.underlying)
        return out

    def option_underlyings(self) -> list[str]:
        out: list[str] = []
        for p in self.positions:
            if isinstance(p, EuropeanOption) and p.underlying not in out:
                out.append(p.underlying)
        return out

    def factor_names(self) -> list[str]:
        """``logS:<u>`` for every underlying, then ``logIV:<u>`` for underlyings with options."""
        return ([f"logS:{u}" for u in self.underlyings()]
                + [f"logIV:{u}" for u in self.option_underlyings()])

    def position_values(self, state: MarketState) -> NDArray[np.float64]:
        vals = []
        for p in self.positions:
            S = state.spot[p.underlying]
            if isinstance(p, Stock):
                vals.append(p.quantity * S)
            else:
                v = bsm_price(S, p.strike, p.expiry, state.rate, state.q(p.underlying),
                              state.vol[p.underlying], p.is_call)
                vals.append(p.quantity * p.multiplier * float(v))
        return np.array(vals)

    def value(self, state: MarketState) -> float:
        return float(self.position_values(state).sum())

    def _factor_columns(self, moves: ArrayLike) -> tuple[NDArray[np.float64], dict[str, int]]:
        x = np.atleast_2d(np.asarray(moves, dtype=float))
        names = self.factor_names()
        if x.shape[1] != len(names):
            raise ValueError(f"moves have {x.shape[1]} columns, expected {len(names)}: {names}")
        return x, {n: i for i, n in enumerate(names)}

    def pnl_full(self, state: MarketState, moves: ArrayLike,
                 horizon: float = 1.0 / TRADING_DAYS) -> NDArray[np.float64]:
        """Full-revaluation P&L per scenario and position, shape ``(n_scenarios, n_positions)``.

        ``moves`` has one column per factor, in the order of :meth:`factor_names`.
        """
        x, col = self._factor_columns(moves)
        out = np.empty((x.shape[0], len(self.positions)))
        for j, p in enumerate(self.positions):
            S = state.spot[p.underlying]
            S1 = S * np.exp(x[:, col[f"logS:{p.underlying}"]])
            if isinstance(p, Stock):
                out[:, j] = p.quantity * (S1 - S)
                continue
            sig = state.vol[p.underlying]
            sig1 = sig * np.exp(x[:, col[f"logIV:{p.underlying}"]])
            r, q = state.rate, state.q(p.underlying)
            v0 = bsm_price(S, p.strike, p.expiry, r, q, sig, p.is_call)
            v1 = bsm_price(S1, p.strike, p.expiry - horizon, r, q, sig1, p.is_call)
            out[:, j] = p.quantity * p.multiplier * (v1 - v0)
        return out

    def pnl_greeks(self, state: MarketState, moves: ArrayLike,
                   horizon: float = 1.0 / TRADING_DAYS, vega: bool = True,
                   theta: bool = True, second_order_vol: bool = False) -> NDArray[np.float64]:
        """Delta-gamma(-vega)(-theta) P&L per scenario and position (module docstring)."""
        x, col = self._factor_columns(moves)
        out = np.empty((x.shape[0], len(self.positions)))
        for j, p in enumerate(self.positions):
            S = state.spot[p.underlying]
            dS = S * np.expm1(x[:, col[f"logS:{p.underlying}"]])
            if isinstance(p, Stock):
                out[:, j] = p.quantity * dS
                continue
            sig = state.vol[p.underlying]
            g = bsm_greeks(S, p.strike, p.expiry, state.rate, state.q(p.underlying), sig,
                           p.is_call)
            approx = g.delta * dS + 0.5 * g.gamma * dS**2
            dsig = sig * np.expm1(x[:, col[f"logIV:{p.underlying}"]])
            if vega:
                approx = approx + g.vega * dsig
            if second_order_vol:
                approx = approx + g.vanna * dS * dsig + 0.5 * g.volga * dsig**2
            if theta:
                approx = approx + g.theta * horizon
            out[:, j] = p.quantity * p.multiplier * approx
        return out

    def factor_sensitivities(self, state: MarketState) -> NDArray[np.float64]:
        """First-order P&L sensitivity to each factor move: ``dV/dx`` at ``x = 0``.

        For ``logS:u`` this is ``sum(delta_j * S_u)`` over positions on ``u`` (a stock has delta
        1); for ``logIV:u`` it is ``sum(vega_j * sigma_u)``. Used by the delta-normal
        (EWMA-parametric) model.
        """
        names = self.factor_names()
        col = {n: i for i, n in enumerate(names)}
        sens = np.zeros(len(names))
        for p in self.positions:
            S = state.spot[p.underlying]
            if isinstance(p, Stock):
                sens[col[f"logS:{p.underlying}"]] += p.quantity * S
                continue
            sig = state.vol[p.underlying]
            g = bsm_greeks(S, p.strike, p.expiry, state.rate, state.q(p.underlying), sig,
                           p.is_call)
            m = p.quantity * p.multiplier
            sens[col[f"logS:{p.underlying}"]] += m * float(g.delta) * S
            sens[col[f"logIV:{p.underlying}"]] += m * float(g.vega) * sig
        return sens


class LinearPnL:
    """P&L linear in the factor moves: ``pnl = moves @ exposure``.

    With ``exposure = [1.0]`` the P&L of a scenario is the return itself, which is how the
    validation experiments use the models (VaR and ES of a return series).
    """

    def __init__(self, exposure: ArrayLike):
        self.exposure = np.atleast_1d(np.asarray(exposure, dtype=float))

    def __call__(self, moves: ArrayLike) -> NDArray[np.float64]:
        return np.atleast_2d(np.asarray(moves, dtype=float)) @ self.exposure

    def linear_exposure(self) -> NDArray[np.float64]:
        return self.exposure

    def by_position(self, moves: ArrayLike) -> NDArray[np.float64]:
        """P&L per factor (each factor treated as one position), shape ``(n, k)``."""
        return np.atleast_2d(np.asarray(moves, dtype=float)) * self.exposure


class PortfolioPnL:
    """Full-revaluation P&L of a :class:`Portfolio` at a fixed :class:`MarketState`."""

    def __init__(self, portfolio: Portfolio, state: MarketState,
                 horizon: float = 1.0 / TRADING_DAYS):
        self.portfolio = portfolio
        self.state = state
        self.horizon = horizon

    def __call__(self, moves: ArrayLike) -> NDArray[np.float64]:
        return self.by_position(moves).sum(axis=1)

    def by_position(self, moves: ArrayLike) -> NDArray[np.float64]:
        return self.portfolio.pnl_full(self.state, moves, self.horizon)

    def linear_exposure(self) -> NDArray[np.float64]:
        return self.portfolio.factor_sensitivities(self.state)
