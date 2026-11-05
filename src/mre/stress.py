"""Historical and hypothetical stress scenarios applied by full revaluation.

A stress scenario is a set of factor moves (log changes of spot and implied volatility, in the
factor naming of :class:`mre.portfolio.Portfolio`) applied **instantaneously** to today's
portfolio: options are repriced at the shocked spot and volatility with unchanged time to
expiry (``horizon = 0`` by default), so the P&L contains no time decay.

**Historical scenarios** use fixed, pre-specified calendar windows. The move of a factor over a
window ``[start, end]`` is the sum of its daily log changes dated ``start`` to ``end``
inclusive, i.e. ``log level(last day <= end) - log level(last day < start)``. If the data end
before ``end`` the window is cut at the last available day and the result says so
(``adjusted = True``); if the data do not cover ``start`` the scenario is not computed.

**Hypothetical scenarios**: all spots down 10%, 20% and 30% (``log(0.9)`` etc.), all implied
volatilities up by 10 volatility points (``sigma -> sigma + 0.10``), and spots down 20% together
with volatilities up 10 points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mre.portfolio import MarketState, Portfolio

__all__ = ["HISTORICAL_WINDOWS", "HistoricalScenario", "historical_scenario",
           "hypothetical_scenarios", "run_stress"]

HISTORICAL_WINDOWS: dict[str, tuple[str, str]] = {
    "2008 crisis": ("2008-09-02", "2008-11-20"),
    "2020 Covid crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}
"""Pre-specified windows (inclusive) for the historical scenarios."""


@dataclass(frozen=True)
class HistoricalScenario:
    """Cumulative factor moves over a window, with the dates actually used."""

    name: str
    moves: pd.Series
    first_day: pd.Timestamp
    last_day: pd.Timestamp
    requested: tuple[str, str]
    adjusted: bool


def historical_scenario(log_levels: pd.DataFrame, start: str, end: str,
                        name: str = "") -> HistoricalScenario:
    """Cumulative log moves of every column of ``log_levels`` over ``[start, end]``.

    ``log_levels`` holds log factor levels (for example ``logS:<u>`` = log price) indexed by
    trading date. Raises ValueError if the data have no day before ``start`` or no day inside the
    window.
    """
    idx = log_levels.index
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    before = idx[idx < t0]
    inside = idx[(idx >= t0) & (idx <= t1)]
    if before.empty or inside.empty:
        raise ValueError(f"data ({idx[0].date()} to {idx[-1].date()}) do not cover the window "
                         f"{start} to {end}")
    base, last = before[-1], inside[-1]
    moves = log_levels.loc[last] - log_levels.loc[base]
    if moves.isna().any():
        raise ValueError(f"missing factor levels at {base.date()} or {last.date()}")
    return HistoricalScenario(name=name, moves=moves, first_day=inside[0], last_day=last,
                              requested=(start, end), adjusted=bool(last < t1 and idx[-1] < t1))


def hypothetical_scenarios(portfolio: Portfolio, state: MarketState) -> dict[str, pd.Series]:
    """The five pre-specified hypothetical scenarios as factor-move vectors (module docstring)."""
    names = portfolio.factor_names()
    spot_cols = [n for n in names if n.startswith("logS:")]
    vol_cols = [n for n in names if n.startswith("logIV:")]

    def scen(spot_mult: float, vol_points: float) -> pd.Series:
        s = pd.Series(0.0, index=names)
        s[spot_cols] = np.log(spot_mult)
        for c in vol_cols:
            sig = state.vol[c.split(":", 1)[1]]
            s[c] = np.log((sig + vol_points) / sig)
        return s

    return {
        "equity -10%": scen(0.90, 0.0),
        "equity -20%": scen(0.80, 0.0),
        "equity -30%": scen(0.70, 0.0),
        "vol +10 pts": scen(1.0, 0.10),
        "equity -20%, vol +10 pts": scen(0.80, 0.10),
    }


def run_stress(portfolio: Portfolio, state: MarketState, scenarios: Mapping[str, pd.Series],
               horizon: float = 0.0) -> pd.DataFrame:
    """Full-revaluation P&L of each scenario: one row per scenario, one column per position plus
    ``total``. Factors missing from a scenario are not moved."""
    names = portfolio.factor_names()
    rows = {}
    for label, mv in scenarios.items():
        x = mv.reindex(names).fillna(0.0).to_numpy()[None, :]
        rows[label] = portfolio.pnl_full(state, x, horizon)[0]
    out = pd.DataFrame(rows, index=[f"pos{j}" for j in range(len(portfolio.positions))]).T
    out["total"] = out.sum(axis=1)
    return out
