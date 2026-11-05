"""Risk and P&L attribution.

* :func:`euler_gaussian` - analytic Euler contributions to Gaussian VaR and ES.
* :func:`euler_historical` - Euler contributions estimated from P&L scenarios: ES contributions
  are the (weighted) average position P&L over the tail scenarios that define the empirical ES;
  VaR contributions use the Nadaraya-Watson kernel estimator of ``E[X_i | X = -VaR]`` (Tasche
  2008, sections 3.3-3.4).
* :func:`factor_regression` - OLS attribution of portfolio P&L to factor returns.
* :func:`option_pnl_explain` - option P&L split into delta, gamma, vega, theta and unexplained.

For P&L ``X = sum_i X_i`` (positive = gain) the Euler contributions are
``VaR_i = -E[X_i | X = -VaR]`` and ``ES_i = -E[X_i | X <= -VaR]``; they sum to VaR and ES.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from mre.measures import empirical_var_es
from mre.portfolio import EuropeanOption, MarketState, Portfolio
from mre.pricing import bsm_greeks

__all__ = ["euler_gaussian", "euler_historical", "FactorRegression", "factor_regression",
           "option_pnl_explain"]


def euler_gaussian(exposure: ArrayLike, cov: ArrayLike, level_var: float = 0.99,
                   level_es: float = 0.975) -> pd.DataFrame:
    """Euler contributions for zero-mean Gaussian P&L ``X = e' f``, ``f ~ N(0, Sigma)``.

    ``sigma_P = sqrt(e' Sigma e)``; the contribution of factor ``i`` is
    ``e_i (Sigma e)_i / sigma_P`` times ``z_a`` (VaR) or ``phi(z_a)/(1-a)`` (ES). The columns sum
    to the portfolio VaR and ES exactly.
    """
    e = np.asarray(exposure, dtype=float)
    S = np.asarray(cov, dtype=float)
    sd = float(np.sqrt(e @ S @ e))
    marg = e * (S @ e) / sd
    z_v = stats.norm.ppf(level_var)
    z_e = stats.norm.ppf(level_es)
    es_mult = stats.norm.pdf(z_e) / (1.0 - level_es)
    return pd.DataFrame({"var": marg * z_v, "es": marg * es_mult})


def euler_historical(pnl_by_position: ArrayLike, level_var: float = 0.99,
                     level_es: float = 0.975, bandwidth: float | None = None) -> pd.DataFrame:
    """Euler contributions from equally weighted scenarios, shape ``(n_scenarios, n_positions)``.

    **ES**: with sorted portfolio P&L, ``k = n (1 - level_es)`` and ``m = floor(k)``, the
    empirical ES of :mod:`mre.measures` gives weight ``1/k`` to the ``m`` worst scenarios and
    ``(k - m)/k`` to the next one. The same weights applied to each position's P&L give
    contributions that sum to the portfolio ES exactly.

    **VaR**: ``VaR_i = -sum_s K_s X_{s,i} / sum_s K_s`` with Gaussian kernel weights
    ``K_s = phi((X_s + VaR)/h)``. The default bandwidth is Silverman's rule
    ``h = 1.06 sd(X) n^(-1/5)``. The contributions sum to the kernel-smoothed
    ``-E[X | X = -VaR]``, which is close to but not exactly the VaR; the difference is reported
    as the attribute ``var_sum_gap`` in ``DataFrame.attrs``.
    """
    P = np.atleast_2d(np.asarray(pnl_by_position, dtype=float))
    X = P.sum(axis=1)
    n = X.size
    var, es = empirical_var_es(X, level_var, level_es)
    order = np.argsort(X, kind="stable")
    k = round(n * (1.0 - level_es), 9)
    m = int(np.floor(k))
    w = np.zeros(n)
    w[order[:m]] = 1.0 / k
    if m < n:
        w[order[m]] = (k - m) / k
    es_c = -(w @ P)
    h = 1.06 * float(np.std(X)) * n ** (-0.2) if bandwidth is None else float(bandwidth)
    kern = stats.norm.pdf((X + var) / h)
    var_c = -(kern @ P) / kern.sum()
    out = pd.DataFrame({"var": var_c, "es": es_c})
    out.attrs.update({"portfolio_var": var, "portfolio_es": es, "bandwidth": h,
                      "var_sum_gap": float(var_c.sum() - var)})
    return out


@dataclass(frozen=True)
class FactorRegression:
    """OLS of P&L on factors with intercept.

    ``variance_share[f] = beta_f Cov(f, y) / Var(y)``; the shares sum to R^2 and
    ``1 - R^2`` is the residual (idiosyncratic) share. ``t_stat`` uses classical OLS standard
    errors (homoskedastic, no autocorrelation).
    """

    beta: pd.Series
    t_stat: pd.Series
    r_squared: float
    variance_share: pd.Series
    n_obs: int


def factor_regression(y: pd.Series, factors: pd.DataFrame) -> FactorRegression:
    """Regress ``y`` on ``factors`` (inner-joined on the index, rows with NaN dropped)."""
    df = pd.concat([y.rename("__y__"), factors], axis=1, join="inner").dropna()
    Y = df["__y__"].to_numpy()
    F = df[factors.columns].to_numpy()
    n, k = F.shape
    A = np.column_stack([np.ones(n), F])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    s2 = float(resid @ resid) / (n - k - 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    names = ["const", *factors.columns]
    beta = pd.Series(coef, index=names)
    tstat = pd.Series(coef / se, index=names)
    var_y = float(np.var(Y, ddof=1))
    cov_fy = np.array([np.cov(F[:, i], Y, ddof=1)[0, 1] for i in range(k)])
    share = pd.Series(coef[1:] * cov_fy / var_y, index=list(factors.columns))
    r2 = 1.0 - float(resid @ resid) / float(((Y - Y.mean()) ** 2).sum())
    return FactorRegression(beta=beta, t_stat=tstat, r_squared=r2, variance_share=share,
                            n_obs=n)


def option_pnl_explain(portfolio: Portfolio, state0: MarketState, state1: MarketState,
                       horizon: float) -> pd.DataFrame:
    """Split each position's P&L from ``state0`` to ``state1`` (``horizon`` years later).

    Full P&L is the change in BSM value (expiry shortened by ``horizon``). With Greeks at
    ``state0``, ``dS = S1 - S0`` and ``dsigma = sigma1 - sigma0``: delta term ``delta dS``,
    gamma term ``0.5 gamma dS^2``, vega term ``vega dsigma``, theta term ``theta horizon``, and
    ``unexplained = full - (sum of the four)``. Stocks have only a delta term.
    """
    rows = []
    for j, p in enumerate(portfolio.positions):
        S0, S1 = state0.spot[p.underlying], state1.spot[p.underlying]
        dS = S1 - S0
        if not isinstance(p, EuropeanOption):
            rows.append({"position": j, "full": p.quantity * dS, "delta": p.quantity * dS,
                         "gamma": 0.0, "vega": 0.0, "theta": 0.0})
            continue
        m = p.quantity * p.multiplier
        s0, s1 = state0.vol[p.underlying], state1.vol[p.underlying]
        q = state0.q(p.underlying)
        g = bsm_greeks(S0, p.strike, p.expiry, state0.rate, q, s0, p.is_call)
        v1 = bsm_greeks(S1, p.strike, p.expiry - horizon, state0.rate, q, s1, p.is_call).price
        rows.append({"position": j, "full": m * float(v1 - g.price),
                     "delta": m * float(g.delta) * dS, "gamma": 0.5 * m * float(g.gamma) * dS**2,
                     "vega": m * float(g.vega) * (s1 - s0), "theta": m * float(g.theta) * horizon})
    out = pd.DataFrame(rows).set_index("position")
    out["unexplained"] = out["full"] - out[["delta", "gamma", "vega", "theta"]].sum(axis=1)
    return out
