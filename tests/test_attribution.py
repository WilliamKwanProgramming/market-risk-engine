"""Euler allocation, factor regression and option P&L explain."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mre.attribution import (
    euler_gaussian,
    euler_historical,
    factor_regression,
    option_pnl_explain,
)
from mre.measures import LocationScalePnL, empirical_var_es
from mre.portfolio import EuropeanOption, MarketState, Portfolio, Stock
from mre.pricing import bsm_greeks

COV = np.array([[0.04, 0.012], [0.012, 0.09]])
EXPO = np.array([2.0, -1.0])


def test_euler_gaussian_by_hand():
    out = euler_gaussian(EXPO, COV, 0.99, 0.975)
    sd = np.sqrt(EXPO @ COV @ EXPO)
    assert out["var"].sum() == pytest.approx(LocationScalePnL(0, sd).var(0.99))
    assert out["es"].sum() == pytest.approx(LocationScalePnL(0, sd).es(0.975))
    # factor 1: e1 (Sigma e)_1 / sd * z = 2 * (0.08 - 0.012) / sd * z
    assert out["var"].iloc[0] == pytest.approx(2 * 0.068 / sd * stats.norm.ppf(0.99))
    assert out["var"].iloc[1] == pytest.approx(-1 * (0.024 - 0.09) / sd * stats.norm.ppf(0.99))


def test_euler_historical_es_sums_exactly_and_var_close_to_gaussian():
    rng = np.random.default_rng(0)
    f = rng.multivariate_normal([0, 0], COV, 200_000)
    P = f * EXPO
    out = euler_historical(P, 0.99, 0.975)
    v, e = empirical_var_es(P.sum(axis=1), 0.99, 0.975)
    assert out["es"].sum() == pytest.approx(e, rel=1e-12)
    assert abs(out.attrs["var_sum_gap"]) < 0.01 * v
    ref = euler_gaussian(EXPO, COV, 0.99, 0.975)
    np.testing.assert_allclose(out["var"], ref["var"], rtol=0.05)
    np.testing.assert_allclose(out["es"], ref["es"], rtol=0.03)


def test_euler_historical_fractional_tail_by_hand():
    # 250 scenarios, two positions; k = 6.25 for ES 97.5%
    rng = np.random.default_rng(1)
    P = rng.standard_normal((250, 2))
    X = P.sum(axis=1)
    order = np.argsort(X)
    w = np.zeros(250)
    w[order[:6]] = 1 / 6.25
    w[order[6]] = 0.25 / 6.25
    out = euler_historical(P, 0.99, 0.975)
    np.testing.assert_allclose(out["es"], -(w @ P))


def test_factor_regression_recovers_betas_and_shares_sum_to_r2():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    F = pd.DataFrame(rng.standard_normal((3000, 2)) * [0.01, 0.005], index=idx,
                     columns=["mkt", "smb"])
    y = 0.0001 + 1.2 * F["mkt"] - 0.5 * F["smb"] + rng.standard_normal(3000) * 0.004
    res = factor_regression(y, F)
    assert res.beta["mkt"] == pytest.approx(1.2, abs=0.03)
    assert res.beta["smb"] == pytest.approx(-0.5, abs=0.05)
    assert res.variance_share.sum() == pytest.approx(res.r_squared, rel=1e-10)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(3000), F.to_numpy()]), y.to_numpy(),
                               rcond=None)
    np.testing.assert_allclose(res.beta.to_numpy(), coef)
    assert res.n_obs == 3000 and res.t_stat["mkt"] > 50


def test_option_pnl_explain_small_and_large_moves():
    port = Portfolio([EuropeanOption("A", 100.0, 0.25, True, 10.0), Stock("A", -5.0)])
    s0 = MarketState({"A": 100.0}, {"A": 0.2}, rate=0.01)
    h = 1 / 252
    small = MarketState({"A": 100.3}, {"A": 0.201}, rate=0.01)
    out = option_pnl_explain(port, s0, small, h)
    g = bsm_greeks(100, 100, 0.25, 0.01, 0, 0.2, True)
    full = 10 * (bsm_greeks(100.3, 100, 0.25 - h, 0.01, 0, 0.201, True).price - g.price)
    assert out.loc[0, "full"] == pytest.approx(full)
    assert out.loc[0, "delta"] == pytest.approx(10 * g.delta * 0.3)
    assert out.loc[0, "theta"] == pytest.approx(10 * g.theta * h)
    assert abs(out.loc[0, "unexplained"]) < 1e-2 * abs(out.loc[0, "full"])
    assert out.loc[1, "full"] == pytest.approx(-1.5) and out.loc[1, "unexplained"] == 0.0
    big = option_pnl_explain(port, s0, MarketState({"A": 85.0}, {"A": 0.35}, rate=0.01), h)
    assert abs(big.loc[0, "unexplained"]) > 100 * abs(out.loc[0, "unexplained"])
    cols = ["delta", "gamma", "vega", "theta", "unexplained"]
    np.testing.assert_allclose(big[cols].sum(axis=1), big["full"])
