"""Positions, factors, full revaluation and the Greek approximation."""

import numpy as np
import pytest

from mre.portfolio import (
    TRADING_DAYS,
    EuropeanOption,
    LinearPnL,
    MarketState,
    Portfolio,
    PortfolioPnL,
    Stock,
)
from mre.pricing import bsm_greeks, bsm_price

STATE = MarketState(spot={"A": 100.0, "B": 50.0}, vol={"A": 0.25, "B": 0.4}, rate=0.03,
                    div_yield={"A": 0.01})
PORT = Portfolio([
    Stock("A", 10.0),
    EuropeanOption("A", 95.0, 0.25, False, 20.0, multiplier=1.0),
    EuropeanOption("B", 55.0, 0.5, True, -5.0, multiplier=100.0),
    Stock("B", -3.0),
])
H = 1.0 / TRADING_DAYS


def test_factor_names_order():
    assert PORT.factor_names() == ["logS:A", "logS:B", "logIV:A", "logIV:B"]
    assert Portfolio([Stock("X", 1.0)]).factor_names() == ["logS:X"]
    with pytest.raises(ValueError):
        Portfolio([])


def test_value_and_full_revaluation_against_direct_pricing():
    x = np.array([[0.02, -0.05, 0.10, -0.03]])
    pnl = PORT.pnl_full(STATE, x)[0]
    SA, SB = 100 * np.exp(0.02), 50 * np.exp(-0.05)
    vA, vB = 0.25 * np.exp(0.10), 0.4 * np.exp(-0.03)
    put0 = bsm_price(100, 95, 0.25, 0.03, 0.01, 0.25, False)
    put1 = bsm_price(SA, 95, 0.25 - H, 0.03, 0.01, vA, False)
    call0 = bsm_price(50, 55, 0.5, 0.03, 0.0, 0.4, True)
    call1 = bsm_price(SB, 55, 0.5 - H, 0.03, 0.0, vB, True)
    np.testing.assert_allclose(pnl, [10 * (SA - 100), 20 * (put1 - put0),
                                     -500 * (call1 - call0), -3 * (SB - 50)], rtol=1e-12)
    assert PORT.value(STATE) == pytest.approx(1000 + 20 * put0 - 500 * call0 - 150)


def test_time_to_maturity_reduced_by_one_day():
    zero = np.zeros((1, 4))
    pnl = PORT.pnl_full(STATE, zero)[0]
    put0 = bsm_price(100, 95, 0.25, 0.03, 0.01, 0.25, False)
    put1 = bsm_price(100, 95, 0.25 - 1 / 252, 0.03, 0.01, 0.25, False)
    assert pnl[1] == pytest.approx(20 * (put1 - put0), rel=1e-12)
    assert pnl[1] < 0  # a long put loses time value when nothing moves
    assert pnl[0] == 0.0 and pnl[3] == 0.0
    np.testing.assert_array_equal(PORT.pnl_full(STATE, zero, horizon=0.0), 0.0)


def test_option_expiring_within_horizon_is_intrinsic():
    p = Portfolio([EuropeanOption("A", 100.0, 0.5 / 252, False, 1.0)])
    pnl = p.pnl_full(STATE, np.array([[np.log(0.9), 0.0]]))[0, 0]
    v0 = bsm_price(100.0, 100.0, 0.5 / 252, 0.03, 0.01, 0.25, False)
    assert pnl == pytest.approx(10.0 - v0)


def test_greek_approximation_converges_to_full_revaluation():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((50, 4))
    errs = []
    for scale in (0.02, 0.01, 0.005):
        x = base * scale
        full = PORT.pnl_full(STATE, x, horizon=0.0).sum(axis=1)
        approx = PORT.pnl_greeks(STATE, x, horizon=0.0, theta=False).sum(axis=1)
        errs.append(np.max(np.abs(full - approx)))
    # the remainder is third order in the move size (cross terms vanna/volga are omitted,
    # which are second order: so the error falls at least by a factor ~4 per halving)
    assert errs[1] < errs[0] / 3.5 and errs[2] < errs[1] / 3.5


def test_second_order_vol_terms_make_the_error_third_order():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((50, 4))
    errs = []
    for scale in (0.02, 0.01, 0.005):
        x = base * scale
        full = PORT.pnl_full(STATE, x, horizon=0.0).sum(axis=1)
        approx = PORT.pnl_greeks(STATE, x, horizon=0.0, theta=False,
                                 second_order_vol=True).sum(axis=1)
        errs.append(np.max(np.abs(full - approx)))
    # a complete second-order expansion leaves a third-order remainder: a factor ~8 per halving
    assert errs[1] < errs[0] / 7.0 and errs[2] < errs[1] / 7.0


def test_greek_approximation_vanna_volga_terms():
    x = np.array([[0.01, 0.0, 0.05, 0.0]])
    g = bsm_greeks(100, 95, 0.25, 0.03, 0.01, 0.25, False)
    dS = 100 * np.expm1(0.01)
    dv = 0.25 * np.expm1(0.05)
    extra = (PORT.pnl_greeks(STATE, x, second_order_vol=True)[0, 1]
             - PORT.pnl_greeks(STATE, x)[0, 1])
    assert extra == pytest.approx(20 * (g.vanna * dS * dv + 0.5 * g.volga * dv**2))


def test_greek_approximation_terms():
    x = np.array([[0.01, 0.0, 0.05, 0.0]])
    g = bsm_greeks(100, 95, 0.25, 0.03, 0.01, 0.25, False)
    dS = 100 * np.expm1(0.01)
    dv = 0.25 * np.expm1(0.05)
    full = PORT.pnl_greeks(STATE, x)[0, 1]
    assert full == pytest.approx(20 * (g.delta * dS + 0.5 * g.gamma * dS**2 + g.vega * dv
                                       + g.theta * H))
    no_vega = PORT.pnl_greeks(STATE, x, vega=False)[0, 1]
    assert full - no_vega == pytest.approx(20 * g.vega * dv)
    no_theta = PORT.pnl_greeks(STATE, x, theta=False)[0, 1]
    assert full - no_theta == pytest.approx(20 * g.theta * H)
    # theta term matches the one-day full revaluation at zero move to first order
    zero = np.zeros((1, 4))
    assert PORT.pnl_full(STATE, zero)[0, 1] == pytest.approx(20 * g.theta * H, rel=1e-2)


def test_factor_sensitivities_are_first_derivatives():
    h = 1e-6
    sens = PORT.factor_sensitivities(STATE)
    for j in range(4):
        e = np.zeros((1, 4))
        e[0, j] = h
        up = PORT.pnl_full(STATE, e, horizon=0.0).sum()
        dn = PORT.pnl_full(STATE, -e, horizon=0.0).sum()
        assert sens[j] == pytest.approx((up - dn) / (2 * h), rel=1e-5, abs=1e-6)


def test_pnl_function_wrappers():
    f = PortfolioPnL(PORT, STATE)
    x = np.random.default_rng(1).standard_normal((5, 4)) * 0.01
    np.testing.assert_allclose(f(x), PORT.pnl_full(STATE, x).sum(axis=1))
    np.testing.assert_allclose(f.linear_exposure(), PORT.factor_sensitivities(STATE))
    lin = LinearPnL([2.0, -1.0])
    np.testing.assert_allclose(lin(np.array([[1.0, 3.0]])), [-1.0])
    np.testing.assert_allclose(lin.by_position(np.array([[1.0, 3.0]])), [[2.0, -3.0]])
    with pytest.raises(ValueError):
        PORT.pnl_full(STATE, np.zeros((1, 3)))
