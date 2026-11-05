"""Black-Scholes-Merton prices, Greeks and implied volatility.

Reference values: worked example and practice problems of T. Plihal, "Financial Derivatives"
course notes, week 9 (https://tomasplihal.github.io/financial_derivatives/week_09.html), quoted
to 2 decimals there (d1, d2 to 4 decimals).
"""

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from mre.pricing import bsm_greeks, bsm_price, implied_vol


def test_worked_example_s42_k40():
    S, K, r, sig, T = 42.0, 40.0, 0.10, 0.20, 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    assert d1 == pytest.approx(0.7693, abs=5e-5)
    assert d1 - sig * np.sqrt(T) == pytest.approx(0.6278, abs=5e-5)
    assert bsm_price(S, K, T, r, 0.0, sig, True) == pytest.approx(4.76, abs=0.005)
    assert bsm_price(S, K, T, r, 0.0, sig, False) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize(("S", "K", "r", "sig", "T", "is_call", "expected"), [
    (50, 50, 0.10, 0.30, 0.25, False, 2.37),
    (52, 50, 0.12, 0.30, 0.25, True, 5.06),
    (69, 70, 0.05, 0.35, 0.50, False, 6.40),
    (30, 29, 0.05, 0.25, 1 / 3, True, 2.52),
    (30, 29, 0.05, 0.25, 1 / 3, False, 1.05),
])
def test_practice_problems(S, K, r, sig, T, is_call, expected):
    # The quoted answers were computed with N(d) from a rounded table, so they can differ
    # from the exact value by up to 0.01 (2.3759 is quoted as 2.37, 2.5251 as 2.52).
    assert bsm_price(S, K, T, r, 0.0, sig, is_call) == pytest.approx(expected, abs=0.01)


def test_implied_vol_practice_problem():
    assert implied_vol(2.5, 15, 13, 0.25, 0.05, 0.0, True) == pytest.approx(0.396, abs=5e-4)


def _grid(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return dict(S=rng.uniform(50, 150, n), K=rng.uniform(60, 140, n), T=rng.uniform(0.02, 3, n),
                r=rng.uniform(-0.01, 0.08, n), q=rng.uniform(0.0, 0.05, n),
                sigma=rng.uniform(0.05, 0.8, n))


def test_put_call_parity_with_dividends():
    g = _grid()
    c = bsm_price(**g, is_call=True)
    p = bsm_price(**g, is_call=False)
    rhs = g["S"] * np.exp(-g["q"] * g["T"]) - g["K"] * np.exp(-g["r"] * g["T"])
    np.testing.assert_allclose(c - p, rhs, atol=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_price_matches_numerical_integration(is_call):
    """Independent implementation: discounted risk-neutral expectation of the payoff."""
    S, K, T, r, q, sig = 100.0, 95.0, 0.7, 0.03, 0.02, 0.35
    mu = np.log(S) + (r - q - 0.5 * sig**2) * T
    sd = sig * np.sqrt(T)

    def integrand(z):
        ST = np.exp(mu + sd * z)
        pay = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)
        return pay * stats.norm.pdf(z)

    val = np.exp(-r * T) * integrate.quad(integrand, -12, 12, limit=200, points=[
        (np.log(K) - mu) / sd])[0]
    assert bsm_price(S, K, T, r, q, sig, is_call) == pytest.approx(val, rel=1e-9)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_central_differences(is_call):
    g = _grid(200, seed=2)
    G = bsm_greeks(**g, is_call=is_call)

    def f(**kw):
        a = dict(g)
        a.update(kw)
        return bsm_price(**a, is_call=is_call)

    hS = 1e-4 * g["S"]
    np.testing.assert_allclose(G.delta, (f(S=g["S"] + hS) - f(S=g["S"] - hS)) / (2 * hS),
                               atol=1e-7)
    np.testing.assert_allclose(
        G.gamma, (f(S=g["S"] + hS) - 2 * G.price + f(S=g["S"] - hS)) / hS**2, atol=1e-6)
    h = 1e-5
    np.testing.assert_allclose(
        G.vega, (f(sigma=g["sigma"] + h) - f(sigma=g["sigma"] - h)) / (2 * h), atol=1e-4)
    np.testing.assert_allclose(G.rho, (f(r=g["r"] + h) - f(r=g["r"] - h)) / (2 * h), atol=1e-4)
    # theta = -dV/dT
    np.testing.assert_allclose(G.theta, -(f(T=g["T"] + h) - f(T=g["T"] - h)) / (2 * h),
                               atol=1e-4)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_satisfy_bsm_pde(is_call):
    """theta + (r - q) S delta + 0.5 sigma^2 S^2 gamma - r V = 0 (theta = dV/dt)."""
    g = _grid(300, seed=3)
    G = bsm_greeks(**g, is_call=is_call)
    S, r, q, sig = g["S"], g["r"], g["q"], g["sigma"]
    resid = G.theta + (r - q) * S * G.delta + 0.5 * sig**2 * S**2 * G.gamma - r * G.price
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_call_put_greek_relations():
    g = _grid(100, seed=4)
    c = bsm_greeks(**g, is_call=True)
    p = bsm_greeks(**g, is_call=False)
    np.testing.assert_allclose(c.delta - p.delta, np.exp(-g["q"] * g["T"]), atol=1e-12)
    np.testing.assert_allclose(c.gamma, p.gamma, rtol=1e-12)
    np.testing.assert_allclose(c.vega, p.vega, rtol=1e-12)
    assert np.all(c.vega > 0) and np.all(c.gamma > 0)


def test_expired_and_negative_time_are_intrinsic():
    S = np.array([90.0, 100.0, 110.0])
    for T in (0.0, -0.01):
        np.testing.assert_allclose(bsm_price(S, 100.0, T, 0.05, 0.0, 0.3, True), [0, 0, 10])
        np.testing.assert_allclose(bsm_price(S, 100.0, T, 0.05, 0.0, 0.3, False), [10, 0, 0])
        G = bsm_greeks(S, 100.0, T, 0.05, 0.0, 0.3, False)
        np.testing.assert_allclose(G.delta, [-1, -0.5, 0])
        assert np.all(G.gamma == 0) and np.all(G.vega == 0) and np.all(G.theta == 0)


def test_limit_t_to_zero_converges_to_intrinsic():
    for T in (1e-4, 1e-6, 1e-8):
        c = bsm_price([95.0, 105.0], 100.0, T, 0.02, 0.0, 0.25, True)
        p = bsm_price([95.0, 105.0], 100.0, T, 0.02, 0.0, 0.25, False)
        np.testing.assert_allclose(c, [0.0, 5.0], atol=5 * np.sqrt(T))
        np.testing.assert_allclose(p, [5.0, 0.0], atol=5 * np.sqrt(T))


def test_limit_sigma_to_zero_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 98.0, 1.0, 0.05, 0.01
    fwd_intr_call = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    fwd_intr_put = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    for sig in (0.0, 1e-9):
        assert bsm_price(S, K, T, r, q, sig, True) == pytest.approx(fwd_intr_call, abs=1e-9)
        assert bsm_price(S, K, T, r, q, sig, False) == pytest.approx(fwd_intr_put, abs=1e-9)
    G0 = bsm_greeks(S, K, T, r, q, 0.0, True)
    Gs = bsm_greeks(S, K, T, r, q, 1e-6, True)
    assert G0.delta == pytest.approx(np.exp(-q * T))
    assert G0.theta == pytest.approx(Gs.theta, abs=1e-6)
    assert G0.rho == pytest.approx(Gs.rho, abs=1e-6)


def test_price_bounds():
    g = _grid(300, seed=5)
    c = bsm_price(**g, is_call=True)
    p = bsm_price(**g, is_call=False)
    disc_S = g["S"] * np.exp(-g["q"] * g["T"])
    disc_K = g["K"] * np.exp(-g["r"] * g["T"])
    assert np.all(c >= np.maximum(disc_S - disc_K, 0) - 1e-12) and np.all(c <= disc_S + 1e-12)
    assert np.all(p >= np.maximum(disc_K - disc_S, 0) - 1e-12) and np.all(p <= disc_K + 1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip_and_brentq(is_call):
    g = _grid(200, seed=6)
    price = bsm_price(**g, is_call=is_call)
    iv = implied_vol(price, g["S"], g["K"], g["T"], g["r"], g["q"], is_call)
    ok = np.isfinite(iv)
    assert ok.mean() > 0.95
    # The price always round-trips; the volatility itself is only identified to ~1e-7 where
    # vega is not tiny (deep in or out of the money the price barely depends on sigma).
    rep = bsm_price(g["S"], g["K"], g["T"], g["r"], g["q"], iv, is_call)
    np.testing.assert_allclose(rep[ok], price[ok], atol=1e-9)
    well = ok & (bsm_greeks(**g, is_call=is_call).vega > 1e-2)
    assert well.sum() > 150
    np.testing.assert_allclose(iv[well], g["sigma"][well], atol=1e-7)
    for i in np.flatnonzero(ok)[:20]:
        a = {k: v[i] for k, v in g.items()}
        root = optimize.brentq(lambda s, a=a, i=i: bsm_price(a["S"], a["K"], a["T"], a["r"],
                                                             a["q"], s, is_call) - price[i],
                               1e-6, 5.0, xtol=1e-14)
        assert iv[i] == pytest.approx(root, abs=1e-8)


def test_implied_vol_outside_bounds_is_nan():
    assert np.isnan(implied_vol(0.5, 100, 90, 1.0, 0.0, 0.0, True))  # below intrinsic
    assert np.isnan(implied_vol(101.0, 100, 90, 1.0, 0.0, 0.0, True))  # above spot
    assert np.isnan(implied_vol(5.0, 100, 100, 0.0, 0.0, 0.0, True))  # expired


def test_vectorised_over_scenarios_matches_loop():
    S = np.linspace(80, 120, 9)
    vec = bsm_price(S, 100.0, 0.5, 0.01, 0.0, 0.2, False)
    loop = [float(bsm_price(s, 100.0, 0.5, 0.01, 0.0, 0.2, False)) for s in S]
    np.testing.assert_allclose(vec, loop, rtol=0, atol=0)
