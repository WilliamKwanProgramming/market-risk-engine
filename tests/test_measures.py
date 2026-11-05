"""Finite-sample VaR/ES definitions and parametric VaR/ES of the normal and Student t."""

import numpy as np
import pytest
from scipy import integrate, stats

from mre.measures import (
    EmpiricalPnL,
    LocationScalePnL,
    empirical_var_es,
    normal_es,
    std_t_es,
    std_t_quantile,
)


def _losses_1_to_n(n, seed=0):
    """P&L sample whose losses are exactly 1, 2, ..., n, shuffled."""
    return np.random.default_rng(seed).permutation(-np.arange(1.0, n + 1))


def test_hs_250_by_hand():
    x = _losses_1_to_n(250)
    var, es = empirical_var_es(x, 0.99, 0.975)
    assert var == 248.0  # k = 2.5 -> 3rd worst loss
    # k = 6.25: the six worst losses plus a quarter of the seventh, over 6.25
    assert es == pytest.approx((250 + 249 + 248 + 247 + 246 + 245 + 0.25 * 244) / 6.25)


def test_hs_500_and_1000_by_hand():
    var, es = empirical_var_es(_losses_1_to_n(500), 0.99, 0.975)
    assert var == 496.0  # k = 5 -> 5th worst
    assert es == pytest.approx((500 + 499 + 498 + 497 + 496 + 495 + 494 + 493 + 492 + 491
                                + 490 + 489 + 0.5 * 488) / 12.5)
    var, es = empirical_var_es(_losses_1_to_n(1000), 0.99, 0.975)
    assert var == 991.0  # k = 10 -> 10th worst
    assert es == pytest.approx(np.mean(np.arange(976, 1001)))  # k = 25, plain mean


def test_es_is_lower_tail_and_at_least_var():
    rng = np.random.default_rng(1)
    x = rng.standard_t(4, 2000) + 0.3  # skew the sample away from symmetric
    for lv in (0.95, 0.975, 0.99):
        var, es = empirical_var_es(x, lv, lv)
        assert es >= var
        assert var == pytest.approx(-np.sort(x)[int(np.ceil(round(2000 * (1 - lv), 9))) - 1])
    # An upside-only sample: losses never occur, so VaR and ES are negative numbers
    var, es = empirical_var_es(np.arange(1.0, 101.0), 0.99, 0.975)
    assert var == -1.0 and es == pytest.approx(-(1 + 2 + 0.5 * 3) / 2.5)


def test_empirical_small_and_empty_samples():
    # n p < 1: the VaR and ES are the worst scenario
    assert empirical_var_es(np.arange(10.0), 0.99, 0.975) == (0.0, 0.0)
    with pytest.raises(ValueError):
        empirical_var_es(np.array([]), 0.99, 0.975)
    with pytest.raises(ValueError):
        empirical_var_es(np.arange(10.0), 1.0, 0.975)


def test_normal_var_es_against_integration():
    for lv in (0.975, 0.99):
        p = 1 - lv
        z = stats.norm.ppf(p)
        tail = integrate.quad(lambda u: u * stats.norm.pdf(u), -40, z)[0] / p
        assert normal_es(lv) == pytest.approx(-tail, rel=1e-9)
    d = LocationScalePnL(0.001, 0.02)
    assert d.var(0.99) == pytest.approx(-(0.001 + 0.02 * stats.norm.ppf(0.01)))
    assert d.var(0.99) == pytest.approx(0.02 * 2.3263478740408408 - 0.001)


@pytest.mark.parametrize("nu", [3.0, 5.0, 8.0, 30.0])
def test_student_t_unit_variance_quantile_and_es(nu):
    c = np.sqrt((nu - 2) / nu)
    assert stats.t.var(nu) * c**2 == pytest.approx(1.0)
    for lv in (0.975, 0.99):
        p = 1 - lv
        q = std_t_quantile(p, nu)
        assert stats.t.cdf(q / c, nu) == pytest.approx(p)
        tail = integrate.quad(lambda u: u * stats.t.pdf(u, nu), -np.inf, q / c)[0] / p
        assert std_t_es(lv, nu) == pytest.approx(-tail * c, rel=1e-8)
    # heavier tails at the same variance: t ES exceeds normal ES at 97.5%
    assert std_t_es(0.975, nu) > normal_es(0.975) - 1e-3


def test_location_scale_sampling_consistent_with_var_es():
    rng = np.random.default_rng(2)
    for d in (LocationScalePnL(0.0, 1.5), LocationScalePnL(0.2, 1.0, "t", 5.0)):
        x = d.sample(rng, 400_000)
        v, e = empirical_var_es(x, 0.99, 0.975)
        assert v == pytest.approx(d.var(0.99), rel=0.02)
        assert e == pytest.approx(d.es(0.975), rel=0.02)
        assert np.std(x) == pytest.approx(d.scale, rel=0.02)


def test_location_scale_validation():
    with pytest.raises(ValueError):
        LocationScalePnL(0.0, 1.0, "t", 2.0)
    with pytest.raises(ValueError):
        LocationScalePnL(0.0, 1.0, "cauchy")


def test_empirical_predictive_sampling_draws_from_scenarios():
    d = EmpiricalPnL(np.array([-3.0, -1.0, 0.0, 2.0]))
    s = d.sample(np.random.default_rng(3), 1000)
    assert set(np.unique(s)) <= {-3.0, -1.0, 0.0, 2.0}
