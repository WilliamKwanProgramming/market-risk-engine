"""VaR and ES backtests: hand-computed statistics, exact distributions and degenerate cases."""

from dataclasses import dataclass
from math import log

import numpy as np
import pytest
from scipy import stats

from mre.backtest import (
    BASEL_PLUS_FACTOR,
    AcerbiSzekelySimulation,
    acerbi_szekely,
    binomial_zone_probabilities,
    christoffersen,
    diebold_mariano,
    exceptions,
    fz0_loss,
    holm,
    kupiec_pof,
    newey_west_lag,
    traffic_light,
    wilson_interval,
    zone_occupancy,
)
from mre.measures import EmpiricalPnL, LocationScalePnL, Predictive


def test_exceptions_strict_inequality():
    np.testing.assert_array_equal(exceptions([-2.0, -1.0, -0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
                                  [True, False, False, False])


def test_kupiec_by_hand():
    zero = kupiec_pof(np.zeros(250, bool), 0.99)
    assert zero.statistic == pytest.approx(-2 * 250 * log(0.99))
    assert zero.statistic == pytest.approx(5.025167926, abs=1e-8)
    assert zero.df == 1
    assert zero.p_value == pytest.approx(stats.chi2.sf(5.025167926, 1))
    e = np.zeros(250, bool)
    e[[3, 50, 51, 120, 200]] = True
    five = kupiec_pof(e, 0.99)
    lr = -2 * (5 * log(0.01) + 245 * log(0.99) - 5 * log(0.02) - 245 * log(0.98))
    assert five.statistic == pytest.approx(lr)
    assert five.p_value == pytest.approx(stats.chi2.sf(lr, 1))
    # p-value from chi2(1), not chi2(2)
    assert five.p_value != pytest.approx(stats.chi2.sf(lr, 2))
    allexc = kupiec_pof(np.ones(10, bool), 0.99)
    assert allexc.statistic == pytest.approx(-2 * 10 * log(0.01))
    assert np.isnan(kupiec_pof(np.array([], bool)).statistic)


def test_kupiec_rejection_probability_matches_exact_binomial():
    """The test is a function of the count only: its exact size is a binomial sum."""
    n, p = 250, 0.01
    rejects = [kupiec_pof(np.arange(n) < x, 0.99).reject(0.05) for x in range(n + 1)]
    exact_size = sum(stats.binom.pmf(x, n, p) for x in range(n + 1) if rejects[x])
    # rejection region at n = 250: x = 0 or x >= 7 (p-values 0.025 at x=0, 0.12 at x=1)
    assert [x for x in range(12) if rejects[x]] == [0, 7, 8, 9, 10, 11]
    assert exact_size == pytest.approx(stats.binom.pmf(0, n, p) + stats.binom.sf(6, n, p))


def test_christoffersen_by_hand():
    e = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], bool)
    res = christoffersen(e, 0.99)
    d = res["ind"].details
    assert (d["n00"], d["n01"], d["n10"], d["n11"]) == (5, 1, 2, 1)
    pi01, pi11, pi = 1 / 6, 1 / 3, 2 / 9
    ll_markov = 5 * log(1 - pi01) + log(pi01) + 2 * log(1 - pi11) + log(pi11)
    ll_iid = 7 * log(1 - pi) + 2 * log(pi)
    ll_null = 7 * log(0.99) + 2 * log(0.01)
    assert res["ind"].statistic == pytest.approx(-2 * (ll_iid - ll_markov))
    assert res["cc"].statistic == pytest.approx(-2 * (ll_null - ll_markov))
    assert res["ind"].df == 1 and res["cc"].df == 2
    assert res["cc"].p_value == pytest.approx(stats.chi2.sf(res["cc"].statistic, 2))


def test_christoffersen_counts_not_symmetric():
    """A sequence starting with an exception has n10 = n01 + 1: the counts must not be swapped."""
    e = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], bool)
    d = christoffersen(e)["ind"].details
    assert (d["n00"], d["n01"], d["n10"], d["n11"]) == (6, 1, 2, 2)
    assert d["pi11"] == pytest.approx(1 / 2) and d["pi01"] == pytest.approx(1 / 7)


def test_cc_equals_uc_plus_ind_on_transitions():
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.random(300) < 0.03
        res = christoffersen(e, 0.99)
        uc = kupiec_pof(e[1:], 0.99).statistic
        assert res["cc"].statistic == pytest.approx(uc + res["ind"].statistic, abs=1e-9)


def test_christoffersen_degenerate_cases():
    zero = christoffersen(np.zeros(250, bool), 0.99)
    assert zero["ind"].statistic == 0.0 and zero["ind"].p_value == 1.0
    assert not zero["ind"].details["ind_testable"]
    assert zero["cc"].statistic == pytest.approx(-2 * 249 * log(0.99))
    last_only = np.zeros(100, bool)
    last_only[-1] = True  # no exception among the first n-1 days: pi11 not identified
    r = christoffersen(last_only)
    assert r["ind"].statistic == pytest.approx(0.0, abs=1e-12)
    assert not r["ind"].details["ind_testable"]
    one = christoffersen(np.array([True]))
    assert np.isnan(one["ind"].statistic) and np.isnan(one["cc"].p_value)
    assert not one["ind"].reject()


def test_christoffersen_detects_clustering():
    e = np.zeros(1000, bool)
    for s in (100, 300, 500, 700, 900):
        e[s:s + 2] = True  # five pairs of consecutive exceptions, rate 1%
    res = christoffersen(e, 0.99)
    assert kupiec_pof(e, 0.99).p_value > 0.5  # right unconditional rate ...
    assert res["ind"].p_value < 0.001  # ... but clustered


def test_traffic_light_boundaries_and_plus_factors():
    zones = [traffic_light(k).zone for k in range(13)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 3
    # Basel Committee (1996), Table 2
    assert [traffic_light(k).plus_factor for k in range(11)] == [
        0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.00]
    assert traffic_light(25).plus_factor == 1.0
    assert BASEL_PLUS_FACTOR[9] == 0.85
    with pytest.raises(ValueError):
        traffic_light(-1)


def test_binomial_probabilities_match_basel_table():
    """Cumulative probabilities printed in Basel Committee (1996), Table 2, to 0.01%."""
    table = [8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99]
    ours = [100 * stats.binom.cdf(k, 250, 0.01) for k in range(11)]
    np.testing.assert_allclose(ours, table, atol=0.005 + 1e-9)
    z = binomial_zone_probabilities(250, 0.01)
    assert z["green"] == pytest.approx(stats.binom.cdf(4, 250, 0.01))
    assert z["red"] == pytest.approx(stats.binom.sf(9, 250, 0.01))
    assert sum(z.values()) == pytest.approx(1.0)


def test_zone_occupancy_against_loop():
    rng = np.random.default_rng(1)
    e = rng.random(800) < 0.02
    counts, frac = zone_occupancy(e, 250)
    brute = [int(e[t - 249:t + 1].sum()) for t in range(249, 800)]
    np.testing.assert_array_equal(counts.to_numpy(), brute)
    assert counts.index[0] == 249
    zones = [traffic_light(c).zone for c in brute]
    assert frac["yellow"] == pytest.approx(zones.count("yellow") / len(zones))
    with pytest.raises(ValueError):
        zone_occupancy(e[:100], 250)


@dataclass(frozen=True)
class _Fixed(Predictive):
    v: float
    e: float

    def var(self, level):
        return self.v

    def es(self, level):
        return self.e

    def sample(self, rng, size):
        return np.zeros(size)


def test_acerbi_szekely_by_hand():
    x = np.array([-3.0, 1.0, -0.5, -2.5])
    preds = [_Fixed(2, 3), _Fixed(2, 3), _Fixed(1, 2), _Fixed(2, 3)]
    res = acerbi_szekely(x, preds, level=0.975, n_sim=10)
    ratio = -3 / 3 + -2.5 / 3
    assert res["Z1"].statistic == pytest.approx(ratio / 2 + 1)
    assert res["Z2"].statistic == pytest.approx(ratio / (4 * 0.025) + 1)
    assert res["Z2"].details["n_exceptions"] == 2
    # simulated paths are all zero: no exceptions, Z1 = 0 (convention) and Z2 = 1
    assert res["Z2"].p_value == pytest.approx(1 / 11)  # only the "+1" counts
    none = acerbi_szekely(np.zeros(4), preds, n_sim=10)
    assert none["Z1"].statistic == 0.0 and none["Z2"].statistic == 1.0


def test_acerbi_szekely_size_and_power():
    """Model N(0, 1) for 250 days. Null: data N(0, 1). Alternative: data N(0, 1.25^2)."""
    rng = np.random.default_rng(2)
    model = [LocationScalePnL(0.0, 1.0)] * 250
    rej_null, rej_alt = [], []
    for i in range(100):
        x_null = rng.standard_normal(250)
        x_alt = 1.25 * rng.standard_normal(250)
        rej_null.append(acerbi_szekely(x_null, model, n_sim=300, seed=i)["Z2"].reject())
        rej_alt.append(acerbi_szekely(x_alt, model, n_sim=300, seed=i)["Z2"].reject())
    # 100 replications: binomial sd of the rejection rate at 5% is 2.2 percentage points
    assert 0.0 < np.mean(rej_null) < 0.12
    assert np.mean(rej_alt) > 0.6


def _as_dense_reference(x, preds, level, n_sim, seed):
    """Wave-1 implementation: the whole (T, n_sim) array at once (independent reference)."""
    p = 1.0 - level
    var = np.array([d.var(level) for d in preds])
    es = np.array([d.es(level) for d in preds])
    rng = np.random.default_rng(seed)
    sims = np.vstack([d.sample(rng, n_sim) for d in preds])

    def stats_(z):
        ind = (z + var[:, None]) < 0
        ratio = np.where(ind, z / es[:, None], 0.0)
        n = ind.sum(axis=0)
        z1 = np.where(n > 0, ratio.sum(axis=0) / np.maximum(n, 1) + 1.0, 0.0)
        return z1, ratio.sum(axis=0) / (z.shape[0] * p) + 1.0

    z1o, z2o = (float(v[0]) for v in stats_(x[:, None]))
    z1s, z2s = stats_(sims)
    return (z1o, (1 + np.sum(z1s <= z1o)) / (n_sim + 1), z2o,
            (1 + np.sum(z2s <= z2o)) / (n_sim + 1))


def test_acerbi_szekely_matches_dense_reference():
    rng = np.random.default_rng(7)
    T = 300
    preds = [LocationScalePnL(0.0, s, "t", 5.0) for s in rng.uniform(0.5, 2.0, T // 2)]
    preds += [EmpiricalPnL(rng.standard_normal(400) * s) for s in rng.uniform(0.5, 2, T // 2)]
    x = 1.2 * np.array([d.sample(rng, 1)[0] for d in preds])
    res = acerbi_szekely(x, preds, level=0.975, n_sim=500, seed=3)
    z1, p1, z2, p2 = _as_dense_reference(x, preds, 0.975, 500, 3)
    assert res["Z1"].statistic == pytest.approx(z1, abs=1e-12)
    assert res["Z2"].statistic == pytest.approx(z2, abs=1e-12)
    assert res["Z1"].p_value == p1 and res["Z2"].p_value == p2


def test_acerbi_szekely_subset_equals_running_on_the_subset():
    rng = np.random.default_rng(8)
    preds = [LocationScalePnL(0.0, s) for s in rng.uniform(0.5, 2.0, 200)]
    x = 1.3 * np.array([d.sample(rng, 1)[0] for d in preds])
    sim = AcerbiSzekelySimulation(0.975, 400, seed=1)
    for t in range(200):
        sim.add(t, x[t], preds[t])
    keep = np.arange(0, 200, 2)
    sub = sim.result(keep)
    # the same simulated draws restricted to the kept days, recomputed densely
    rng2 = np.random.default_rng(1)
    draws = np.vstack([d.sample(rng2, 400) for d in preds])[keep]
    var = np.array([preds[t].var(0.975) for t in keep])
    es = np.array([preds[t].es(0.975) for t in keep])
    ratio = np.where(draws + var[:, None] < 0, draws / es[:, None], 0.0)
    z2s = ratio.sum(axis=0) / (keep.size * 0.025) + 1.0
    xo = x[keep]
    z2o = np.sum(np.where(xo + var < 0, xo / es, 0.0)) / (keep.size * 0.025) + 1.0
    assert sub["Z2"].statistic == pytest.approx(z2o)
    assert sub["Z2"].p_value == (1 + np.sum(z2s <= z2o)) / 401
    assert sub["Z2"].details["n_days"] == 100


def test_acerbi_szekely_memory_does_not_grow_with_days_times_sims():
    import tracemalloc

    T, M = 1000, 5000
    preds = [LocationScalePnL(0.0, 1.0)] * T
    x = np.zeros(T)
    tracemalloc.start()
    acerbi_szekely(x, preds, n_sim=M)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # a (T, M) float array alone would take 40 MB; kept tail events are about 2.5% of draws
    assert peak < 12e6


def test_fz0_by_hand_and_minimised_at_truth():
    # Y below v: L = -(v - Y)/(a e) + v/e + log(-e) - 1, with v = -VaR, e = -ES
    y, var, es, a = -3.0, 2.0, 2.5, 0.025
    expected = -(-2.0 + 3.0) / (a * -2.5) + (-2.0 / -2.5) + log(2.5) - 1
    assert fz0_loss([y], [var], [es], 0.975)[0] == pytest.approx(expected)
    above = fz0_loss([1.0], [var], [es], 0.975)[0]
    assert above == pytest.approx(0.8 + log(2.5) - 1)
    y = np.random.default_rng(3).standard_normal(400_000)
    v0, e0 = stats.norm.ppf(0.975), stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025
    n = y.size
    truth = fz0_loss(y, np.full(n, v0), np.full(n, e0), 0.975).mean()
    for dv, de in ((1.1, 1.0), (0.9, 1.0), (1.0, 1.1), (1.0, 0.9), (0.9, 0.9)):
        assert truth < fz0_loss(y, np.full(n, v0 * dv), np.full(n, e0 * de), 0.975).mean()
    with pytest.raises(ValueError):
        fz0_loss([0.0], [1.0], [-1.0], 0.975)


def test_diebold_mariano_against_statsmodels_hac():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(4)
    u = rng.standard_normal(1001)
    d = u[1:] + 0.5 * u[:-1] + 0.05  # MA(1) loss differential with a small mean
    res = diebold_mariano(d, np.zeros(1000))
    assert res.details["lag"] == newey_west_lag(1000) == 6
    ols = sm.OLS(d, np.ones((1000, 1))).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": 6, "use_correction": False})
    assert res.statistic == pytest.approx(float(ols.tvalues[0]), rel=1e-10)
    assert res.p_value == pytest.approx(2 * stats.norm.sf(abs(res.statistic)))
    zero_lag = diebold_mariano(d, np.zeros(1000), lag=0)
    assert zero_lag.statistic == pytest.approx(d.mean() / (d.std() / np.sqrt(1000)))


def test_newey_west_lag_rule():
    assert [newey_west_lag(n) for n in (100, 250, 1000, 8000)] == [4, 4, 6, 10]
    assert np.isnan(diebold_mariano(np.ones(50), np.ones(50)).statistic)


def test_wilson_interval_by_hand_and_against_statsmodels():
    lo, hi = wilson_interval(10, 1000)
    z = stats.norm.ppf(0.975)
    ph = 0.01
    c, h = ph + z * z / 2000, z * np.sqrt(ph * 0.99 / 1000 + z * z / 4e6)
    assert lo == pytest.approx((c - h) / (1 + z * z / 1000))
    assert hi == pytest.approx((c + h) / (1 + z * z / 1000))
    smp = pytest.importorskip("statsmodels.stats.proportion")
    for x, n in ((0, 250), (3, 250), (87, 7300), (250, 250)):
        ref = smp.proportion_confint(x, n, alpha=0.05, method="wilson")
        np.testing.assert_allclose(wilson_interval(x, n), ref, rtol=1e-10, atol=1e-14)


def test_holm_by_hand_and_against_statsmodels():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    # sorted finite: 0.005, 0.01, 0.03, 0.04 with m = 4 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02, np.nan])
    mt = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(3)
    q = rng.uniform(0, 0.2, 40) ** 2
    np.testing.assert_allclose(holm(q), mt.multipletests(q, method="holm")[1], rtol=1e-12)
    assert np.isnan(holm([np.nan])).all()
