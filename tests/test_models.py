"""Risk models: hand-computed values, cross-model consistency and the information convention."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mre.garch import GarchFit, GarchParams, fit_garch_t, garch_filter, simulate_garch_t
from mre.measures import empirical_var_es, normal_es, std_t_es, std_t_quantile
from mre.models import (
    EWMAParametric,
    FilteredHistoricalSimulation,
    HistoricalSimulation,
    MonteCarlo,
    ewma_covariance,
    rolling_forecasts,
)
from mre.portfolio import EuropeanOption, LinearPnL, MarketState, Portfolio, PortfolioPnL, Stock

P = GarchParams(mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)


def _returns(n=1600, seed=0):
    return simulate_garch_t(n, P, np.random.default_rng(seed))[0]


def test_ewma_covariance_by_hand():
    x = np.array([[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5]])
    lam = 0.9
    w = (1 - lam) * np.array([lam**2, lam, 1.0]) / (1 - lam**3)
    expected = sum(w[i] * np.outer(x[i], x[i]) for i in range(3))
    np.testing.assert_allclose(ewma_covariance(x, lam), expected, rtol=1e-14)
    np.testing.assert_allclose(ewma_covariance(x, lam, window=2), ewma_covariance(x[1:], lam))
    # univariate input
    assert ewma_covariance(x[:, 0], lam)[0, 0] == pytest.approx(expected[0, 0])


def test_ewma_matches_riskmetrics_recursion_for_long_window():
    r = _returns(1200, seed=1)
    lam = 0.94
    s2 = r[0] ** 2
    for v in r[1:]:
        s2 = lam * s2 + (1 - lam) * v**2  # sigma^2 after observing v
    assert ewma_covariance(r, lam)[0, 0] == pytest.approx(s2, rel=1e-12)


def test_hs_uses_exactly_last_window():
    r = _returns(800)
    m = HistoricalSimulation(250)
    pred, _ = m.predict(r[:600, None], None, LinearPnL([1.0]), 600)
    v, e = empirical_var_es(r[350:600], 0.99, 0.975)
    assert pred.var(0.99) == v and pred.es(0.975) == e
    with pytest.raises(ValueError):
        m.predict(r[:100, None], None, LinearPnL([1.0]), 100)


def test_ewma_parametric_normal_and_t():
    r = _returns(800)
    sd = np.sqrt(ewma_covariance(r[:700], 0.94, 500)[0, 0])
    pn, _ = EWMAParametric().predict(r[:700, None], None, LinearPnL([1.0]), 700)
    assert pn.var(0.99) == pytest.approx(stats.norm.ppf(0.99) * sd)
    assert pn.es(0.975) == pytest.approx(normal_es(0.975) * sd)
    pt, _ = EWMAParametric(dist="t", nu=5.0).predict(r[:700, None], None, LinearPnL([1.0]), 700)
    assert pt.var(0.99) == pytest.approx(-std_t_quantile(0.01, 5.0) * sd)
    assert pt.es(0.975) == pytest.approx(std_t_es(0.975, 5.0) * sd)
    # a position of size 3 has three times the risk
    p3, _ = EWMAParametric().predict(r[:700, None], None, LinearPnL([3.0]), 700)
    assert p3.var(0.99) == pytest.approx(3 * pn.var(0.99))


def test_fhs_scenarios_by_hand():
    r = _returns(1200)
    m = FilteredHistoricalSimulation(window=1000, refit_every=250)
    hist = r[:1100, None]
    fitted = m.fit(hist)
    pred, diag = m.predict(hist, fitted, LinearPnL([1.0]), 1100)
    p = fitted[0].params
    s2, s2n = garch_filter(r[100:1100], p)
    scen = p.mu + np.sqrt(s2n) * (r[100:1100] - p.mu) / np.sqrt(s2)
    v, e = empirical_var_es(scen, 0.99, 0.975)
    assert pred.var(0.99) == pytest.approx(v) and pred.es(0.975) == pytest.approx(e)
    assert diag["converged"] is True
    assert fitted[0].params == fit_garch_t(r[100:1100]).params


def test_fhs_bootstrap_draws_residual_dates_with_the_day_seed():
    r = _returns(1200)
    exact = FilteredHistoricalSimulation(window=1000, refit_every=20)
    boot = FilteredHistoricalSimulation(window=1000, refit_every=20, n_boot=5000, seed=11)
    hist = r[:1100, None]
    fitted = exact.fit(hist)
    full, _ = exact.predict(hist, fitted, LinearPnL([1.0]), 1100)
    pred, _ = boot.predict(hist, fitted, LinearPnL([1.0]), 1100)
    idx = np.random.default_rng([11, 1100]).integers(0, 1000, size=5000)
    np.testing.assert_array_equal(pred.scenarios, full.scenarios[idx])
    again, _ = boot.predict(hist, fitted, LinearPnL([1.0]), 1100)
    np.testing.assert_array_equal(again.scenarios, pred.scenarios)
    other, _ = boot.predict(hist, fitted, LinearPnL([1.0]), 1101)
    assert not np.array_equal(other.scenarios, pred.scenarios)


def test_rolling_forecasts_callback_memory_and_failures():
    x = _returns(400, seed=4)
    seen = []
    path = rolling_forecasts(HistoricalSimulation(250), x, 300, 320,
                             on_forecast=lambda t, f, xr: seen.append((t, f.predictive, xr)),
                             keep_predictive=False)
    assert [s[0] for s in seen] == list(range(300, 320))
    assert all(f.predictive is None for f in path.forecasts)
    assert all(s[1] is not None for s in seen)
    np.testing.assert_allclose([s[2] for s in seen], path.realized)
    fr = path.frame()
    expected = [empirical_var_es(x[t - 250:t], 0.975, 0.975)[0] for t in range(300, 320)]
    np.testing.assert_allclose(fr["var_at_es_level"], expected)

    class Flaky(HistoricalSimulation):
        def predict(self, history, fitted, pnl, index):
            if index == 305:
                raise np.linalg.LinAlgError("boom")
            return super().predict(history, fitted, pnl, index)

    with pytest.raises(np.linalg.LinAlgError):
        rolling_forecasts(Flaky(250), x, 300, 320)
    calls = []
    rec = rolling_forecasts(Flaky(250), x, 300, 320, on_error="record",
                            on_forecast=lambda t, f, xr: calls.append(t))
    assert np.isnan(rec.forecasts[5].var) and "boom" in rec.forecasts[5].diagnostics["error"]
    assert 305 not in calls and len(calls) == 19
    assert rec.forecasts[6].var == path.forecasts[6].var


def test_fhs_with_true_parameters_tracks_true_var():
    """With the true parameters and many residuals, FHS VaR is close to the true VaR."""
    r, s2 = simulate_garch_t(3000, P, np.random.default_rng(5))
    m = FilteredHistoricalSimulation(window=2999)
    fit = [GarchFit(P, 0.0, True, "", 0, 2999)]
    pred, _ = m.predict(r[:2999, None], fit, LinearPnL([1.0]), 2999)
    true_var = -np.sqrt(s2[2999]) * std_t_quantile(0.01, P.nu)
    # 2999 residuals: the 1% empirical quantile has a relative sd of roughly 5%
    assert pred.var(0.99) == pytest.approx(true_var, rel=0.15)


def test_monte_carlo_antithetic_and_reproducible():
    r = np.random.default_rng(2).standard_normal((700, 2)) * [0.01, 0.02]
    m = MonteCarlo(n_draws=2000, seed=7)
    cov = ewma_covariance(r, 0.94, 500)
    x = m.draw_moves(cov, 5)
    np.testing.assert_allclose(x[:1000], -x[1000:])
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_array_equal(x, m.draw_moves(cov, 5))
    assert not np.array_equal(x, m.draw_moves(cov, 6))
    # linear P&L: the sample mean is exactly zero thanks to the antithetic pairs
    pred, _ = m.predict(r, None, LinearPnL([1.0, 1.0]), 5)
    assert abs(pred.scenarios.mean()) < 1e-17
    with pytest.raises(ValueError):
        MonteCarlo(n_draws=101)


@pytest.mark.parametrize(("dist", "nu"), [("normal", 6.0), ("t", 5.0)])
def test_monte_carlo_converges_to_parametric(dist, nu):
    r = np.random.default_rng(3).standard_normal((600, 2)) @ np.array([[0.01, 0.0],
                                                                      [0.006, 0.008]])
    e = LinearPnL([2.0, -1.0])
    mc, _ = MonteCarlo(dist=dist, nu=nu, n_draws=400_000, seed=1).predict(r, None, e, 600)
    pa, _ = EWMAParametric(dist=dist, nu=nu).predict(r, None, e, 600)
    assert mc.var(0.99) == pytest.approx(pa.var(0.99), rel=0.01)
    assert mc.es(0.975) == pytest.approx(pa.es(0.975), rel=0.01)
    np.testing.assert_allclose(np.cov(MonteCarlo(dist=dist, nu=nu, n_draws=400_000).draw_moves(
        ewma_covariance(r, 0.94, 500), 0).T), ewma_covariance(r, 0.94, 500), rtol=0.03)


def test_rolling_forecasts_realized_and_exceptions():
    r = pd.Series(_returns(700), index=pd.bdate_range("2001-01-01", periods=700))
    path = rolling_forecasts(HistoricalSimulation(250), r, start=600)
    fr = path.frame()
    assert len(fr) == 100 and fr.index[0] == r.index[600]
    np.testing.assert_array_equal(fr["realized"], r.iloc[600:].to_numpy())
    np.testing.assert_array_equal(fr["exception"], fr["realized"] < -fr["var"])
    for i in (0, 57, 99):
        v, e = empirical_var_es(r.iloc[350 + i:600 + i], 0.99, 0.975)
        assert fr["var"].iloc[i] == v and fr["es"].iloc[i] == e
    with pytest.raises(ValueError):
        rolling_forecasts(HistoricalSimulation(250), r, start=100)


def test_refit_schedule():
    r = _returns(1700)
    path = rolling_forecasts(FilteredHistoricalSimulation(1000, refit_every=250), r, 1000, 1600)
    assert len(path.fits) == 3  # refits at rows 1000, 1250, 1500
    assert path.fits[1][0].params == fit_garch_t(r[250:1250]).params


def test_rolling_with_portfolio_state_per_day():
    spot = 100 * np.exp(np.cumsum(_returns(400, seed=4)))
    moves = np.column_stack([np.diff(np.log(spot)), np.zeros(399)])
    port = Portfolio([Stock("A", 1.0), EuropeanOption("A", 100.0, 1.0, False, 2.0)])

    def pnl_at(t):
        return PortfolioPnL(port, MarketState({"A": spot[t]}, {"A": 0.2}))

    path = rolling_forecasts(HistoricalSimulation(250), moves, 300, 310, pnl=pnl_at)
    t = 305
    real = port.pnl_full(MarketState({"A": spot[t]}, {"A": 0.2}), moves[t:t + 1]).sum()
    assert path.realized[5] == pytest.approx(real)


MODELS = [
    HistoricalSimulation(250),
    HistoricalSimulation(1000),
    EWMAParametric(),
    EWMAParametric(dist="t"),
    FilteredHistoricalSimulation(1000, refit_every=100),
    FilteredHistoricalSimulation(1000, refit_every=20, n_boot=3000, seed=4),
    MonteCarlo(n_draws=2000, seed=3),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_lookahead_guard(model):
    """The forecast for day t (made at the close of t-1) does not change when the data of day t
    and later are changed or deleted."""
    x = _returns(1400, seed=9)
    t = 1250
    base = rolling_forecasts(model, x, 1100, 1300)
    changed = x.copy()
    changed[t:] = np.random.default_rng(1).standard_normal(1400 - t) * 0.05
    changed[t] = -0.25  # a loss far in the tail: any leak of day t into its forecast shows
    alt = rolling_forecasts(model, changed, 1100, 1300)
    cut = rolling_forecasts(model, x[:t + 1], 1100, t + 1)  # nothing after day t
    for i in range(t - 1100 + 1):  # every forecast up to and including the one for day t
        assert alt.forecasts[i].var == base.forecasts[i].var
        assert alt.forecasts[i].es == base.forecasts[i].es
        assert cut.forecasts[i].var == base.forecasts[i].var
    # ... the realised P&L of day t does use day t, and the forecast for day t depends on
    # day t - 1 (so the guard is not vacuous)
    assert alt.realized[t - 1100] != base.realized[t - 1100]
    shifted = x.copy()
    shifted[t - 1] = -0.08
    later = rolling_forecasts(model, shifted, 1100, 1300)
    assert later.forecasts[t - 1100].var != base.forecasts[t - 1100].var
