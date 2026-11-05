"""The real-data protocol code: STRANGLE roll, sample alignment, look-ahead and the summary.

Tests that read the skfolio data skip when skfolio is not installed (pip install skfolio).
"""

import numpy as np
import pandas as pd
import pytest

from mre.backtest import holm, newey_west_lag
from mre.experiments import realdata as rd
from mre.models import EWMAParametric, HistoricalSimulation, rolling_forecasts
from mre.portfolio import LinearPnL
from mre.pricing import bsm_price


def _spot(n=120, seed=0):
    idx = pd.bdate_range("1993-12-01", periods=n)
    r = np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.Series(450 * np.exp(np.cumsum(r)), index=idx)


def test_strangle_schedule_rolls_on_month_starts():
    s = _spot()
    sch = rd.StrangleSchedule.build(s, first=pd.Timestamp("1994-01-03"), last_days=21)
    d = sch.dates
    opens = sorted(set(sch.open_pos[sch.open_pos >= 0]))
    assert [d[o].strftime("%m-%d") for o in opens] == ["01-03", "02-01", "03-01", "04-01",
                                                       "05-02"]
    assert (sch.open_pos[d < "1994-01-03"] == -1).all()
    o1, o2 = opens[0], opens[1]
    n = o2 - o1
    assert sch.days_left[o1] == n and sch.days_left[o2 - 1] == 1
    assert sch.days_left[o2] == opens[2] - o2
    assert sch.strike_put[o1 + 3] == pytest.approx(0.95 * s.iloc[o1])
    assert sch.strike_call[o2] == pytest.approx(1.05 * s.iloc[o2])
    last = opens[-1]
    assert sch.days_left[last] == 21 and sch.days_left[-1] == 21 - (len(d) - 1 - last)


def test_strangle_realised_pnl_is_bsm_revaluation_including_expiry_day():
    s = _spot(300, seed=1)
    s.index = pd.bdate_range("1993-01-01", periods=300)
    inp = rd.strangle_input(s.rename("SP500"))
    spot = s.iloc[1:]
    iv = inp.extra["iv"]
    # pick the day on which the January 1994 pair expires (first trading day of February)
    t_date = pd.Timestamp("1994-02-01")
    r = inp.moves.index.get_loc(t_date)
    f = inp.pnl(r)
    prev = spot.index[spot.index.get_loc(t_date) - 1]
    assert f.state.spot["SP500"] == pytest.approx(spot[prev])
    assert all(p.expiry == pytest.approx(1 / 252) for p in f.portfolio.positions)
    x = f(inp.moves.iloc[[r]].to_numpy())[0]
    k_put, k_call = f.portfolio.positions[0].strike, f.portfolio.positions[1].strike
    s1, v0 = spot[t_date], iv[prev]
    v_prev = (bsm_price(spot[prev], k_put, 1 / 252, 0.03, 0.02, v0, False)
              + bsm_price(spot[prev], k_call, 1 / 252, 0.03, 0.02, v0, True))
    intrinsic = max(k_put - s1, 0) + max(s1 - k_call, 0)
    assert x == pytest.approx(-(intrinsic - v_prev), abs=1e-9)


@pytest.fixture(scope="module")
def market():
    pytest.importorskip("skfolio", reason="pip install skfolio to run the real-data tests")
    return rd.load_market()


def test_evaluation_samples_are_aligned(market):
    stocks, index, rep = market
    eq, spx, st = rd.eq20_input(stocks), rd.spx_input(index), rd.strangle_input(index)
    assert eq.moves.index[eq.start] == pd.Timestamp("1994-01-03")
    assert spx.moves.index[spx.start] == pd.Timestamp("1994-01-03")
    assert st.moves.index[st.start] == pd.Timestamp("1994-01-04")
    assert len(eq.moves) - eq.start == 7300 and len(st.moves) - st.start == 7299
    assert eq.start >= 1000 and st.start >= 1000  # HS1000 and FHS have their history
    assert rep == {"stocks_rows_dropped": 0, "index_rows_dropped": 0}
    assert eq.moves.index[-1] == st.moves.index[-1] == pd.Timestamp("2022-12-28")
    assert newey_west_lag(7300) == 10
    np.testing.assert_allclose(eq.moves.iloc[:, 0],
                               stocks.pct_change().iloc[1:].mean(axis=1).to_numpy())


def test_eq20_multivariate_ewma_equals_univariate(market):
    stocks, _, _ = market
    eq = rd.eq20_input(stocks)
    r = stocks.pct_change().iloc[1:]
    m = EWMAParametric(lam=0.94, window=500)
    uni = rolling_forecasts(m, eq.moves, eq.start, eq.start + 60)
    multi = rolling_forecasts(m, r, eq.start, eq.start + 60, pnl=LinearPnL(np.full(20, 0.05)))
    np.testing.assert_allclose([f.var for f in multi.forecasts],
                               [f.var for f in uni.forecasts], rtol=1e-12)
    np.testing.assert_allclose(multi.realized, uni.realized, rtol=1e-12)


def test_strangle_forecast_has_no_lookahead(market):
    _, index, _ = market
    inp = rd.strangle_input(index)
    t = inp.start + 30
    base = rolling_forecasts(HistoricalSimulation(250), inp.moves, t, t + 1, inp.pnl)
    # change every price from day t on: the IV rule and the moves of day t and later change
    changed = index.copy()
    changed.iloc[changed.index.get_loc(inp.moves.index[t]):] *= 0.7
    alt_inp = rd.strangle_input(changed)
    alt = rolling_forecasts(HistoricalSimulation(250), alt_inp.moves, t, t + 1, alt_inp.pnl)
    assert alt.forecasts[0].var == base.forecasts[0].var
    assert alt.realized[0] != base.realized[0]


def test_run_forecasts_and_evaluate_on_a_short_sample(market):
    stocks, _, _ = market
    inp = rd.eq20_input(stocks)
    methods = {k: v for k, v in rd.make_methods().items() if k in ("HS250", "EWMA-N")}
    fc, runs, es = rd.run_forecasts(inp, methods, end=inp.start + 300)
    assert list(runs["failed_days"]) == [0, 0] and len(fc) == 300
    assert es["days"].tolist() == [300, 300] and (es["n_sim"] == rd.N_SIM_AS).all()
    x = fc["realized"].to_numpy()
    exp = (x < -fc["HS250:var99"].to_numpy()).sum()
    for m in rd.METHODS:  # evaluate needs every method: reuse the two computed ones
        if m not in methods:
            for k in ("var99", "es975", "var975"):
                fc[f"{m}:{k}"] = fc[f"HS250:{k}"]
    es_all = pd.concat([es] + [es.iloc[[0]].assign(method=m) for m in rd.METHODS
                               if m not in methods])
    full, by_year, stress = rd.evaluate("EQ20", fc, es_all)
    row = full.set_index("method").loc["HS250"]
    assert row["exceptions"] == exp and row["days"] == 300
    assert set(by_year["year"]) == {1994, 1995}
    assert by_year[by_year.method == "HS250"]["exceptions"].sum() == exp
    assert (stress["days"] == 0).all()  # no stress window in 1994-1995


def test_failed_day_is_removed_for_every_method():
    idx = pd.bdate_range("2000-01-03", periods=5)
    fc = pd.DataFrame({"realized": [1.0, 2, 3, 4, 5], "A:var99": [1.0, np.nan, 1, 1, 1],
                       "B:var99": [1.0, 1, 1, 1, np.inf]}, index=idx)
    assert list(rd.common_sample(fc).index) == list(idx[[0, 2, 3]])


def test_shortfall_ratio_uses_the_975_exceedances():
    x = np.array([-3.0, -1.0, 0.5, -2.5])
    v = np.full(4, 2.0)
    e = np.array([2.5, 2.5, 2.5, 3.5])
    ratio, n = rd.shortfall_ratio(x, v, e)
    assert n == 2 and ratio == pytest.approx(2.75 / 3.0)
    assert np.isnan(rd.shortfall_ratio(np.zeros(3), np.ones(3), np.ones(3))[0])


def test_holm_family_is_all_cells():
    p = np.array([0.001, 0.02, 0.5])
    np.testing.assert_allclose(holm(p), [0.003, 0.04, 0.5])


def test_es_positive_requires_every_method():
    fc = pd.DataFrame({"realized": [0.0, 0, 0], "A:es975": [1.0, -0.1, 2.0],
                       "B:es975": [0.5, 1.0, 0.0], "A:var99": [-5.0, -5, -5]})
    assert list(rd.es_positive(fc)) == [True, False, False]
