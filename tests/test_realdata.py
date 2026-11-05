"""The real-data protocol code: STRANGLE roll, sample alignment, look-ahead and the summary.

Tests that read the skfolio data skip when skfolio is not installed (pip install skfolio).
"""

import numpy as np
import pandas as pd
import pytest

from mre.backtest import AcerbiSzekelySimulation, holm, newey_west_lag, z2_contributions
from mre.experiments import realdata as rd
from mre.measures import LocationScalePnL
from mre.models import EWMAParametric, rolling_forecasts
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


@pytest.mark.parametrize("method", rd.METHODS)
def test_strangle_forecast_has_no_lookahead(market, method):
    """Changing every price from day t on changes the realised P&L of day t, not the forecast."""
    _, index, _ = market
    inp = rd.strangle_input(index)
    t = inp.start + 30
    model = rd.make_methods()[method]  # the protocol's parameters, one forecast day
    base = rolling_forecasts(model, inp.moves, t, t + 1, inp.pnl)
    # change every price from day t on: the IV rule and the moves of day t and later change
    changed = index.copy()
    changed.iloc[changed.index.get_loc(inp.moves.index[t]):] *= 0.7
    alt_inp = rd.strangle_input(changed)
    alt = rolling_forecasts(rd.make_methods()[method], alt_inp.moves, t, t + 1, alt_inp.pnl)
    a, b = base.forecasts[0], alt.forecasts[0]
    assert np.isfinite(a.var)
    assert (a.var, a.es, a.var_at_es_level) == (b.var, b.es, b.var_at_es_level)
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
    # the post hoc dominance columns are computed, and Z2 is 1 + the sum of its daily terms
    assert set(rd.Z2_DOMINANCE_COLUMNS) <= set(es.columns)
    assert set(rd.DM_DOMINANCE_COLUMNS) <= set(full.columns)
    for m in ("HS250", "EWMA-N"):
        c = z2_contributions(fc["realized"], fc[f"{m}:var975"], fc[f"{m}:es975"], rd.LEVEL_ES)
        assert es.set_index("method").loc[m, "z2"] == pytest.approx(1.0 + c.sum())
    assert np.isnan(full.set_index("method").loc["EWMA-N", "dm_top_share"])  # the reference
    assert 0.0 < full.set_index("method").loc["HS250", "dm_top_share"] <= 1.0
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


def test_z2_dominance_names_the_day_that_makes_the_statistic():
    rng = np.random.default_rng(11)
    n, bad = 300, 120
    sigma = np.ones(n)
    sigma[bad] = 1e-4  # predicted ES close to zero on one day
    preds = [LocationScalePnL(0.0, s) for s in sigma]
    x = rng.standard_normal(n)
    x[bad] = -1.0  # an ordinary loss on that day: 10,000 predicted standard deviations
    idx = pd.bdate_range("2001-01-01", periods=n)
    fc = pd.DataFrame({"realized": x, "A:var975": [d.var(0.975) for d in preds],
                       "A:es975": [d.es(0.975) for d in preds]}, index=idx)
    sim = AcerbiSzekelySimulation(rd.LEVEL_ES, 400, seed=5)
    for t in range(n):
        sim.add(t, x[t], preds[t])
    ok, rows = np.ones(n, dtype=bool), np.arange(n)
    out = rd.z2_dominance(sim, fc, ok, rows, "A")
    z2 = sim.result(rows)["Z2"]
    c = z2_contributions(x, fc["A:var975"], fc["A:es975"], rd.LEVEL_ES)
    assert z2.statistic == pytest.approx(1.0 + c.sum())
    assert out["z2_top_day"] == str(idx[bad].date())
    assert out["z2_top_share"] > 0.99 and out["z2_top5_share"] >= out["z2_top_share"]
    assert out["z2_top_es_ratio"] < 1e-3
    # without that day the statistic is an ordinary one; with it the statistic is its term
    assert z2.statistic < -100.0 and abs(out["z2_drop1"]) < 5.0
    assert out["z2_drop1"] == pytest.approx(
        sim.result(np.delete(rows, bad))["Z2"].statistic)
    assert out["z2_drop1_p"] == sim.result(np.delete(rows, bad))["Z2"].p_value
    assert 0.0 < out["z2_drop5_p"] <= 1.0
    # a sample without a 97.5% exceedance has nothing to attribute
    fc2 = fc.assign(realized=0.1)
    sim2 = AcerbiSzekelySimulation(rd.LEVEL_ES, 50, seed=5)
    for t in range(n):
        sim2.add(t, 0.1, preds[t])
    none = rd.z2_dominance(sim2, fc2, ok, rows, "A")
    assert none["z2_top_day"] == "" and np.isnan(none["z2_top_share"])


def test_dm_dominance_names_the_day_that_makes_the_statistic():
    rng = np.random.default_rng(3)
    n = 400
    ref = rng.standard_normal(n) * 0.1
    fz = ref + rng.standard_normal(n) * 0.1
    fz[77] += 500.0  # one day with a huge loss differential
    idx = pd.bdate_range("2001-01-01", periods=n)
    out = rd.dm_dominance(fz, ref, idx)
    assert out["dm_top_day"] == str(idx[77].date()) and out["dm_top_share"] > 0.9
    # the statistic with the day is that of a single spike; without it, of the noise
    from mre.backtest import diebold_mariano

    assert diebold_mariano(fz, ref).statistic == pytest.approx(1.0, abs=0.5)
    assert out["dm_drop1_stat"] == pytest.approx(
        diebold_mariano(np.delete(fz, 77), np.delete(ref, 77)).statistic)
    assert abs(out["dm_drop1_stat"]) < 3.0 and np.isfinite(out["dm_drop5_p"])


def test_data_fingerprint_check_stops_on_changed_data(monkeypatch):
    good = dict(rd.DATA_SHA256)
    monkeypatch.delenv("MRE_ALLOW_DATA_CHANGE", raising=False)
    rd.verify_data(good)  # the pinned fingerprints pass
    bad = {**good, "index": "0" * 64}
    with pytest.raises(RuntimeError, match="index data differ"):
        rd.verify_data(bad)
    monkeypatch.setenv("MRE_ALLOW_DATA_CHANGE", "1")
    rd.verify_data(bad)  # the override lets the run go ahead


def test_installed_skfolio_data_match_the_pinned_fingerprints(market):
    from mre import data

    got = rd.data_fingerprints(data.load_skfolio_prices(), data.load_skfolio_index())
    assert got == rd.DATA_SHA256


def test_iv_coupling_checks_on_a_synthetic_index():
    idx = pd.bdate_range("1990-01-02", periods=2200)
    r = np.random.default_rng(5).standard_t(5, len(idx)) * 0.008
    index = pd.Series(300 * np.exp(np.cumsum(r)), index=idx, name="SP500")
    rows = {row["check"].split(": ")[1]: row for row in rd.iv_coupling_checks(index)}
    err = rows["max abs error of log IV(t) rebuilt from IV(t-1) and the return of t"]
    assert err["value"] < 1e-9 and "above the floor" in err["note"]
    corr_abs = rows["correlation of the log IV move with the absolute log return"]["value"]
    corr_r = rows["correlation of the log IV move with the log return"]["value"]
    assert corr_abs > 0.4 and abs(corr_r) < 0.3


def test_stale_price_checks_find_a_stock_with_zero_returns_and_a_large_move():
    idx = pd.bdate_range("1990-01-02", periods=1500)
    rng = np.random.default_rng(2)
    stocks = pd.DataFrame({"AAA": 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))),
                           "BBB": 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))}, index=idx)
    stocks.loc[idx[10:310], "BBB"] = stocks.loc[idx[10], "BBB"]  # stale for 300 days
    stocks.loc[idx[400]:, "AAA"] *= 0.3  # a -70% day
    rows = rd.stale_price_checks(stocks)
    assert rows[0]["value"] > 0.25 and rows[0]["note"].startswith("BBB")
    expected = abs(stocks["AAA"].iloc[400] / stocks["AAA"].iloc[399] - 1.0)
    assert rows[1]["value"] == pytest.approx(expected) and 0.69 < expected < 0.72
    assert rows[1]["note"] == f"AAA on {idx[400].date()}"
