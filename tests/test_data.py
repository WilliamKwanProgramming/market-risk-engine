"""Cleaning, returns, the implied-volatility rule and the real-data loaders (smoke checks)."""

import numpy as np
import pandas as pd
import pytest

from mre.data import (
    clean_prices,
    ewma_vol_rule,
    factor_levels,
    load_arch,
    load_csv,
    to_returns,
)


def _prices():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-06",
                            "2020-01-07", "2020-01-08"])
    return pd.DataFrame({"A": [10.0, 11.0, 11.5, np.nan, 12.0, 12.5],
                         "B": [np.nan, 5.0, 5.0, 5.5, np.nan, 6.0]}, index=idx)


def test_clean_drop_policy():
    df, rep = clean_prices(_prices(), "drop")
    assert list(df.index.strftime("%m-%d")) == ["01-03", "01-08"]
    assert rep["weekend_rows_dropped"] == 1  # 2020-01-04 is a Saturday
    assert rep["rows_dropped"] == 3 and rep["rows_out"] == 2


def test_clean_ffill_policy_respects_limit_and_leading_gaps():
    df, rep = clean_prices(_prices(), "ffill", max_fill=1)
    assert df.index[0] == pd.Timestamp("2020-01-03")  # B has no price before 01-03
    assert df.loc["2020-01-06", "A"] == 11.0 and df.loc["2020-01-07", "B"] == 5.5
    assert rep["values_filled"] == 2
    gap = pd.DataFrame({"A": [1.0, np.nan, np.nan, 2.0]}, index=pd.bdate_range("2021-01-04",
                                                                                periods=4))
    df2, _ = clean_prices(gap, "ffill", max_fill=1)
    assert len(df2) == 3  # the second missing day exceeds the fill limit and is dropped


def test_clean_rejects_bad_input():
    p = _prices()
    with pytest.raises(ValueError):
        clean_prices(pd.concat([p, p.iloc[:1]]))
    bad = p.copy()
    bad.iloc[0, 0] = -1.0
    with pytest.raises(ValueError):
        clean_prices(bad)
    with pytest.raises(ValueError):
        clean_prices(p, "zero")


def test_returns_log_and_simple():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=3))
    lr = to_returns(p, "log")
    sr = to_returns(p, "simple")
    np.testing.assert_allclose(lr["A"], [np.log(1.1), np.log(0.9)])
    np.testing.assert_allclose(sr["A"], [0.1, -0.1])
    assert lr.index[0] == p.index[1]
    with pytest.raises(ValueError):
        to_returns(_prices())


def test_ewma_vol_rule_matches_ewma_covariance_and_is_causal():
    from mre.models import ewma_covariance

    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 700),
                  index=pd.bdate_range("2020-01-01", periods=700))
    r.iloc[200:260] = 0.0005  # a calm stretch that hits the floor
    v = ewma_vol_rule(r, multiple=1.2, floor=0.10, window=500)
    for t in (0, 1, 40, 230, 499, 500, 650, 699):
        s2 = float(ewma_covariance(r.iloc[:t + 1].to_numpy(), 0.94, 500)[0, 0])
        assert v.iloc[t] == pytest.approx(max(0.10, 1.2 * np.sqrt(252 * s2)), rel=1e-10)
    assert (v.iloc[250:260] == 0.10).all()
    r2 = r.copy()
    r2.iloc[601:] = 0.5  # change the future
    v2 = ewma_vol_rule(r2)
    pd.testing.assert_series_equal(v.iloc[:601], v2.iloc[:601])


def test_factor_levels_names():
    p = pd.DataFrame({"A": [100.0, 101.0]}, index=pd.bdate_range("2020-01-01", periods=2))
    iv = pd.DataFrame({"A": [0.2, 0.25]}, index=p.index)
    lv = factor_levels(p, iv)
    assert list(lv.columns) == ["logS:A", "logIV:A"]
    assert lv.iloc[1, 1] == pytest.approx(np.log(0.25))


def test_load_csv(tmp_path):
    f = tmp_path / "px.csv"
    f.write_text("Date,X,Y\n2020-01-03,2,3\n2020-01-02,1,2.5\n")
    df = load_csv(f)
    assert list(df.columns) == ["X", "Y"] and df.index[0] == pd.Timestamp("2020-01-02")
    assert df.dtypes.unique().tolist() == [np.dtype(float)]


def test_skfolio_loaders_smoke():
    pytest.importorskip("skfolio")
    from mre.data import load_skfolio_factors, load_skfolio_index, load_skfolio_prices

    p = load_skfolio_prices()
    assert p.shape == (8313, 20) and p.index[0] == pd.Timestamp("1990-01-02")
    assert p.index[-1] == pd.Timestamp("2022-12-28")
    # dividend-adjusted: MSFT's USD 3 special dividend (ex-date 2004-11-15) leaves no drop
    assert p.loc["2004-11-15", "MSFT"] / p.loc["2004-11-12", "MSFT"] > 0.97
    idx = load_skfolio_index()
    assert idx.name == "SP500" and len(idx) == 8313
    assert load_skfolio_factors().shape[1] == 5


def test_arch_loaders_smoke():
    pytest.importorskip("arch")
    sp = load_arch("sp500")
    assert "Adj Close" in sp.columns and sp.index[0] == pd.Timestamp("1999-01-04")
    ff = load_arch("frenchdata")
    assert ff.index[0] == pd.Timestamp("1926-07-31") and list(ff.columns)[0] == "Mkt-RF"
    assert load_arch("vix").index[-1] == pd.Timestamp("2019-01-03")
    with pytest.raises(ValueError):
        load_arch("gold")


def test_ewma_vol_rule_truncation_with_a_short_window():
    from mre.models import ewma_covariance

    r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 40),
                  index=pd.bdate_range("2020-01-01", periods=40))
    v = ewma_vol_rule(r, multiple=1.0, floor=0.0, lam=0.8, window=5)
    for t in (2, 4, 5, 6, 20, 39):
        s2 = float(ewma_covariance(r.iloc[:t + 1].to_numpy(), 0.8, 5)[0, 0])
        assert v.iloc[t] == pytest.approx(np.sqrt(252 * s2), rel=1e-10)
