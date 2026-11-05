"""Historical and hypothetical stress scenarios with full revaluation."""

import numpy as np
import pandas as pd
import pytest

from mre.portfolio import EuropeanOption, MarketState, Portfolio, Stock
from mre.pricing import bsm_price
from mre.stress import (
    HISTORICAL_WINDOWS,
    historical_scenario,
    hypothetical_scenarios,
    run_stress,
)

STATE = MarketState({"A": 100.0}, {"A": 0.2}, rate=0.02)
PORT = Portfolio([Stock("A", 10.0), EuropeanOption("A", 90.0, 0.5, False, 30.0)])


def _levels():
    idx = pd.bdate_range("2008-08-01", "2008-12-31")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"logS:A": np.cumsum(rng.normal(0, 0.02, idx.size)),
                         "logIV:A": np.cumsum(rng.normal(0, 0.05, idx.size))}, index=idx)


def test_historical_scenario_is_cumulative_move_over_window():
    lv = _levels()
    sc = historical_scenario(lv, "2008-09-02", "2008-11-20", "x")
    base = lv.loc[: "2008-09-01"].iloc[-1]
    last = lv.loc[: "2008-11-20"].iloc[-1]
    pd.testing.assert_series_equal(sc.moves, last - base, check_names=False)
    daily = lv.diff().loc["2008-09-02":"2008-11-20"].sum()
    np.testing.assert_allclose(sc.moves.to_numpy(), daily.to_numpy(), atol=1e-12)
    assert sc.first_day == pd.Timestamp("2008-09-02") and not sc.adjusted


def test_historical_scenario_cut_when_data_end_early_and_error_when_not_covered():
    lv = _levels().loc[:"2008-10-31"]
    sc = historical_scenario(lv, "2008-09-02", "2008-11-20")
    assert sc.adjusted and sc.last_day == pd.Timestamp("2008-10-31")
    with pytest.raises(ValueError):
        historical_scenario(lv, "2020-02-19", "2020-03-23")
    with pytest.raises(ValueError):
        historical_scenario(lv, "2008-07-01", "2008-07-31")


def test_hypothetical_scenarios_definitions():
    sc = hypothetical_scenarios(PORT, STATE)
    assert list(sc) == ["equity -10%", "equity -20%", "equity -30%", "vol +10 pts",
                        "equity -20%, vol +10 pts"]
    assert 100 * np.exp(sc["equity -20%"]["logS:A"]) == pytest.approx(80.0)
    assert 0.2 * np.exp(sc["vol +10 pts"]["logIV:A"]) == pytest.approx(0.30)
    assert sc["vol +10 pts"]["logS:A"] == 0.0


def test_run_stress_full_revaluation():
    out = run_stress(PORT, STATE, hypothetical_scenarios(PORT, STATE))
    p0 = bsm_price(100, 90, 0.5, 0.02, 0.0, 0.2, False)
    p1 = bsm_price(70, 90, 0.5, 0.02, 0.0, 0.2, False)
    assert out.loc["equity -30%", "pos0"] == pytest.approx(-300.0)
    assert out.loc["equity -30%", "pos1"] == pytest.approx(30 * (p1 - p0))
    assert out.loc["equity -30%", "total"] == pytest.approx(-300 + 30 * (p1 - p0))
    p_vol = bsm_price(100, 90, 0.5, 0.02, 0.0, 0.3, False)
    assert out.loc["vol +10 pts", "total"] == pytest.approx(30 * (p_vol - p0))
    # the protective puts gain more the larger the fall
    assert out["pos1"].iloc[:3].is_monotonic_increasing
    # instantaneous: no time decay when nothing moves
    zero = run_stress(PORT, STATE, {"none": pd.Series({"logS:A": 0.0})})
    assert zero.loc["none", "total"] == 0.0


def test_windows_are_the_prespecified_ones():
    assert HISTORICAL_WINDOWS == {
        "2008 crisis": ("2008-09-02", "2008-11-20"),
        "2020 Covid crash": ("2020-02-19", "2020-03-23"),
        "2022 rate shock": ("2022-01-03", "2022-10-12"),
    }


def test_windows_inside_skfolio_data():
    """Smoke check on real data: all three windows are inside 1990-01-02 .. 2022-12-28."""
    pytest.importorskip("skfolio")
    from mre.data import clean_prices, factor_levels, load_skfolio_prices

    prices, _ = clean_prices(load_skfolio_prices()[["JPM", "XOM"]])
    lv = factor_levels(prices)
    for name, (a, b) in HISTORICAL_WINDOWS.items():
        sc = historical_scenario(lv, a, b, name)
        assert not sc.adjusted and sc.last_day <= pd.Timestamp(b)
