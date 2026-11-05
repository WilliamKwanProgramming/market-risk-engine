"""Data loaders, cleaning and returns. Nothing is stored in the repository.

Sources (all read at run time from installed packages or a user file):

* ``skfolio`` (``pip install skfolio``): :func:`load_skfolio_prices` - daily prices of 20 US
  stocks, 1990-01-02 to 2022-12-28. Checked in this sandbox: the prices are adjusted for splits
  **and dividends** (no drop on the MSFT special-dividend ex-date 2004-11-15; the package
  docstring calls them "adjusted closing prices"), so their returns are total returns.
  :func:`load_skfolio_index` - the S&P 500 index level (a price index: no dividends), same dates.
  :func:`load_skfolio_factors` - five factor ETFs, 2014-01-02 to 2022-12-28.
* ``arch`` (``pip install arch``): :func:`load_arch` - ``sp500`` and ``nasdaq`` (OHLC and Adj
  Close, 1999-2018), ``vix`` (2014-01-03 to 2019-01-03), ``wti`` (FRED DCOILWTICO daily),
  ``crude`` (Brent and WTI monthly), ``frenchdata`` (monthly Fama-French factors in percent).
* :func:`load_csv` - a user CSV with a date column and one price column per asset (for data
  that cannot be distributed, for example option or insurer prices).

**Missing-data policy** (:func:`clean_prices`). Rows are sorted by date; duplicate dates and
non-positive prices raise an error; weekend dates are dropped (business-day calendar, holidays
are whatever the source omits). Then either

* ``policy="drop"`` (default): drop every date on which any asset is missing. The return after a
  dropped date spans more than one trading day; the number of dropped dates is reported.
* ``policy="ffill"``: carry the last price forward for at most ``max_fill`` consecutive dates
  (zero return on filled dates), then drop dates that are still missing. Leading missing values
  (before an asset's first price) are never filled.

Every function returns a new object and leaves its input unchanged.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_skfolio_prices", "load_skfolio_index", "load_skfolio_factors", "load_arch",
           "load_csv", "clean_prices", "to_returns", "realized_vol_proxy", "factor_levels"]

_ARCH_SETS = ("sp500", "nasdaq", "vix", "wti", "crude", "frenchdata")


def _require(module: str, pip_name: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(f"{module} is not installed: pip install {pip_name}") from exc


def load_skfolio_prices() -> pd.DataFrame:
    """Daily adjusted prices of 20 S&P 500 stocks (1990-01-02 to 2022-12-28) from skfolio."""
    return _require("skfolio.datasets", "skfolio").load_sp500_dataset().copy()


def load_skfolio_index() -> pd.Series:
    """Daily S&P 500 index level (price index) from skfolio, as a Series named ``SP500``."""
    return _require("skfolio.datasets", "skfolio").load_sp500_index()["SP500"].copy()


def load_skfolio_factors() -> pd.DataFrame:
    """Daily prices of the factor ETFs MTUM, QUAL, SIZE, USMV, VLUE (2014-2022) from skfolio."""
    return _require("skfolio.datasets", "skfolio").load_factors_dataset().copy()


def load_arch(name: str) -> pd.DataFrame:
    """One of the ``arch.data`` sets: sp500, nasdaq, vix, wti, crude, frenchdata.

    ``frenchdata`` comes with an integer YYYYMM index stored as a datetime; it is converted to
    month-end dates. Values are returned as stored (Fama-French returns are in percent).
    """
    if name not in _ARCH_SETS:
        raise ValueError(f"unknown arch dataset {name!r}; choose from {_ARCH_SETS}")
    df = _require(f"arch.data.{name}", "arch").load().copy()
    if name == "frenchdata":
        yyyymm = df.index.asi8 if isinstance(df.index, pd.DatetimeIndex) else df.index
        yyyymm = np.asarray(yyyymm, dtype=np.int64)
        df.index = pd.PeriodIndex.from_fields(year=yyyymm // 100, month=yyyymm % 100,
                                              freq="M").to_timestamp(how="end").normalize()
    df.index.name = "date"
    return df


def load_csv(path: str | Path, date_column: str | int = 0) -> pd.DataFrame:
    """Read a CSV with one date column and one numeric price column per asset."""
    df = pd.read_csv(path)
    col = df.columns[date_column] if isinstance(date_column, int) else date_column
    df[col] = pd.to_datetime(df[col])
    df = df.set_index(col).sort_index()
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="raise").astype(float)


def clean_prices(prices: pd.DataFrame | pd.Series, policy: str = "drop",
                 max_fill: int = 5) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the missing-data policy of the module docstring; returns (prices, report)."""
    df = prices.to_frame() if isinstance(prices, pd.Series) else prices.copy()
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate dates in price data")
    if (df <= 0).any().any():
        raise ValueError("non-positive prices")
    report = {"rows_in": len(df)}
    weekday = df.index.dayofweek < 5
    report["weekend_rows_dropped"] = int((~weekday).sum())
    df = df[weekday]
    report["missing_values"] = int(df.isna().sum().sum())
    if policy == "ffill":
        before = df.isna()
        df = df.ffill(limit=max_fill)
        report["values_filled"] = int((before & df.notna()).sum().sum())
    elif policy != "drop":
        raise ValueError("policy must be 'drop' or 'ffill'")
    keep = df.notna().all(axis=1)
    report["rows_dropped"] = int((~keep).sum())
    df = df[keep]
    report["rows_out"] = len(df)
    return df, report


def to_returns(prices: pd.DataFrame | pd.Series, kind: str = "log") -> pd.DataFrame:
    """Returns between consecutive rows: ``log(P_t/P_{t-1})`` or ``P_t/P_{t-1} - 1``.

    The first row (no previous price) is dropped. Rows must already be clean (no NaN).
    """
    df = prices.to_frame() if isinstance(prices, pd.Series) else prices
    if df.isna().any().any():
        raise ValueError("clean the prices first (clean_prices)")
    if kind == "log":
        out = np.log(df).diff()
    elif kind == "simple":
        out = df.pct_change(fill_method=None)
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return out.iloc[1:]


def realized_vol_proxy(log_returns: pd.DataFrame | pd.Series, window: int = 21,
                       multiple: float = 1.0, periods_per_year: int = 252) -> pd.DataFrame:
    """Implied-volatility proxy: ``multiple * sqrt(periods_per_year) * sd(last window returns)``.

    This is an explicit assumption for option positions when no implied-volatility history
    exists (it is not an implied volatility). The value on day t uses returns up to and including
    day t (sample standard deviation, ``ddof = 1``); the first ``window - 1`` values are NaN.
    """
    df = log_returns.to_frame() if isinstance(log_returns, pd.Series) else log_returns
    return multiple * np.sqrt(periods_per_year) * df.rolling(window).std(ddof=1)


def factor_levels(prices: pd.DataFrame, implied_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Log factor levels named as :meth:`mre.portfolio.Portfolio.factor_names` expects.

    ``logS:<u>`` = log price of column ``u`` of ``prices``; ``logIV:<u>`` = log of column ``u``
    of ``implied_vol`` (annualised, 0.2 = 20%), if given. Rows are inner-joined on dates.
    """
    out = np.log(prices).add_prefix("logS:")
    if implied_vol is not None:
        out = out.join(np.log(implied_vol).add_prefix("logIV:"), how="inner")
    return out
