"""Real-data backtests of the seven risk methods on three portfolios (protocol: docs/protocol.md).

Real data: the skfolio S&P 500 panel (20 stocks and the index, daily, 1990-01-02 to 2022-12-28),
read at run time from the installed package. The STRANGLE portfolio is model P&L under an assumed
implied-volatility rule. Usage (from the repository root)::

    python -m mre.experiments.realdata --portfolio EQ20 --out results/realdata   # stage 1
    python -m mre.experiments.realdata --portfolio SPX --out results/realdata
    python -m mre.experiments.realdata --portfolio STRANGLE --out results/realdata
    python -m mre.experiments.realdata --summarise --out results/realdata        # stage 2
    python -m mre.experiments.realdata --checks --out results/realdata           # data checks

Stage 1 writes ``forecasts_<P>.csv``, ``run_<P>.csv``, ``es_test_<P>.csv`` and the per-portfolio
test results in ``stage1/``; stage 2 applies the Holm adjustment across portfolios and writes
``summary.csv``, ``by_year.csv``, ``stress.csv`` and ``failures.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mre import data
from mre.attribution import euler_historical, option_pnl_explain
from mre.backtest import (
    AcerbiSzekelySimulation,
    christoffersen,
    diebold_mariano,
    fz0_loss,
    holm,
    kupiec_pof,
    newey_west_lag,
    wilson_interval,
    zone_occupancy,
)
from mre.experiments.common import Manifest
from mre.garch import garch_filter
from mre.models import (
    EWMAParametric,
    FilteredHistoricalSimulation,
    HistoricalSimulation,
    MonteCarlo,
    PnLFunction,
    RiskModel,
    rolling_forecasts,
)
from mre.portfolio import (
    TRADING_DAYS,
    EuropeanOption,
    LinearPnL,
    MarketState,
    Portfolio,
    PortfolioPnL,
)
from mre.stress import HISTORICAL_WINDOWS

PORTFOLIOS = ("EQ20", "SPX", "STRANGLE")
METHODS = ("HS250", "HS500", "HS1000", "EWMA-N", "EWMA-T", "FHS", "MC")
EVAL_START, EVAL_END = pd.Timestamp("1994-01-03"), pd.Timestamp("2022-12-28")
LEVEL_VAR, LEVEL_ES = 0.99, 0.975
SEED_FHS, SEED_MC, SEED_AS = 20261111, 20261112, 20261113
N_SIM_AS = 5000
RATE, DIV_YIELD, IV_MULTIPLE, IV_FLOOR = 0.03, 0.02, 1.2, 0.10
STRANGLE_LAST_DAYS = 21
INDEX = "SP500"
H = 1.0 / TRADING_DAYS
TITLE = "real data: skfolio S&P 500 panel 1990-2022"


def make_methods() -> dict[str, RiskModel]:
    """The seven methods of the protocol with their fixed parameters and output names."""
    models: dict[str, RiskModel] = {
        "HS250": HistoricalSimulation(250),
        "HS500": HistoricalSimulation(500),
        "HS1000": HistoricalSimulation(1000),
        "EWMA-N": EWMAParametric(lam=0.94, dist="normal", window=500),
        "EWMA-T": EWMAParametric(lam=0.94, dist="t", nu=6.0, window=500),
        "FHS": FilteredHistoricalSimulation(window=1000, refit_every=20, n_boot=10_000,
                                            seed=SEED_FHS),
        "MC": MonteCarlo(lam=0.94, dist="t", nu=6.0, n_draws=20_000, seed=SEED_MC, window=500),
    }
    for name, m in models.items():
        m.name = name
    return models


@dataclass
class PortfolioInput:
    """Factor moves, the P&L function per forecast row, and the first forecast row."""

    name: str
    moves: pd.DataFrame
    pnl: PnLFunction | Callable[[int], PnLFunction]
    start: int
    extra: pd.DataFrame | None = None


def load_market() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """skfolio stock prices and the S&P 500 index, cleaned with the "drop" policy."""
    stocks, rep_s = data.clean_prices(data.load_skfolio_prices(), "drop")
    index, rep_i = data.clean_prices(data.load_skfolio_index(), "drop")
    report = {"stocks_rows_dropped": rep_s["rows_dropped"],
              "index_rows_dropped": rep_i["rows_dropped"]}
    return stocks, index[INDEX], report


def first_row_on_or_after(index: pd.Index, date: pd.Timestamp) -> int:
    """Position of the first date in ``index`` on or after ``date``."""
    return int(np.searchsorted(index.values, np.datetime64(date), side="left"))


def eq20_input(stocks: pd.DataFrame) -> PortfolioInput:
    """EQ20: X_t = mean of the 20 simple returns (one factor, P&L = the factor move)."""
    x = data.to_returns(stocks, "simple").mean(axis=1).to_frame("EQ20")
    return PortfolioInput("EQ20", x, LinearPnL([1.0]), first_row_on_or_after(x.index, EVAL_START))


def spx_input(index: pd.Series) -> PortfolioInput:
    """SPX: X_t = simple return of the index."""
    x = data.to_returns(index, "simple")
    x.columns = ["SPX"]
    return PortfolioInput("SPX", x, LinearPnL([1.0]), first_row_on_or_after(x.index, EVAL_START))


@dataclass(frozen=True)
class StrangleSchedule:
    """Which short strangle is held at the close of each date (positions in ``dates``).

    ``open_pos[i]`` is the position of the opening day of the pair held at the close of date
    ``i`` (-1 before the first opening), ``days_left[i]`` its remaining trading days to expiry and
    ``strike_put[i]``, ``strike_call[i]`` its strikes.
    """

    dates: pd.DatetimeIndex
    open_pos: np.ndarray
    days_left: np.ndarray
    strike_put: np.ndarray
    strike_call: np.ndarray

    @classmethod
    def build(cls, spot: pd.Series, first: pd.Timestamp = EVAL_START,
              last_days: int = STRANGLE_LAST_DAYS) -> StrangleSchedule:
        """Openings on the first date of each month from ``first``; each pair expires at the
        close of the next opening date; the last pair gets ``last_days`` trading days."""
        dates = pd.DatetimeIndex(spot.index)
        pos = np.arange(len(dates))
        month = dates.to_period("M")
        is_first = np.r_[True, month[1:] != month[:-1]] & (dates >= first)
        opens = pos[is_first]
        n_days = np.r_[np.diff(opens), last_days]
        open_pos = np.full(len(dates), -1)
        days_left = np.zeros(len(dates), dtype=int)
        s = spot.to_numpy(dtype=float)
        kp = np.full(len(dates), np.nan)
        kc = np.full(len(dates), np.nan)
        for o, n in zip(opens, n_days, strict=True):
            seg = slice(o, min(o + n, len(dates)))  # the pair is held at the close of o..o+n-1
            open_pos[seg] = o
            days_left[seg] = n - (pos[seg] - o)
            kp[seg], kc[seg] = 0.95 * s[o], 1.05 * s[o]
        return cls(dates, open_pos, days_left, kp, kc)

    def portfolio_at(self, i: int) -> Portfolio:
        """The short put and short call held at the close of date position ``i``."""
        if self.open_pos[i] < 0:
            raise ValueError(f"no strangle is held at {self.dates[i].date()}")
        expiry = self.days_left[i] / TRADING_DAYS
        return Portfolio([EuropeanOption(INDEX, float(self.strike_put[i]), expiry, False, -1.0),
                          EuropeanOption(INDEX, float(self.strike_call[i]), expiry, True, -1.0)])


def strangle_input(index: pd.Series) -> PortfolioInput:
    """STRANGLE: factors logS and logIV of the index; P&L by full revaluation (model P&L)."""
    lr = data.to_returns(index, "log").iloc[:, 0]
    iv = data.ewma_vol_rule(lr, IV_MULTIPLE, IV_FLOOR)
    spot = index.loc[iv.index]
    levels = data.factor_levels(spot.to_frame(INDEX), iv.to_frame(INDEX))
    moves = levels.diff().iloc[1:]
    sched = StrangleSchedule.build(spot)
    s_arr, v_arr = spot.to_numpy(float), iv.to_numpy(float)
    offset = 1  # moves row r is dated spot.index[r + 1]; the state is at spot.index[r]

    def pnl_at(r: int) -> PnLFunction:
        i = r + offset - 1  # close of the previous date
        state = MarketState({INDEX: s_arr[i]}, {INDEX: v_arr[i]}, RATE, {INDEX: DIV_YIELD})
        return PortfolioPnL(sched.portfolio_at(i), state, H)

    start = first_row_on_or_after(moves.index, EVAL_START + pd.Timedelta(days=1))
    extra = pd.DataFrame({"spot": spot, "iv": iv, "days_left": sched.days_left},
                         index=spot.index).loc[moves.index]
    return PortfolioInput("STRANGLE", moves, pnl_at, start, extra)


def build_input(name: str) -> tuple[PortfolioInput, pd.DataFrame]:
    stocks, index, _ = load_market()
    if name == "EQ20":
        return eq20_input(stocks), stocks
    if name == "SPX":
        return spx_input(index), stocks
    if name == "STRANGLE":
        return strangle_input(index), stocks
    raise ValueError(f"unknown portfolio {name!r}")


def run_forecasts(inp: PortfolioInput, methods: dict[str, RiskModel], end: int | None = None,
                  fits_out: dict | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling forecasts of every method; Acerbi-Szekely simulated on the common sample.

    Returns (forecasts in wide form, one row per method of run information, ES test results).
    If ``fits_out`` is a dict, the GARCH fits of each method are stored in it by method name.
    """
    end = len(inp.moves) if end is None else end
    cols: dict[str, np.ndarray] = {}
    runs, sims, realized = [], {}, None
    for name, model in methods.items():
        sim = AcerbiSzekelySimulation(LEVEL_ES, N_SIM_AS, SEED_AS)

        def feed(t, f, xr, sim=sim):
            sim.add(t, xr, f.predictive)

        t0 = time.time()
        path = rolling_forecasts(model, inp.moves, inp.start, end, inp.pnl, LEVEL_VAR, LEVEL_ES,
                                 on_forecast=feed, keep_predictive=False, on_error="record")
        realized = path.realized
        fr = path.frame()
        cols[f"{name}:var99"] = fr["var"].to_numpy()
        cols[f"{name}:es975"] = fr["es"].to_numpy()
        cols[f"{name}:var975"] = fr["var_at_es_level"].to_numpy()
        failed = [(str(f.date.date()), f.diagnostics["error"]) for f in path.forecasts
                  if "error" in f.diagnostics]
        fits = [g for batch in path.fits for g in batch]
        if fits_out is not None:
            fits_out[name] = path.fits
        runs.append({"method": name, "forecast_days": len(path.forecasts),
                     "failed_days": len(failed), "failed": json.dumps(failed),
                     "garch_fits": len(fits),
                     "garch_nonconverged": sum(not g.converged for g in fits),
                     "seconds": round(time.time() - t0, 1)})
        sims[name] = sim
    dates = inp.moves.index[inp.start:end]
    fc = pd.DataFrame({"realized": realized, **cols}, index=dates)
    fc.index.name = "date"
    ok = np.all(np.isfinite(fc.to_numpy()), axis=1)
    common_rows = np.arange(inp.start, end)[ok]
    es_rows = []
    for name, sim in sims.items():
        res = sim.result(common_rows)
        es_rows.append({"method": name, "z1": res["Z1"].statistic, "z1_p": res["Z1"].p_value,
                        "z2": res["Z2"].statistic, "z2_p": res["Z2"].p_value,
                        "exceedances975": res["Z2"].details["n_exceptions"],
                        "days": res["Z2"].details["n_days"], "n_sim": N_SIM_AS})
    return fc, pd.DataFrame(runs), pd.DataFrame(es_rows)


def common_sample(fc: pd.DataFrame) -> pd.DataFrame:
    """Rows on which every method produced a finite forecast."""
    return fc[np.all(np.isfinite(fc.to_numpy()), axis=1)]


def shortfall_ratio(realized: np.ndarray, var975: np.ndarray, es975: np.ndarray
                    ) -> tuple[float, int]:
    """Mean realised loss over mean predicted ES on days with loss above the 97.5% VaR."""
    hit = realized < -var975
    if not hit.any():
        return float("nan"), 0
    return float(np.mean(-realized[hit]) / np.mean(es975[hit])), int(hit.sum())


def evaluate(portfolio: str, fc: pd.DataFrame, es: pd.DataFrame
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full-sample, by-year and stress-window results of one portfolio (raw p-values)."""
    fc = common_sample(fc)
    x = fc["realized"].to_numpy()
    years = fc.index.year.to_numpy()
    fz_ref = fz0_loss(x, fc["EWMA-N:var975"], fc["EWMA-N:es975"], LEVEL_ES)
    es = es.set_index("method")
    full, by_year, stress = [], [], []
    for m in METHODS:
        var, es975, var975 = (fc[f"{m}:{k}"].to_numpy() for k in ("var99", "es975", "var975"))
        exc = x < -var
        n, k = exc.size, int(exc.sum())
        lo, hi = wilson_interval(k, n)
        kup = kupiec_pof(exc, LEVEL_VAR)
        chr_ = christoffersen(exc, LEVEL_VAR)
        _, zones = zone_occupancy(exc, 250)
        ratio, n975 = shortfall_ratio(x, var975, es975)
        fz = fz0_loss(x, var975, es975, LEVEL_ES)
        dm = diebold_mariano(fz, fz_ref) if m != "EWMA-N" else None
        full.append({
            "portfolio": portfolio, "method": m, "days": n, "first_day": fc.index[0].date(),
            "last_day": fc.index[-1].date(), "exceptions": k, "rate": k / n,
            "wilson_lo": lo, "wilson_hi": hi,
            "kupiec_lr": kup.statistic, "kupiec_p": kup.p_value,
            "ind_lr": chr_["ind"].statistic, "ind_p": chr_["ind"].p_value,
            "cc_lr": chr_["cc"].statistic, "cc_p": chr_["cc"].p_value,
            "green": zones["green"], "yellow": zones["yellow"], "red": zones["red"],
            "z1": es.loc[m, "z1"], "z1_p": es.loc[m, "z1_p"],
            "z2": es.loc[m, "z2"], "z2_p": es.loc[m, "z2_p"], "z2_n_sim": es.loc[m, "n_sim"],
            "exceedances975": n975, "shortfall_ratio": ratio, "fz0_mean": float(np.mean(fz)),
            "dm_stat": np.nan if dm is None else dm.statistic,
            "dm_p": np.nan if dm is None else dm.p_value,
            "dm_lag": newey_west_lag(n),
        })
        for y in np.unique(years):
            sel = years == y
            ey = exc[sel]
            ky = kupiec_pof(ey, LEVEL_VAR)
            cy = christoffersen(ey, LEVEL_VAR)
            by_year.append({"portfolio": portfolio, "method": m, "year": int(y),
                            "days": int(sel.sum()), "exceptions": int(ey.sum()),
                            "kupiec_p": ky.p_value, "ind_p": cy["ind"].p_value,
                            "cc_p": cy["cc"].p_value})
        for wname, (a, b) in HISTORICAL_WINDOWS.items():
            sel = (fc.index >= pd.Timestamp(a)) & (fc.index <= pd.Timestamp(b))
            r, nr = shortfall_ratio(x[sel], var975[sel], es975[sel])
            stress.append({"portfolio": portfolio, "method": m, "window": wname,
                           "start": a, "end": b, "days": int(sel.sum()),
                           "exceptions": int(exc[sel].sum()),
                           "expected": round(0.01 * sel.sum(), 2),
                           "exceedances975": nr, "shortfall_ratio": r})
    return pd.DataFrame(full), pd.DataFrame(by_year), pd.DataFrame(stress)


def fhs_diagnostics(inp: PortfolioInput, fits: list, window: int = 1000,
                    refit_every: int = 20) -> pd.DataFrame:
    """Per FHS refit day and factor: GARCH parameters, the share of zero moves in the window,
    the smallest and the next-day conditional volatility, the largest absolute standardised
    residual and the largest absolute scenario move it implies (``sigma_next * max |z|``)."""
    x = inp.moves.to_numpy(dtype=float)
    rows = []
    for k, batch in enumerate(fits):
        t = inp.start + k * refit_every
        h = x[t - window:t]
        for j, fit in enumerate(batch):
            s2, s2n = garch_filter(h[:, j], fit.params)
            z = (h[:, j] - fit.params.mu) / np.sqrt(s2)
            pr = fit.params
            rows.append({"date": inp.moves.index[t].date(), "factor": inp.moves.columns[j],
                         "omega": pr.omega, "alpha": pr.alpha, "beta": pr.beta, "nu": pr.nu,
                         "converged": fit.converged,
                         "zero_move_share": float(np.mean(h[:, j] == 0)),
                         "sigma_min": float(np.sqrt(s2.min())), "sigma_next": float(np.sqrt(s2n)),
                         "max_abs_z": float(np.abs(z).max()),
                         "max_abs_scenario_move": float(np.sqrt(s2n) * np.abs(z).max())})
    return pd.DataFrame(rows)


def strangle_explain(inp: PortfolioInput, fc: pd.DataFrame) -> pd.DataFrame:
    """Daily delta/gamma/vega/theta/unexplained split of the realised STRANGLE model P&L."""
    ex = inp.extra
    rows = []
    for r in range(inp.start, len(inp.moves)):
        f = inp.pnl(r)
        s1 = MarketState({INDEX: float(ex["spot"].iloc[r])}, {INDEX: float(ex["iv"].iloc[r])},
                         RATE, {INDEX: DIV_YIELD})
        e = option_pnl_explain(f.portfolio, f.state, s1, H).sum()
        rows.append({"date": inp.moves.index[r].date(), "spot": s1.spot[INDEX],
                     "iv": s1.vol[INDEX], "days_left_prev_close": int(ex["days_left"].iloc[r - 1])
                     if r > 0 else np.nan, **{k: e[k] for k in
                                              ("full", "delta", "gamma", "vega", "theta",
                                               "unexplained")}})
    out = pd.DataFrame(rows)
    gap = np.max(np.abs(out["full"].to_numpy() - fc["realized"].to_numpy()))
    if gap > 1e-8:
        raise AssertionError(f"P&L explain does not reproduce the realised P&L (gap {gap})")
    return out


def eq20_attribution(stocks: pd.DataFrame, fc: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                        pd.DataFrame]:
    """Realised P&L contribution by stock and year; HS500 Euler contributions on the first
    evaluation day of each stress window."""
    r = data.to_returns(stocks, "simple") / stocks.shape[1]
    ev = r.loc[fc.index]
    year = ev.groupby(ev.index.year).sum()
    year.index.name = "year"
    rows = []
    for wname, (a, _) in HISTORICAL_WINDOWS.items():
        day = fc.index[fc.index >= pd.Timestamp(a)][0]
        pos = r.index.get_loc(day)
        scen = r.iloc[pos - 500:pos].to_numpy()
        c = euler_historical(scen, LEVEL_VAR, LEVEL_ES)
        for j, stock in enumerate(stocks.columns):
            rows.append({"window": wname, "day": day.date(), "stock": stock,
                         "var_contribution": c["var"].iloc[j],
                         "es_contribution": c["es"].iloc[j],
                         "portfolio_var": c.attrs["portfolio_var"],
                         "portfolio_es": c.attrs["portfolio_es"]})
    return year.reset_index(), pd.DataFrame(rows)


def data_checks() -> pd.DataFrame:
    """The data checks of the protocol: dividend adjustment, index cross-check, VIX check."""
    stocks, index, rep = load_market()
    rows = [{"check": k, "value": v, "note": "clean_prices(policy='drop')"}
            for k, v in rep.items()]
    rs = data.to_returns(stocks, "simple")
    rows.append({"check": "MSFT simple return 2004-11-15", "value": rs.loc["2004-11-15", "MSFT"],
                 "note": "ex-date of the USD 3.00 special dividend; unadjusted prices would "
                         "fall by about 10%"})
    ri = data.to_returns(index, "simple").iloc[:, 0]
    arch = data.to_returns(data.load_arch("sp500")["Adj Close"], "simple").iloc[:, 0]
    both = pd.concat([ri, arch], axis=1, join="inner").dropna()
    d = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    rows += [{"check": "arch sp500 common return dates", "value": len(both),
              "note": f"{both.index[0].date()} to {both.index[-1].date()}"},
             {"check": "arch sp500 max abs return difference", "value": d.max(),
              "note": f"on {d.idxmax().date()}"}]
    iv = data.ewma_vol_rule(data.to_returns(index, "log").iloc[:, 0], IV_MULTIPLE, IV_FLOOR)
    vix = data.load_arch("vix")["vix"].dropna() / 100.0
    vix = vix[(vix.index >= "2014-01-03") & (vix.index <= "2019-01-03")]
    j = pd.concat([iv, vix], axis=1, join="inner").dropna()
    rows += [{"check": "IV rule vs VIX/100 common dates", "value": len(j),
              "note": f"{j.index[0].date()} to {j.index[-1].date()}"},
             {"check": "IV rule vs VIX/100 correlation", "value": j.iloc[:, 0].corr(j.iloc[:, 1]),
              "note": "Pearson, levels; sanity check only, the rule is not recalibrated"},
             {"check": "IV rule / (VIX/100) mean ratio",
              "value": float((j.iloc[:, 0] / j.iloc[:, 1]).mean()), "note": "mean of daily ratios"},
             {"check": "IV rule share of days at the 10% floor",
              "value": float((iv.loc[EVAL_START:] == IV_FLOOR).mean()),
              "note": "evaluation period 1994-01-03 to 2022-12-28"}]
    return pd.DataFrame(rows)


def stage1(name: str, out: Path) -> list[str]:
    inp, stocks = build_input(name)
    fits: dict = {}
    fc, runs, es = run_forecasts(inp, make_methods(), fits_out=fits)
    full, by_year, stress = evaluate(name, fc, es)
    part = out / "stage1"
    part.mkdir(parents=True, exist_ok=True)
    if inp.extra is not None:
        fc = fc.join(inp.extra[["spot", "iv"]])
    fc.to_csv(out / f"forecasts_{name}.csv", float_format="%.8g")
    runs.to_csv(out / f"run_{name}.csv", index=False)
    es.to_csv(out / f"es_test_{name}.csv", index=False, float_format="%.6g")
    full.to_csv(part / f"full_{name}.csv", index=False, float_format="%.10g")
    by_year.to_csv(part / f"by_year_{name}.csv", index=False, float_format="%.10g")
    stress.to_csv(part / f"stress_{name}.csv", index=False, float_format="%.10g")
    files = [f"forecasts_{name}.csv", f"run_{name}.csv", f"es_test_{name}.csv",
             f"stage1/full_{name}.csv", f"stage1/by_year_{name}.csv", f"stage1/stress_{name}.csv"]
    fhs_diagnostics(inp, fits["FHS"]).to_csv(out / f"fhs_fits_{name}.csv", index=False,
                                             float_format="%.6g")
    files.append(f"fhs_fits_{name}.csv")
    if name == "STRANGLE":
        strangle_explain(inp, fc).to_csv(out / "strangle_explain.csv", index=False,
                                         float_format="%.8g")
        files.append("strangle_explain.csv")
    if name == "EQ20":
        year, euler = eq20_attribution(stocks, fc)
        year.to_csv(out / "eq20_attribution_year.csv", index=False, float_format="%.8g")
        euler.to_csv(out / "eq20_euler_stress.csv", index=False, float_format="%.8g")
        files += ["eq20_attribution_year.csv", "eq20_euler_stress.csv"]
    return files


def stage2(out: Path) -> list[str]:
    part = out / "stage1"
    full = pd.concat([pd.read_csv(part / f"full_{p}.csv") for p in PORTFOLIOS])
    by_year = pd.concat([pd.read_csv(part / f"by_year_{p}.csv") for p in PORTFOLIOS])
    stress = pd.concat([pd.read_csv(part / f"stress_{p}.csv") for p in PORTFOLIOS])
    for col in ("kupiec_p", "ind_p", "cc_p", "z2_p", "dm_p"):
        full[col.replace("_p", "_holm")] = holm(full[col].to_numpy())
    for col in ("kupiec_p", "ind_p", "cc_p"):
        by_year[col.replace("_p", "_holm")] = holm(by_year[col].to_numpy())
    fails = []
    for p in PORTFOLIOS:
        for _, r in pd.read_csv(out / f"run_{p}.csv").iterrows():
            fails += [{"portfolio": p, "method": r["method"], "date": d, "error": e}
                      for d, e in json.loads(r["failed"])]
    full.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    by_year.to_csv(out / "by_year.csv", index=False, float_format="%.6g")
    stress.to_csv(out / "stress.csv", index=False, float_format="%.6g")
    pd.DataFrame(fails, columns=["portfolio", "method", "date", "error"]).to_csv(
        out / "failures.csv", index=False)
    return ["summary.csv", "by_year.csv", "stress.csv", "failures.csv"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--portfolio", choices=PORTFOLIOS)
    g.add_argument("--summarise", action="store_true")
    g.add_argument("--checks", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/realdata"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = (f"FHS bootstrap default_rng([{SEED_FHS}, t]); MC default_rng([{SEED_MC}, t]); "
             f"Acerbi-Szekely default_rng({SEED_AS}) with {N_SIM_AS} simulations")
    man = Manifest("mre.experiments.realdata", seeds)
    if args.portfolio:
        files, tag = stage1(args.portfolio, args.out), f"realdata_{args.portfolio}"
    elif args.summarise:
        files, tag = stage2(args.out), "realdata_summary"
    else:
        data_checks().to_csv(args.out / "checks.csv", index=False, float_format="%.8g")
        files, tag = ["checks.csv"], "realdata_checks"
    man.write(args.out, tag, files, notes=f"{TITLE}; protocol docs/protocol.md")


if __name__ == "__main__":
    main()
