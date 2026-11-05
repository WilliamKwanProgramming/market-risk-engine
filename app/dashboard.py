"""Streamlit dashboard of the precomputed backtest results (no network, no model runs).

It reads only the files in results/realdata/ and results/validation/ written by
scripts/reproduce.sh, so it starts in a second. Run from the repository root:

    pip install -e '.[dashboard]'
    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
RD = REPO / "results" / "realdata"
VAL = REPO / "results" / "validation"
PORTFOLIOS = ["EQ20", "SPX", "STRANGLE"]
METHODS = ["HS250", "HS500", "HS1000", "EWMA-N", "EWMA-T", "FHS", "MC"]
DESCRIPTION = {
    "EQ20": "equal weights in 20 US stocks, rebalanced daily; P&L = portfolio simple return",
    "SPX": "the S&P 500 index (price index); P&L = daily simple return",
    "STRANGLE": "short 1-month 95% put and 105% call on the S&P 500, rolled monthly; P&L in "
                "index points. Model P&L under an assumed implied-volatility rule "
                "(1.2 x EWMA volatility, floor 10%), not traded option prices",
}
WINDOWS = {"2008 crisis": ("2008-09-02", "2008-11-20"),
           "2020 Covid crash": ("2020-02-19", "2020-03-23"),
           "2022 rate shock": ("2022-01-03", "2022-10-12")}
# colours: categorical slots and status colours of the default palette (dataviz skill)
C_LOSS, C_VAR, C_ES = "#8a8984", "#2a78d6", "#eb6834"
C_EXC = "#d03b3b"
ZONE_COLORS = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}
COMPONENT_ORDER = ["delta", "gamma", "vega", "theta", "unexplained"]
COMPONENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"]


@st.cache_data
def read_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


@st.cache_data
def index_returns() -> pd.DataFrame:
    """Daily simple returns of the S&P 500 index, read from the installed skfolio package
    (the repository stores no price or index data)."""
    from mre import data

    r = data.to_returns(data.load_skfolio_index(), "simple").iloc[:, 0]
    return pd.DataFrame({"date": pd.to_datetime(r.index), "realized": r.to_numpy()})


def forecasts(portfolio: str) -> pd.DataFrame:
    fc = read_csv(str(RD / f"forecasts_{portfolio}.csv"), parse_dates=["date"])
    if "realized" not in fc.columns:  # SPX: its P&L is the index return, which is data
        try:
            fc = fc.merge(index_returns(), on="date", how="left", validate="one_to_one")
        except ImportError:
            st.error("The SPX page needs the index returns from the `skfolio` package "
                     "(not stored in this repository): `pip install -e '.[dashboard]'`.")
            st.stop()
    return fc


def method_frame(fc: pd.DataFrame, method: str) -> pd.DataFrame:
    """Loss, VaR, ES and the exception flag of one method (cheap column selection)."""
    d = pd.DataFrame({"date": fc["date"], "loss": -fc["realized"],
                      "VaR 99%": fc[f"{method}:var99"], "ES 97.5%": fc[f"{method}:es975"]})
    d = d.dropna()
    d["exception"] = d["loss"] > d["VaR 99%"]
    return d


def zones(d: pd.DataFrame) -> pd.DataFrame:
    """Exceptions in the last 250 days and the Basel zone, from day 250 on."""
    c = d["exception"].astype(int).rolling(250).sum()
    z = pd.DataFrame({"date": d["date"], "count": c}).dropna()
    z["zone"] = np.select([z.iloc[:, 1] <= 4, z.iloc[:, 1] <= 9], ["green", "yellow"], "red")
    return z


def loss_chart(d: pd.DataFrame, title: str, height: int = 320,
               date_format: str = "%Y") -> alt.LayerChart:
    long = d.melt(id_vars=["date"], value_vars=["loss", "VaR 99%", "ES 97.5%"],
                  var_name="series", value_name="value")
    color = alt.Color("series:N", scale=alt.Scale(domain=["loss", "VaR 99%", "ES 97.5%"],
                                                  range=[C_LOSS, C_VAR, C_ES]),
                      legend=alt.Legend(orient="top", title=None))
    lines = alt.Chart(long).mark_line(strokeWidth=1).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format=date_format)),
        y=alt.Y("value:Q", title="loss (positive) / risk"),
        color=color, tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".4g")])
    exc = alt.Chart(d[d["exception"]]).mark_point(size=60, filled=True, color=C_EXC,
                                                  opacity=1).encode(
        x="date:T", y="loss:Q",
        tooltip=["date:T", alt.Tooltip("loss:Q", format=".4g"),
                 alt.Tooltip("VaR 99%:Q", format=".4g")])
    return (lines + exc).properties(title=title, height=height)


def zone_chart(z: pd.DataFrame) -> alt.LayerChart:
    """Rolling exception count as a step line over the Basel zone bands."""
    top = max(12.0, float(z.iloc[:, 1].max()) + 1.0)
    bands = pd.DataFrame({"zone": ["green", "yellow", "red"], "lo": [0.0, 4.5, 9.5],
                          "hi": [4.5, 9.5, top]})
    band = alt.Chart(bands).mark_rect(opacity=0.18).encode(
        y=alt.Y("lo:Q", title="exceptions", scale=alt.Scale(domain=[0, top])),
        y2="hi:Q",
        color=alt.Color("zone:N", scale=alt.Scale(domain=list(ZONE_COLORS),
                                                  range=list(ZONE_COLORS.values())),
                        legend=alt.Legend(orient="right", title="Basel zone")))
    line = alt.Chart(z).mark_line(interpolate="step-after", strokeWidth=1.5,
                                  color="#0b0b0b").encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y")),
        y="count:Q",
        tooltip=["date:T", "count:Q", "zone:N"])
    return (band + line).properties(
        title="Exceptions over the last 250 days (green 0-4, yellow 5-9, red 10+)", height=240)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def about(summary: pd.DataFrame) -> None:
    with st.expander("About this data", expanded=False):
        spans = summary.groupby("portfolio")[["first_day", "last_day", "days"]].first()
        st.markdown(
            "- **Real data**: daily prices of 20 US stocks (split- and dividend-adjusted) and "
            "the S&P 500 index (price index), 1990-01-02 to 2022-12-28, from the public "
            "Python package `skfolio` (`skfolio.datasets`); `arch` (`arch.data`) for two "
            "cross-checks. Nothing else; no intraday data.\n"
            "- **What is stored**: forecasts and test results, no prices and no index levels. "
            "The P&L of EQ20 (mean of 20 daily returns) and the model P&L of STRANGLE are "
            "results and are stored with the forecasts; the S&P 500 index's own daily returns "
            "(the SPX P&L) are read from `skfolio` when the SPX page is shown.\n"
            "- **STRANGLE is model P&L**: option values come from Black-Scholes-Merton with an "
            "assumed implied volatility (1.2 x EWMA volatility of the index, floor 10%), "
            "dividend yield 2% and rate 3%. There are no option prices in these data.\n"
            "- **Evaluation dates**: "
            + "; ".join(f"{p}: {r.first_day} to {r.last_day} ({r.days} days)"
                        for p, r in spans.iterrows())
            + ". One-day forecasts for day t use data up to t-1 only.\n"
            "- Protocol fixed before any real-data run: `docs/protocol.md`. Results are "
            "about these data and this period only.")


def main() -> None:
    st.set_page_config(page_title="Market-risk backtests", layout="wide")
    st.markdown("<style>.block-container {padding-top: 2rem; padding-bottom: 1rem}</style>",
                unsafe_allow_html=True)
    st.header("Market-risk backtests")
    st.caption("Real data: skfolio S&P 500 panel 1990-2022 (precomputed results; "
               "this page runs no model).")
    summary = read_csv(str(RD / "summary.csv"))
    about(summary)
    c1, c2 = st.columns(2)
    portfolio = c1.selectbox("Portfolio", PORTFOLIOS)
    method = c2.selectbox("Method", METHODS, index=METHODS.index("FHS"))
    st.caption(f"**{portfolio}**: {DESCRIPTION[portfolio]}.")
    fc = forecasts(portfolio)
    d = method_frame(fc, method)
    tabs = st.tabs(["VaR and ES over time", "Tests by method", "ES backtest",
                    "Stress windows", "P&L attribution", "Validation (simulated)"])

    with tabs[0]:
        row = summary[(summary.portfolio == portfolio) & (summary.method == method)].iloc[0]
        k = st.columns(4)
        k[0].metric(f"Exceptions of 99% VaR (expected {0.01 * row.days:.0f})",
                    f"{int(row.exceptions)}")
        k[1].metric("Exception rate", f"{100 * row.rate:.2f}%",
                    help=f"Wilson 95% interval {100 * row.wilson_lo:.2f}-"
                         f"{100 * row.wilson_hi:.2f}%")
        k[2].metric("Kupiec p (Holm)", fmt_p(row.kupiec_holm))
        k[3].metric("Red-zone windows", f"{100 * row.red:.0f}%")
        st.altair_chart(loss_chart(d, f"{portfolio}, {method}: realised loss against the "
                                      "forecasts (red points: exceptions)", 300),
                        use_container_width=True)
        st.altair_chart(zone_chart(zones(d)), use_container_width=True)

    with tabs[1]:
        s = summary[summary.portfolio == portfolio].set_index("method").loc[METHODS]
        rates = s.reset_index()[["method", "rate", "wilson_lo", "wilson_hi"]]
        base = alt.Chart(rates).encode(y=alt.Y("method:N", sort=METHODS, title=None))
        bars = base.mark_bar(color=C_VAR, cornerRadiusEnd=4, height=14).encode(
            x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="exception rate"),
            tooltip=["method", alt.Tooltip("rate:Q", format=".2%"),
                     alt.Tooltip("wilson_lo:Q", format=".2%"),
                     alt.Tooltip("wilson_hi:Q", format=".2%")])
        err = base.mark_rule(color="#0b0b0b").encode(x="wilson_lo:Q", x2="wilson_hi:Q")
        nominal = alt.Chart(pd.DataFrame({"x": [0.01]})).mark_rule(
            strokeDash=[4, 3], color="#52514e").encode(x="x:Q")
        st.altair_chart((bars + err + nominal).properties(
            title="Exception rate with 95% Wilson interval (dashed: 1% nominal)", height=260),
            use_container_width=True)
        tbl = pd.DataFrame({
            "exceptions": s.exceptions, "days": s.days,
            "Kupiec p": s.kupiec_p.map(fmt_p), "Kupiec Holm": s.kupiec_holm.map(fmt_p),
            "Ind. p": s.ind_p.map(fmt_p), "Ind. Holm": s.ind_holm.map(fmt_p),
            "CC p": s.cc_p.map(fmt_p), "CC Holm": s.cc_holm.map(fmt_p),
            "green %": (100 * s.green).round(0), "yellow %": (100 * s.yellow).round(0),
            "red %": (100 * s.red).round(0)})
        st.dataframe(tbl, use_container_width=True)
        st.caption("Holm adjustment over all 21 portfolio x method cells of each test. A cell "
                   "with Holm-adjusted p below 0.05 is a finding; anything else is 'not "
                   "rejected', which is not evidence that the model is correct.")

    with tabs[2]:
        s = summary[summary.portfolio == portfolio].set_index("method").loc[METHODS]
        tbl = pd.DataFrame({
            "Z2": s.z2.round(3), "Z2 p": s.z2_p.map(fmt_p), "Z2 Holm": s.z2_holm.map(fmt_p),
            "shortfall ratio": s.shortfall_ratio.round(2),
            "97.5% exceedances": s.exceedances975, "mean FZ0": s.fz0_mean.round(4),
            "DM vs EWMA-N": s.dm_stat.round(2), "DM Holm": s.dm_holm.map(fmt_p)})
        st.dataframe(tbl, use_container_width=True)
        st.caption(f"Acerbi-Szekely Z2 with {int(s.z2_n_sim.iloc[0])} simulations under each "
                   "method's own predictive distributions (one-sided; negative Z2 = tail "
                   "losses larger than predicted). Shortfall ratio = mean realised loss over "
                   "mean predicted ES on days with loss above the 97.5% VaR (1 for a correct "
                   "model). FZ0: lower is better; DM negative = lower score than EWMA-N.")

    with tabs[3]:
        stress = read_csv(str(RD / "stress.csv"))
        w = st.radio("Window", list(WINDOWS), horizontal=True)
        a, b = WINDOWS[w]
        sub = d[(d.date >= a) & (d.date <= b)]
        st.altair_chart(loss_chart(sub, f"{portfolio}, {method}, {w} ({a} to {b})", 300,
                                   "%d %b %Y"),
                        use_container_width=True)
        t = stress[(stress.portfolio == portfolio) & (stress.window == w)].set_index("method")
        t = t.loc[METHODS, ["days", "exceptions", "expected", "exceedances975",
                            "shortfall_ratio"]].round(2)
        t.columns = ["days", "99% VaR exceptions", "expected", "97.5% exceedances",
                     "shortfall ratio"]
        st.dataframe(t, use_container_width=True)

    with tabs[4]:
        if portfolio == "STRANGLE":
            ex = read_csv(str(RD / "strangle_explain.csv"), parse_dates=["date"])
            cum = ex[["date", *COMPONENT_ORDER]].copy()
            cum[COMPONENT_ORDER] = cum[COMPONENT_ORDER].cumsum()
            long = cum.melt("date", var_name="component", value_name="cumulative P&L")
            st.altair_chart(alt.Chart(long).mark_line(strokeWidth=1.5).encode(
                x=alt.X("date:T", title=None), y="cumulative P&L:Q",
                color=alt.Color("component:N", scale=alt.Scale(domain=COMPONENT_ORDER,
                                                                range=COMPONENT_COLORS),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["date:T", "component:N",
                         alt.Tooltip("cumulative P&L:Q", format=".1f")]).properties(
                title="STRANGLE model P&L explained: cumulative delta, gamma, vega, theta and "
                      "unexplained parts (index points)", height=320), use_container_width=True)
            tot = ex[["full", *COMPONENT_ORDER]].sum().rename("sum over the sample").to_frame()
            st.dataframe(tot.T.round(1), use_container_width=True)
            st.altair_chart(alt.Chart(ex).mark_line(strokeWidth=1, color=C_VAR).encode(
                x=alt.X("date:T", title=None), y=alt.Y("iv:Q", axis=alt.Axis(format="%"),
                                                        title="assumed implied volatility"),
                tooltip=["date:T", alt.Tooltip("iv:Q", format=".1%")]).properties(
                title="Implied-volatility rule (an assumption, not data)", height=180),
                use_container_width=True)
        elif portfolio == "EQ20":
            yr = read_csv(str(RD / "eq20_attribution_year.csv"))
            long = yr.melt("year", var_name="stock", value_name="contribution")
            st.altair_chart(alt.Chart(long).mark_rect().encode(
                x=alt.X("year:O", title=None), y=alt.Y("stock:N", title=None),
                color=alt.Color("contribution:Q", scale=alt.Scale(scheme="redblue",
                                                                  domainMid=0),
                                legend=alt.Legend(format="%", title="P&L")),
                tooltip=["year:O", "stock:N", alt.Tooltip("contribution:Q", format=".2%")]
            ).properties(title="Realised P&L by stock and year (sum of daily return / 20)",
                         height=420), use_container_width=True)
            eu = read_csv(str(RD / "eq20_euler_stress.csv"))
            w = st.radio("Euler ES contributions at the start of", list(WINDOWS),
                         horizontal=True, key="euler")
            e = eu[eu.window == w]
            st.altair_chart(alt.Chart(e).mark_bar(color=C_VAR, cornerRadiusEnd=4).encode(
                x=alt.X("es_contribution:Q", axis=alt.Axis(format="%"),
                        title="contribution to 97.5% ES"),
                y=alt.Y("stock:N", sort="-x", title=None),
                tooltip=["stock:N", alt.Tooltip("es_contribution:Q", format=".3%")]
            ).properties(title=f"HS500 Euler ES contributions on {e.day.iloc[0]} (sum = "
                               f"portfolio ES {100 * e.portfolio_es.iloc[0]:.2f}%)",
                         height=420), use_container_width=True)
        else:
            st.info("SPX is a single factor: there is nothing to attribute. Choose EQ20 (by "
                    "stock) or STRANGLE (option P&L explain).")

    with tabs[5]:
        st.caption("Simulated data with known truth (docs/validation.md): GARCH(1,1)-t "
                   "returns, 200 replications of 1000 forecast days.")
        sset = st.radio("Parameter set", ["calm", "typical", "persistent_fat"],
                        horizontal=True, index=1)
        v = read_csv(str(VAL / f"garch_dgp_{sset}_summary.csv")).set_index("method")
        cols = ["var_bias", "var_rmse", "es_bias", "exc_rate", "kupiec_rej", "cc_rej"]
        st.dataframe((100 * v[cols]).round(2), use_container_width=True)
        st.caption("Percent: relative bias and RMSE of VaR/ES against the true conditional "
                   "values, exception rate, rejection rates at 5%.")


main()
