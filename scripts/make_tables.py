#!/usr/bin/env python3
"""Render the results as markdown tables.

Simulated data: results/validation/*.csv -> results/validation/tables.md and the block between
``<!-- tables:begin -->`` and ``<!-- tables:end -->`` in docs/validation.md.
Real data: results/realdata/*.csv -> results/realdata/tables.md and the same block in
docs/findings.md (if that file exists). Every number in the tables comes from the CSV files;
nothing is typed in.

Usage: python3 scripts/make_tables.py [--results results/validation] [--doc docs/validation.md]
       [--realdata results/realdata] [--findings docs/findings.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SETS = ("calm", "typical", "persistent_fat")
METHOD_ORDER = ["Oracle", "HS250", "HS500", "HS1000", "EWMA-N", "EWMA-t6", "FHS1000", "MC-N"]


def pct(x: float, se: float | None = None, digits: int = 1) -> str:
    s = f"{100 * x:.{digits}f}"
    return s if se is None else f"{s} ({100 * se:.{digits}f})"


def num(x: float, se: float | None = None, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s if se is None else f"{s} ({se:.{digits}f})"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def garch_tables(res: Path) -> list[str]:
    parts = []
    for s in SETS:
        f = res / f"garch_dgp_{s}_summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("method").loc[METHOD_ORDER]
        raw = pd.read_csv(res / f"garch_dgp_{s}.csv")
        n_rep = int(d["n_rep"].iloc[0])
        rows = []
        for m, r in d.iterrows():
            rows.append([
                m, pct(r.var_bias, r.var_bias_se), pct(r.var_rmse, r.var_rmse_se),
                pct(r.es_bias, r.es_bias_se), pct(r.es_rmse, r.es_rmse_se),
                pct(r.exc_rate, r.exc_rate_se, 2), pct(r.kupiec_rej, r.kupiec_rej_se, 0),
                pct(r.ind_rej, r.ind_rej_se, 0), pct(r.cc_rej, r.cc_rej_se, 0),
                pct(r.yellow + r.red, None, 0), pct(r.red, r.red_se, 1),
                "-" if m == "Oracle" else num(r.fz_diff, r.fz_diff_se),
                "-" if m == "Oracle" else pct(r.dm_rej, r.dm_rej_se, 0),
            ])
        fhs = raw[raw.method == "FHS1000"]
        parts.append(
            f"**Table A-{s}. Simulated data: GARCH(1,1)-t set `{s}`, {n_rep} replications of "
            "1000 forecast days.** Mean over replications (Monte Carlo standard error). Bias and "
            "RMSE are relative to the true conditional VaR/ES, in percent. Exc. = 99% VaR "
            "exception rate (%). Kupiec, Ind., CC = rejection rate at 5% (%). Y+R = share of "
            "250-day windows in the yellow or red zone (%); Red = share in the red zone (%). "
            "FZ0 = mean FZ0 loss minus the Oracle's; DM = share of replications in which "
            f"Diebold-Mariano rejects equal FZ0 loss vs the Oracle at 5% (%). FHS: "
            f"{int(fhs.nonconverged_fits.sum())} of {int(fhs.n_fits.sum())} GARCH fits did not "
            "converge.\n\n"
            + md_table(["Method", "VaR bias", "VaR RMSE", "ES bias", "ES RMSE", "Exc.",
                        "Kupiec", "Ind.", "CC", "Y+R", "Red", "FZ0", "DM"], rows))
    return parts


def size_power_tables(res: Path) -> list[str]:
    parts = []
    f = res / "size_power_coverage.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [[str(int(r.n)), pct(r.true_rate, None, 2), pct(r.kupiec_rej, r.kupiec_rej_se),
                 pct(r.kupiec_exact), pct(r.ind_rej, r.ind_rej_se), pct(r.cc_rej, r.cc_rej_se)]
                for r in d.itertuples()]
        parts.append(
            "**Table B1. Simulated data: rejection rates (%) at 5% of 99% VaR backtests when "
            "exceptions are iid with the given true rate** (binomial standard error in "
            "parentheses; 'exact' is the binomial probability of the Kupiec rejection region). "
            "The row with true rate 1.00% is the size; the other rows are power.\n\n"
            + md_table(["n", "true rate (%)", "Kupiec", "Kupiec exact", "Christoffersen ind.",
                        "Christoffersen CC"], rows))
    f = res / "size_power_clustering.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [[str(int(r.n)), f"{r.pi11:.2f}", pct(r.mean_rate, None, 2),
                 pct(r.kupiec_rej, r.kupiec_rej_se), pct(r.ind_rej, r.ind_rej_se),
                 pct(r.cc_rej, r.cc_rej_se)] for r in d.itertuples()]
        parts.append(
            "**Table B2. Simulated data: rejection rates (%) at 5% when exceptions cluster** "
            "(Markov chain with stationary rate 1% and P(exception | exception yesterday) = "
            "pi11; pi11 = 0.01 is independence).\n\n"
            + md_table(["n", "pi11", "observed rate (%)", "Kupiec", "Christoffersen ind.",
                        "Christoffersen CC"], rows))
    return parts


def basel_tables(res: Path) -> list[str]:
    f = res / "basel_zones.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f, dtype={"item": str})
    gof = pd.read_csv(res / "basel_zones_gof.csv").set_index("true_rate")
    rows = []
    for item in d["item"].unique():
        r = [item]
        for rate in sorted(d["true_rate"].unique()):
            x = d[(d.item == item) & (d.true_rate == rate)].iloc[0]
            r += [pct(x.simulated, x.se, 2), pct(x.exact, None, 2)]
        rows.append(r)
    rates = sorted(d["true_rate"].unique())
    header = ["exceptions / zone"]
    for rate in rates:
        header += [f"simulated, rate {100 * rate:g}%", f"exact, rate {100 * rate:g}%"]
    g = "; ".join(f"rate {100 * k:g}%: chi2 = {v.pearson_chi2:.1f}, p = {v.p_value:.2f}"
                  for k, v in gof.iterrows())
    n = int(gof["windows"].iloc[0])
    return [f"**Table C. Simulated data: Basel zone and exception-count probabilities (%) for "
            f"250 days of 99% VaR, {n} simulated windows per rate** (binomial standard error in "
            f"parentheses). Pearson goodness of fit of the 11 count cells, 10 df: {g}.\n\n"
            + md_table(header, rows)]


def option_tables(res: Path) -> list[str]:
    f = res / "option_tail_summary.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f)
    raw = pd.read_csv(res / "option_tail.csv")
    nb = raw["batch"].nunique()
    parts = []
    for dist in ("normal", "t4"):
        rows = []
        sub = d[d.dist == dist]
        for (port, sig), g in sub.groupby(["portfolio", "sigma0"], sort=False):
            dg = g[g.method == "DG"].iloc[0]
            dgv = g[g.method == "DGV"].iloc[0]
            dgvx = g[g.method == "DGVX"].iloc[0]
            rows.append([port, f"{sig:.2f}", num(dg.var_full, None, 3),
                         pct(dg.var_rel_err, dg.var_rel_err_se),
                         pct(dgv.var_rel_err, dgv.var_rel_err_se),
                         pct(dgvx.var_rel_err, dgvx.var_rel_err_se),
                         pct(dg.es_rel_err, dg.es_rel_err_se),
                         pct(dgv.es_rel_err, dgv.es_rel_err_se),
                         pct(dgvx.es_rel_err, dgvx.es_rel_err_se),
                         pct(dgv.tail_mae_rel, dgv.tail_mae_rel_se)])
        parts.append(
            f"**Table D-{dist}. Simulated data: error of Greek approximations relative to full "
            f"revaluation, one-day P&L, {dist} innovations.** Mean over {nb} batches of "
            "scenarios (Monte Carlo standard error), in percent. 'VaR full' is the "
            "full-revaluation 99% VaR in index points (index = 100). DG = delta-gamma-theta, "
            "DGV = delta-gamma-vega-theta, DGVX = DGV plus vanna and volga (section d'). "
            "Tail MAE = mean absolute P&L error of DGV over the "
            "scenarios in the full-revaluation 97.5% tail, divided by the full-revaluation "
            "97.5% ES.\n\n"
            + md_table(["Portfolio", "sigma0", "VaR full", "DG VaR err", "DGV VaR err",
                        "DGVX VaR err", "DG ES err", "DGV ES err", "DGVX ES err",
                        "DGV tail MAE"], rows))
    return parts


def es_power_tables(res: Path) -> list[str]:
    files = sorted(res.glob("es_power_T*.csv"), key=lambda f: int(f.stem.split("T")[-1]))
    if not files:
        return []
    d = pd.concat([pd.read_csv(f) for f in files])
    rows = []
    for (T, model), g in d.groupby(["T", "model"], sort=False):
        n = len(g)
        r2, r1 = (g.z2_p < 0.05).mean(), (g.z1_p < 0.05).mean()
        rows.append([str(T), model, pct(r2, np.sqrt(r2 * (1 - r2) / n)), num(g.z2.mean(), None, 3),
                     pct(r1, np.sqrt(r1 * (1 - r1) / n)), num(g.z1.mean(), None, 3),
                     num(g.n_exceptions.mean(), None, 1), str(n)])
    return ["**Table E. Simulated data: size and power of the Acerbi-Szekely tests at 97.5% "
            "with 5000 simulations per test (GARCH(1,1)-t set `typical`).** Rejection rate at "
            "5% (%) with its binomial standard error; mean of the statistic; mean number of "
            "97.5% VaR exceedances. `correct` = the true conditional distribution (size); "
            "`normal` = right variance, normal shape (ES understated); `scale0.9` = true shape "
            "with 0.9 times the volatility.\n\n"
            + md_table(["T", "model", "Z2 rejection", "mean Z2", "Z1 rejection", "mean Z1",
                        "exceedances", "replications"], rows)]


def hs_regime_tables(res: Path) -> list[str]:
    parts = []
    for s_ in SETS:
        f = res / f"hs_regime_{s_}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        n_rep = d["rep"].nunique()
        rows = []
        order = {"level": ["low", "middle", "high"], "change": ["falling", "stable", "rising"]}
        for m in ("HS250", "HS500", "HS1000"):
            for kind, classes in order.items():
                dk = d[(d.method == m) & (d.classification == kind)]
                tot = dk.n_days.sum()
                for c in classes:
                    g = dk[dk["class"] == c]
                    k = len(g)
                    if k == 0:
                        rows.append([m, kind, c, "-", "-", "0", "0.0"])
                        continue
                    se_e = g.var_rel_err.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
                    se_x = g.exc_rate.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
                    rows.append([m, kind, c, pct(g.var_rel_err.mean(), se_e),
                                 pct(g.exc_rate.mean(), se_x, 2), str(k),
                                 pct(g.n_days.sum() / tot)])
        parts.append(
            f"**Table F-{s_}. Simulated data: historical-simulation 99% VaR error by volatility "
            f"regime, GARCH(1,1)-t set `{s_}`, {n_rep} replications of 2000 forecast days.** "
            "Mean over replications of the class mean (Monte Carlo standard error), in percent. "
            "Level: true sigma below 0.8 / above 1.25 times the unconditional volatility. "
            "Change: true sigma divided by the root mean true variance over the method's own "
            "window, below 0.8 (falling) / above 1.25 (rising). Reps = replications with days "
            "in the class; Days = share of forecast days in the class (%).\n\n"
            + md_table(["Method", "classification", "class", "VaR error", "Exc. rate", "Reps",
                        "Days"], rows))
    return parts


RD_TITLE = "real data: skfolio S&P 500 panel 1990-2022"
RD_METHODS = ["HS250", "HS500", "HS1000", "EWMA-N", "EWMA-T", "FHS", "MC"]
RD_PORTS = ["EQ20", "SPX", "STRANGLE"]


def fp(p: float) -> str:
    """p-value: three decimals, or scientific notation below 0.001."""
    if not np.isfinite(p):
        return "-"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def fp_holm(p: float, h: float) -> str:
    """Raw p-value with the Holm-adjusted one; a finding (Holm < 0.05) is in bold."""
    if not np.isfinite(p):
        return "-"
    cell = f"{fp(p)} ({fp(h)})"
    return f"**{cell}**" if h < 0.05 else cell


def realdata_tables(res: Path) -> list[str]:
    f = res / "summary.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f)
    d["order"] = d.method.map(RD_METHODS.index) + 10 * d.portfolio.map(RD_PORTS.index)
    d = d.sort_values("order")
    n_fam = {c: int(d[c].notna().sum()) for c in ("kupiec_p", "ind_p", "cc_p", "z2_p", "dm_p")}
    by_year = pd.read_csv(res / "by_year.csv")
    stress = pd.read_csv(res / "stress.csv")
    fails = pd.read_csv(res / "failures.csv")
    parts = []
    span = {p: f"{g.first_day.iloc[0]} to {g.last_day.iloc[0]}, {int(g.days.iloc[0])} days"
            for p, g in d.groupby("portfolio")}
    rows = []
    for r in d.itertuples():
        rows.append([r.portfolio, r.method, f"{r.exceptions} ({0.01 * r.days:.0f})",
                     f"{100 * r.rate:.2f} [{100 * r.wilson_lo:.2f}, {100 * r.wilson_hi:.2f}]",
                     fp_holm(r.kupiec_p, r.kupiec_holm), fp_holm(r.ind_p, r.ind_holm),
                     fp_holm(r.cc_p, r.cc_holm),
                     f"{100 * r.green:.0f} / {100 * r.yellow:.0f} / {100 * r.red:.0f}"])
    parts.append(
        f"**Table R1. {RD_TITLE}: 99% VaR backtests over the full evaluation sample** (EQ20 and "
        f"SPX: {span['EQ20']}; STRANGLE: {span['STRANGLE']}; {len(fails)} failed forecasts). "
        "Exceptions with the 1% expectation in parentheses; exception rate (%) with the 95% "
        "Wilson interval; p-values raw (Holm-adjusted within the family of "
        f"{n_fam['kupiec_p']} cells), bold where the Holm-adjusted p-value is below 0.05 (a "
        "finding). Zones: share (%) of rolling 250-day windows green / yellow / red. STRANGLE "
        "is model P&L under an assumed implied-volatility rule.\n\n"
        + md_table(["Portfolio", "Method", "Exceptions", "Rate % [Wilson 95%]", "Kupiec",
                    "Christoffersen ind.", "Christoffersen CC", "Zones G / Y / R"], rows))
    rows = []
    for r in d.itertuples():
        rows.append([r.portfolio, r.method, str(r.es_days), f"{r.z2:.3f}",
                     fp_holm(r.z2_p, r.z2_holm),
                     f"{r.shortfall_ratio:.2f} ({r.exceedances975})", f"{r.fz0_mean:.4f}",
                     "-" if not np.isfinite(r.dm_stat) else f"{r.dm_stat:.2f}",
                     fp_holm(r.dm_p, r.dm_holm)])
    parts.append(
        f"**Table R2. {RD_TITLE}: 97.5% ES backtest and joint VaR/ES scores.** Z2 = "
        f"Acerbi-Szekely Z2 ({int(d.z2_n_sim.iloc[0])} simulations under each method's own "
        "predictive distributions; negative = realised tail losses larger than predicted), "
        f"p-value raw (Holm over {n_fam['z2_p']} cells). Shortfall ratio = mean realised loss "
        "over mean predicted ES on the days with loss above the 97.5% VaR (number of such days "
        "in parentheses; 1 for a correct model). FZ0 = mean FZ0 score (lower is better; scale "
        "differs by portfolio). DM = Diebold-Mariano statistic of the FZ0 difference against "
        f"EWMA-N (negative = lower score than EWMA-N), Newey-West lag {int(d.dm_lag.iloc[0])}, "
        f"p-value raw (Holm over {n_fam['dm_p']} cells). ES days = common days on which "
        "every method's 97.5% ES is a positive loss; all columns of this table use only those "
        "days (docs/protocol.md, Part 3).\n\n"
        + md_table(["Portfolio", "Method", "ES days", "Z2", "Z2 p (Holm)", "Shortfall ratio (n)",
                    "FZ0", "DM", "DM p (Holm)"], rows))
    windows = list(dict.fromkeys(stress.window))
    rows = []
    for (port, m), g in stress.groupby(["portfolio", "method"], sort=False):
        cells = [port, m]
        for w in windows:
            x = g[g.window == w].iloc[0]
            ratio = "-" if not np.isfinite(x.shortfall_ratio) else f"{x.shortfall_ratio:.2f}"
            cells += [f"{x.exceptions} / {x.days}", f"{ratio} ({x.exceedances975})"]
        rows.append(cells)
    head = ["Portfolio", "Method"]
    for w in windows:
        x = stress[stress.window == w].iloc[0]
        head += [f"{w}: exc. / days", f"{w}: shortfall ratio (n)"]
    wdesc = "; ".join(f"{w} = {stress[stress.window == w].start.iloc[0]} to "
                      f"{stress[stress.window == w].end.iloc[0]}" for w in windows)
    parts.append(
        f"**Table R3. {RD_TITLE}: pre-specified stress windows.** 99% VaR exceptions / "
        "evaluation days in the window (1% of the days is the expectation); shortfall ratio as "
        f"in R2 over the 97.5% exceedances inside the window (their number in parentheses). "
        f"Windows: {wdesc}. No test is applied inside the windows.\n\n"
        + md_table(head, rows))
    rows = []
    n_years = by_year.year.nunique()
    for (port, m), g in by_year.groupby(["portfolio", "method"], sort=False):
        rows.append([port, m, str(int((g.kupiec_p < 0.05).sum())),
                     str(int((g.kupiec_holm < 0.05).sum())), str(int((g.cc_p < 0.05).sum())),
                     str(int((g.cc_holm < 0.05).sum())), str(int((g.ind_holm < 0.05).sum())),
                     str(int(g.exceptions.max())), str(int(g.loc[g.exceptions.idxmax(), "year"]))])
    parts.append(
        f"**Table R4. {RD_TITLE}: calendar-year backtests ({n_years} years, "
        f"{len(by_year)} cells per test family).** Number of years in which the test rejects at "
        "5% before (raw) and after the Holm adjustment over all cells of the family; largest "
        "number of exceptions in one year and that year.\n\n"
        + md_table(["Portfolio", "Method", "Kupiec raw", "Kupiec Holm", "CC raw", "CC Holm",
                    "Ind. Holm", "Max exc. in a year", "Year"], rows))
    return parts


def realdata_check_tables(res: Path) -> list[str]:
    parts = []
    f = res / "checks.csv"
    if f.exists():
        c = pd.read_csv(f)
        rows = [[r.check, f"{r.value:.6g}", str(r.note)] for r in c.itertuples()]
        parts.append(f"**Table R5. {RD_TITLE}: data checks.**\n\n"
                     + md_table(["Check", "Value", "Note"], rows))
    fits = sorted(res.glob("fhs_fits_*.csv"))
    if fits:
        rows = []
        for f in fits:
            g = pd.read_csv(f)
            for fac, h in g.groupby("factor", sort=False):
                rows.append([f.stem.replace("fhs_fits_", ""), fac, str(len(h)),
                             str(int((~h.converged).sum())),
                             f"{h.zero_move_share.max():.2f}", f"{h.max_abs_z.median():.3g}",
                             f"{h.max_abs_z.max():.3g}",
                             f"{h.max_abs_scenario_move.max():.3g}"])
        parts.append(
            f"**Table R6. {RD_TITLE}: FHS GARCH(1,1)-t fits, one per 20 trading days per "
            "factor.** Non-converged = fits whose optimiser did not report success; zero-move "
            "share = largest share of exactly-zero moves in a 1000-day window; |z| = largest "
            "absolute standardised residual in the window (median and maximum over refits); "
            "largest scenario move = maximum of next-day volatility times largest |z| (a log "
            "move of the factor).\n\n"
            + md_table(["Portfolio", "Factor", "Fits", "Non-converged", "Zero-move share",
                        "max |z| median", "max |z| max", "Largest scenario move"], rows))
    return parts


def replace_block(doc: Path, text: str) -> None:
    body = doc.read_text()
    new = re.sub(r"<!-- tables:begin -->.*<!-- tables:end -->",
                 lambda _: "<!-- tables:begin -->\n" + text + "<!-- tables:end -->", body,
                 flags=re.S)
    doc.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO / "results" / "validation")
    ap.add_argument("--doc", type=Path, default=REPO / "docs" / "validation.md")
    ap.add_argument("--realdata", type=Path, default=REPO / "results" / "realdata")
    ap.add_argument("--findings", type=Path, default=REPO / "docs" / "findings.md")
    args = ap.parse_args()
    parts = (garch_tables(args.results) + size_power_tables(args.results)
             + basel_tables(args.results) + option_tables(args.results)
             + es_power_tables(args.results) + hs_regime_tables(args.results))
    text = "\n\n".join(parts) + "\n"
    (args.results / "tables.md").write_text(
        "<!-- Generated by scripts/make_tables.py from the CSV files in this directory. -->\n\n"
        + text)
    replace_block(args.doc, text)
    rd = realdata_tables(args.realdata) + realdata_check_tables(args.realdata)
    if rd:
        rtext = "\n\n".join(rd) + "\n"
        (args.realdata / "tables.md").write_text(
            "<!-- Generated by scripts/make_tables.py from the CSV files in this directory. "
            "-->\n\n" + rtext)
        if args.findings.exists():
            replace_block(args.findings, rtext)
    print(f"wrote {len(parts)} simulated-data tables and {len(rd)} real-data tables")


if __name__ == "__main__":
    main()
