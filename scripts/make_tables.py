#!/usr/bin/env python3
"""Render the validation results (results/validation/*.csv) as markdown tables.

Writes results/validation/tables.md and replaces the block between ``<!-- tables:begin -->`` and
``<!-- tables:end -->`` in docs/validation.md with the same text. Every number in the tables comes
from the CSV files; nothing is typed in.

Usage: python3 scripts/make_tables.py [--results results/validation] [--doc docs/validation.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
            rows.append([port, f"{sig:.2f}", num(dg.var_full, None, 3),
                         pct(dg.var_rel_err, dg.var_rel_err_se),
                         pct(dgv.var_rel_err, dgv.var_rel_err_se),
                         pct(dg.es_rel_err, dg.es_rel_err_se),
                         pct(dgv.es_rel_err, dgv.es_rel_err_se),
                         pct(dgv.tail_mae_rel, dgv.tail_mae_rel_se)])
        parts.append(
            f"**Table D-{dist}. Simulated data: error of Greek approximations relative to full "
            f"revaluation, one-day P&L, {dist} innovations.** Mean over {nb} batches of "
            "scenarios (Monte Carlo standard error), in percent. 'VaR full' is the "
            "full-revaluation 99% VaR in index points (index = 100). DG = delta-gamma-theta, "
            "DGV = delta-gamma-vega-theta. Tail MAE = mean absolute P&L error of DGV over the "
            "scenarios in the full-revaluation 97.5% tail, divided by the full-revaluation "
            "97.5% ES.\n\n"
            + md_table(["Portfolio", "sigma0", "VaR full", "DG VaR err", "DGV VaR err",
                        "DG ES err", "DGV ES err", "DGV tail MAE"], rows))
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO / "results" / "validation")
    ap.add_argument("--doc", type=Path, default=REPO / "docs" / "validation.md")
    args = ap.parse_args()
    parts = (garch_tables(args.results) + size_power_tables(args.results)
             + basel_tables(args.results) + option_tables(args.results))
    text = "\n\n".join(parts) + "\n"
    (args.results / "tables.md").write_text(
        "<!-- Generated by scripts/make_tables.py from the CSV files in this directory. -->\n\n"
        + text)
    doc = args.doc.read_text()
    new = re.sub(r"<!-- tables:begin -->.*<!-- tables:end -->",
                 lambda _: "<!-- tables:begin -->\n" + text + "<!-- tables:end -->", doc,
                 flags=re.S)
    args.doc.write_text(new)
    print(f"wrote {len(parts)} tables")


if __name__ == "__main__":
    main()
