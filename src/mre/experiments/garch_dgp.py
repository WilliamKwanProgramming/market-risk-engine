"""Experiment (a): every risk model on a GARCH(1,1)-t process with known conditional VaR and ES.

Protocol: docs/validation.md, section (a). Usage::

    python -m mre.experiments.garch_dgp --set calm --reps 200 --out results/validation

Writes ``garch_dgp_<set>.csv`` (one row per replication and method),
``garch_dgp_<set>_summary.csv`` (mean and Monte Carlo standard error per method) and a manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mre.backtest import christoffersen, diebold_mariano, fz0_loss, kupiec_pof, traffic_light
from mre.experiments.common import Manifest, mean_se
from mre.garch import GarchParams, simulate_garch_t
from mre.measures import LocationScalePnL
from mre.models import (
    EWMAParametric,
    FilteredHistoricalSimulation,
    HistoricalSimulation,
    MonteCarlo,
    RiskModel,
    rolling_forecasts,
)

BASE_SEED = 20261105
N_HIST, N_TEST, BURN = 1000, 1000, 500
SETS = {  # name: (alpha, beta, nu, unconditional daily volatility)
    "calm": (0.05, 0.90, 30.0, 0.006),
    "typical": (0.08, 0.90, 8.0, 0.010),
    "persistent_fat": (0.10, 0.89, 4.0, 0.012),
}
METRICS = ["var_bias", "var_rmse", "es_bias", "es_rmse", "exc_rate", "kupiec_rej", "ind_rej",
           "cc_rej", "ind_testable", "green", "yellow", "red", "fz_diff", "dm_rej",
           "nonconverged_fits"]


def params_for(name: str) -> GarchParams:
    a, b, nu, vol = SETS[name]
    return GarchParams(mu=0.0, omega=vol**2 * (1.0 - a - b), alpha=a, beta=b, nu=nu)


class Oracle(RiskModel):
    """The true conditional distribution ``sigma_t z`` (uses the known variance path)."""

    name = "Oracle"

    def __init__(self, sigma2: np.ndarray, nu: float):
        self.sigma2, self.nu = sigma2, nu

    def predict(self, history, fitted, pnl, index):
        return LocationScalePnL(0.0, float(np.sqrt(self.sigma2[index])), "t", self.nu), {}


def run_replication(set_name: str, rep: int) -> list[dict]:
    p = params_for(set_name)
    j = list(SETS).index(set_name)
    r, s2 = simulate_garch_t(N_HIST + N_TEST, p, np.random.default_rng([BASE_SEED, j, rep]),
                             burn=BURN)
    models: list[RiskModel] = [
        Oracle(s2, p.nu), HistoricalSimulation(250), HistoricalSimulation(500),
        HistoricalSimulation(1000), EWMAParametric(), EWMAParametric(dist="t", nu=6.0),
        FilteredHistoricalSimulation(1000, refit_every=250), MonteCarlo(n_draws=5000, seed=rep),
    ]
    truth = [LocationScalePnL(0.0, float(np.sqrt(v)), "t", p.nu) for v in s2[N_HIST:]]
    var_true = np.array([d.var(0.99) for d in truth])
    es_true = np.array([d.es(0.975) for d in truth])
    rows, fz_oracle = [], None
    for m in models:
        path = rolling_forecasts(m, r, N_HIST)
        var = np.array([f.var for f in path.forecasts])
        es = np.array([f.es for f in path.forecasts])
        var975 = np.array([f.var_at_es_level for f in path.forecasts])
        x = path.realized
        exc = x < -var
        chr_ = christoffersen(exc, 0.99)
        zones = [traffic_light(int(exc[k:k + 250].sum())).zone for k in range(0, N_TEST, 250)]
        fz = fz0_loss(x, var975, es, 0.975)
        if fz_oracle is None:
            fz_oracle = fz
        dm = diebold_mariano(fz, fz_oracle) if m.name != "Oracle" else None
        nonconv = sum(not f.converged for fits in path.fits for f in fits)
        rows.append({
            "set": set_name, "rep": rep, "method": m.name,
            "var_bias": np.mean(var / var_true - 1.0),
            "var_rmse": np.sqrt(np.mean((var / var_true - 1.0) ** 2)),
            "es_bias": np.mean(es / es_true - 1.0),
            "es_rmse": np.sqrt(np.mean((es / es_true - 1.0) ** 2)),
            "exc_rate": exc.mean(),
            "kupiec_rej": kupiec_pof(exc, 0.99).reject(0.05),
            "ind_rej": chr_["ind"].reject(0.05),
            "cc_rej": chr_["cc"].reject(0.05),
            "ind_testable": chr_["ind"].details.get("ind_testable", False),
            "green": zones.count("green"), "yellow": zones.count("yellow"),
            "red": zones.count("red"),
            "fz_diff": float(np.mean(fz - fz_oracle)),
            "dm_rej": bool(dm.reject(0.05)) if dm is not None else False,
            "nonconverged_fits": nonconv, "n_fits": sum(len(f) for f in path.fits),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    j = list(SETS).index(args.set)
    man = Manifest("mre.experiments.garch_dgp", f"default_rng([{BASE_SEED}, {j}, rep]) for "
                   f"rep = 0..{args.reps - 1}; MC-N seed = rep")
    rows = [row for rep in range(args.reps) for row in run_replication(args.set, rep)]
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    raw = f"garch_dgp_{args.set}.csv"
    summ = f"garch_dgp_{args.set}_summary.csv"
    df.to_csv(args.out / raw, index=False, float_format="%.6g")
    df["green"] /= 4
    df["yellow"] /= 4
    df["red"] /= 4
    mean_se(df, ["set", "method"], METRICS).to_csv(args.out / summ, index=False,
                                                   float_format="%.6g")
    man.write(args.out, f"garch_dgp_{args.set}", [raw, summ],
              notes="green/yellow/red in the summary are fractions of the four 250-day windows")


if __name__ == "__main__":
    main()
