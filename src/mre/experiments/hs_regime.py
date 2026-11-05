"""Experiment (f): bias of historical-simulation VaR by volatility regime (level and change).

Protocol: docs/validation.md, section (f). Usage::

    python -m mre.experiments.hs_regime --set typical --reps 20 --out results/validation

Writes ``hs_regime_<set>.csv``: one row per replication, method, classification and class with
the class mean of the relative VaR error, the exception rate and the number of days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mre.experiments.common import Manifest
from mre.experiments.garch_dgp import SETS, params_for
from mre.garch import simulate_garch_t
from mre.measures import LocationScalePnL
from mre.models import HistoricalSimulation, rolling_forecasts

SEED = 20261110
N_HIST, N_TEST, BURN = 1000, 2000, 500
WINDOWS = (250, 500, 1000)
LOW, HIGH = 0.8, 1.25


def classify_level(sigma: np.ndarray, v: float) -> np.ndarray:
    """``low`` if sigma < 0.8 v, ``high`` if sigma > 1.25 v, else ``middle``."""
    return np.where(sigma < LOW * v, "low", np.where(sigma > HIGH * v, "high", "middle"))


def classify_change(s2: np.ndarray, start: int, window: int) -> np.ndarray:
    """For days t >= start: rho_t = sigma_t / sqrt(mean of sigma_s^2, s = t-window..t-1);
    ``rising`` if rho > 1.25, ``falling`` if rho < 0.8, else ``stable``."""
    c = np.concatenate(([0.0], np.cumsum(s2)))
    t = np.arange(start, s2.size)
    rho = np.sqrt(s2[t]) / np.sqrt((c[t] - c[t - window]) / window)
    return np.where(rho > HIGH, "rising", np.where(rho < LOW, "falling", "stable"))


def run_replication(set_name: str, rep: int) -> list[dict]:
    p = params_for(set_name)
    j = list(SETS).index(set_name)
    r, s2 = simulate_garch_t(N_HIST + N_TEST, p, np.random.default_rng([SEED, j, rep]),
                             burn=BURN)
    v = float(np.sqrt(p.unconditional_variance))
    sig = np.sqrt(s2[N_HIST:])
    var_true = np.array([LocationScalePnL(0.0, s, "t", p.nu).var(0.99) for s in sig])
    level = classify_level(sig, v)
    rows = []
    for w in WINDOWS:
        path = rolling_forecasts(HistoricalSimulation(w), r, N_HIST)
        err = np.array([f.var for f in path.forecasts]) / var_true - 1.0
        exc = path.realized < -np.array([f.var for f in path.forecasts])
        for kind, cls in (("level", level), ("change", classify_change(s2, N_HIST, w))):
            for c in np.unique(cls):
                m = cls == c
                rows.append({"set": set_name, "rep": rep, "method": f"HS{w}",
                             "classification": kind, "class": c, "n_days": int(m.sum()),
                             "var_rel_err": float(err[m].mean()),
                             "exc_rate": float(exc[m].mean())})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    j = list(SETS).index(args.set)
    man = Manifest("mre.experiments.hs_regime",
                   f"default_rng([{SEED}, {j}, rep]) for rep = 0..{args.reps - 1}")
    rows = [row for rep in range(args.reps) for row in run_replication(args.set, rep)]
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"hs_regime_{args.set}"
    pd.DataFrame(rows).to_csv(args.out / f"{name}.csv", index=False, float_format="%.6g")
    man.write(args.out, name, [f"{name}.csv"],
              notes=f"{N_TEST} forecast days per replication after {N_HIST} history days")


if __name__ == "__main__":
    main()
