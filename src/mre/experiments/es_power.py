"""Experiment (e): size and power of the Acerbi-Szekely Z1/Z2 tests with 5000 simulations.

Protocol: docs/validation.md, section (e). Usage::

    python -m mre.experiments.es_power --T 250 --reps 200 --out results/validation

Writes ``es_power_T<T>.csv`` (one row per replication and model) and a manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mre.backtest import acerbi_szekely
from mre.experiments.common import Manifest
from mre.experiments.garch_dgp import params_for
from mre.garch import simulate_garch_t
from mre.measures import LocationScalePnL

SEED = 20261109
N_SIM = 5000
LEVEL = 0.975
LENGTHS = (250, 1000)
MODELS = {"correct": ("t", 1.0), "normal": ("normal", 1.0), "scale0.9": ("t", 0.9)}


def run_replication(T: int, rep: int) -> list[dict]:
    p = params_for("typical")
    r, s2 = simulate_garch_t(T, p, np.random.default_rng([SEED, LENGTHS.index(T), rep]),
                             burn=500)
    sd = np.sqrt(s2)
    rows = []
    for name, (dist, mult) in MODELS.items():
        preds = [LocationScalePnL(0.0, mult * s, dist, p.nu) for s in sd]
        res = acerbi_szekely(r, preds, LEVEL, n_sim=N_SIM, seed=rep)
        rows.append({"T": T, "rep": rep, "model": name,
                     "z1": res["Z1"].statistic, "z1_p": res["Z1"].p_value,
                     "z2": res["Z2"].statistic, "z2_p": res["Z2"].p_value,
                     "n_exceptions": res["Z2"].details["n_exceptions"]})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--T", type=int, choices=LENGTHS, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    man = Manifest("mre.experiments.es_power",
                   f"returns default_rng([{SEED}, {LENGTHS.index(args.T)}, rep]); Monte Carlo "
                   f"p-value seed = rep; rep = 0..{args.reps - 1}")
    rows = [row for rep in range(args.reps) for row in run_replication(args.T, rep)]
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"es_power_T{args.T}"
    pd.DataFrame(rows).to_csv(args.out / f"{name}.csv", index=False, float_format="%.6g")
    man.write(args.out, name, [f"{name}.csv"],
              notes=f"GARCH(1,1)-t set typical; {N_SIM} simulations per test; level {LEVEL}")


if __name__ == "__main__":
    main()
