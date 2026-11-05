"""Experiment (b): size and power of the Kupiec and Christoffersen tests at 5%.

Protocol: docs/validation.md, section (b). Usage::

    python -m mre.experiments.size_power --reps 5000 --out results/validation

Writes ``size_power_coverage.csv`` and ``size_power_clustering.csv`` and a manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mre.backtest import christoffersen, kupiec_pof
from mre.experiments.common import Manifest

BASE_SEED = 20261106
LENGTHS = (250, 1000, 8000)
COVERAGE_RATES = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03)
PI11 = (0.01, 0.05, 0.1, 0.2)
P0 = 0.01


def kupiec_rejection_region(n: int, size: float = 0.05) -> list[int]:
    """Exception counts for which the Kupiec test rejects at ``size`` with ``n`` days."""
    return [x for x in range(n + 1) if kupiec_pof(np.arange(n) < x, 1 - P0).reject(size)]


def _rates(exc: np.ndarray) -> dict[str, float]:
    kup, ind, cc = [], [], []
    for row in exc:
        kup.append(kupiec_pof(row, 1 - P0).reject())
        c = christoffersen(row, 1 - P0)
        ind.append(c["ind"].reject())
        cc.append(c["cc"].reject())
    out = {}
    for name, v in (("kupiec", kup), ("ind", ind), ("cc", cc)):
        m = float(np.mean(v))
        out[f"{name}_rej"] = m
        out[f"{name}_rej_se"] = float(np.sqrt(m * (1 - m) / len(v)))
    return out


def markov_exceptions(rng: np.random.Generator, reps: int, n: int, pi11: float) -> np.ndarray:
    """Two-state Markov chain with stationary exception rate P0 and P(1|1) = pi11."""
    pi01 = P0 * (1.0 - pi11) / (1.0 - P0)
    exc = np.empty((reps, n), dtype=bool)
    exc[:, 0] = rng.random(reps) < P0
    for t in range(1, n):
        u = rng.random(reps)
        exc[:, t] = np.where(exc[:, t - 1], u < pi11, u < pi01)
    return exc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    man = Manifest("mre.experiments.size_power",
                   f"coverage: default_rng([{BASE_SEED}, 0, i_n, i_p]); "
                   f"clustering: default_rng([{BASE_SEED}, 1, i_n, i_pi11])")
    cov_rows, clu_rows = [], []
    for i_n, n in enumerate(LENGTHS):
        region = kupiec_rejection_region(n)
        for i_p, p in enumerate(COVERAGE_RATES):
            rng = np.random.default_rng([BASE_SEED, 0, i_n, i_p])
            exc = rng.random((args.reps, n)) < p
            cov_rows.append({"n": n, "true_rate": p, **_rates(exc),
                             "kupiec_exact": float(stats.binom.pmf(region, n, p).sum())})
        for i_q, q in enumerate(PI11):
            rng = np.random.default_rng([BASE_SEED, 1, i_n, i_q])
            exc = markov_exceptions(rng, args.reps, n, q)
            clu_rows.append({"n": n, "pi11": q, "mean_rate": float(exc.mean()), **_rates(exc)})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov_rows).to_csv(args.out / "size_power_coverage.csv", index=False,
                                  float_format="%.6g")
    pd.DataFrame(clu_rows).to_csv(args.out / "size_power_clustering.csv", index=False,
                                  float_format="%.6g")
    man.write(args.out, "size_power", ["size_power_coverage.csv", "size_power_clustering.csv"],
              notes=f"{args.reps} replications per cell; nominal VaR level 99%; tests at 5%")


if __name__ == "__main__":
    main()
