"""Experiment (c): distribution of the Basel zone under the null, simulated vs exact binomial.

Protocol: docs/validation.md, section (c). Usage::

    python -m mre.experiments.basel_zones --windows 200000 --out results/validation

Writes ``basel_zones.csv`` (simulated and exact probabilities of each count and zone) and
``basel_zones_gof.csv`` (Pearson chi-square test of the 11 count cells 0..9 and 10+, 10 degrees
of freedom, per true rate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mre.backtest import binomial_zone_probabilities, traffic_light
from mre.experiments.common import Manifest

SEED = 20261107
RATES = (0.01, 0.02)
CHUNK = 20_000


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    man = Manifest("mre.experiments.basel_zones", f"default_rng([{SEED}, i_rate])")
    rows, gof = [], []
    for i, p in enumerate(RATES):
        rng = np.random.default_rng([SEED, i])
        counts = []
        for start in range(0, args.windows, CHUNK):
            m = min(CHUNK, args.windows - start)
            counts.append((rng.random((m, 250)) < p).sum(axis=1))
        c = np.concatenate(counts)
        zone = np.array([traffic_light(k).zone for k in range(251)])[c]
        n = c.size
        for k in range(11):
            sim = float(np.mean(c >= 10)) if k == 10 else float(np.mean(c == k))
            exact = float(stats.binom.sf(9, 250, p)) if k == 10 else float(stats.binom.pmf(k,
                                                                                        250, p))
            rows.append({"true_rate": p, "item": "10+" if k == 10 else str(k),
                         "simulated": sim, "se": np.sqrt(sim * (1 - sim) / n), "exact": exact})
        observed = np.array([np.sum(c == k) for k in range(10)] + [np.sum(c >= 10)])
        expected = n * np.append(stats.binom.pmf(np.arange(10), 250, p), stats.binom.sf(9, 250, p))
        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        gof.append({"true_rate": p, "windows": n, "pearson_chi2": chi2, "df": 10,
                    "p_value": float(stats.chi2.sf(chi2, 10))})
        exact_z = binomial_zone_probabilities(250, p)
        for z in ("green", "yellow", "red"):
            sim = float(np.mean(zone == z))
            rows.append({"true_rate": p, "item": z, "simulated": sim,
                         "se": np.sqrt(sim * (1 - sim) / n), "exact": exact_z[z]})
    df = pd.DataFrame(rows)
    df["z_score"] = np.where(df["se"] > 0, (df["simulated"] - df["exact"]) / df["se"], 0.0)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "basel_zones.csv", index=False, float_format="%.6g")
    pd.DataFrame(gof).to_csv(args.out / "basel_zones_gof.csv", index=False, float_format="%.6g")
    man.write(args.out, "basel_zones", ["basel_zones.csv", "basel_zones_gof.csv"],
              notes=f"{args.windows} windows of 250 iid Bernoulli days per rate")


if __name__ == "__main__":
    main()
