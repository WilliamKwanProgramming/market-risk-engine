"""Experiment (d): full revaluation vs delta-gamma(-vega) approximations in the 99% tail.

Protocol: docs/validation.md, sections (d) and (d') (DGVX adds vanna and volga). Usage::

    python -m mre.experiments.option_tail --batches 20 --batch-size 50000 --out results/validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mre.experiments.common import Manifest, mean_se
from mre.measures import empirical_var_es
from mre.portfolio import TRADING_DAYS, EuropeanOption, MarketState, Portfolio, Stock
from mre.pricing import bsm_greeks

SEED = 20261108
KAPPA, THETA, XI, RHO = 5.0, 0.20, 1.0, -0.7
SIGMA0 = (0.12, 0.20, 0.40)
DISTS = ("normal", "t4")
S0, RATE, EXPIRY = 100.0, 0.02, 21 / TRADING_DAYS
H = 1.0 / TRADING_DAYS


def portfolios(sigma: float) -> dict[str, Portfolio]:
    """The six pre-specified portfolios on the index ``X`` (delta hedge at the given state)."""
    def opt(k, call, qty):
        return EuropeanOption("X", k, EXPIRY, call, qty)

    straddle = [opt(100, True, -1), opt(100, False, -1)]
    delta = sum(float(bsm_greeks(S0, p.strike, EXPIRY, RATE, 0.0, sigma, p.is_call).delta)
                * p.quantity for p in straddle)
    return {
        "long ATM call": Portfolio([opt(100, True, 1)]),
        "short ATM put": Portfolio([opt(100, False, -1)]),
        "short ATM straddle": Portfolio(straddle),
        "delta-hedged short straddle": Portfolio([*straddle, Stock("X", -delta)]),
        "long 90 put, short 110 call": Portfolio([opt(90, False, 1), opt(110, True, -1)]),
        "short 90 put": Portfolio([opt(90, False, -1)]),
    }


def draw_moves(rng: np.random.Generator, n: int, sigma: float, dist: str) -> np.ndarray:
    """One-day (log index, log implied vol) moves of the stochastic-volatility model."""
    if dist == "normal":
        e = rng.standard_normal((n, 2))
    else:
        e = rng.standard_t(4.0, (n, 2)) * np.sqrt(0.5)  # unit variance
    x = -0.5 * sigma**2 * H + sigma * np.sqrt(H) * e[:, 0]
    y = (KAPPA * (np.log(THETA) - np.log(sigma)) * H
         + XI * np.sqrt(H) * (RHO * e[:, 0] + np.sqrt(1 - RHO**2) * e[:, 1]))
    return np.column_stack([x, y])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--out", type=Path, default=Path("results/validation"))
    args = ap.parse_args(argv)
    man = Manifest("mre.experiments.option_tail", f"default_rng([{SEED}, batch]); the same "
                   "draws are rescaled for every state and portfolio")
    rows = []
    for b in range(args.batches):
        for dist in DISTS:
            rng = np.random.default_rng([SEED, b, DISTS.index(dist)])
            state_draws = rng.bit_generator.state
            for sigma in SIGMA0:
                rng.bit_generator.state = state_draws  # common random numbers across states
                moves = draw_moves(rng, args.batch_size, sigma, dist)
                state = MarketState({"X": S0}, {"X": sigma}, rate=RATE)
                for name, port in portfolios(sigma).items():
                    full = port.pnl_full(state, moves, H).sum(axis=1)
                    var_f, es_f = empirical_var_es(full, 0.99, 0.975)
                    tail = full <= -empirical_var_es(full, 0.975, 0.975)[0]
                    for method, vega, second in (("DG", False, False), ("DGV", True, False),
                                                 ("DGVX", True, True)):
                        approx = port.pnl_greeks(state, moves, H, vega=vega,
                                                 second_order_vol=second).sum(axis=1)
                        var_a, es_a = empirical_var_es(approx, 0.99, 0.975)
                        rows.append({
                            "dist": dist, "sigma0": sigma, "portfolio": name, "method": method,
                            "batch": b, "var_full": var_f, "es_full": es_f,
                            "var_rel_err": var_a / var_f - 1.0, "es_rel_err": es_a / es_f - 1.0,
                            "tail_mae_rel": float(np.mean(np.abs(approx - full)[tail])) / es_f,
                        })
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "option_tail.csv", index=False, float_format="%.6g")
    summ = mean_se(df, ["dist", "sigma0", "portfolio", "method"],
                   ["var_full", "es_full", "var_rel_err", "es_rel_err", "tail_mae_rel"])
    summ.to_csv(args.out / "option_tail_summary.csv", index=False, float_format="%.6g")
    man.write(args.out, "option_tail", ["option_tail.csv", "option_tail_summary.csv"],
              notes=f"{args.batches} batches of {args.batch_size} scenarios")


if __name__ == "__main__":
    main()
