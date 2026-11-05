#!/usr/bin/env python3
"""Mutation check: does the test suite notice deliberate bugs?

Copies src/, tests/ and pyproject.toml into a temporary directory, then applies ONE deliberate
bug ("mutant") at a time to the copy and runs pytest there (tests/conftest.py puts the copy's
src/ first on sys.path). A mutant is "caught" if at least one test fails. The working tree is
never modified. The mutants are bugs that would silently produce plausible numbers: wrong degrees
of freedom, off-by-one, sign and unit errors, look-ahead, wrong tail or level. Exits with status 1
if a mutant survives or the unmutated baseline fails.

Usage: python3 tests/mutation_check.py [--out results/mutation_check.txt]
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPY = ["src", "tests", "pyproject.toml"]
M = "src/mre/"

# (name, [(file, text that must occur exactly once, replacement), ...])
MUTANTS = [
    ("M01 Kupiec p-value from chi2(2) instead of chi2(1)", [
        (M + "backtest.py", "float(chdtrc(1, lr))", "float(chdtrc(2, lr))")]),
    ("M02 Christoffersen conditional coverage with 1 degree of freedom", [
        (M + "backtest.py", "float(chdtrc(2, lr_cc))", "float(chdtrc(1, lr_cc))")]),
    ("M03 Christoffersen transition counts n01 and n10 swapped", [
        (M + "backtest.py", "n01 = int(np.sum((prev == 0) & (cur == 1)))",
         "n01 = int(np.sum((prev == 1) & (cur == 0)))"),
        (M + "backtest.py", "n10 = int(np.sum((prev == 1) & (cur == 0)))",
         "n10 = int(np.sum((prev == 0) & (cur == 1)))")]),
    ("M04 traffic light: green zone up to 5 exceptions", [
        (M + "backtest.py", "if k <= 4:", "if k <= 5:")]),
    ("M05 Basel plus factor for 7 exceptions 0.60 instead of 0.65", [
        (M + "backtest.py", "7: 0.65", "7: 0.60")]),
    ("M06 EWMA weights with lambda and (1 - lambda) exchanged", [
        (M + "models.py", "w = (1.0 - lam) * lam**i / (1.0 - lam**n)",
         "w = lam * (1.0 - lam)**i / (1.0 - (1.0 - lam)**n)")]),
    ("M07 GARCH variance recursion uses eps_t instead of eps_{t-1}", [
        (M + "garch.py", "x = params.omega + params.alpha * np.concatenate(([b], eps**2))",
         "x = params.omega + params.alpha * np.concatenate((eps**2, [b]))")]),
    ("M08 GARCH-t likelihood scaled with nu instead of nu - 2", [
        (M + "garch.py", "np.log1p(eps**2 / (s2 * (nu - 2.0)))", "np.log1p(eps**2 / (s2 * nu))")]),
    ("M09 ES summed over the upper (profit) tail", [
        (M + "measures.py", "tail = x[:m].sum() + (ke - m) * (x[m] if m < n else 0.0)",
         "tail = -x[n - m:].sum() + (ke - m) * (x[m] if m < n else 0.0)")]),
    ("M10 forecast ES computed at the VaR level (99%) instead of 97.5%", [
        (M + "models.py", "es=pred.es(level_es), diagnostics=diag",
         "es=pred.es(level_var), diagnostics=diag")]),
    ("M11 empirical VaR one order statistic too far in the tail", [
        (M + "measures.py", "var = -x[int(np.ceil(kv)) - 1]", "var = -x[int(np.ceil(kv)) - 2]")]),
    ("M12 Student t quantile not rescaled to unit variance", [
        (M + "measures.py", "return stdtrit(nu, p) * np.sqrt((nu - 2.0) / nu)",
         "return stdtrit(nu, p)")]),
    ("M13 put price with the sign of the payoff reversed", [
        (M + "pricing.py", "put = K * dr * ndtr(-d2) - S * dq * ndtr(-d1)",
         "put = S * dq * ndtr(-d1) - K * dr * ndtr(-d2)")]),
    ("M14 time to maturity not reduced by one day in full revaluation", [
        (M + "portfolio.py", "v1 = bsm_price(S1, p.strike, p.expiry - horizon, r, q, sig1",
         "v1 = bsm_price(S1, p.strike, p.expiry, r, q, sig1")]),
    ("M15 trading days per year 365 instead of 252", [
        (M + "portfolio.py", "TRADING_DAYS = 252", "TRADING_DAYS = 365")]),
    ("M16 theta term added with the wrong sign in the Greek approximation", [
        (M + "portfolio.py", "approx = approx + g.theta * horizon",
         "approx = approx - g.theta * horizon")]),
    ("M17 look-ahead: the history for day t includes day t", [
        (M + "models.py", "history = x[:t]\n", "history = x[:t + 1]\n")]),
    ("M18 historical simulation window one day too long", [
        (M + "models.py", "scen = history[-self.window:]\n        if",
         "scen = history[-self.window - 1:]\n        if")]),
    ("M19 Monte Carlo antithetic half reused instead of negated", [
        (M + "models.py", "return np.vstack([x, -x])", "return np.vstack([x, x])")]),
    ("M20 Monte Carlo Student t draws not rescaled to the EWMA covariance", [
        (M + "models.py", "x = x * np.sqrt((self.nu - 2.0) / w)[:, None]",
         "x = x * np.sqrt(self.nu / w)[:, None]")]),
    ("M21 FHS scaled by the last in-sample volatility instead of the forecast", [
        (M + "models.py", "sig_next[j] = np.sqrt(s2_next)", "sig_next[j] = np.sqrt(s2[-1])")]),
    ("M22 exception flag on the profit side", [
        (M + "models.py", '"exception": self.realized < -np.array([x.var for x in f]),',
         '"exception": self.realized > np.array([x.var for x in f]),')]),
    ("M23 Acerbi-Szekely Z2 normalised by T instead of T p", [
        (M + "backtest.py", "z2_obs = ratio_obs / (T * p) + 1.0", "z2_obs = ratio_obs / T + 1.0"),
        (M + "backtest.py", "z2_sim = tot / (T * p) + 1.0", "z2_sim = tot / T + 1.0")]),
    ("M24 Diebold-Mariano long-run variance without Bartlett weights", [
        (M + "backtest.py", "s += 2.0 * (1.0 - j / (L + 1.0)) * float(u[j:] @ u[:-j]) / n",
         "s += 2.0 * float(u[j:] @ u[:-j]) / n")]),
    ("M25 implied-vol proxy uses a centred (look-ahead) window", [
        (M + "data.py", "df.rolling(window).std(ddof=1)",
         "df.rolling(window, center=True).std(ddof=1)")]),
    ("M26 historical stress window starts after the first day's move", [
        (M + "stress.py", "base, last = before[-1], inside[-1]",
         "base, last = inside[0], inside[-1]")]),
    ("M27 Gaussian Euler contributions ignore correlations", [
        (M + "attribution.py", "marg = e * (S @ e) / sd", "marg = e * np.diag(S) * e / sd")]),
]


def run_pytest(work: Path, timeout: float):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "--tb=no",
                            "-p", "no:cacheprovider", "tests"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, ["(timeout)"], time.monotonic() - t0, "timeout"
    failed = [ln.split(" - ")[0].replace("FAILED ", "") for ln in p.stdout.splitlines()
              if ln.startswith("FAILED") or ln.startswith("ERROR")]
    summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
    return p.returncode, failed, time.monotonic() - t0, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Mutation check for the mre test suite.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per pytest run")
    args = ap.parse_args()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                              text=True).stdout.strip()

    import numpy

    dirty = git("status", "--porcelain", "--", "src", "tests")
    log("# Generated by tests/mutation_check.py; do not edit by hand.")
    log("# command: python3 tests/mutation_check.py " + " ".join(
        shlex.quote(a) for a in sys.argv[1:]))
    log(f"# git:     {git('rev-parse', 'HEAD') or 'unknown'} "
        f"({'uncommitted changes in src/ or tests/' if dirty else 'clean'})")
    log(f"# date:    {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}")
    log(f"# python:  {platform.python_version()}; numpy {numpy.__version__}")
    log("# For each mutant: the pytest summary and up to three of the failing tests.")
    log("")
    survived = 0
    with tempfile.TemporaryDirectory(prefix="mre_mutation_") as tmp:
        work = Path(tmp)
        for item in COPY:
            src = REPO / item
            if src.is_dir():
                shutil.copytree(src, work / item, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))
            else:
                shutil.copy2(src, work / item)
        rc, _, secs, summary = run_pytest(work, args.timeout)
        log(f"baseline (no mutant): {summary} [{secs:.0f} s]")
        if rc != 0:
            log("baseline must pass; aborting")
            return 1
        log("")
        originals: dict[str, str] = {}
        for name, edits in MUTANTS:
            touched: dict[str, str] = {}
            try:
                for rel, old, new in edits:
                    if rel not in originals:
                        originals[rel] = (work / rel).read_text()
                    text = touched.get(rel, originals[rel])
                    if text.count(old) != 1:
                        log(f"{name}: SETUP ERROR, pattern found {text.count(old)} times in {rel}")
                        return 1
                    touched[rel] = text.replace(old, new)
                for rel, text in touched.items():
                    (work / rel).write_text(text)
                rc, failed, secs, summary = run_pytest(work, args.timeout)
                caught = rc != 0
                survived += not caught
                log(f"{name}: {'caught' if caught else 'SURVIVED'} - {summary} [{secs:.0f} s]")
                for f in failed[:3]:
                    log(f"    {f}")
            finally:
                for rel in touched:
                    (work / rel).write_text(originals[rel])
    log("")
    log(f"{len(MUTANTS) - survived} of {len(MUTANTS)} mutants caught.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
