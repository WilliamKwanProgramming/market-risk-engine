"""Shared helpers for the experiments: run manifests and Monte Carlo summaries."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

REPO = Path(__file__).resolve().parents[3]


def git_state() -> tuple[str, str]:
    """(commit hash, 'clean' or 'dirty') of the repository, or ('unknown', 'unknown')."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src",
                                "scripts", "pyproject.toml"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return commit, "dirty" if dirty else "clean"


class Manifest:
    """Collects what is needed to reproduce one experiment run and writes it next to the CSVs.

    The command line is recorded as ``python -m <module> <arguments>`` with paths as given (the
    reproduce script passes paths relative to the repository root).
    """

    def __init__(self, module: str, seeds: str):
        self.module = module
        self.seeds = seeds
        self.args = sys.argv[1:]
        self.t0 = time.time()
        self.started = datetime.datetime.now(datetime.UTC)
        self.load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")

    def write(self, out_dir: Path, name: str, outputs: Iterable[str], notes: str = "") -> Path:
        commit, tree = git_state()
        lines = [
            f"experiment: {name}",
            "command: python -m " + " ".join([self.module, *self.args]),
            f"seeds: {self.seeds}",
            f"git commit: {commit} (working tree {tree})",
            f"python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, pandas {pd.__version__}",
            f"machine: {platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs, "
            f"load average at start {self.load:.2f}",
            f"started: {self.started:%Y-%m-%dT%H:%M:%SZ}",
            f"runtime: {time.time() - self.t0:.1f} s",
            "outputs: " + ", ".join(outputs),
        ]
        if notes:
            lines.append(f"notes: {notes}")
        path = Path(out_dir) / f"{name}.manifest.txt"
        path.write_text("\n".join(lines) + "\n")
        return path


def mean_se(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean and Monte Carlo standard error (sd / sqrt(count)) of ``cols`` within groups."""
    g = df.groupby(by, sort=False)[cols]
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    out = mean.join(se, rsuffix="_se")
    out["n_rep"] = g.size()
    return out.reset_index()
