#!/usr/bin/env python3
"""Check that the numbers quoted in README.md are the numbers in the committed results files.

Each entry of CHECKS names a results file, a text that must occur in that file (a table row or a
value) and a text that must occur in README.md. When a result is regenerated and its number
changes, the first check fails; when the README is edited by hand and no longer quotes the number,
the second fails. It also fails when a relative link or image in README.md points to a file that
does not exist. (Modelled on scripts/check_readme_numbers.py of limit-order-book-market-maker.)

BOUNDS covers statements of the form "no value is larger than x": a column of a results CSV
(optionally for some rows only) must have a maximum not above the bound quoted in the README.

Usage: python3 scripts/check_readme_numbers.py      (exit status 1 on any problem)
Standard library only.
"""
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = "results/validation/tables.md"
RD = "results/realdata/tables.md"

CHECKS = [
    # Simulated data, GARCH set `typical` (Table A-typical) and the Z2 size/power study (Table E).
    (VAL, "| FHS1000 | 1.2 (0.4) | 8.0 (0.2) |", "VaR RMSE 8.0%"),
    (VAL, "| EWMA-N | -10.3 (0.2) | 12.9 (0.1) |", "understates VaR by 10.3%"),
    (VAL, "| 1.68 (0.03) | 53 (4) |", "Kupiec rejects it in 53% of"),
    (VAL, "| 250 | correct | 4.5 (1.5) |", "(4.5% and 6.0%)"),
    (VAL, "| 1000 | correct | 6.0 (1.7) |", "(4.5% and 6.0%)"),
    (VAL, "| 1000 | normal | 20.5 (2.9) |", "20.5-70%"),
    (VAL, "| 1000 | scale0.9 | 70.0 (3.2) |", "20.5-70%"),
    (VAL, "| 8000 | 1.25 | 55.7 (0.7) | 55.4 |", "1.25% in 55.7% of samples"),
    (VAL, "| 8000 | 1.50 | 98.4 (0.2) | 97.9 |", "a true 1.5% in 98.4%"),
    # Real data, Table R1.
    (RD, "| EQ20 | FHS | 92 (73) | 1.26 [1.03, 1.54] |", "92 exceptions, 1.26% [1.03, 1.54]"),
    (RD, "| SPX | FHS | 90 (73) | 1.23 [1.00, 1.51] |", "90, 1.23%\n  [1.00, 1.51]"),
    # Raw Kupiec p of FHS on EQ20 is below 0.05: "not rejected" holds only after Holm.
    (RD, "| EQ20 | FHS | 92 (73) | 1.26 [1.03, 1.54] | 0.032 (0.158) |",
     "not rejected by Kupiec on either\n  after Holm"),
    (RD, "| EQ20 | EWMA-N | 156 (73) | 2.14 [", "EWMA-N has 2.14%"),
    (RD, "| SPX | EWMA-N | 156 (73) | 2.14 [", "EWMA-N has 2.14%"),
    (RD, "| STRANGLE | EWMA-N | 374 (73) | 5.12 [", "(5.12% and 4.37%)"),
    (RD, "| STRANGLE | EWMA-T | 319 (73) | 4.37 [", "(5.12% and 4.37%)"),
    (RD, "| STRANGLE | FHS | 50 (73) | 0.69 [", "too few exceptions (0.69%)"),
    (RD, "| STRANGLE | HS1000 | 74 (73) | 1.01 [0.81, 1.27] | 0.906 (0.906) | 0.007 (0.103) | "
         "0.027 (0.082) |", "HS1000 is the only method not\n  rejected by any VaR test after Holm"
                            " (its raw independence p-value is 0.007)"),
    # Real data, Table R2 (Z2 not rejected) and R3 (2008 window).
    (RD, "| EQ20 | FHS | 7300 | -0.078 | 0.099 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| SPX | FHS | 7300 | -0.107 | 0.051 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| STRANGLE | HS1000 | 7148 | -0.112 | 0.073 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| EQ20 | HS1000 | 17 / 58 |", "HS1000 had 17 exceptions in 58 days"),
    (RD, "| SPX | HS1000 | 17 / 58 |", "HS1000 had 17 exceptions in 58 days"),
    (RD, "| EQ20 | FHS | 2 / 58 |", "SPX, FHS 2."),
    (RD, "| SPX | FHS | 2 / 58 |", "SPX, FHS 2."),
    # Real data, Table R2b (which days make Z2 and the DM statistic; STRANGLE).
    (RD, "| STRANGLE | FHS | -36.370 | 2007-02-27 (98%) |", "98% of FHS's Z2 of"),
    (RD, "| STRANGLE | FHS | -36.370 | 2007-02-27 (98%) | 9.3e-06 | 0.424 (1.000) |",
     "(without it Z2 = 0.42, p = 1.000)"),
    (RD, "| STRANGLE | EWMA-N | -89.382 | 2007-02-27 (58%) |", "58% of EWMA-N's and EWMA-T's"),
    (RD, "| STRANGLE | EWMA-T | -78.421 | 2007-02-27 (58%) |", "58% of EWMA-N's and EWMA-T's"),
    (RD, "| STRANGLE | HS250 | -0.664 | 2007-02-27 (29%) |", "29-38% of HS250's"),
    (RD, "| STRANGLE | HS500 | -0.734 | 2007-02-27 (38%) |", "29-38% of HS250's"),
    (RD, "| STRANGLE | MC | -1.884 | 2007-02-27 (30%) |", "29-38% of HS250's"),
    (RD, "| STRANGLE | HS1000 | -0.112 | 2007-02-27 (7%) |", "and 7% of HS1000's"),
    (RD, "| -0.024 (0.273) |", "(raw p = 0.273)"),
    (RD, "| -0.077 (0.124) | 0.032 (0.540) |", "(raw p = 0.540)"),
    (RD, "| -1.032 (0.002) | -0.833 (0.004) |", "(raw p <= 0.004)"),
    (RD, "| STRANGLE | HS250 | -0.664 | 2007-02-27 (29%) | 0.0011 |",
     "predicted ES was below 0.6% of its median"),
    (RD, "| STRANGLE | HS1000 | -0.112 | 2007-02-27 (7%) | 0.0057 |",
     "predicted ES was below 0.6% of its median"),
    (RD, "| -1.43 | 2007-02-27 (59%) | -1.12 (0.262) |", "58-59% of"),
    (RD, "| -1.43 | 2014-07-31 (58%) | -1.19 (0.234) |", "2014-07-31 for FHS"),
    (RD, "| STRANGLE | EWMA-T | -78.421 | 2007-02-27 (58%) | 3.3e-05 | -32.459 (2.0e-04) | "
         "-3.903 (2.0e-04) | -1.46 |", "(-1.43 to -1.46)"),
    (RD, "| -3.69 (2.3e-04) |", "(statistics -3.7 to -8.8)"),
    (RD, "| -8.84 (9.9e-19) |", "(statistics -3.7 to -8.8)"),
    (RD, "| STRANGLE | FHS | -36.370 |", "FHS has Z2 = -36.4 and a shortfall ratio of 0.06"),
    (RD, "| STRANGLE | FHS | 7148 | -36.370 | **2.0e-04 (0.004)** | 0.06 (141) |",
     "FHS has Z2 = -36.4 and a shortfall ratio of 0.06"),
    # Real data, checks and FHS diagnostics (Tables R5, R6, R7).
    (RD, "| IV rule: max abs error of log IV(t) rebuilt from IV(t-1) and the return of t | "
         "1.49214e-13 | 6498 of 7299 evaluation days with IV(t-1) above the floor |",
     "maximum error 1.5e-13 on 6,498 of"),
    (RD, "| IV rule: correlation of the log IV move with the log return | -0.162507 |",
     "correlation with the return -0.16, with its absolute value 0.68"),
    (RD, "| IV rule: correlation of the log IV move with the absolute log return | 0.682755 |",
     "correlation with the return -0.16, with its absolute value 0.68"),
    (RD, "| stocks: largest share of exactly-zero daily returns in one stock, 1990-01-02 to "
         "1993-12-31 | 0.69664 | RRC;", "69.7% exactly-zero daily"),
    (RD, "| stocks: largest absolute daily simple return in one stock | 0.666767 | RRC on "
         "1990-04-10 |", "a -66.7% day on 1990-04-10"),
    (RD, "| STRANGLE | HS1000 | 7299 | 128 | 37 |", "128-200 days for five of the seven"),
    (RD, "| STRANGLE | MC | 7299 | 200 | 20 |", "128-200 days for five of the seven"),
    (RD, "| IV rule vs VIX/100 correlation | 0.822538 |", "correlation 0.82 to VIX/100"),
    (RD, "| STRANGLE | logIV:SP500 | 365 | 31 | 0.42 | 6.85 | 1.13e+05 |",
     "31 of 365 GARCH fits did not converge"),
    (RD, "| STRANGLE | logIV:SP500 | 365 | 31 | 0.42 | 6.85 | 1.13e+05 |", "reach 1.13e+05"),
    # Reproduction time.
    ("results/MANIFEST.txt", "total runtime: 1357 s wall, 1008 s CPU",
     "1357 s wall time and 1008 s"),
    # Mutation check.
    ("results/mutation_check.txt", "38 of 38 mutants caught", "38 deliberate bugs"),
]


# (CSV file, column, {column: allowed values} selecting the rows, bound, README text)
BOUNDS = [
    ("results/realdata/summary.csv", "z2_top_share", {"portfolio": {"EQ20", "SPX"}}, 0.015,
     "no single day contributes more than 1.5% of any method's Z2"),
]


def bound_problems(readme: str) -> list[str]:
    """Problems with the BOUNDS entries: a maximum above its bound, or a missing README text."""
    out = []
    for path, column, select, bound, text in BOUNDS:
        try:
            with open(ROOT / path, encoding="utf-8", newline="") as f:
                rows = [r for r in csv.DictReader(f)
                        if all(r[k] in v for k, v in select.items())]
        except FileNotFoundError:
            out.append(f"{path}: file not found")
            continue
        values = [float(r[column]) for r in rows if r[column] != ""]
        if not values:
            out.append(f"{path}: no values of {column} for {select}")
        elif max(values) > bound:
            out.append(f"{path}: max of {column} for {select} is {max(values):.4g}, "
                       f"above the {bound} quoted in the README")
        if text not in readme:
            out.append(f"README.md does not contain {text!r} (checked against {path})")
    return out


def broken_links(readme: str) -> list[str]:
    """Relative links and images in the README whose target file or directory does not exist."""
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#", 1)[0]).exists():
            missing.append(target)
    return missing


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    texts: dict[str, str] = {}
    problems = 0
    for target in broken_links(readme):
        print(f"README.md links to {target!r}, which does not exist")
        problems += 1
    for check in CHECKS:
        path, in_file, in_readme = check[:3]
        expected = check[3] if len(check) > 3 else None
        if path not in texts:
            try:
                texts[path] = (ROOT / path).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"{path}: file not found")
                problems += 1
                continue
        found = texts[path].count(in_file)
        if found == 0 or (expected is not None and found != expected):
            print(f"{path}: {in_file!r} occurs {found} times (expected "
                  f"{'at least 1' if expected is None else expected})")
            problems += 1
        if in_readme not in readme:
            print(f"README.md does not contain {in_readme!r} (checked against {path})")
            problems += 1
    for message in bound_problems(readme):
        print(message)
        problems += 1
    print(f"{len(CHECKS) + len(BOUNDS)} checks, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
