#!/usr/bin/env python3
"""Check that the numbers quoted in README.md are the numbers in the committed results files.

Each entry of CHECKS names a results file, a text that must occur in that file (a table row or a
value) and a text that must occur in README.md. When a result is regenerated and its number
changes, the first check fails; when the README is edited by hand and no longer quotes the number,
the second fails. It also fails when a relative link or image in README.md points to a file that
does not exist. (Modelled on scripts/check_readme_numbers.py of limit-order-book-market-maker.)

Usage: python3 scripts/check_readme_numbers.py      (exit status 1 on any problem)
Standard library only.
"""
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
    # Real data, Table R1.
    (RD, "| EQ20 | FHS | 92 (73) | 1.26 [1.03, 1.54] |", "92 exceptions, 1.26% [1.03, 1.54]"),
    (RD, "| SPX | FHS | 90 (73) | 1.23 [1.00, 1.51] |", "90, 1.23%\n  [1.00, 1.51]"),
    (RD, "| EQ20 | EWMA-N | 156 (73) | 2.14 [", "EWMA-N has 2.14%"),
    (RD, "| SPX | EWMA-N | 156 (73) | 2.14 [", "EWMA-N has 2.14%"),
    (RD, "| STRANGLE | EWMA-N | 374 (73) | 5.12 [", "(5.12% and 4.37%)"),
    (RD, "| STRANGLE | EWMA-T | 319 (73) | 4.37 [", "(5.12% and 4.37%)"),
    (RD, "| STRANGLE | FHS | 50 (73) | 0.69 [", "too few exceptions (0.69%)"),
    (RD, "| STRANGLE | HS1000 | 74 (73) | 1.01 [0.81, 1.27] | 0.906 (0.906) | 0.007 (0.103) | "
         "0.027 (0.082) |", "HS1000 is the only method not\n  rejected by any VaR test"),
    # Real data, Table R2 (Z2 not rejected) and R3 (2008 window).
    (RD, "| EQ20 | FHS | 7300 | -0.078 | 0.099 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| SPX | FHS | 7300 | -0.107 | 0.051 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| STRANGLE | HS1000 | 7148 | -0.112 | 0.073 (0.153) |", "Z2 rejects 18 of 21 cells"),
    (RD, "| EQ20 | HS1000 | 17 / 58 |", "HS1000 had 17 exceptions in 58 days"),
    (RD, "| SPX | HS1000 | 17 / 58 |", "HS1000 had 17 exceptions in 58 days"),
    (RD, "| EQ20 | FHS | 2 / 58 |", "SPX, FHS 2."),
    (RD, "| SPX | FHS | 2 / 58 |", "SPX, FHS 2."),
    # Real data, checks and FHS diagnostics (Tables R5, R6).
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
    print(f"{len(CHECKS)} checks, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
