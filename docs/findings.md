# Findings of the real-data backtests

**Data: real daily prices** of 20 US stocks and the S&P 500 index from the `skfolio` package
(1990-01-02 to 2022-12-28), evaluated from 1994-01-03 to 2022-12-28. The STRANGLE portfolio is
**model P&L under an assumed implied-volatility rule**, not traded option prices. Everything
below is about these data and this period; nothing here says how the methods behave on other
markets, portfolios or years.

The protocol (`docs/protocol.md`) was committed before any real-data backtest was run (commit
"Fix the real-data protocol before any real-data backtest"). One deviation was needed (Part 3 of
the protocol): the ES-based measures (Z1/Z2, shortfall ratio, FZ0, Diebold-Mariano) are computed
on the days on which every method forecasts a positive 97.5% ES. That leaves 7,148 of the 7,299
STRANGLE days and all EQ20 and SPX days. Every method produced a forecast on every day (0 failed
forecasts).

**Rule for reading.** A result is a *finding* if its Holm-adjusted p-value is below 0.05 (bold in
the tables). "Not rejected" is not evidence that a method is correct. Tests run: 21 cells each
for Kupiec, Christoffersen independence, Christoffersen conditional coverage (CC) and
Acerbi-Szekely Z2 on the full sample, 18 Diebold-Mariano cells, and 609 cells (21 x 29 years) for
each of the three by-year VaR tests: 1,929 tests in 8 families. Z1 is computed but not tested.
Each method had one parameter set; nothing was tuned.

## What the backtests show

### 99% VaR (Table R1)

- **EQ20 and SPX: every method has more exceptions than 1%**, from 1.23% (FHS, SPX) to 2.14%
  (EWMA-N, both portfolios). Kupiec rejects six of seven methods on each portfolio after Holm.
  The one method not rejected on either is **FHS**: 92 exceptions in 7,300 days on EQ20 (1.26%,
  Wilson 95% interval 1.03-1.54%, Holm p = 0.158) and 90 on SPX (1.23%, 1.00-1.51%, Holm
  p = 0.161). Its intervals reach 1.00% only at the edge; FHS is "not rejected", not shown to be
  correct.
- **Clustering**: the Christoffersen independence test rejects the three historical-simulation
  methods on SPX and HS500 and HS1000 on EQ20; it rejects none of the EWMA, FHS or MC methods.
  CC rejects every method except FHS on both portfolios.
- **Basel zones**: FHS has 85% of rolling 250-day windows green on EQ20 and SPX; EWMA-N has 35-38%
  green and 57-59% yellow; HS1000 has 10-14% red windows.
- **STRANGLE (model P&L)**: the delta-normal methods fail clearly, EWMA-N with 5.12% exceptions
  (374 against 73 expected) and EWMA-T with 4.37%, red in 83% and 69% of windows. MC with full
  revaluation is rejected too (2.03%). FHS is rejected in the other direction: 50 exceptions
  (0.69%, interval 0.52-0.90%), too few. The three historical-simulation methods are not
  rejected by Kupiec (1.01-1.26%), but HS250 and HS500 are rejected by the independence and CC
  tests; **HS1000 is the only method not rejected by any VaR test on STRANGLE** (74 exceptions,
  1.01%, interval 0.81-1.27%).

### 97.5% ES (Table R2)

- Z2 rejects **18 of the 21 cells** after Holm. Not rejected: FHS on EQ20 (raw p = 0.099), FHS on
  SPX (raw p = 0.051) and HS1000 on STRANGLE (raw p = 0.073); each has Holm p = 0.153. Many
  p-values sit at the Monte Carlo floor 1/5001 = 2.0e-04 of the 5,000-simulation test.
- The descriptive shortfall ratios for EQ20 and SPX lie between 0.99 and 1.13: on days with a
  loss above the 97.5% VaR the realised loss averages up to 13% more than the predicted ES;
  EWMA-N has the largest ratio (1.13 on both).
- In simulation (docs/validation.md, Table E) the same Z2 test held its size (4.5% and 6.0%) and
  had 20.5-70% power at 1,000 days against a misspecified tail; its power at the 7,300 days used
  here was not measured.

### Which days make the STRANGLE ES statistics (Table R2b, post hoc)

Z2 is a sum of one term per day with a 97.5% exceedance, each divided by that day's predicted ES;
Table R2b shows which days make it. It was added after the results, as a sensitivity: it is not
in the protocol, and its p-values are not corrected for the choice of days.

- **EQ20 and SPX**: no day contributes more than 1.5% of any method's Z2 (the five largest days
  at most 6.4%), and leaving out the largest day or the five largest changes no conclusion (the
  Z2 of FHS stays non-rejected, the others stay rejected).
- **STRANGLE**: for every method the largest term is 2007-02-27, when the held strangle had 3
  trading days to expiry, the index fell 3.47% and every method's predicted ES was between 9e-06
  and 0.006 of its median (0.00015 index points for EWMA-N and 0.097 for HS1000; the loss was
  1.44 index points).
  That day is 98% of FHS's Z2 (-36.4; without it +0.42, p = 1.000), 58% of EWMA-N's (-89.4) and
  EWMA-T's (-78.4), 38% of HS500's, 30% of MC's, 29% of HS250's and 7% of HS1000's. After leaving out the
  five most negative days the rejections of HS250 (raw p = 0.273), HS500 (0.540) and FHS (1.000)
  are gone, while those of EWMA-N, EWMA-T (Z2 about -4) and MC (-0.83, raw p = 0.004) remain. So
  the Z2 values of -36 to -89 are one or two near-expiry days in which a position with almost no
  predicted tail loss lost something; they do not measure the average ES forecast. The reading of the
  sign of Z2 ("negative = realised tail losses larger than predicted") holds when no day
  dominates, which is not the case here: FHS has Z2 = -36.4 and a shortfall ratio of 0.06.
- The earlier statement of this document that "Z2 divides by the predicted ES, and days with a
  predicted ES close to zero would dominate them; possible, not tested" is now tested: it is
  true for STRANGLE (R2b).

### Joint VaR/ES scores (Table R2)

- On EQ20 and SPX the FZ0 score of EWMA-T, FHS and MC is lower (better) than EWMA-N's, and that
  of HS500 and HS1000 higher, all with Holm p below 0.05; HS250 is not significantly different.
  FHS has the lowest mean score on both portfolios (-3.6218 and -3.6168).
- On STRANGLE every DM statistic is about -1.43 (-1.43 to -1.46; Holm p 0.871): the mean FZ0 of
  EWMA-N (89.3091 in the table, against 2.8-3.5 for HS and MC) dominates every difference, so the
  comparison is not informative for this portfolio. One day (2007-02-27; 2014-07-31 for FHS) is 58-59%
  of the sum of absolute FZ0 differences to EWMA-N (R2b); without it the statistics are -1.12
  to -1.19 (raw p 0.23-0.26), without the five largest days -3.7 to -8.8 (every method scores
  significantly lower than EWMA-N, post hoc).

### Stress windows (Table R3)

- **2008-09-02 to 2008-11-20 (58 days, 0.6 exceptions expected)**: HS1000 had 17 exceptions on
  both EQ20 and SPX, HS500 14, HS250 9; EWMA-T, FHS and MC had 2-3. The order is the same in the
  **2020 Covid window** (24 days): HS1000 10 exceptions, FHS 1-2.
- **2022 rate shock (196 days)**: HS1000 had 2-3 exceptions on EQ20 and SPX and HS250 the most
  (9-10).
- STRANGLE: FHS had 0 exceptions in the 2020 and 2022 windows and shortfall ratios of 0.09-0.71
  (realised tail losses far below its predicted ES); EWMA-N had 15 exceptions in the 2022 window.
- No test is applied inside the windows; they are 24 to 196 days long.

### Calendar years (Table R4)

After Holm over 609 cells per family, few single years are rejections on EQ20 and SPX: at most
4 years per method (HS1000, Kupiec, EQ20). The largest yearly counts are HS1000's 29 (EQ20) and
25 (SPX) exceptions in 2008. STRANGLE is different: Kupiec rejects EWMA-N in 23 of 29 years and
EWMA-T in 15. No by-year independence test survives Holm.

### Data checks (Table R5)

- The skfolio stock prices are **dividend-adjusted**: the MSFT return on 2004-11-15, the ex-date
  of its USD 3.00 special dividend (about 10% of the price), is +1.86%, not a 10% drop.
- The skfolio index returns agree with the `arch` S&P 500 series on 5,030 common days to within
  1.1e-07.
- The implied-volatility rule has a correlation of 0.82 with VIX/100 over 2014-2019 and a mean
  ratio of 0.99; it sits at its 10% floor on 11% of the evaluation days. This is a sanity check
  only; the rule was not recalibrated.

### Implied-volatility coupling and stale prices (Table R5, added after the results)

- Above the 10% floor, log IV(t) - log IV(t-1) is an exact function of IV(t-1) and the day's index
  return: the maximum error of the reconstruction is 1.5e-13 over 6,498 of the 7,299 evaluation
  days. The correlation of the move with the return is -0.16 and with its absolute value 0.68; on
  10.1% of the days the move is exactly zero (the rule at its floor).
- One stock, RRC, has 69.7% exactly-zero daily returns in 1990-1993 (stale prices) and a -66.7%
  day on 1990-04-10; both are inside the 1,000-day histories of the first EQ20 forecasts. The
  largest absolute daily return in the 20 stocks is that one.
- On 128-200 days per method (HS250, HS500, HS1000, FHS, MC) the STRANGLE 99% VaR is zero or
  negative (Table R7); EWMA-N and EWMA-T have none.

## FHS on the implied-volatility factor (Table R6)

On STRANGLE, FHS fits a GARCH(1,1)-t to the daily log changes of the assumed implied volatility.
Up to 42% of a 1,000-day window of those changes are exactly zero (the rule at its floor), 31 of
365 fits did not report convergence, and the largest standardised residual in a window reaches
1.13e+05; multiplied by the next-day volatility it gives log-volatility moves up to 1.86e+04 (the
options are then priced at their infinite-volatility limit). The STRANGLE FHS results (too few
exceptions, shortfall ratios far below 1) are consistent with such scenarios driving its
forecasts; that link is possible, not tested. For EQ20, SPX and the spot factor of STRANGLE the
largest standardised residuals are 6.9-7.4 and every fit converged.

## Mechanisms that are *not* tested here

- Why EWMA-N and EWMA-T fail on STRANGLE: a delta-normal forecast ignores the short gamma and
  vega of the position, which is the usual explanation; possible, not tested here.
- Why the delta-normal methods and FHS miss on STRANGLE beyond gamma and vega: under the
  implied-volatility rule the log volatility move is an exact function of the same day's index
  return (Table R5), so STRANGLE has one source of randomness. EWMA-N, EWMA-T and MC model two
  correlated factors and FHS two separate GARCH filters, which misspecifies the coupling by
  construction; historical simulation keeps the joint pairs and HS1000 is the one method not
  rejected on STRANGLE. That this coupling, rather than gamma and vega alone, explains the
  ranking is possible, not tested (a one-factor variant would be post hoc).
- Why historical simulation has many exceptions in 2008 and 2020: on simulated data with known
  truth (docs/validation.md, Tables F) HS errs low after volatility rises; that this is the cause
  here is possible, not tested.

## What is not shown

- That any method is correct: "not rejected" (FHS on EQ20 and SPX, HS1000 on STRANGLE) means only
  that these tests did not find a deviation in about 7,300 days.
- Anything about other periods, markets, portfolios, horizons (only one day) or confidence levels;
  any firm's risk model; intraday risk.
- Anything about real option prices: STRANGLE uses Black-Scholes-Merton values under an assumed
  implied volatility (1.2 x EWMA volatility, floor 10%), a 2% dividend yield and a 3% rate.
- Robustness to the protocol's choices (window lengths, lambda, nu = 6): one parameter set per
  method was run, by design.
- The power of Z2 at 7,300 days (Table E measures it at 250 and 1,000 days only). Kupiec's power
  at 8,000 days, close to 7,300, is 55.7% against a true exception rate of 1.25% and 98.4% against
  1.5% (docs/validation.md, Table B1), at the 5% level and before Holm: a method with a true rate
  of 1.25%, which FHS's 1.26% and 1.23% are consistent with, passes the unadjusted test in 44% of
  samples, so "FHS is not rejected" is weak evidence of correct coverage.
- That the STRANGLE ES results describe the average ES forecast: they rest on one or two days
  (Table R2b).
- Survivorship: the 20 stocks survived to 2022 and were chosen by the package authors; the direction
  of the bias in the exception counts was not tested.

## Tables

Generated by `scripts/make_tables.py` from `results/realdata/*.csv`.

<!-- tables:begin -->
**Table R1. real data: skfolio S&P 500 panel 1990-2022: 99% VaR backtests over the full evaluation sample** (EQ20 and SPX: 1994-01-03 to 2022-12-28, 7300 days; STRANGLE: 1994-01-04 to 2022-12-28, 7299 days; 0 failed forecasts). Exceptions with the 1% expectation in parentheses; exception rate (%) with the 95% Wilson interval; p-values raw (Holm-adjusted within the family of 21 cells), bold where the Holm-adjusted p-value is below 0.05 (a finding). Zones: share (%) of rolling 250-day windows green / yellow / red. STRANGLE is model P&L under an assumed implied-volatility rule.

| Portfolio | Method | Exceptions | Rate % [Wilson 95%] | Kupiec | Christoffersen ind. | Christoffersen CC | Zones G / Y / R |
|---|---|---|---|---|---|---|---|
| EQ20 | HS250 | 108 (73) | 1.48 [1.23, 1.78] | **1.2e-04 (0.001)** | 0.028 (0.365) | **5.5e-05 (5.0e-04)** | 77 / 16 / 8 |
| EQ20 | HS500 | 104 (73) | 1.42 [1.18, 1.72] | **6.1e-04 (0.005)** | **1.2e-04 (0.002)** | **1.7e-06 (2.1e-05)** | 66 / 26 / 8 |
| EQ20 | HS1000 | 113 (73) | 1.55 [1.29, 1.86] | **1.3e-05 (1.7e-04)** | **1.5e-08 (3.2e-07)** | **8.4e-12 (1.3e-10)** | 69 / 17 / 14 |
| EQ20 | EWMA-N | 156 (73) | 2.14 [1.83, 2.49] | **2.3e-17 (4.3e-16)** | 0.072 (0.722) | **4.8e-17 (9.2e-16)** | 38 / 57 / 5 |
| EQ20 | EWMA-T | 104 (73) | 1.42 [1.18, 1.72] | **6.1e-04 (0.005)** | 0.265 (1.000) | **0.002 (0.011)** | 75 / 23 / 2 |
| EQ20 | FHS | 92 (73) | 1.26 [1.03, 1.54] | 0.032 (0.158) | 0.473 (1.000) | 0.077 (0.153) | 85 / 13 / 2 |
| EQ20 | MC | 104 (73) | 1.42 [1.18, 1.72] | **6.1e-04 (0.005)** | 0.265 (1.000) | **0.002 (0.011)** | 74 / 26 / 1 |
| SPX | HS250 | 112 (73) | 1.53 [1.28, 1.84] | **2.1e-05 (2.5e-04)** | **3.5e-04 (0.006)** | **2.0e-07 (2.6e-06)** | 64 / 31 / 5 |
| SPX | HS500 | 106 (73) | 1.45 [1.20, 1.75] | **2.8e-04 (0.003)** | **2.3e-05 (4.5e-04)** | **1.7e-07 (2.4e-06)** | 72 / 20 / 9 |
| SPX | HS1000 | 115 (73) | 1.58 [1.31, 1.89] | **5.2e-06 (7.8e-05)** | **8.6e-05 (0.002)** | **1.4e-08 (2.1e-07)** | 70 / 19 / 10 |
| SPX | EWMA-N | 156 (73) | 2.14 [1.83, 2.49] | **2.3e-17 (4.3e-16)** | 0.072 (0.722) | **4.8e-17 (9.2e-16)** | 35 / 59 / 6 |
| SPX | EWMA-T | 116 (73) | 1.59 [1.33, 1.90] | **3.2e-06 (5.1e-05)** | 0.050 (0.560) | **2.8e-06 (3.1e-05)** | 61 / 35 / 3 |
| SPX | FHS | 90 (73) | 1.23 [1.00, 1.51] | 0.054 (0.161) | 0.441 (1.000) | 0.115 (0.153) | 85 / 14 / 1 |
| SPX | MC | 115 (73) | 1.58 [1.31, 1.89] | **5.2e-06 (7.8e-05)** | 0.047 (0.560) | **4.2e-06 (4.2e-05)** | 61 / 36 / 3 |
| STRANGLE | HS250 | 92 (73) | 1.26 [1.03, 1.54] | 0.032 (0.158) | **1.6e-04 (0.003)** | **7.9e-05 (6.3e-04)** | 70 / 29 / 1 |
| STRANGLE | HS500 | 81 (73) | 1.11 [0.89, 1.38] | 0.354 (0.709) | **0.002 (0.033)** | **0.006 (0.030)** | 79 / 19 / 1 |
| STRANGLE | HS1000 | 74 (73) | 1.01 [0.81, 1.27] | 0.906 (0.906) | 0.007 (0.103) | 0.027 (0.082) | 77 / 19 / 3 |
| STRANGLE | EWMA-N | 374 (73) | 5.12 [4.64, 5.65] | **1.2e-139 (2.5e-138)** | 0.596 (1.000) | **3.1e-138 (6.5e-137)** | 1 / 16 / 83 |
| STRANGLE | EWMA-T | 319 (73) | 4.37 [3.92, 4.86] | **1.8e-101 (3.5e-100)** | 0.987 (1.000) | **4.5e-100 (9.1e-99)** | 1 / 30 / 69 |
| STRANGLE | FHS | 50 (73) | 0.69 [0.52, 0.90] | **0.004 (0.025)** | 0.358 (1.000) | **0.011 (0.043)** | 100 / 0 / 0 |
| STRANGLE | MC | 148 (73) | 2.03 [1.73, 2.38] | **9.5e-15 (1.6e-13)** | 0.281 (1.000) | **5.2e-14 (8.8e-13)** | 38 / 61 / 2 |

**Table R2. real data: skfolio S&P 500 panel 1990-2022: 97.5% ES backtest and joint VaR/ES scores.** Z2 = Acerbi-Szekely Z2 (5000 simulations under each method's own predictive distributions; negative = realised tail losses larger than predicted when no single day dominates the statistic, see Table R2b), p-value raw (Holm over 21 cells). Shortfall ratio = mean realised loss over mean predicted ES on the days with loss above the 97.5% VaR (number of such days in parentheses; 1 for a correct model). FZ0 = mean FZ0 score (lower is better; scale differs by portfolio). DM = Diebold-Mariano statistic of the FZ0 difference against EWMA-N (negative = lower score than EWMA-N), Newey-West lag 10, p-value raw (Holm over 18 cells). ES days = common days on which every method's 97.5% ES is a positive loss; all columns of this table use only those days (docs/protocol.md, Part 3). STRANGLE is model P&L under an assumed implied-volatility rule; its Z2, FZ0 and DM values rest on very few days (Table R2b).

| Portfolio | Method | ES days | Z2 | Z2 p (Holm) | Shortfall ratio (n) | FZ0 | DM | DM p (Holm) |
|---|---|---|---|---|---|---|---|---|
| EQ20 | HS250 | 7300 | -0.366 | **2.0e-04 (0.004)** | 1.03 (237) | -3.4620 | 2.40 | 0.016 (0.132) |
| EQ20 | HS500 | 7300 | -0.343 | **2.0e-04 (0.004)** | 1.06 (229) | -3.3881 | 3.58 | **3.4e-04 (0.004)** |
| EQ20 | HS1000 | 7300 | -0.318 | **2.0e-04 (0.004)** | 1.08 (220) | -3.3199 | 3.81 | **1.4e-04 (0.002)** |
| EQ20 | EWMA-N | 7300 | -0.582 | **2.0e-04 (0.004)** | 1.13 (252) | -3.5512 | - | - |
| EQ20 | EWMA-T | 7300 | -0.354 | **2.0e-04 (0.004)** | 1.00 (243) | -3.5799 | -4.60 | **4.2e-06 (6.7e-05)** |
| EQ20 | FHS | 7300 | -0.078 | 0.099 (0.153) | 1.03 (191) | -3.6218 | -3.49 | **4.8e-04 (0.005)** |
| EQ20 | MC | 7300 | -0.361 | **2.0e-04 (0.004)** | 1.00 (244) | -3.5775 | -4.24 | **2.3e-05 (3.4e-04)** |
| SPX | HS250 | 7300 | -0.426 | **2.0e-04 (0.004)** | 1.03 (249) | -3.4463 | 1.83 | 0.067 (0.468) |
| SPX | HS500 | 7300 | -0.400 | **2.0e-04 (0.004)** | 1.06 (239) | -3.3828 | 2.97 | **0.003 (0.027)** |
| SPX | HS1000 | 7300 | -0.411 | **2.0e-04 (0.004)** | 1.06 (237) | -3.3034 | 3.49 | **4.8e-04 (0.005)** |
| SPX | EWMA-N | 7300 | -0.751 | **2.0e-04 (0.004)** | 1.13 (274) | -3.5188 | - | - |
| SPX | EWMA-T | 7300 | -0.483 | **2.0e-04 (0.004)** | 1.01 (260) | -3.5575 | -5.49 | **4.0e-08 (7.2e-07)** |
| SPX | FHS | 7300 | -0.107 | 0.051 (0.153) | 0.99 (199) | -3.6168 | -4.00 | **6.3e-05 (8.9e-04)** |
| SPX | MC | 7300 | -0.496 | **2.0e-04 (0.004)** | 1.01 (262) | -3.5542 | -5.10 | **3.4e-07 (5.7e-06)** |
| STRANGLE | HS250 | 7148 | -0.664 | **2.0e-04 (0.004)** | 1.07 (183) | 3.0681 | -1.43 | 0.152 (0.871) |
| STRANGLE | HS500 | 7148 | -0.734 | **2.0e-04 (0.004)** | 1.08 (175) | 3.2921 | -1.43 | 0.152 (0.871) |
| STRANGLE | HS1000 | 7148 | -0.112 | 0.073 (0.153) | 0.97 (177) | 2.8015 | -1.43 | 0.153 (0.871) |
| STRANGLE | EWMA-N | 7148 | -89.382 | **2.0e-04 (0.004)** | 1.48 (450) | 89.3091 | - | - |
| STRANGLE | EWMA-T | 7148 | -78.421 | **2.0e-04 (0.004)** | 1.32 (437) | 78.6631 | -1.46 | 0.145 (0.871) |
| STRANGLE | FHS | 7148 | -36.370 | **2.0e-04 (0.004)** | 0.06 (141) | 40.1057 | -1.43 | 0.152 (0.871) |
| STRANGLE | MC | 7148 | -1.884 | **1.0e-03 (0.004)** | 1.04 (302) | 3.5235 | -1.43 | 0.152 (0.871) |

**Table R2b. real data: skfolio S&P 500 panel 1990-2022: which days make Z2 and the Diebold-Mariano statistic (post hoc sensitivity, not part of the protocol).** Z2 = 1 + the sum of one term per day with a 97.5% exceedance (all negative); the second column gives the day with the most negative term and its share of the sum, the third that day's predicted ES divided by the median predicted ES of the method. Z2 (raw Monte Carlo p-value) is then recomputed on the same simulated paths after leaving out that day and the 5 days with the most negative terms. DM = the statistic of Table R2; the next columns give the day with the largest absolute daily FZ0 difference to EWMA-N and its share of the sum of absolute differences, and the statistic (raw p) without that day and without the 5 days with the largest absolute differences. The p-values are not corrected for the choice of days: the table shows how much a rejection rests on a few days, it is not a test. STRANGLE is model P&L under an assumed implied-volatility rule.

| Portfolio | Method | Z2 | Largest Z2 term: day (share of sum) | Its predicted ES / median ES | Z2 without it (p) | Z2 without the 5 largest (p) | DM | Largest FZ0 difference: day (share) | DM without it (p) | DM without the 5 largest (p) |
|---|---|---|---|---|---|---|---|---|---|---|
| EQ20 | HS250 | -0.366 | 1997-10-27 (1%) | 0.9 | -0.350 (2.0e-04) | -0.294 (2.0e-04) | 2.40 | 2018-10-10 (1%) | 2.68 (0.007) | 2.88 (0.004) |
| EQ20 | HS500 | -0.343 | 2008-09-29 (1%) | 1.1 | -0.327 (2.0e-04) | -0.269 (2.0e-04) | 3.58 | 2020-03-16 (1%) | 3.72 (2.0e-04) | 4.06 (4.9e-05) |
| EQ20 | HS1000 | -0.318 | 2008-09-29 (1%) | 0.89 | -0.299 (2.0e-04) | -0.234 (1.0e-03) | 3.81 | 2020-03-16 (1%) | 3.97 (7.3e-05) | 4.40 (1.1e-05) |
| EQ20 | EWMA-N | -0.582 | 2007-02-27 (1%) | 0.45 | -0.563 (2.0e-04) | -0.512 (2.0e-04) | - | - | - | - |
| EQ20 | EWMA-T | -0.354 | 2007-02-27 (1%) | 0.45 | -0.338 (2.0e-04) | -0.293 (2.0e-04) | -4.60 | 2007-02-27 (2%) | -4.49 (7.1e-06) | -4.31 (1.6e-05) |
| EQ20 | FHS | -0.078 | 2007-02-27 (1%) | 0.53 | -0.064 (0.135) | -0.023 (0.285) | -3.49 | 2007-02-27 (2%) | -3.34 (8.4e-04) | -3.16 (0.002) |
| EQ20 | MC | -0.361 | 2007-02-27 (1%) | 0.45 | -0.345 (2.0e-04) | -0.300 (2.0e-04) | -4.24 | 2007-02-27 (2%) | -4.11 (4.0e-05) | -3.84 (1.2e-04) |
| SPX | HS250 | -0.426 | 1997-10-27 (1%) | 0.81 | -0.410 (2.0e-04) | -0.357 (2.0e-04) | 1.83 | 2018-10-10 (2%) | 2.27 (0.023) | 3.00 (0.003) |
| SPX | HS500 | -0.400 | 1997-10-27 (1%) | 0.79 | -0.383 (2.0e-04) | -0.327 (2.0e-04) | 2.97 | 2018-10-10 (1%) | 3.26 (0.001) | 3.91 (9.4e-05) |
| SPX | HS1000 | -0.411 | 1997-10-27 (1%) | 0.66 | -0.392 (2.0e-04) | -0.326 (2.0e-04) | 3.49 | 2020-03-16 (1%) | 3.63 (2.9e-04) | 4.28 (1.8e-05) |
| SPX | EWMA-N | -0.751 | 2007-02-27 (1%) | 0.48 | -0.732 (2.0e-04) | -0.670 (2.0e-04) | - | - | - | - |
| SPX | EWMA-T | -0.483 | 2007-02-27 (1%) | 0.48 | -0.466 (2.0e-04) | -0.411 (2.0e-04) | -5.49 | 2007-02-27 (2%) | -5.40 (6.7e-08) | -5.03 (4.9e-07) |
| SPX | FHS | -0.107 | 2007-02-27 (1%) | 0.48 | -0.091 (0.077) | -0.046 (0.197) | -4.00 | 2018-10-10 (2%) | -3.90 (9.7e-05) | -3.50 (4.6e-04) |
| SPX | MC | -0.496 | 2007-02-27 (1%) | 0.48 | -0.479 (2.0e-04) | -0.424 (2.0e-04) | -5.10 | 2007-02-27 (2%) | -5.00 (5.9e-07) | -4.61 (4.0e-06) |
| STRANGLE | HS250 | -0.664 | 2007-02-27 (29%) | 0.0011 | -0.179 (0.009) | -0.024 (0.273) | -1.43 | 2007-02-27 (59%) | -1.12 (0.262) | -6.09 (1.1e-09) |
| STRANGLE | HS500 | -0.734 | 2007-02-27 (38%) | 0.00076 | -0.077 (0.124) | 0.032 (0.540) | -1.43 | 2007-02-27 (59%) | -1.12 (0.263) | -5.76 (8.6e-09) |
| STRANGLE | HS1000 | -0.112 | 2007-02-27 (7%) | 0.0057 | -0.029 (0.281) | 0.076 (0.726) | -1.43 | 2007-02-27 (59%) | -1.12 (0.264) | -5.28 (1.3e-07) |
| STRANGLE | EWMA-N | -89.382 | 2007-02-27 (58%) | 3.3e-05 | -37.112 (2.0e-04) | -4.638 (2.0e-04) | - | - | - | - |
| STRANGLE | EWMA-T | -78.421 | 2007-02-27 (58%) | 3.3e-05 | -32.459 (2.0e-04) | -3.903 (2.0e-04) | -1.46 | 2007-02-27 (59%) | -1.17 (0.242) | -8.84 (9.9e-19) |
| STRANGLE | FHS | -36.370 | 2007-02-27 (98%) | 9.3e-06 | 0.424 (1.000) | 0.534 (1.000) | -1.43 | 2014-07-31 (58%) | -1.19 (0.234) | -3.69 (2.3e-04) |
| STRANGLE | MC | -1.884 | 2007-02-27 (30%) | 0.001 | -1.032 (0.002) | -0.833 (0.004) | -1.43 | 2007-02-27 (59%) | -1.12 (0.263) | -7.02 (2.2e-12) |

**Table R3. real data: skfolio S&P 500 panel 1990-2022: pre-specified stress windows.** 99% VaR exceptions / evaluation days in the window (1% of the days is the expectation); shortfall ratio as in R2 over the 97.5% exceedances inside the window (their number in parentheses). Windows: 2008 crisis = 2008-09-02 to 2008-11-20; 2020 Covid crash = 2020-02-19 to 2020-03-23; 2022 rate shock = 2022-01-03 to 2022-10-12. No test is applied inside the windows. STRANGLE is model P&L under an assumed implied-volatility rule.

| Portfolio | Method | 2008 crisis: exc. / days | 2008 crisis: shortfall ratio (n) | 2020 Covid crash: exc. / days | 2020 Covid crash: shortfall ratio (n) | 2022 rate shock: exc. / days | 2022 rate shock: shortfall ratio (n) |
|---|---|---|---|---|---|---|---|
| EQ20 | HS250 | 9 / 58 | 1.12 (15) | 6 / 24 | 1.29 (11) | 9 / 196 | 1.11 (14) |
| EQ20 | HS500 | 14 / 58 | 1.19 (20) | 7 / 24 | 1.43 (12) | 5 / 196 | 0.99 (14) |
| EQ20 | HS1000 | 17 / 58 | 1.46 (21) | 10 / 24 | 1.65 (12) | 2 / 196 | 0.78 (6) |
| EQ20 | EWMA-N | 3 / 58 | 1.16 (5) | 6 / 24 | 1.24 (6) | 6 / 196 | 1.10 (9) |
| EQ20 | EWMA-T | 2 / 58 | 1.02 (5) | 4 / 24 | 1.09 (6) | 4 / 196 | 0.97 (9) |
| EQ20 | FHS | 2 / 58 | 1.16 (4) | 2 / 24 | 1.03 (3) | 5 / 196 | 1.13 (5) |
| EQ20 | MC | 2 / 58 | 1.03 (5) | 4 / 24 | 1.07 (6) | 4 / 196 | 0.97 (9) |
| SPX | HS250 | 9 / 58 | 1.07 (17) | 8 / 24 | 1.33 (11) | 10 / 196 | 1.10 (16) |
| SPX | HS500 | 14 / 58 | 1.20 (19) | 9 / 24 | 1.54 (11) | 6 / 196 | 1.01 (13) |
| SPX | HS1000 | 17 / 58 | 1.41 (21) | 10 / 24 | 1.65 (12) | 3 / 196 | 0.79 (10) |
| SPX | EWMA-N | 5 / 58 | 1.10 (7) | 6 / 24 | 1.34 (6) | 4 / 196 | 1.00 (12) |
| SPX | EWMA-T | 2 / 58 | 0.97 (7) | 6 / 24 | 1.17 (6) | 2 / 196 | 0.88 (12) |
| SPX | FHS | 2 / 58 | 1.05 (5) | 1 / 24 | 0.89 (3) | 2 / 196 | 0.95 (4) |
| SPX | MC | 3 / 58 | 0.98 (7) | 6 / 24 | 1.16 (6) | 2 / 196 | 0.88 (12) |
| STRANGLE | HS250 | 5 / 58 | 1.22 (9) | 5 / 24 | 1.44 (5) | 4 / 196 | 1.07 (8) |
| STRANGLE | HS500 | 7 / 58 | 1.22 (11) | 4 / 24 | 1.40 (6) | 3 / 196 | 0.94 (8) |
| STRANGLE | HS1000 | 9 / 58 | 1.35 (14) | 4 / 24 | 1.21 (8) | 1 / 196 | 0.61 (6) |
| STRANGLE | EWMA-N | 6 / 58 | 1.46 (7) | 5 / 24 | 1.48 (5) | 15 / 196 | 1.22 (17) |
| STRANGLE | EWMA-T | 5 / 58 | 1.28 (7) | 5 / 24 | 1.30 (5) | 11 / 196 | 1.08 (16) |
| STRANGLE | FHS | 2 / 58 | 0.21 (2) | 0 / 24 | 0.09 (1) | 0 / 196 | 0.71 (3) |
| STRANGLE | MC | 4 / 58 | 1.07 (7) | 4 / 24 | 1.13 (5) | 1 / 196 | 0.82 (9) |

**Table R4. real data: skfolio S&P 500 panel 1990-2022: calendar-year backtests (29 years, 609 cells per test family).** Number of years in which the test rejects at 5% before (raw) and after the Holm adjustment over all cells of the family; largest number of exceptions in one year and that year.

| Portfolio | Method | Kupiec raw | Kupiec Holm | CC raw | CC Holm | Ind. Holm | Max exc. in a year | Year |
|---|---|---|---|---|---|---|---|---|
| EQ20 | HS250 | 7 | 2 | 3 | 1 | 0 | 15 | 2008 |
| EQ20 | HS500 | 13 | 2 | 4 | 2 | 0 | 22 | 2008 |
| EQ20 | HS1000 | 15 | 4 | 5 | 3 | 0 | 29 | 2008 |
| EQ20 | EWMA-N | 7 | 1 | 6 | 1 | 0 | 12 | 2020 |
| EQ20 | EWMA-T | 3 | 0 | 2 | 0 | 0 | 10 | 2020 |
| EQ20 | FHS | 4 | 0 | 1 | 0 | 0 | 9 | 2007 |
| EQ20 | MC | 4 | 0 | 2 | 0 | 0 | 9 | 2020 |
| SPX | HS250 | 8 | 1 | 7 | 1 | 0 | 12 | 2008 |
| SPX | HS500 | 15 | 2 | 6 | 1 | 0 | 18 | 2008 |
| SPX | HS1000 | 16 | 3 | 6 | 3 | 0 | 25 | 2008 |
| SPX | EWMA-N | 11 | 2 | 9 | 1 | 0 | 12 | 2020 |
| SPX | EWMA-T | 6 | 1 | 3 | 0 | 0 | 11 | 2020 |
| SPX | FHS | 2 | 0 | 1 | 0 | 0 | 10 | 2007 |
| SPX | MC | 5 | 1 | 2 | 0 | 0 | 11 | 2020 |
| STRANGLE | HS250 | 7 | 1 | 4 | 1 | 0 | 12 | 1994 |
| STRANGLE | HS500 | 11 | 0 | 4 | 0 | 0 | 10 | 2008 |
| STRANGLE | HS1000 | 12 | 1 | 2 | 1 | 0 | 14 | 2008 |
| STRANGLE | EWMA-N | 28 | 23 | 28 | 18 | 0 | 21 | 1996 |
| STRANGLE | EWMA-T | 27 | 15 | 26 | 11 | 0 | 19 | 1994 |
| STRANGLE | FHS | 5 | 0 | 0 | 0 | 0 | 4 | 2000 |
| STRANGLE | MC | 6 | 1 | 5 | 0 | 0 | 11 | 1994 |

**Table R5. real data: skfolio S&P 500 panel 1990-2022: data checks.**

| Check | Value | Note |
|---|---|---|
| stocks_rows_dropped | 0 | clean_prices(policy='drop') |
| index_rows_dropped | 0 | clean_prices(policy='drop') |
| MSFT simple return 2004-11-15 | 0.0185931 | ex-date of the USD 3.00 special dividend; unadjusted prices would fall by about 10% |
| arch sp500 common return dates | 5030 | 1999-01-05 to 2018-12-31 |
| arch sp500 max abs return difference | 1.1407e-07 | on 2016-05-05 |
| IV rule vs VIX/100 common dates | 1259 | 2014-01-03 to 2019-01-03 |
| IV rule vs VIX/100 correlation | 0.822538 | Pearson, levels; sanity check only, the rule is not recalibrated |
| IV rule / (VIX/100) mean ratio | 0.986611 | mean of daily ratios |
| IV rule share of days at the 10% floor | 0.109726 | evaluation period 1994-01-03 to 2022-12-28 |
| IV rule: max abs error of log IV(t) rebuilt from IV(t-1) and the return of t | 1.49214e-13 | 6498 of 7299 evaluation days with IV(t-1) above the floor |
| IV rule: correlation of the log IV move with the log return | -0.162507 | 7299 evaluation days |
| IV rule: correlation of the log IV move with the absolute log return | 0.682755 | 7299 evaluation days |
| IV rule: share of days with a log IV move of exactly zero | 0.101247 | 7299 evaluation days |
| stocks: largest share of exactly-zero daily returns in one stock, 1990-01-02 to 1993-12-31 | 0.69664 | RRC; inside the 1000-day history of the first EQ20 forecasts |
| stocks: largest absolute daily simple return in one stock | 0.666767 | RRC on 1990-04-10 |

**Table R7. real data: skfolio S&P 500 panel 1990-2022: forecasts that are not losses.** Days on which a method's 99% VaR or 97.5% ES is zero or negative, i.e. the position is forecast not to lose in the tail (the short strangle gains from time decay close to expiry). An exception on such a day is a gain smaller than predicted; ES-based measures leave out the days on which any method's ES is not positive (protocol Part 3). Cells not listed are 0.

| Portfolio | Method | Forecast days | VaR99 <= 0 days | ES97.5 <= 0 days |
|---|---|---|---|---|
| STRANGLE | HS250 | 7299 | 163 | 117 |
| STRANGLE | HS500 | 7299 | 148 | 87 |
| STRANGLE | HS1000 | 7299 | 128 | 37 |
| STRANGLE | FHS | 7299 | 194 | 118 |
| STRANGLE | MC | 7299 | 200 | 20 |

**Table R6. real data: skfolio S&P 500 panel 1990-2022: FHS GARCH(1,1)-t fits, one per 20 trading days per factor.** Non-converged = fits whose optimiser did not report success; zero-move share = largest share of exactly-zero moves in a 1000-day window; |z| = largest absolute standardised residual in the window (median and maximum over refits); largest scenario move = maximum of next-day volatility times largest |z| (a log move of the factor).

| Portfolio | Factor | Fits | Non-converged | Zero-move share | max |z| median | max |z| max | Largest scenario move |
|---|---|---|---|---|---|---|---|
| EQ20 | EQ20 | 365 | 0 | 0.00 | 4.5 | 6.86 | 0.326 |
| SPX | SPX | 365 | 0 | 0.00 | 5.36 | 7.27 | 0.403 |
| STRANGLE | logS:SP500 | 365 | 0 | 0.00 | 5.45 | 7.4 | 0.389 |
| STRANGLE | logIV:SP500 | 365 | 31 | 0.42 | 6.85 | 1.13e+05 | 1.86e+04 |
<!-- tables:end -->
