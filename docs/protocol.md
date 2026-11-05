# Pre-specified protocol for the real-data backtests

This file was committed before any backtest was run on real data. Part 1 is the protocol as
written by the project lead on 2026-11-05, copied unchanged except for one spelling correction
(the scoring function is due to Fissler and Ziegel; the original text said "Fissler-Ziegler").
Part 2 makes every choice precise (columns, dates, seeds, simulation sizes, output files) without
changing any choice of Part 1; where Part 1 left something open, Part 2 says which reading was
taken and why. Part 3 lists deviations; it is empty unless a choice turned out to be impossible.

## Part 1. Protocol as written by the lead

Anything below may be made more precise by the implementing agent, but no choice may be changed
after real-data results have been seen. If a choice turns out to be impossible (data missing,
method not converging), say so in the protocol's "Deviations" section with the date and the
reason, and report the deviation next to every affected result.

### Data (all real; read at runtime from installed packages, nothing copied into the repository)
- skfolio `load_sp500_dataset()`: 20 US stocks, daily prices 1990-01-02 to 2022-12-28;
  `load_sp500_index()`. State in the README whether the prices are dividend-adjusted (find out
  from the data: compare a known dividend payer's total return with its price return, and say
  what was found).
- arch `sp500` (1999-2018) only as an independent cross-check of the index returns (report the
  maximum absolute daily difference). arch `vix` (2014-01-03 to 2019-01-03) only for the check of
  the implied-volatility rule below.

### Portfolios (fixed; constant weights rebalanced daily; loss = -(weighted simple return))
- EQ20: equal weight in the 20 stocks.
- SPX: the S&P 500 index.
- STRANGLE (model P&L, not traded prices): short one 1-month put struck at 95% of spot and one
  1-month call struck at 105% of spot on the index, opened at the first trading day of each month
  at the then spot, held to the next month's first trading day (a new pair is opened then), valued
  by full revaluation each day with time to maturity reduced by one day. Implied-volatility rule
  (an explicit assumption, not data): sigma_imp(t) = 1.2 * EWMA volatility of SPX at t (lambda
  0.94, annualised, 252 days) with a floor of 10%; dividend yield 2%, risk-free rate 3% constant.
  Everything about STRANGLE is labelled "model P&L under an assumed implied-volatility rule". In
  2014-01-03 to 2019-01-03 report the correlation and the mean ratio between the rule and VIX/100,
  as a sanity check only; do not recalibrate the rule from it.

### Risk measures and methods (fixed)
VaR level 99% and ES level 97.5%, one-day horizon, forecast for date t from data up to t-1 only.
Methods: HS-250, HS-500, HS-1000 (historical simulation); EWMA-N (lambda 0.94, Gaussian); EWMA-T
(lambda 0.94, Student-t quantiles with 6 degrees of freedom, scaled to unit variance); FHS
(GARCH(1,1) with Student-t innovations, parameters re-estimated every 20 trading days on the last
1000 days, residual bootstrap with 10,000 draws, fixed seed); MC (EWMA covariance, multivariate
Student-t with 6 degrees of freedom, 20,000 draws with antithetic variates, fixed seed; STRANGLE
only: full revaluation per draw). For EQ20 and SPX the univariate versions of the parametric
methods act on the portfolio return series; EQ20 also has the multivariate EWMA-covariance
version, and the two must agree to numerical precision for EWMA-N (a test, not a result).

### Evaluation sample
All methods are evaluated on the same days: from the first trading day on or after 1994-01-03 (so
that HS-1000 has 1000 observations) to 2022-12-28. The STRANGLE portfolio starts at the first
month start in that range. A method that fails to produce a forecast on a day is reported, and
that day is excluded for every method (so the samples stay common) with the count shown.

### Tests (fixed)
- Exceptions: loss_t > VaR_t. Kupiec unconditional coverage; Christoffersen independence and
  conditional coverage; all at the 5% level, on the full sample and by calendar year (one row per
  year). Basel traffic light on every rolling 250-day window; report the share of windows in
  green, yellow and red.
- ES: Acerbi-Szekely Z2 with a p-value from simulating under the method's own predictive
  distribution (state the simulation size); mean exceedance shortfall ratio (realised mean loss on
  exception days divided by mean predicted ES on those days) as a descriptive.
- Model comparison: Fissler-Ziegel (0-homogeneous) joint VaR/ES score; Diebold-Mariano versus
  EWMA-N with Newey-West variance, lag floor(4 (n/100)^(2/9)); two-sided p-values.
- Multiple testing: all (method x portfolio) p-values for each test family are listed; report both
  raw and Holm-adjusted values. Nothing is selected after the fact.
- Stress windows (fixed): 2008-09-02 to 2008-11-20; 2020-02-19 to 2020-03-23; 2022-01-03 to
  2022-10-12. Report, for each method and portfolio, the number of exceptions and the
  realised/predicted ES ratio inside each window. The windows are reported even if no method is
  rejected in them.

### What counts as a finding
A cell of the result tables is a finding if its Holm-adjusted p-value is below 0.05. Everything
else is described as "not rejected" and is not evidence that the model is correct. Report the
number of tests run. The README states which methods were rejected on which portfolios and which
were not, with the exception counts and intervals (Wilson for exception rates). No method is
promoted or demoted on tuning; parameters above are the only ones tried (the number of parameter
sets tried is 1 per method).

### Not done and not claimed
No claim about any firm's actual risk model. No intraday data. No claim that results carry to
other years, markets or portfolios. Options are model P&L.

## Part 2. Precise specification (fixed before any real-data run)

Code: `src/mre/experiments/realdata.py` (stage 1 per portfolio, stage 2 summary) and the library
functions it calls. Output directory: `results/realdata/`.

### Data
- Stocks: `skfolio.datasets.load_sp500_dataset()`, columns `AAPL AMD BAC BBY CVX GE HD JNJ JPM KO
  LLY MRK MSFT PEP PFE PG RRC UNH WMT XOM`; index: `load_sp500_index()`, column `SP500`. Both
  have 8,313 rows from 1990-01-02 to 2022-12-28 on the same dates and no missing value (loader
  smoke check). `mre.data.clean_prices(policy="drop")` is applied anyway; the number of rows it
  drops is written to `checks.csv`.
- Simple returns $R_t = P_t / P_{t-1} - 1$ for the P&L of EQ20 and SPX. Log returns
  $\ln(P_t/P_{t-1})$ for the factors of STRANGLE and for the implied-volatility rule. The first
  return is dated 1990-01-03.
- Dividend adjustment: `checks.csv` reports the skfolio MSFT simple return on 2004-11-15 (the
  ex-date of Microsoft's USD 3.00 special dividend, about 10% of the price then); an unadjusted
  price series would show a drop of that size. This replaces "compare total return with price
  return", because no unadjusted stock prices are available in the sandbox.
- Index cross-check: arch `sp500` column `Adj Close`, simple returns, on the dates common to both
  series (1999-01-05 to 2018-12-31 after the first return); `checks.csv` reports the number of
  common dates, the maximum absolute difference of daily returns and its date.
- VIX check: arch `vix` column `vix`, missing values dropped, dates 2014-01-03 to 2019-01-03
  common with the rule; Pearson correlation of $\sigma_{imp}(t)$ and VIX/100, and the mean of
  $\sigma_{imp}(t) / (\text{VIX}_t/100)$. Reported in `checks.csv` only.

### Portfolios
- **EQ20**: P&L $X_t = \frac{1}{20}\sum_i R_{i,t}$ (portfolio value 1 at every close).
- **SPX**: $X_t = R_t$ of `SP500` (value 1).
- **STRANGLE**: P&L in index points of a position of $-1$ European put and $-1$ European call on
  `SP500` (multiplier 1), priced by `mre.pricing.bsm_price` with $q = 0.02$, $r = 0.03$.
  - Opening days $d_m$: the first date in the data of each calendar month, from 1994-01-03 to
    2022-12-01 (348 pairs). Strikes $0.95\,S(d_m)$ and $1.05\,S(d_m)$ at the close of $d_m$.
  - Time to expiry at opening $N_m/252$ years, $N_m$ = number of trading days in the data after
    $d_m$ up to and including $d_{m+1}$, so each pair expires at the close of the next opening
    day, where it is worth its intrinsic value and the next pair is opened. For the last pair
    (opened 2022-12-01) there is no $d_{m+1}$ in the data and $N = 21$; the data end before it
    expires. "1-month" is read as "until the next month's first trading day" because the protocol
    holds each pair exactly that long; a fixed 21-day maturity would let some pairs expire
    before the roll.
  - Each trading day the time to expiry falls by $1/252$.
  - Implied volatility $\sigma_{imp}(t) = \max(0.10,\ 1.2\sqrt{252\, s_t^2})$ with $s_t^2$ =
    `mre.models.ewma_covariance` of the `SP500` log returns dated up to and including $t$
    ($\lambda = 0.94$, zero mean, the last 500 returns with weights renormalised to sum to 1;
    while fewer than 500 returns exist, i.e. in 1990-1991, all available returns). The value at
    $t$ uses no data after $t$.
  - Risk factors (`mre.portfolio` convention): `logS:SP500` = log return of the index and
    `logIV:SP500` = $\ln \sigma_{imp}(t) - \ln \sigma_{imp}(t-1)$.
  - Realised P&L of day $t$: the pair held at the close of $t-1$ revalued at the close of $t$
    (spot $S_t$, volatility $\sigma_{imp}(t)$, expiry shortened by $1/252$) minus its value at
    the close of $t-1$ (`Portfolio.pnl_full` with the realised factor moves and horizon 1/252).
  - First STRANGLE forecast day: 1994-01-04 (the first pair is opened at the close of
    1994-01-03).

### Methods (names used in the outputs)
All forecasts for day $t$ use factor-move rows dated before $t$ only
(`mre.models.rolling_forecasts`). For EQ20 and SPX the factor is the portfolio return $X_t$ and
the P&L of a scenario is the scenario itself; for STRANGLE the factors are the two moves above and
every scenario is fully revalued at the state of the close of $t-1$ (horizon 1/252).

| Name | Class and arguments |
|---|---|
| `HS250`, `HS500`, `HS1000` | `HistoricalSimulation(window)` |
| `EWMA-N` | `EWMAParametric(lam=0.94, dist="normal", window=500)`; zero mean |
| `EWMA-T` | `EWMAParametric(lam=0.94, dist="t", nu=6, window=500)` |
| `FHS` | `FilteredHistoricalSimulation(window=1000, refit_every=20, n_boot=10000, seed=20261111)` |
| `MC` | `MonteCarlo(lam=0.94, dist="t", nu=6, n_draws=20000, seed=20261112, window=500)` |

- EWMA weights beyond 500 days are dropped (weight $0.94^{500} < 10^{-13}$).
- For STRANGLE, EWMA-N and EWMA-T are delta-normal in the two factors: $\sigma_P^2 = e'\Sigma e$
  with the 2x2 EWMA covariance and the first-order sensitivities $e$ (delta times spot, vega
  times volatility) at the close of $t-1$. Part 1 names no other form for a parametric method on
  an option position.
- FHS: GARCH(1,1)-t (`mre.garch.fit_garch_t`) per factor on the last 1000 rows, estimated on the
  first forecast day and every 20th forecast day after it; each day the filter is run with the
  current parameters over the last 1000 rows, and 10,000 residual dates are drawn with
  replacement (rows kept together) with `numpy.random.default_rng([20261111, t])`, $t$ the row
  number of the forecast day in the factor-move matrix. Scenario $= \mu + \sigma_{t}\, z$. A fit
  whose optimiser does not report convergence is used as returned and counted.
- MC: draws from `numpy.random.default_rng([20261112, t])`; 10,000 draws and their antithetic
  partners. For EQ20 and SPX the "univariate version" is the one-factor case (Student t(6)
  scaled to the EWMA variance of $X$); for STRANGLE it is bivariate with full revaluation.
- "Parametric methods" in Part 1 is read as every method other than HS (EWMA-N, EWMA-T, FHS, MC):
  for EQ20 all of them act on the portfolio return series. HS is the same in both versions for a
  linear portfolio of simple returns.
- The multivariate EWMA-N check for EQ20 (20 stock simple returns, `LinearPnL` with weights 1/20)
  is the test `tests/test_realdata.py::test_eq20_multivariate_ewma_equals_univariate`; it is not
  a result.
- A forecast **fails** if the model raises an exception or returns a non-finite VaR or ES at
  either level. Failed days are written to `failures.csv` and removed from every method of that
  portfolio before any test.

### Evaluation sample
- EQ20 and SPX: the 7,300 trading days 1994-01-03 to 2022-12-28.
- STRANGLE: the 7,299 trading days 1994-01-04 to 2022-12-28.
- Minus failed days (reported).

### Tests and descriptives
- Exception: $X_t < -\text{VaR}_{99,t}$ (loss strictly above VaR). Exception rate with a 95%
  Wilson interval.
- Kupiec, Christoffersen independence and conditional coverage (`mre.backtest`) on the full
  sample and for each calendar year 1994-2022 (29 years; 1994 starts on the first evaluation day
  and 2022 ends on 2022-12-28).
- Basel traffic light: exception counts over every window of 250 consecutive evaluation days
  (`mre.backtest.zone_occupancy`); shares of windows green (0-4), yellow (5-9), red (10+).
- Acerbi-Szekely Z2 (and Z1, reported alongside) at 97.5% with VaR and ES at 97.5% read from each
  day's predictive distribution: `mre.backtest.AcerbiSzekelySimulation(level=0.975,
  n_sim=5000, seed=20261113)`, one-sided Monte Carlo p-value (small Z2 = losses beyond the
  model's ES). Its size and power at 5000 simulations were measured on simulated data first
  (docs/validation.md, table E).
- Exceedance shortfall ratio (descriptive): over the days with realised loss above the 97.5%
  VaR, mean realised loss divided by mean predicted 97.5% ES. Part 1 says "exception days"; the
  97.5% exceedances are used because ES at 97.5% is the expected loss beyond the 97.5% VaR, so
  the ratio is 1 for a correct model; over 99% exceptions it would exceed 1 for a correct model.
  The number of 97.5% exceedances is reported with it.
- FZ0 score (`mre.backtest.fz0_loss`, Patton, Ziegel and Chen 2019) with VaR and ES at 97.5%;
  mean score per method. Diebold-Mariano of each method against EWMA-N on the daily FZ0 scores
  (`mre.backtest.diebold_mariano`, Newey-West Bartlett lag $\lfloor 4 (n/100)^{2/9} \rfloor$),
  two-sided normal p-value; negative statistic = lower score than EWMA-N.
- Holm adjustment within each family. Full-sample families, one cell per (portfolio, method): (1)
  Kupiec, (2) Christoffersen independence, (3) Christoffersen conditional coverage, (4)
  Acerbi-Szekely Z2 (21 cells each), (5) Diebold-Mariano (18 cells). By-year families, one cell
  per (portfolio, method, year): (6) Kupiec, (7) independence, (8) conditional coverage (609
  cells each). Holm: with ordered p-values $p_{(1)} \le \dots \le p_{(m)}$, adjusted
  $\tilde p_{(k)} = \max_{j \le k} \min(1, (m - j + 1)\, p_{(j)})$. An independence test with no
  exception in the first $n-1$ days has statistic 0 and p-value 1 and stays in its family.
- Stress windows (inclusive, from `mre.stress.HISTORICAL_WINDOWS`): 2008-09-02 to 2008-11-20,
  2020-02-19 to 2020-03-23, 2022-01-03 to 2022-10-12. For each portfolio, method and window:
  number of evaluation days, 99% VaR exceptions (and the 1% expectation), 97.5% exceedances and
  the exceedance shortfall ratio inside the window (empty if no exceedance).
- Finding: Holm-adjusted p-value below 0.05. The total number of tests per family is reported.

### Descriptive outputs for the dashboard (no tests)
- STRANGLE daily P&L explain (`mre.attribution.option_pnl_explain`): delta, gamma, vega, theta
  and unexplained parts of the realised model P&L, with spot, implied volatility and days to
  expiry.
- EQ20: each stock's contribution $R_{i,t}/20$ to the realised P&L summed by calendar year; and
  Euler ES contributions by stock (`mre.attribution.euler_historical`, HS500 scenarios of the
  20 stock returns times 1/20) on the first evaluation day of each stress window.

### Output files in `results/realdata/`
| File | Content |
|---|---|
| `forecasts_<P>.csv` | per day: `date`, `realized` (not for SPX, Part 5), and for each method `<M>:var99`, `<M>:es975`, `<M>:var975` (P = EQ20, SPX, STRANGLE); STRANGLE also `iv` |
| `run_<P>.csv` | per method: forecast days, failed days, GARCH fits and non-converged fits, runtime |
| `es_test_<P>.csv` | per method: Z1, Z2, their p-values, 97.5% exceedances, days, simulations |
| `failures.csv` | portfolio, method, date and error of every failed forecast (header only if none) |
| `summary.csv` | per portfolio and method: days, exceptions, rate and Wilson interval, Kupiec/independence/CC statistics with raw and Holm p-values, zone shares, Z2 with raw and Holm p, exceedance shortfall ratio, mean FZ0, DM statistic with raw and Holm p |
| `by_year.csv` | per portfolio, method and year: days, exceptions, Kupiec/independence/CC raw and Holm p |
| `stress.csv` | per portfolio, method and window: the stress-window quantities above |
| `checks.csv` | data checks (dividend check, arch cross-check, VIX check, rows dropped, sample dates) |
| `strangle_explain.csv`, `eq20_attribution_year.csv`, `eq20_euler_stress.csv` | descriptives |
| `fhs_fits_<P>.csv` | per FHS refit day and factor: GARCH parameters and residual diagnostics (Part 4) |
| `*.manifest.txt`, `tables.md` | command, seeds, commit; tables rendered by `scripts/make_tables.py` |

## Part 3. Deviations

1. **ES-based measures on days with positive predicted ES only (2026-11-05).** The first full
   run stopped in the STRANGLE evaluation because some methods forecast a 97.5% ES that is not a
   positive loss on some days (the short strangle gains in every scenario of the predictive
   distribution, mostly through time decay close to expiry). The FZ0 score is undefined for
   ES <= 0 and the Acerbi-Szekely statistics divide by ES, so the protocol's choices cannot be
   applied on those days. The FZ0 means, the Diebold-Mariano tests, Z1/Z2 and the shortfall
   ratios (full sample and stress windows) are therefore computed on the common days on which
   **every** method's 97.5% ES is positive; the number of those days is reported
   (`es_days` in `summary.csv`, `days` in `es_test_<P>.csv`). The VaR tests (Kupiec,
   Christoffersen, traffic light, by year) use the full common sample as specified. No test
   result had been seen when this was decided (the run stopped before writing any test).

## Part 4. Additions made after the protocol commit, before any backtest result (2026-11-05)

These add outputs or fix numerical handling; no choice of Parts 1-2 is changed.

1. **Infinite volatility in a scenario.** A timing run of the STRANGLE methods on 150 days (no
   test was computed or looked at) raised overflow warnings in FHS: the GARCH(1,1)-t fitted to the
   `logIV:SP500` moves of 1990-1993 (17% of them exactly zero, because the rule sat at its 10%
   floor) has conditional volatilities down to about $5 \times 10^{-6}$, so standardised residuals
   reach about $7 \times 10^{4}$; rescaled by the next-day volatility they give log-volatility moves
   so large that $\sigma e^{y}$ overflows. `bsm_price` returned NaN for $\sigma = \infty$, and a NaN
   scenario sorted as the largest gain, i.e. it would have been dropped from the loss tail
   silently. Fixed before any backtest: `bsm_price` now returns the $\sigma \to \infty$ limits (call
   $S e^{-qT}$, put $K e^{-rT}$; tested), and `EmpiricalPnL` refuses NaN scenarios (so a NaN would
   be a failed forecast under Part 2). The FHS method itself is run exactly as specified.
2. **FHS fit diagnostics.** Because of item 1, stage 1 also writes `fhs_fits_<P>.csv`: for every
   FHS refit day and factor, the GARCH parameters, the share of exactly-zero moves in the
   1000-day window, the smallest and the next-day conditional volatility, the largest absolute
   standardised residual and the scenario move it implies. These are descriptive; they are not
   tests and do not change any forecast.

## Part 5. Correction of what the result files store, after the results (2026-11-05)

The data are read at run time from installed packages and nothing is copied into the repository
(Part 2, data). The first complete run nevertheless wrote two kinds of series into the result files:
the S&P 500 index level (`spot`, in `forecasts_STRANGLE.csv` and `strangle_explain.csv`) and the
index's daily return (`realized` in `forecasts_SPX.csv`), which is the input data itself. Neither is
stored any more. `forecasts_SPX.csv` has no `realized` column (the dashboard reads the index return
from the installed `skfolio` package, so `skfolio` is part of the `dashboard` extra), and no file
has a price or index-level column (`tests/test_results_hygiene.py`). The daily P&L of the two derived
test portfolios is still stored with the forecasts, because it is a result of this study and the
dashboard plots it: EQ20 (the mean of 20 stock returns) and STRANGLE (model P&L under the assumed
implied-volatility rule). No forecast, test or number in any table changed; the results were
regenerated by `scripts/reproduce.sh` and the tables compared (see the commit message).
