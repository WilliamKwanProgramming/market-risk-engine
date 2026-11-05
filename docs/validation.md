# Validation on simulated data with known truth

Everything on this page uses **simulated data only**. Nothing here says how the models perform on
market data.

The protocol below (data-generating processes, parameter sets, methods, sample lengths, numbers of
replications, metrics, seeds) was written and committed before any of the experiments were run.
It is not changed after looking at results; if it ever is, the change and its reason are listed
under "Deviations from the protocol" at the end.

## Protocol

### (a) Risk models on a GARCH(1,1)-t process (`mre.experiments.garch_dgp`)

Daily returns $r_t = \sigma_t z_t$ with $z_t$ iid Student t($\nu$) scaled to unit variance and
$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$ (mean zero). The true one-day
99% VaR and 97.5% ES on day $t$ are those of the true conditional distribution
$\sigma_t z$, computed with `mre.measures.LocationScalePnL`.

| Set | $\alpha$ | $\beta$ | $\nu$ | unconditional daily volatility |
|---|---|---|---|---|
| `calm` | 0.05 | 0.90 | 30 | 0.6% |
| `typical` | 0.08 | 0.90 | 8 | 1.0% |
| `persistent_fat` | 0.10 | 0.89 | 4 | 1.2% |

$\omega$ is set from the unconditional volatility: $\omega = v^2 (1 - \alpha - \beta)$.

Each replication simulates 2000 days (after a discarded burn-in of 500): days 0-999 are history
only, days 1000-1999 are the 1000 forecast days (four 250-day Basel windows). 200 replications
per set; replication $i$ of set $j$ uses the seed `(20261105, j, i)`.

Methods (library defaults unless stated): **Oracle** (the true conditional distribution, as a
check of the checker), **HS250**, **HS500**, **HS1000**, **EWMA-N** ($\lambda = 0.94$, window
500), **EWMA-t6** (same with unit-variance t(6) quantiles), **FHS1000** (GARCH(1,1)-t on the last
1000 days, refit every 250 days), **MC-N** (EWMA covariance, normal, 5000 antithetic draws, seed
= replication number).

Metrics per replication and method, over the 1000 forecast days: relative bias
$\overline{\widehat{VaR}_t / VaR_t - 1}$ and relative RMSE $\sqrt{\overline{(\widehat{VaR}_t /
VaR_t - 1)^2}}$ of the 99% VaR, the same for the 97.5% ES; the 99% VaR exception rate; rejection
at 5% by Kupiec, Christoffersen independence and Christoffersen conditional coverage (1000 days);
the Basel zone of each of the four non-overlapping 250-day windows; the mean FZ0 loss (VaR and ES
at 97.5%) minus that of the Oracle, and whether a Diebold-Mariano test (lag rule
$\lfloor 4(n/100)^{2/9} \rfloor = 6$) of method vs Oracle rejects at 5%. For FHS, the number of
GARCH fits that did not converge. Reported: the mean over replications with its Monte Carlo
standard error (standard deviation over replications divided by $\sqrt{200}$).

### (b) Size and power of the VaR backtests (`mre.experiments.size_power`)

Exception indicators are simulated directly, for sample lengths $n \in \{250, 1000, 8000\}$ and
nominal coverage 99% ($p_0 = 0.01$), 5000 replications per cell, seed `(20261106, ...)`.

1. Coverage: iid Bernoulli($p$) exceptions with true rate
   $p \in \{0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03\}$. $p = 0.01$ gives the size, the
   others the power curve. For Kupiec, the exact rejection probability (binomial sum over the
   rejection region) is computed alongside the simulated one.
2. Clustering: a two-state Markov chain with $P(I_t = 1 \mid I_{t-1} = 1) = \pi_{11} \in
   \{0.01, 0.05, 0.1, 0.2\}$ and $\pi_{01}$ chosen so that the stationary exception rate is
   exactly 1% ($\pi_{01} = 0.01 (1 - \pi_{11}) / 0.99$); the first day is drawn from the
   stationary distribution. $\pi_{11} = 0.01$ is independence.

Reported: rejection rates at 5% of Kupiec, Christoffersen independence and conditional coverage,
with binomial standard errors.

### (c) Basel zones under the null (`mre.experiments.basel_zones`)

200,000 simulated 250-day windows of iid Bernoulli(0.01) exceptions (seed 20261107): frequency of
each exception count 0-9 and 10+, and of each zone, against the exact binomial probabilities.
The same for a true exception rate of 2% (a VaR that is too low).

### (d) Option portfolios: full revaluation vs Greek approximations (`mre.experiments.option_tail`)

One-day joint moves of the log index $x$ and the log implied volatility $y$ under a stochastic
volatility model in which the index diffuses at the current implied volatility $\sigma$
(horizon $h = 1/252$):

$$x = -\tfrac12 \sigma^2 h + \sigma \sqrt{h}\, \varepsilon_1, \qquad
y = \kappa(\ln\theta - \ln\sigma) h + \xi \sqrt{h}\,(\rho\, \varepsilon_1 +
\sqrt{1-\rho^2}\, \varepsilon_2),$$

$\kappa = 5$, $\theta = 0.20$, $\xi = 1.0$, $\rho = -0.7$, with $(\varepsilon_1, \varepsilon_2)$
either independent standard normal or independent unit-variance Student t(4). Starting states
$\sigma \in \{0.12, 0.20, 0.40\}$; index 100; rate 2%; no dividends.

Portfolios (all options European, 21 trading days to expiry, one unit each):

1. long ATM call; 2. short ATM put; 3. short ATM straddle; 4. delta-hedged short ATM straddle
(index position equal to minus the straddle delta); 5. long 90% put and short 110% call;
6. short 90% put.

For each combination: 99% VaR and 97.5% ES of the full-revaluation P&L (time to expiry reduced by
one day), of the delta-gamma-theta approximation (DG) and of the delta-gamma-vega-theta
approximation (DGV). Reported: the relative error of the approximate VaR and ES, (approx -
full)/full, and the mean absolute P&L error over the scenarios in the full-revaluation 97.5% tail
divided by the full-revaluation ES; mean and Monte Carlo standard error over 20 independent
batches of 50,000 scenarios (seed `(20261108, batch)`).

### Additions written in wave 2 (2026-11-05), before these experiments were run

Two sentences of the wave-1 reading stated a mechanism that no experiment had tested, and the
Acerbi-Szekely ES tests had been checked only in unit tests. The three experiments below were
written down and committed before they were run.

**(d') Second-order terms in volatility (part of `mre.experiments.option_tail`).** Same draws,
states and portfolios as (d). One more approximation, DGVX = DGV plus the vanna and volga terms
$\text{vanna}\, dS\, d\sigma + \tfrac12 \text{volga}\, d\sigma^2$ (Greeks at the current state;
$dS = S(e^x - 1)$, $d\sigma = \sigma(e^y - 1)$), i.e. a full second-order expansion in spot and
volatility. Reported next to DG and DGV: its relative VaR and ES errors. Reading fixed in advance:
the wave-1 explanation for the short 90% put ("convexity in spot and volatility outside a
second-order expansion in spot and first-order in volatility") is supported for a cell if DGVX
removes at least half of the DGV VaR error ($|\text{err}_{DGVX}| \le 0.5\, |\text{err}_{DGV}|$);
otherwise the remaining error is attributed to terms of order three and higher, which are not
separated further.

**(e) Size and power of the Acerbi-Szekely Z2 test (`mre.experiments.es_power`).** Returns follow
the GARCH(1,1)-t set `typical` of (a) (seed `(20261109, i_T, rep)`, burn-in 500, mean zero).
Sample lengths $T \in \{250, 1000\}$ forecast days, 200 replications each. Three forecasting
models, each given the true conditional variance $\sigma_t^2$:

1. `correct`: unit-variance Student t(8) scaled by $\sigma_t$ (the true distribution: size);
2. `normal`: normal with standard deviation $\sigma_t$ (right variance, thinner tail: ES
   understated);
3. `scale0.9`: unit-variance t(8) scaled by $0.9\,\sigma_t$ (VaR and ES understated by 10%).

Z1 and Z2 at 97.5% are computed by `mre.backtest.acerbi_szekely` with **5000 simulations**, the
number fixed for the real-data protocol, and the Monte Carlo p-value of the function (seed = the
replication number). Reported: rejection rate at 5% with its binomial standard error, and the mean
of the statistic. Reading fixed in advance: the size check passes if the rejection rate of
`correct` is within two standard errors of 5%; power is reported as measured.

**(f) Historical-simulation bias by volatility regime (`mre.experiments.hs_regime`).** The three
GARCH(1,1)-t sets of (a); 20 replications per set of 3000 days after the burn-in (seed
`(20261110, j, rep)`), days 0-999 history only, days 1000-2999 forecast days. Methods HS250,
HS500, HS1000 (`mre.models.HistoricalSimulation`). For each forecast day: relative error of the
99% VaR $\widehat{VaR}_t / VaR_t - 1$ and the exception indicator. Days are classified twice:

- by **level**: `low` if the true $\sigma_t < 0.8\,v$, `high` if $\sigma_t > 1.25\,v$, `middle`
  otherwise ($v$ = unconditional volatility of the set);
- by **change** relative to the method's own window of $w$ days: $\rho_t = \sigma_t /
  \sqrt{\tfrac1w \sum_{s=t-w}^{t-1} \sigma_s^2}$; `rising` if $\rho_t > 1.25$, `falling` if
  $\rho_t < 0.8$, `stable` otherwise.

Reported per set, method and class: the mean relative VaR error and the exception rate (mean over
replications of the per-replication class mean, Monte Carlo standard error over replications; the
number of replications with days in the class and the share of days in the class). Reading fixed in
advance: the wave-1 sentence "too high in calm periods and too low after volatility rises" is
supported for a set and method if the error is positive by more than two standard errors in
`low` and negative by more than two standard errors in `rising`.

### What is pre-specified about the reading

The checks of the checker are: the Oracle in (a) must show exception rates near 1% and Kupiec
rejection near the exact size of the test; the simulated Kupiec rejection in (b) must match the
exact binomial value within Monte Carlo error; the simulated zone frequencies in (c) must match
the binomial probabilities within Monte Carlo error. If any of these fails, the experiment
results are not reported as findings until the cause is found.

## Results

Generated by `scripts/make_tables.py` from `results/validation/*.csv`.

<!-- tables:begin -->
**Table A-calm. Simulated data: GARCH(1,1)-t set `calm`, 200 replications of 1000 forecast days.** Mean over replications (Monte Carlo standard error). Bias and RMSE are relative to the true conditional VaR/ES, in percent. Exc. = 99% VaR exception rate (%). Kupiec, Ind., CC = rejection rate at 5% (%). Y+R = share of 250-day windows in the yellow or red zone (%); Red = share in the red zone (%). FZ0 = mean FZ0 loss minus the Oracle's; DM = share of replications in which Diebold-Mariano rejects equal FZ0 loss vs the Oracle at 5% (%). FHS: 0 of 800 GARCH fits did not converge.

| Method | VaR bias | VaR RMSE | ES bias | ES RMSE | Exc. | Kupiec | Ind. | CC | Y+R | Red | FZ0 | DM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.99 (0.02) | 7 (2) | 1 (1) | 4 (1) | 11 | 0.0 (0.0) | - | - |
| HS250 | 1.5 (0.4) | 15.8 (0.3) | 0.3 (0.3) | 14.5 (0.2) | 1.25 (0.02) | 7 (2) | 6 (2) | 6 (2) | 22 | 0.0 (0.0) | 0.045 (0.002) | 34 (3) |
| HS500 | 4.2 (0.4) | 14.6 (0.3) | 1.7 (0.4) | 13.1 (0.2) | 1.03 (0.02) | 1 (1) | 8 (2) | 8 (2) | 15 | 0.1 (0.1) | 0.035 (0.002) | 24 (3) |
| HS1000 | 3.2 (0.4) | 12.8 (0.2) | 2.3 (0.4) | 12.2 (0.2) | 1.03 (0.03) | 9 (2) | 8 (2) | 9 (2) | 16 | 0.8 (0.3) | 0.031 (0.002) | 22 (3) |
| EWMA-N | -3.8 (0.2) | 9.1 (0.1) | -4.2 (0.2) | 9.3 (0.1) | 1.37 (0.02) | 20 (3) | 2 (1) | 8 (2) | 24 | 0.0 (0.0) | 0.026 (0.001) | 18 (3) |
| EWMA-t6 | 6.1 (0.2) | 11.0 (0.1) | 9.0 (0.2) | 13.0 (0.1) | 0.80 (0.02) | 6 (2) | 2 (1) | 3 (1) | 3 | 0.0 (0.0) | 0.023 (0.001) | 31 (3) |
| FHS1000 | 0.1 (0.3) | 6.2 (0.1) | -0.9 (0.3) | 5.7 (0.1) | 1.04 (0.02) | 5 (2) | 4 (1) | 4 (1) | 14 | 0.1 (0.1) | 0.008 (0.001) | 11 (2) |
| MC-N | -3.6 (0.2) | 9.4 (0.1) | -4.2 (0.2) | 9.5 (0.1) | 1.38 (0.02) | 20 (3) | 1 (1) | 10 (2) | 24 | 0.0 (0.0) | 0.027 (0.001) | 17 (3) |

**Table A-typical. Simulated data: GARCH(1,1)-t set `typical`, 200 replications of 1000 forecast days.** Mean over replications (Monte Carlo standard error). Bias and RMSE are relative to the true conditional VaR/ES, in percent. Exc. = 99% VaR exception rate (%). Kupiec, Ind., CC = rejection rate at 5% (%). Y+R = share of 250-day windows in the yellow or red zone (%); Red = share in the red zone (%). FZ0 = mean FZ0 loss minus the Oracle's; DM = share of replications in which Diebold-Mariano rejects equal FZ0 loss vs the Oracle at 5% (%). FHS: 0 of 800 GARCH fits did not converge.

| Method | VaR bias | VaR RMSE | ES bias | ES RMSE | Exc. | Kupiec | Ind. | CC | Y+R | Red | FZ0 | DM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.98 (0.02) | 4 (1) | 2 (1) | 2 (1) | 9 | 0.0 (0.0) | - | - |
| HS250 | 11.6 (0.9) | 37.3 (1.2) | 9.5 (0.9) | 35.8 (1.2) | 1.40 (0.02) | 22 (3) | 14 (2) | 22 (3) | 30 | 2.4 (0.5) | 0.168 (0.008) | 70 (3) |
| HS500 | 19.7 (1.2) | 40.9 (1.5) | 15.3 (1.2) | 37.4 (1.3) | 1.15 (0.03) | 14 (2) | 16 (3) | 18 (3) | 24 | 2.4 (0.5) | 0.173 (0.009) | 73 (3) |
| HS1000 | 20.7 (1.4) | 39.1 (1.4) | 18.9 (1.4) | 37.9 (1.3) | 1.11 (0.04) | 28 (3) | 21 (3) | 34 (3) | 22 | 4.4 (0.7) | 0.182 (0.009) | 77 (3) |
| EWMA-N | -10.3 (0.2) | 12.9 (0.1) | -12.1 (0.2) | 14.3 (0.2) | 1.68 (0.03) | 53 (4) | 4 (1) | 36 (3) | 40 | 0.5 (0.2) | 0.034 (0.002) | 16 (3) |
| EWMA-t6 | -1.0 (0.2) | 8.8 (0.1) | -0.0 (0.2) | 8.8 (0.1) | 1.13 (0.02) | 4 (1) | 2 (1) | 2 (1) | 14 | 0.0 (0.0) | 0.019 (0.001) | 14 (2) |
| FHS1000 | 1.2 (0.4) | 8.0 (0.2) | -0.4 (0.4) | 7.3 (0.2) | 1.00 (0.02) | 2 (1) | 2 (1) | 1 (1) | 11 | 0.1 (0.1) | 0.010 (0.001) | 12 (2) |
| MC-N | -10.2 (0.2) | 13.0 (0.1) | -12.1 (0.2) | 14.4 (0.2) | 1.69 (0.03) | 51 (4) | 5 (2) | 37 (3) | 41 | 0.6 (0.3) | 0.035 (0.002) | 14 (2) |

**Table A-persistent_fat. Simulated data: GARCH(1,1)-t set `persistent_fat`, 200 replications of 1000 forecast days.** Mean over replications (Monte Carlo standard error). Bias and RMSE are relative to the true conditional VaR/ES, in percent. Exc. = 99% VaR exception rate (%). Kupiec, Ind., CC = rejection rate at 5% (%). Y+R = share of 250-day windows in the yellow or red zone (%); Red = share in the red zone (%). FZ0 = mean FZ0 loss minus the Oracle's; DM = share of replications in which Diebold-Mariano rejects equal FZ0 loss vs the Oracle at 5% (%). FHS: 0 of 800 GARCH fits did not converge.

| Method | VaR bias | VaR RMSE | ES bias | ES RMSE | Exc. | Kupiec | Ind. | CC | Y+R | Red | FZ0 | DM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 1.05 (0.02) | 8 (2) | 3 (1) | 4 (1) | 12 | 0.0 (0.0) | - | - |
| HS250 | 27.1 (2.1) | 67.3 (3.4) | 21.0 (1.9) | 62.4 (3.2) | 1.43 (0.02) | 22 (3) | 18 (3) | 24 (3) | 31 | 2.8 (0.6) | 0.271 (0.014) | 71 (3) |
| HS500 | 43.0 (3.1) | 78.5 (4.3) | 33.4 (2.8) | 69.7 (3.8) | 1.25 (0.03) | 22 (3) | 24 (3) | 26 (3) | 27 | 5.1 (0.7) | 0.309 (0.018) | 74 (3) |
| HS1000 | 48.0 (3.6) | 78.0 (4.2) | 43.3 (3.5) | 74.3 (4.1) | 1.18 (0.05) | 37 (3) | 26 (3) | 44 (4) | 24 | 6.8 (0.9) | 0.343 (0.022) | 78 (3) |
| EWMA-N | -17.3 (0.2) | 20.2 (0.2) | -22.1 (0.2) | 24.1 (0.2) | 2.00 (0.03) | 86 (2) | 2 (1) | 69 (3) | 57 | 2.1 (0.5) | 0.077 (0.003) | 20 (3) |
| EWMA-t6 | -8.8 (0.3) | 14.6 (0.1) | -11.4 (0.2) | 16.0 (0.1) | 1.50 (0.02) | 30 (3) | 2 (1) | 18 (3) | 31 | 0.0 (0.0) | 0.039 (0.002) | 12 (2) |
| FHS1000 | 3.6 (0.7) | 11.8 (0.4) | 0.4 (0.6) | 10.8 (0.3) | 1.02 (0.02) | 6 (2) | 2 (1) | 2 (1) | 12 | 0.0 (0.0) | 0.012 (0.001) | 14 (2) |
| MC-N | -17.2 (0.2) | 20.2 (0.2) | -22.1 (0.2) | 24.2 (0.2) | 1.98 (0.03) | 82 (3) | 3 (1) | 68 (3) | 56 | 2.1 (0.5) | 0.078 (0.003) | 24 (3) |

**Table B1. Simulated data: rejection rates (%) at 5% of 99% VaR backtests when exceptions are iid with the given true rate** (binomial standard error in parentheses; 'exact' is the binomial probability of the Kupiec rejection region). The row with true rate 1.00% is the size; the other rows are power.

| n | true rate (%) | Kupiec | Kupiec exact | Christoffersen ind. | Christoffersen CC |
|---|---|---|---|---|---|
| 250 | 0.50 | 29.0 (0.6) | 28.6 | 0.5 (0.1) | 0.2 (0.1) |
| 250 | 0.75 | 15.2 (0.5) | 15.5 | 1.0 (0.1) | 0.4 (0.1) |
| 250 | 1.00 | 9.5 (0.4) | 9.5 | 1.2 (0.2) | 0.9 (0.1) |
| 250 | 1.25 | 7.7 (0.4) | 8.2 | 1.5 (0.2) | 2.2 (0.2) |
| 250 | 1.50 | 11.2 (0.4) | 10.8 | 1.9 (0.2) | 5.7 (0.3) |
| 250 | 2.00 | 23.9 (0.6) | 24.3 | 1.8 (0.2) | 16.4 (0.5) |
| 250 | 3.00 | 62.2 (0.7) | 62.5 | 1.5 (0.2) | 50.9 (0.7) |
| 1000 | 0.50 | 43.7 (0.7) | 44.0 | 1.7 (0.2) | 26.3 (0.6) |
| 1000 | 0.75 | 13.7 (0.5) | 13.3 | 2.5 (0.2) | 6.9 (0.4) |
| 1000 | 1.00 | 6.3 (0.3) | 5.5 | 1.8 (0.2) | 2.8 (0.2) |
| 1000 | 1.25 | 13.2 (0.5) | 13.4 | 1.7 (0.2) | 6.8 (0.4) |
| 1000 | 1.50 | 34.8 (0.7) | 33.6 | 1.7 (0.2) | 21.5 (0.6) |
| 1000 | 2.00 | 76.7 (0.6) | 78.2 | 1.8 (0.2) | 64.1 (0.7) |
| 1000 | 3.00 | 99.6 (0.1) | 99.7 | 2.0 (0.2) | 99.1 (0.1) |
| 8000 | 0.50 | 100.0 (0.0) | 100.0 | 2.2 (0.2) | 99.9 (0.0) |
| 8000 | 0.75 | 68.8 (0.7) | 68.1 | 1.5 (0.2) | 55.5 (0.7) |
| 8000 | 1.00 | 4.9 (0.3) | 5.0 | 2.2 (0.2) | 4.1 (0.3) |
| 8000 | 1.25 | 55.7 (0.7) | 55.4 | 2.4 (0.2) | 48.4 (0.7) |
| 8000 | 1.50 | 98.4 (0.2) | 97.9 | 7.1 (0.4) | 96.7 (0.3) |
| 8000 | 2.00 | 100.0 (0.0) | 100.0 | 6.6 (0.4) | 100.0 (0.0) |
| 8000 | 3.00 | 100.0 (0.0) | 100.0 | 5.4 (0.3) | 100.0 (0.0) |

**Table B2. Simulated data: rejection rates (%) at 5% when exceptions cluster** (Markov chain with stationary rate 1% and P(exception | exception yesterday) = pi11; pi11 = 0.01 is independence).

| n | pi11 | observed rate (%) | Kupiec | Christoffersen ind. | Christoffersen CC |
|---|---|---|---|---|---|
| 250 | 0.01 | 1.00 | 9.5 (0.4) | 1.4 (0.2) | 1.2 (0.2) |
| 250 | 0.05 | 0.99 | 10.8 (0.4) | 7.5 (0.4) | 3.9 (0.3) |
| 250 | 0.10 | 1.00 | 12.7 (0.5) | 14.6 (0.5) | 8.9 (0.4) |
| 250 | 0.20 | 1.01 | 15.4 (0.5) | 28.6 (0.6) | 17.3 (0.5) |
| 1000 | 0.01 | 1.00 | 6.2 (0.3) | 1.7 (0.2) | 3.1 (0.2) |
| 1000 | 0.05 | 1.00 | 6.8 (0.4) | 15.4 (0.5) | 13.7 (0.5) |
| 1000 | 0.10 | 1.00 | 8.2 (0.4) | 36.0 (0.7) | 31.7 (0.7) |
| 1000 | 0.20 | 0.99 | 12.4 (0.5) | 67.4 (0.7) | 62.4 (0.7) |
| 8000 | 0.01 | 1.00 | 4.9 (0.3) | 2.1 (0.2) | 4.1 (0.3) |
| 8000 | 0.05 | 1.00 | 5.9 (0.3) | 65.1 (0.7) | 60.1 (0.7) |
| 8000 | 0.10 | 1.00 | 7.1 (0.4) | 97.4 (0.2) | 96.8 (0.3) |
| 8000 | 0.20 | 1.00 | 10.1 (0.4) | 100.0 (0.0) | 100.0 (0.0) |

**Table C. Simulated data: Basel zone and exception-count probabilities (%) for 250 days of 99% VaR, 200000 simulated windows per rate** (binomial standard error in parentheses). Pearson goodness of fit of the 11 count cells, 10 df: rate 1%: chi2 = 12.3, p = 0.27; rate 2%: chi2 = 9.3, p = 0.50.

| exceptions / zone | simulated, rate 1% | exact, rate 1% | simulated, rate 2% | exact, rate 2% |
|---|---|---|---|---|
| 0 | 8.05 (0.06) | 8.11 | 0.67 (0.02) | 0.64 |
| 1 | 20.33 (0.09) | 20.47 | 3.27 (0.04) | 3.27 |
| 2 | 25.77 (0.10) | 25.74 | 8.32 (0.06) | 8.30 |
| 3 | 21.43 (0.09) | 21.49 | 13.89 (0.08) | 14.01 |
| 4 | 13.51 (0.08) | 13.41 | 17.69 (0.09) | 17.65 |
| 5 | 6.77 (0.06) | 6.66 | 17.76 (0.09) | 17.72 |
| 6 | 2.78 (0.04) | 2.75 | 14.68 (0.08) | 14.77 |
| 7 | 0.94 (0.02) | 0.97 | 10.49 (0.07) | 10.51 |
| 8 | 0.30 (0.01) | 0.30 | 6.50 (0.06) | 6.51 |
| 9 | 0.09 (0.01) | 0.08 | 3.65 (0.04) | 3.57 |
| 10+ | 0.03 (0.00) | 0.03 | 3.07 (0.04) | 3.04 |
| green | 89.09 (0.07) | 89.22 | 43.84 (0.11) | 43.87 |
| yellow | 10.88 (0.07) | 10.76 | 53.09 (0.11) | 53.09 |
| red | 0.03 (0.00) | 0.03 | 3.07 (0.04) | 3.04 |

**Table D-normal. Simulated data: error of Greek approximations relative to full revaluation, one-day P&L, normal innovations.** Mean over 20 batches of scenarios (Monte Carlo standard error), in percent. 'VaR full' is the full-revaluation 99% VaR in index points (index = 100). DG = delta-gamma-theta, DGV = delta-gamma-vega-theta, DGVX = DGV plus vanna and volga (section d'). Tail MAE = mean absolute P&L error of DGV over the scenarios in the full-revaluation 97.5% tail, divided by the full-revaluation 97.5% ES.

| Portfolio | sigma0 | VaR full | DG VaR err | DGV VaR err | DGVX VaR err | DG ES err | DGV ES err | DGVX ES err | DGV tail MAE |
|---|---|---|---|---|---|---|---|---|---|
| long ATM call | 0.12 | 0.650 | 20.0 (0.1) | -1.7 (0.0) | -2.0 (0.0) | 19.8 (0.1) | -1.9 (0.0) | -2.1 (0.0) | 2.1 (0.0) |
| short ATM put | 0.12 | 1.130 | -13.8 (0.1) | 1.5 (0.0) | 1.8 (0.0) | -13.8 (0.0) | 1.7 (0.0) | 1.9 (0.0) | 1.7 (0.0) |
| short ATM straddle | 0.12 | 0.557 | -27.2 (0.3) | 6.1 (0.1) | 7.2 (0.1) | -26.3 (0.3) | 6.6 (0.0) | 7.7 (0.0) | 7.3 (0.0) |
| delta-hedged short straddle | 0.12 | 0.635 | -41.8 (0.2) | 5.8 (0.0) | 6.7 (0.0) | -41.0 (0.2) | 6.2 (0.0) | 7.1 (0.0) | 6.3 (0.0) |
| long 90 put, short 110 call | 0.12 | 0.007 | 48.8 (0.4) | 21.2 (0.3) | -0.7 (0.2) | 47.3 (0.3) | 20.1 (0.2) | -1.8 (0.1) | 24.3 (0.1) |
| short 90 put | 0.12 | 0.012 | -76.1 (0.1) | -63.5 (0.1) | -32.2 (0.1) | -77.3 (0.0) | -65.4 (0.1) | -35.2 (0.1) | 65.7 (0.1) |
| long ATM call | 0.20 | 1.092 | 17.6 (0.1) | -1.7 (0.0) | -1.7 (0.0) | 17.4 (0.1) | -1.7 (0.0) | -1.7 (0.0) | 2.0 (0.0) |
| short ATM put | 0.20 | 1.854 | -12.7 (0.1) | 1.5 (0.0) | 1.5 (0.0) | -12.6 (0.0) | 1.7 (0.0) | 1.7 (0.0) | 1.7 (0.0) |
| short ATM straddle | 0.20 | 0.886 | -24.4 (0.3) | 6.6 (0.1) | 6.6 (0.1) | -23.6 (0.3) | 7.1 (0.0) | 7.1 (0.0) | 7.8 (0.0) |
| delta-hedged short straddle | 0.20 | 1.000 | -38.3 (0.3) | 6.3 (0.0) | 6.3 (0.0) | -37.5 (0.2) | 6.6 (0.0) | 6.6 (0.0) | 6.8 (0.0) |
| long 90 put, short 110 call | 0.20 | 0.210 | 33.6 (0.3) | 22.5 (0.2) | -8.0 (0.1) | 32.7 (0.2) | 21.5 (0.1) | -8.8 (0.1) | 18.7 (0.1) |
| short 90 put | 0.20 | 0.247 | -48.7 (0.1) | -29.7 (0.1) | -6.2 (0.1) | -49.5 (0.0) | -30.7 (0.0) | -6.9 (0.0) | 30.9 (0.0) |
| long ATM call | 0.40 | 2.249 | 14.5 (0.1) | -1.2 (0.0) | -0.9 (0.0) | 14.3 (0.1) | -1.2 (0.0) | -1.0 (0.0) | 1.8 (0.0) |
| short ATM put | 0.40 | 3.563 | -11.4 (0.1) | 1.4 (0.0) | 1.1 (0.0) | -11.4 (0.0) | 1.6 (0.0) | 1.3 (0.0) | 1.6 (0.0) |
| short ATM straddle | 0.40 | 1.569 | -10.5 (0.4) | 6.8 (0.1) | 5.3 (0.1) | -9.2 (0.4) | 7.2 (0.0) | 5.7 (0.0) | 8.2 (0.0) |
| delta-hedged short straddle | 0.40 | 1.843 | -32.9 (0.3) | 6.2 (0.1) | 5.0 (0.1) | -32.0 (0.2) | 6.6 (0.0) | 5.4 (0.0) | 6.8 (0.0) |
| long 90 put, short 110 call | 0.40 | 2.110 | 13.6 (0.1) | 10.4 (0.1) | -3.4 (0.1) | 13.4 (0.1) | 10.2 (0.0) | -4.0 (0.0) | 9.7 (0.0) |
| short 90 put | 0.40 | 1.646 | -26.4 (0.1) | -8.7 (0.0) | 0.3 (0.1) | -26.7 (0.0) | -9.0 (0.0) | 0.3 (0.0) | 9.0 (0.0) |

**Table D-t4. Simulated data: error of Greek approximations relative to full revaluation, one-day P&L, t4 innovations.** Mean over 20 batches of scenarios (Monte Carlo standard error), in percent. 'VaR full' is the full-revaluation 99% VaR in index points (index = 100). DG = delta-gamma-theta, DGV = delta-gamma-vega-theta, DGVX = DGV plus vanna and volga (section d'). Tail MAE = mean absolute P&L error of DGV over the scenarios in the full-revaluation 97.5% tail, divided by the full-revaluation 97.5% ES.

| Portfolio | sigma0 | VaR full | DG VaR err | DGV VaR err | DGVX VaR err | DG ES err | DGV ES err | DGVX ES err | DGV tail MAE |
|---|---|---|---|---|---|---|---|---|---|
| long ATM call | 0.12 | 0.702 | 21.2 (0.1) | -3.7 (0.1) | -4.1 (0.1) | 19.9 (0.1) | -5.6 (0.1) | -6.1 (0.1) | 6.8 (0.2) |
| short ATM put | 0.12 | 1.305 | -13.1 (0.1) | 2.0 (0.0) | 2.3 (0.0) | -10.8 (0.1) | 3.9 (0.1) | 4.3 (0.1) | 3.6 (0.1) |
| short ATM straddle | 0.12 | 0.785 | -17.4 (0.5) | 4.6 (0.2) | 5.7 (0.2) | -7.9 (0.5) | 11.3 (0.4) | 12.5 (0.4) | 14.6 (0.4) |
| delta-hedged short straddle | 0.12 | 0.844 | -25.3 (0.4) | 5.2 (0.1) | 6.2 (0.1) | -15.3 (0.5) | 11.4 (0.4) | 12.5 (0.4) | 13.5 (0.4) |
| long 90 put, short 110 call | 0.12 | 0.008 | 57.1 (0.7) | 30.4 (0.6) | -13.9 (0.3) | -1.0 (5.9) | -15.5 (5.0) | -46.3 (3.6) | 56.7 (2.4) |
| short 90 put | 0.12 | 0.016 | -77.2 (0.1) | -67.2 (0.1) | -35.6 (0.2) | -86.2 (0.3) | -81.0 (0.4) | -60.1 (0.8) | 81.7 (0.4) |
| long ATM call | 0.20 | 1.177 | 19.0 (0.1) | -3.6 (0.1) | -3.6 (0.1) | 17.8 (0.1) | -5.4 (0.1) | -5.4 (0.1) | 6.5 (0.2) |
| short ATM put | 0.20 | 2.144 | -12.1 (0.1) | 2.0 (0.0) | 2.0 (0.0) | -9.9 (0.2) | 4.0 (0.1) | 4.0 (0.1) | 3.6 (0.1) |
| short ATM straddle | 0.20 | 1.265 | -14.8 (0.5) | 5.0 (0.2) | 5.0 (0.2) | -5.1 (0.5) | 11.8 (0.4) | 11.8 (0.4) | 15.1 (0.4) |
| delta-hedged short straddle | 0.20 | 1.350 | -22.1 (0.4) | 5.4 (0.1) | 5.4 (0.1) | -12.0 (0.5) | 11.9 (0.4) | 11.9 (0.4) | 14.1 (0.4) |
| long 90 put, short 110 call | 0.20 | 0.241 | 34.9 (0.3) | 23.7 (0.2) | -16.4 (0.2) | 16.1 (0.9) | 7.0 (0.8) | -33.3 (0.8) | 25.8 (0.5) |
| short 90 put | 0.20 | 0.305 | -49.3 (0.1) | -32.2 (0.1) | -6.7 (0.2) | -52.9 (0.3) | -39.0 (0.3) | -11.6 (0.5) | 40.3 (0.2) |
| long ATM call | 0.40 | 2.418 | 16.1 (0.1) | -2.8 (0.1) | -2.3 (0.1) | 15.3 (0.1) | -4.5 (0.1) | -3.9 (0.1) | 5.5 (0.1) |
| short ATM put | 0.40 | 4.120 | -11.1 (0.1) | 1.9 (0.1) | 1.6 (0.0) | -8.9 (0.2) | 3.9 (0.2) | 3.5 (0.2) | 3.2 (0.1) |
| short ATM straddle | 0.40 | 2.354 | -6.4 (0.6) | 5.1 (0.2) | 3.5 (0.2) | 3.6 (0.7) | 12.8 (0.6) | 11.1 (0.6) | 15.5 (0.6) |
| delta-hedged short straddle | 0.40 | 2.542 | -17.3 (0.4) | 5.6 (0.1) | 4.3 (0.1) | -6.4 (0.7) | 12.8 (0.6) | 11.3 (0.6) | 14.4 (0.6) |
| long 90 put, short 110 call | 0.40 | 2.422 | 13.4 (0.1) | 10.2 (0.1) | -5.6 (0.1) | 7.7 (0.2) | 4.9 (0.2) | -12.9 (0.3) | 10.3 (0.1) |
| short 90 put | 0.40 | 1.957 | -26.6 (0.1) | -9.4 (0.1) | 0.6 (0.1) | -25.7 (0.3) | -9.8 (0.3) | 2.7 (0.4) | 11.4 (0.1) |

**Table E. Simulated data: size and power of the Acerbi-Szekely tests at 97.5% with 5000 simulations per test (GARCH(1,1)-t set `typical`).** Rejection rate at 5% (%) with its binomial standard error; mean of the statistic; mean number of 97.5% VaR exceedances. `correct` = the true conditional distribution (size); `normal` = right variance, normal shape (ES understated); `scale0.9` = true shape with 0.9 times the volatility.

| T | model | Z2 rejection | mean Z2 | Z1 rejection | mean Z1 | exceedances | replications |
|---|---|---|---|---|---|---|---|
| 250 | correct | 4.5 (1.5) | -0.013 | 4.5 (1.5) | 0.001 | 6.3 | 200 |
| 250 | normal | 13.0 (2.4) | -0.169 | 31.0 (3.3) | -0.084 | 6.7 | 200 |
| 250 | scale0.9 | 31.5 (3.3) | -0.479 | 5.0 (1.5) | -0.023 | 9.0 | 200 |
| 1000 | correct | 6.0 (1.7) | 0.009 | 4.0 (1.4) | 0.004 | 24.9 | 200 |
| 1000 | normal | 20.5 (2.9) | -0.144 | 71.0 (3.2) | -0.081 | 26.5 | 200 |
| 1000 | scale0.9 | 70.0 (3.2) | -0.458 | 8.0 (1.9) | -0.018 | 35.8 | 200 |

**Table F-calm. Simulated data: historical-simulation 99% VaR error by volatility regime, GARCH(1,1)-t set `calm`, 20 replications of 2000 forecast days.** Mean over replications of the class mean (Monte Carlo standard error), in percent. Level: true sigma below 0.8 / above 1.25 times the unconditional volatility. Change: true sigma divided by the root mean true variance over the method's own window, below 0.8 (falling) / above 1.25 (rising). Reps = replications with days in the class; Days = share of forecast days in the class (%).

| Method | classification | class | VaR error | Exc. rate | Reps | Days |
|---|---|---|---|---|---|---|
| HS250 | level | low | 18.5 (3.3) | 0.00 (0.00) | 14 | 0.3 |
| HS250 | level | middle | 2.7 (0.7) | 1.28 (0.03) | 20 | 96.6 |
| HS250 | level | high | -17.5 (1.1) | 1.52 (0.39) | 20 | 3.1 |
| HS250 | change | falling | 35.5 (3.8) | 0.00 (0.00) | 18 | 1.0 |
| HS250 | change | stable | 2.3 (0.7) | 1.27 (0.04) | 20 | 96.3 |
| HS250 | change | rising | -20.3 (0.7) | 2.27 (0.49) | 20 | 2.7 |
| HS500 | level | low | 30.4 (2.7) | 0.00 (0.00) | 14 | 0.3 |
| HS500 | level | middle | 5.2 (0.7) | 1.01 (0.05) | 20 | 96.6 |
| HS500 | level | high | -18.8 (0.8) | 1.85 (0.40) | 20 | 3.1 |
| HS500 | change | falling | 36.4 (2.8) | 0.00 (0.00) | 19 | 0.8 |
| HS500 | change | stable | 5.0 (0.6) | 0.99 (0.05) | 20 | 96.4 |
| HS500 | change | rising | -20.0 (0.8) | 2.77 (0.51) | 20 | 2.9 |
| HS1000 | level | low | 27.9 (1.5) | 0.00 (0.00) | 14 | 0.3 |
| HS1000 | level | middle | 4.3 (0.7) | 1.01 (0.05) | 20 | 96.6 |
| HS1000 | level | high | -22.1 (0.9) | 2.00 (0.42) | 20 | 3.1 |
| HS1000 | change | falling | 31.4 (1.4) | 0.00 (0.00) | 16 | 0.7 |
| HS1000 | change | stable | 4.2 (0.7) | 1.00 (0.05) | 20 | 96.3 |
| HS1000 | change | rising | -22.7 (0.8) | 2.35 (0.42) | 20 | 3.1 |

**Table F-typical. Simulated data: historical-simulation 99% VaR error by volatility regime, GARCH(1,1)-t set `typical`, 20 replications of 2000 forecast days.** Mean over replications of the class mean (Monte Carlo standard error), in percent. Level: true sigma below 0.8 / above 1.25 times the unconditional volatility. Change: true sigma divided by the root mean true variance over the method's own window, below 0.8 (falling) / above 1.25 (rising). Reps = replications with days in the class; Days = share of forecast days in the class (%).

| Method | classification | class | VaR error | Exc. rate | Reps | Days |
|---|---|---|---|---|---|---|
| HS250 | level | low | 29.4 (2.7) | 0.54 (0.08) | 20 | 39.3 |
| HS250 | level | middle | 5.1 (2.2) | 1.43 (0.09) | 20 | 48.2 |
| HS250 | level | high | -18.9 (3.0) | 3.28 (0.37) | 20 | 12.5 |
| HS250 | change | falling | 50.5 (3.2) | 0.24 (0.05) | 20 | 27.3 |
| HS250 | change | stable | 2.3 (1.0) | 1.17 (0.08) | 20 | 58.2 |
| HS250 | change | rising | -27.2 (0.8) | 4.04 (0.31) | 20 | 14.5 |
| HS500 | level | low | 44.2 (3.5) | 0.33 (0.06) | 20 | 39.3 |
| HS500 | level | middle | 11.8 (3.1) | 1.12 (0.09) | 20 | 48.2 |
| HS500 | level | high | -22.8 (2.9) | 3.49 (0.34) | 20 | 12.5 |
| HS500 | change | falling | 59.7 (4.2) | 0.12 (0.03) | 20 | 30.5 |
| HS500 | change | stable | 8.7 (1.2) | 0.91 (0.08) | 20 | 55.0 |
| HS500 | change | rising | -26.9 (0.8) | 3.90 (0.30) | 20 | 14.5 |
| HS1000 | level | low | 52.6 (5.7) | 0.20 (0.05) | 20 | 39.3 |
| HS1000 | level | middle | 13.6 (4.3) | 0.94 (0.08) | 20 | 48.2 |
| HS1000 | level | high | -28.0 (2.5) | 3.72 (0.31) | 20 | 12.5 |
| HS1000 | change | falling | 58.1 (5.9) | 0.14 (0.03) | 20 | 35.1 |
| HS1000 | change | stable | 11.0 (1.3) | 0.81 (0.07) | 20 | 50.9 |
| HS1000 | change | rising | -29.4 (0.8) | 3.91 (0.33) | 20 | 14.0 |

**Table F-persistent_fat. Simulated data: historical-simulation 99% VaR error by volatility regime, GARCH(1,1)-t set `persistent_fat`, 20 replications of 2000 forecast days.** Mean over replications of the class mean (Monte Carlo standard error), in percent. Level: true sigma below 0.8 / above 1.25 times the unconditional volatility. Change: true sigma divided by the root mean true variance over the method's own window, below 0.8 (falling) / above 1.25 (rising). Reps = replications with days in the class; Days = share of forecast days in the class (%).

| Method | classification | class | VaR error | Exc. rate | Reps | Days |
|---|---|---|---|---|---|---|
| HS250 | level | low | 43.8 (4.0) | 0.88 (0.06) | 20 | 69.2 |
| HS250 | level | middle | 5.3 (4.2) | 2.09 (0.23) | 20 | 19.9 |
| HS250 | level | high | -18.7 (3.9) | 4.10 (0.40) | 20 | 10.9 |
| HS250 | change | falling | 85.6 (5.8) | 0.28 (0.05) | 20 | 41.3 |
| HS250 | change | stable | 0.9 (1.4) | 1.37 (0.08) | 20 | 39.7 |
| HS250 | change | rising | -32.4 (1.0) | 4.24 (0.32) | 20 | 19.1 |
| HS500 | level | low | 73.3 (5.4) | 0.51 (0.05) | 20 | 69.2 |
| HS500 | level | middle | 9.9 (5.0) | 1.90 (0.18) | 20 | 19.9 |
| HS500 | level | high | -26.6 (3.8) | 4.85 (0.34) | 20 | 10.9 |
| HS500 | change | falling | 103.9 (5.8) | 0.20 (0.03) | 20 | 50.0 |
| HS500 | change | stable | 8.8 (1.5) | 1.08 (0.08) | 20 | 32.7 |
| HS500 | change | rising | -33.3 (0.9) | 4.63 (0.28) | 20 | 17.3 |
| HS1000 | level | low | 91.3 (8.6) | 0.33 (0.05) | 20 | 69.2 |
| HS1000 | level | middle | 10.6 (5.5) | 1.58 (0.16) | 20 | 19.9 |
| HS1000 | level | high | -35.3 (3.1) | 6.04 (0.54) | 20 | 10.9 |
| HS1000 | change | falling | 110.3 (9.4) | 0.16 (0.03) | 20 | 56.9 |
| HS1000 | change | stable | 11.9 (1.5) | 0.97 (0.09) | 20 | 27.2 |
| HS1000 | change | rising | -36.4 (1.1) | 5.39 (0.48) | 20 | 15.8 |
<!-- tables:end -->

## Reading of the results

All statements refer to the simulated data above and the tables they cite.

**Checks of the checker.** The Oracle (true conditional distribution) has exception rates of
0.98-1.05% and Kupiec rejection rates of 4-8% over 1000 days (exact size of the test at n = 1000:
5.5%, Table B1), so the backtest machinery does not invent failures for a correct model. The
simulated Basel zone frequencies match the binomial probabilities (Table C, Pearson p = 0.27 and
0.50). In Table B1, two of 21 cells differ from the exact Kupiec value by more than 2.5 standard
errors (n = 1000, true rates 1% and 2%). Recomputing those cells from the simulated exception
counts gives the same values, so the deviation is in the simulated draws, not in the test code;
the "exact" column is the reference for the size of the test.

**Models (Tables A).**

- Filtered historical simulation is the only method that is close to the truth in all three
  sets: VaR bias within 4% and relative RMSE 6-12%, exception rates 1.00-1.04%, Kupiec
  rejection at the level of the Oracle. It is also the only method that is given the right model
  class (GARCH-t), so this is a check that the estimator and the filtering work, not evidence that
  FHS is best on market data.
- Historical simulation overstates VaR on average (bias +1.5% to +48%) and still has more exceptions
  than 1% (1.03-1.43%). Its relative RMSE (13-79%) is the largest of all methods, it fails
  Christoffersen's independence test more often than any other method (up to 26%), and a longer
  window does not help in the persistent sets: HS1000 has an RMSE of 39.1 against 37.3 for HS250
  (`typical`) and 78.0 against 67.3 (`persistent_fat`), differences within about two standard
  errors, and more Kupiec rejections (28% against 22%, 37% against 22%). (Wave 1 called HS1000
  "worse in RMSE"; with these standard errors that is not established.)
- *Why* historical simulation errs was stated in wave 1 without a test ("too high in calm
  periods and too low after volatility rises"). Experiment (f) tests it (Tables F): in every set
  and for every window, the VaR error is positive in the `low` volatility class (+18% to +91%)
  and negative in the `rising` class (-20% to -36%), each by more than two standard errors, which
  is the pre-specified criterion; the exception rate is 0-0.9% in `low` and 2.3-5.4% in
  `rising`. In the `calm` set the `low` class holds only 0.3% of the days (14 of 20
  replications).
- EWMA with normal quantiles understates VaR by 4%, 10% and 17% and ES by 4%, 12% and 22% as the
  tails get fatter; its exception rate reaches 2.0% and Kupiec rejects it in 86% of 1000-day
  samples in `persistent_fat`. Monte Carlo with the same covariance (MC-N) reproduces EWMA-N to
  within its simulation noise, as it should.
- EWMA with Student t(6) quantiles is nearly unbiased only when the true tail is close to t(6)
  (`typical`, nu = 8); it overstates VaR by 6% when the truth is near-normal and understates it
  by 9% when the truth is t(4). A fixed nu is a modelling choice with a visible cost either way.
- The FZ0 score ranks FHS first in every set, but the Diebold-Mariano test against the Oracle has
  little power over 1000 days for the methods that are close: it rejects EWMA-N in only 16-20% of
  samples even where Kupiec rejects EWMA-N in 53-86%.

**Backtests (Tables B and C).**

- At n = 250 the Kupiec test is a weak instrument: its exact size is 9.5% (discreteness), and it
  rejects a VaR whose true exception rate is 2% (twice nominal) in only 24% of samples.
- The Christoffersen independence test has size below nominal (0.5-2.5% for n <= 1000) and low
  power: with clustering strong enough that an exception raises tomorrow's probability from 1%
  to 20%, it rejects in 29% of 250-day samples and 67% of 1000-day samples. At n = 8000 its
  rejection rate under independence rises to 5-7% when the true rate is 1.5-3%.
- Under a correct 99% VaR, 10.8% of 250-day windows are yellow (exact binomial) and the Oracle
  lands in yellow in 9-12% of windows in Tables A. A VaR with twice the nominal exception rate
  stays green in 44% of windows (Table C).

**ES backtests (Table E, experiment e).** With 5000 simulations per test the Z2 test rejects
the true model in 4.5% (SE 1.5) of 250-day samples and 6.0% (SE 1.7) of 1000-day samples, within
two standard errors of 5%: the size check passes. Its power is modest against a normal model
with the right variance (13% at 250 days, 20.5% at 1000 days), where Z1 does better (31% and
71%), and larger against a volatility understated by 10% (31.5% and 70%). The real-data protocol
uses Z2 over about 7,300 days; this table does not measure power at that length.

**Option portfolios (Tables D).**

- Without the vega term (DG) the approximation errors are 6-77% of the full-revaluation VaR:
  the volatility factor matters for one-day option P&L under this model.
- With the vega term (DGV) the VaR error is 1-7% for the ATM portfolios under normal innovations.
  It is not small for out-of-the-money options: for a short 90% put with 21 days to expiry DGV
  understates the 99% VaR by 64% (sigma0 = 0.12) and 30% (sigma0 = 0.20). Wave 1 attributed this
  to convexity outside a second-order expansion in spot and first-order in volatility without a
  test. Experiment (d') adds vanna and volga (DGVX): the error falls to -6.2% (sigma0 = 0.20)
  and +0.3% (sigma0 = 0.40), more than halved as the pre-specified criterion requires, but at
  sigma0 = 0.12 it is still -32.2% (normal) and -35.6% (t4), a reduction just short of half
  (criterion not met). So the missing vanna and volga terms explain most of the error at 20% and
  40% volatility; at 12% volatility about half of it comes from terms of order three and higher,
  which were not separated further. For the ATM straddles DGVX changes the VaR error by at most
  1.6 percentage points (DGVX errors 3.5-7.2%).
- With Student t(4) innovations the DGV error in ES for short straddles is 11-13% and its mean
  absolute error over the tail is 13-16% of ES. For the long-90-put / short-110-call position at
  sigma0 = 0.12 the full-revaluation VaR is under 0.01 index points, so its relative errors (and
  the large standard error of its ES error) say little.

## Deviations from the protocol

None.
