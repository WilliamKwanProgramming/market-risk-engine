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

### What is pre-specified about the reading

The checks of the checker are: the Oracle in (a) must show exception rates near 1% and Kupiec
rejection near the exact size of the test; the simulated Kupiec rejection in (b) must match the
exact binomial value within Monte Carlo error; the simulated zone frequencies in (c) must match
the binomial probabilities within Monte Carlo error. If any of these fails, the experiment
results are not reported as findings until the cause is found.

## Results

Generated by `scripts/make_tables.py` from `results/validation/*.csv`.

<!-- tables:begin -->
<!-- tables:end -->

## Deviations from the protocol

None.
