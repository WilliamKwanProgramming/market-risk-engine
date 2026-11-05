# API of the `mre` package (wave 1)

One page for wave 2. Docstrings hold the formulas and sources; this page states the contracts.

## Conventions used everywhere

- **P&L sign.** P&L is positive for a gain. VaR and ES are returned as **positive loss numbers**.
  An exception on day t is `pnl_t < -VaR_t`.
- **Levels.** `level_var=0.99`, `level_es=0.975` by default (confidence, not tail probability).
- **Factors.** `logS:<u>` (log spot of underlying `u`) and `logIV:<u>` (log of one flat implied
  volatility per underlying with options). A scenario is a row of factor *moves* (log changes).
  Column order is `Portfolio.factor_names()`.
- **Time.** Years; one trading day is `1/TRADING_DAYS` with `TRADING_DAYS = 252`.
- **Information.** Row t of the move matrix is the move from close t-1 to close t. The forecast
  for row t uses rows `0..t-1` only; `rolling_forecasts` is the only place that slices history.
- **Option P&L is model P&L**: BSM revaluation at the shocked spot and implied volatility. There is
  no implied-volatility history in the data available here; wave 2 must state its assumption (for
  example VIX/100 where it exists, or `data.realized_vol_proxy` times a stated multiple).

## Modules

| Module | Main objects | Notes |
|---|---|---|
| `mre.data` | `load_skfolio_prices/index/factors()`, `load_arch(name)`, `load_csv(path)`, `clean_prices(prices, policy="drop"\|"ffill", max_fill=5) -> (df, report)`, `to_returns(prices, "log"\|"simple")`, `realized_vol_proxy(log_returns, window=21, multiple=1.0)`, `factor_levels(prices, implied_vol=None)` | skfolio stock prices are split- and dividend-adjusted; the index is a price index |
| `mre.pricing` | `bsm_price(S,K,T,r,q,sigma,is_call)`, `bsm_greeks(...) -> Greeks(price,delta,gamma,vega,theta,rho)`, `implied_vol(price,S,K,T,r,q,is_call)` | broadcasts over arrays; vega per 1.00 vol; theta = -dV/dT per year; `T<=0` is intrinsic |
| `mre.portfolio` | `Stock(u, qty)`, `EuropeanOption(u, strike, expiry, is_call, qty, multiplier=1)`, `MarketState(spot, vol, rate, div_yield)`, `Portfolio(positions)` with `factor_names()`, `value(state)`, `pnl_full(state, moves, horizon)`, `pnl_greeks(state, moves, horizon, vega, theta)`, `factor_sensitivities(state)`; `PortfolioPnL(portfolio, state, horizon)`, `LinearPnL(exposure)` | P&L arrays are `(n_scenarios, n_positions)`; a P&L function maps `(n, k)` moves to `(n,)` and has `linear_exposure()` |
| `mre.measures` | `empirical_var_es(pnl, level_var, level_es)`, `EmpiricalPnL(scenarios)`, `LocationScalePnL(loc, scale, "normal"\|"t", nu)`, `std_t_quantile`, `std_t_es`, `normal_es` | predictive distributions expose `var(level)`, `es(level)`, `sample(rng, size)` |
| `mre.garch` | `GarchParams`, `fit_garch_t(r) -> GarchFit(params, loglik, converged, message, n_iter, n_obs)`, `garch_filter(r, params) -> (sigma2, sigma2_next)`, `simulate_garch_t(n, params, rng, burn)` | same likelihood and backcast as `arch`; `converged` is the optimiser's flag |
| `mre.models` | `HistoricalSimulation(window)`, `EWMAParametric(lam, dist, nu, window)`, `FilteredHistoricalSimulation(window, refit_every)`, `MonteCarlo(lam, dist, nu, n_draws, seed, window)`, `ewma_covariance`, `RiskForecast`, `rolling_forecasts(model, moves, start, end, pnl, level_var, level_es) -> ForecastPath` | `ForecastPath.frame()` gives var, es, var_at_es_level, realized, exception |
| `mre.backtest` | `kupiec_pof(exc, level)`, `christoffersen(exc, level) -> {"ind", "cc"}`, `traffic_light(k)`, `zone_occupancy(exc, 250)`, `binomial_zone_probabilities(n, p)`, `acerbi_szekely(realized, predictives, level, n_sim, seed) -> {"Z1", "Z2"}`, `fz0_loss(realized, var, es, level)`, `diebold_mariano(loss_a, loss_b, lag=None)` | all tests return `TestResult(statistic, p_value, df, details)` with `.reject(size)` |
| `mre.stress` | `HISTORICAL_WINDOWS`, `historical_scenario(log_levels, start, end, name)`, `hypothetical_scenarios(portfolio, state)`, `run_stress(portfolio, state, scenarios, horizon=0.0)` | scenarios are instantaneous (no time decay) unless `horizon` is set |
| `mre.attribution` | `euler_gaussian(exposure, cov, ...)`, `euler_historical(pnl_by_position, ..., bandwidth=None)`, `factor_regression(y, factors)`, `option_pnl_explain(portfolio, state0, state1, horizon)` | ES contributions sum exactly; kernel VaR contributions report their gap |

## Typical use (wave 2)

```python
import numpy as np
from mre import data, models, backtest, portfolio as pf

prices, report = data.clean_prices(data.load_skfolio_prices()[["JPM", "XOM"]])
moves = data.to_returns(prices).add_prefix("logS:")          # factor moves, log returns
port = pf.Portfolio([pf.Stock("JPM", 100), pf.Stock("XOM", -50)])

def pnl_at(t):                                              # state at the close of day t-1
    spot = prices.iloc[t].to_dict()                         # prices has one more row than moves
    return pf.PortfolioPnL(port, pf.MarketState(spot))

path = models.rolling_forecasts(models.HistoricalSimulation(500), moves, start=500, pnl=pnl_at)
fr = path.frame()
backtest.kupiec_pof(fr["exception"]), backtest.christoffersen(fr["exception"])
```

`prices.iloc[t]` is the close of day t-1 relative to `moves` row t because `to_returns` drops the
first price row. Wave 1 ran no backtest on real data; the snippet only shows the wiring.
