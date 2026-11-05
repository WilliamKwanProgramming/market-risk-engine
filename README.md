# Market Risk Engine: VaR, Expected Shortfall and Backtesting

> **Status: work in progress (wave 1).** The Python library (`src/mre`), its tests and a validation on **simulated data** exist. No backtest has been run on market data, and the dashboard does not exist yet. The rest of this page is still the plan; see the Status table below for what is done.

## Status

| Feature | Implemented | Tested | Run on simulated data | Run on real data |
|---|---|---|---|---|
| Data loaders (skfolio, arch, user CSV), missing-data policy, returns | yes | yes | n/a | loaders smoke-checked only |
| Black-Scholes-Merton prices, Greeks, implied volatility | yes | yes | yes | not run |
| Portfolio full revaluation and delta-gamma-vega approximation | yes | yes | yes ([validation](docs/validation.md), table D) | not run |
| Historical simulation, EWMA parametric, filtered HS (GARCH-t), Monte Carlo | yes | yes | yes (table A) | not run |
| Kupiec, Christoffersen, Basel traffic light | yes | yes | yes (tables A-C) | not run |
| Acerbi-Szekely ES tests, FZ0 score, Diebold-Mariano | yes | yes | FZ0 and DM in table A; ES tests only in unit tests | not run |
| Historical and hypothetical stress scenarios | yes | yes | unit tests only | window coverage checked only |
| Euler VaR/ES attribution, factor regression, option P&L explain | yes | yes | unit tests only | not run |
| Streamlit dashboard | no | no | no | no |

Library contracts: [docs/api.md](docs/api.md). Validation protocol and results: [docs/validation.md](docs/validation.md). Tests: `pytest -q`; mutation check: `python3 tests/mutation_check.py`; all simulated results: `scripts/reproduce.sh`.

## Goal

A small, transparent market-risk engine for a multi-asset portfolio that compares risk models by how well they survive backtesting, not just by how they look on a dashboard.

## Planned scope

- **Risk measures.** 1-day 99% value-at-risk (VaR) and 97.5% expected shortfall (ES). The Fundamental Review of the Trading Book (FRTB) moves internal-model capital from 99% VaR to 97.5% ES.
- **Methods.** Historical simulation, EWMA-parametric, filtered historical simulation (GARCH) and Monte Carlo.
- **Full revaluation.** Options are repriced under each scenario instead of being approximated with Greeks. Wave 1 uses its own Black-Scholes-Merton pricer (`mre.pricing`); whether to switch to the [options pricing library](https://github.com/WilliamKwanProgramming/options-pricing-algorithm) is open.
- **Backtesting.** Kupiec proportion-of-failures test, Christoffersen independence and conditional-coverage tests, and the Basel traffic-light zones for 99% VaR over 250 trading days: green is 0 to 4 exceptions, yellow 5 to 9, red 10 or more.
- **Stress testing.** Historical scenarios (2008, March 2020, the 2022 rate shock) and hypothetical shocks.
- **Dashboard.** Streamlit app reporting exceptions, expected shortfall and P&L attribution by risk factor.

## Data

Public daily price and rate data only.

## Disclaimer

Educational project. Not investment advice.
