# Market Risk Engine: VaR, Expected Shortfall and Backtesting

> **Status: work in progress.** This repository currently holds the project plan only. No code or results have been published yet.

## Goal

A small, transparent market-risk engine for a multi-asset portfolio that compares risk models by how well they survive backtesting, not just by how they look on a dashboard.

## Planned scope

- **Risk measures.** 1-day 99% value-at-risk (VaR) and 97.5% expected shortfall (ES). The Fundamental Review of the Trading Book (FRTB) moves internal-model capital from 99% VaR to 97.5% ES.
- **Methods.** Historical simulation, EWMA-parametric, filtered historical simulation (GARCH) and Monte Carlo.
- **Full revaluation.** Options are repriced under each scenario with the [options pricing library](https://github.com/WilliamKwanProgramming/options-pricing-algorithm) instead of being approximated with Greeks.
- **Backtesting.** Kupiec proportion-of-failures test, Christoffersen independence and conditional-coverage tests, and the Basel traffic-light zones for 99% VaR over 250 trading days: green is 0 to 4 exceptions, yellow 5 to 9, red 10 or more.
- **Stress testing.** Historical scenarios (2008, March 2020, the 2022 rate shock) and hypothetical shocks.
- **Dashboard.** Streamlit app reporting exceptions, expected shortfall and P&L attribution by risk factor.

## Data

Public daily price and rate data only.

## Disclaimer

Educational project. Not investment advice.
