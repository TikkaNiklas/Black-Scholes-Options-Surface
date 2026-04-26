# Black–Scholes Options Diagnostics

This project compares real AAPL option prices to theoretical Black–Scholes prices using live market data, then backtests a simple mispricing strategy on the resulting signal.

The script builds option price and implied volatility surfaces, measures pricing error between market and model, and runs a Monte Carlo backtest of a long-short portfolio formed from the mispricing signal.

## What the project does

- Downloads AAPL option chain data using yfinance
- Builds an implied volatility surface across strikes and maturities
- Prices options using the Black–Scholes model
- Computes a vega-normalised mispricing signal (market − model) for every contract
- Forms a long-short portfolio: long the cheapest decile, short the richest decile
- Backtests the portfolio with Monte Carlo paths on the underlying
- Visualises mispricing, model prices, volatility surfaces, equity curve, and PnL distribution

## Key metrics

- **Sharpe ratio** measures risk-adjusted return of the strategy equity curve
- **Max drawdown** measures the largest peak-to-trough loss along the equity curve
- **Hit rate** measures the fraction of MC paths ending with positive PnL
- **5% VaR** and **95% PnL** percentiles describe the tails of the terminal PnL distribution
