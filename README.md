# Black–Scholes Options Diagnostics

This project compares real AAPL option prices to theoretical Black–Scholes prices using live market data.

The script builds option price surfaces, implied volatility surfaces, and measures pricing error between the market and the model.

## What the project does

- Downloads AAPL option chain data using yfinance
- Builds an implied volatility surface across strikes and maturities
- Prices options using the Black–Scholes model
- Measures pricing error using RMSE, MAE, and bias
- Visualizes mispricing, model prices, and volatility surfaces

## Key metrics

- RMSE (root mean squared error) measures overall pricing deviation
- MAE (mean absolute error) measures average absolute pricing difference
- Bias measures whether the model tends to overprice or underprice options
