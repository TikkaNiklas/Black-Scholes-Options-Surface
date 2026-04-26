"""
Options Pricing & Mispricing Backtest
=====================================

Black-Scholes pricing engine for European options, implied-volatility surface
construction across strikes and maturities, and a Monte Carlo backtest of a
cross-sectional mispricing strategy reporting PnL, Sharpe ratio, and drawdown.

Pipeline
--------
1.  Pull live option chain (yfinance) and calibrate underlying realised vol.
2.  Build IV and price surfaces over a (strike, maturity) grid.
3.  Compute a vega-normalised mispricing signal (market - model) per contract.
4.  Form a long-short portfolio: long the most-underpriced decile, short the
    most-overpriced decile, equal-weighted, gross-neutral.
5.  Simulate GBM paths of the underlying and mark the book to model at each
    daily step. PnL = mark-to-model - entry price (minus a fixed cost).
6.  Report Sharpe ratio of the average equity curve, max drawdown, hit rate,
    and 5%/95% PnL percentiles.

Note on data
------------
True historical option chain data requires paid sources (OptionMetrics, OPRA).
This implementation uses the live chain and Monte Carlo dynamics on the
underlying. The backtest module is data-source agnostic — pass any DataFrame
with columns [K, T, market_price, iv, type] into `compute_mispricing` and
`run_backtest`.


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================
TICKER = "AAPL"
RISK_FREE_RATE = 0.04
N_EXPIRIES = 6              # number of expiries to pull from chain
HISTORY_DAYS = 252          # underlying history for realised-vol estimate
N_PATHS = 2000              # MC paths in backtest
GRID_RES = (80, 60)         # (strike resolution, maturity resolution)
PORTFOLIO_DECILE = 0.10     # long bottom 10%, short top 10% by signal
TRANSACTION_COST_BPS = 5    # one-off cost on gross notional
SEED = 42

rng = np.random.default_rng(SEED)


# ============================================================
# 1.  BLACK-SCHOLES PRICING
# ============================================================
def bs_price(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    maturity: np.ndarray | float,
    rate: float,
    vol: np.ndarray | float,
    call: bool = True,
) -> np.ndarray:
    """Black-Scholes price for a European option (vectorised)."""
    eps = 1e-12
    vol = np.maximum(vol, eps)
    T = np.maximum(maturity, eps)
    sqrtT = np.sqrt(T)

    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    if call:
        return spot * norm.cdf(d1) - strike * np.exp(-rate * T) * norm.cdf(d2)
    return strike * np.exp(-rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_vega(spot, strike, maturity, rate, vol):
    """Vega per 1.0 of vol."""
    eps = 1e-12
    vol = np.maximum(vol, eps)
    T = np.maximum(maturity, eps)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return spot * norm.pdf(d1) * np.sqrt(T)


# ============================================================
# 2.  DATA LOADING
# ============================================================
def load_chain(ticker_obj: yf.Ticker, n_expiries: int) -> Tuple[pd.DataFrame, float]:
    """Pull the live option chain for the first `n_expiries` expirations."""
    spot = float(ticker_obj.history(period="1d")["Close"].iloc[-1])
    expiries = ticker_obj.options[:n_expiries]
    if not expiries:
        raise RuntimeError(f"No expiries available for {ticker_obj.ticker}")

    today = pd.Timestamp.now().normalize()
    rows = []

    for exp in expiries:
        chain = ticker_obj.option_chain(exp)
        T = max(0.0, (pd.to_datetime(exp).normalize() - today).days) / 365.0

        calls = chain.calls.assign(type="call")
        puts = chain.puts.assign(type="put")
        df = pd.concat([calls, puts], ignore_index=True)
        df["T"] = T
        df["expiry"] = exp
        rows.append(df)

    opt = pd.concat(rows, ignore_index=True)

    # Prefer bid-ask mid (yfinance lastPrice is often stale)
    bid_ask_valid = (opt["bid"] > 0) & (opt["ask"] > 0)
    mid = (opt["bid"] + opt["ask"]) / 2
    opt["market_price"] = np.where(bid_ask_valid, mid, opt["lastPrice"])

    opt = opt.rename(columns={"strike": "K", "impliedVolatility": "iv"})
    opt = opt[["K", "T", "expiry", "market_price", "iv", "type"]]
    opt = opt.dropna(subset=["K", "T", "market_price"])
    opt = opt[(opt["market_price"] > 0) & (opt["T"] > 0)]
    return opt.reset_index(drop=True), spot


def realized_volatility(prices: pd.Series, window: int = 30) -> float:
    """Annualised realised volatility from log returns."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252))


# ============================================================
# 3.  SURFACE CONSTRUCTION
# ============================================================
def build_grid(opt: pd.DataFrame, res: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """(Strike, Maturity) grid as a meshgrid pair."""
    nK, nT = res
    K = np.linspace(opt["K"].min(), opt["K"].max(), nK)
    T = np.linspace(opt["T"].min(), opt["T"].max(), nT)
    return np.meshgrid(K, T)


def interpolate_surface(points: np.ndarray, values: np.ndarray, grid) -> np.ndarray:
    """Linear interp with nearest-neighbour fallback for NaN gaps."""
    surface = griddata(points, values, grid, method="linear")
    if np.isnan(surface).any():
        nearest = griddata(points, values, grid, method="nearest")
        surface = np.where(np.isnan(surface), nearest, surface)
    return surface


def build_surface(opt: pd.DataFrame, opt_type: str, value_col: str, grid) -> np.ndarray:
    sub = opt[opt["type"] == opt_type].dropna(subset=[value_col])
    pts = np.column_stack([sub["K"], sub["T"]])
    return interpolate_surface(pts, sub[value_col].values, grid)


# ============================================================
# 4.  MISPRICING SIGNAL
# ============================================================
def compute_mispricing(opt: pd.DataFrame, spot: float, rate: float) -> pd.DataFrame:
    """Add BS price, vega-normalised mispricing, and z-score signal."""
    df = opt.copy()
    iv = df["iv"].fillna(df["iv"].median()).values
    is_call = (df["type"] == "call").values

    bs = np.where(
        is_call,
        bs_price(spot, df["K"].values, df["T"].values, rate, iv, call=True),
        bs_price(spot, df["K"].values, df["T"].values, rate, iv, call=False),
    )
    df["bs_price"] = bs
    df["mispricing"] = df["market_price"] - df["bs_price"]

    # Normalise by vega so the signal is in vol-points, not dollars —
    # otherwise long-dated, deep-ITM options dominate by sheer scale.
    df["vega"] = bs_vega(spot, df["K"].values, df["T"].values, rate, iv)
    df["signal"] = df["mispricing"] / np.maximum(df["vega"], 1e-6)

    sig = df["signal"]
    df["signal_z"] = (sig - sig.mean()) / (sig.std(ddof=0) + 1e-12)
    return df


# ============================================================
# 5.  MONTE CARLO BACKTEST
# ============================================================
@dataclass
class BacktestResult:
    equity_curve: np.ndarray   # mean equity across MC paths over time
    pnl_paths: np.ndarray      # full (n_paths, n_steps + 1) PnL matrix
    final_pnl: np.ndarray      # terminal PnL per path
    metrics: dict
    portfolio: pd.DataFrame    # the long-short book


def select_portfolio(scored: pd.DataFrame, decile: float) -> pd.DataFrame:
    """Long the bottom decile (cheap), short the top decile (rich)."""
    n = len(scored)
    k = max(1, int(np.floor(n * decile)))
    sorted_df = scored.sort_values("signal_z").reset_index(drop=True)

    longs = sorted_df.head(k).assign(side=1.0)
    shorts = sorted_df.tail(k).assign(side=-1.0)

    book = pd.concat([longs, shorts], ignore_index=True)
    book["weight"] = book["side"] / k     # equal-weight, gross = 2.0
    return book


def simulate_paths(
    spot: float, vol: float, rate: float, T_max: float, n_steps: int, n_paths: int
) -> np.ndarray:
    """GBM paths of shape (n_paths, n_steps + 1)."""
    dt = T_max / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (rate - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return spot * np.exp(log_paths)


def mark_book_to_model(
    paths: np.ndarray,
    book: pd.DataFrame,
    rate: float,
    times: np.ndarray,
) -> np.ndarray:
    """Mark every option in the book at every timestep along every MC path."""
    n_paths, n_pts = paths.shape
    portfolio_value = np.zeros((n_paths, n_pts))

    for _, row in book.iterrows():
        K = row["K"]
        T_exp = row["T"]
        iv = row["iv"] if not np.isnan(row["iv"]) else 0.3
        is_call = row["type"] == "call"
        w = row["weight"]
        entry = row["market_price"]

        # Step at which the option expires; after that it's a fixed cash payoff.
        expiry_idx = int(np.argmin(np.abs(times - T_exp)))

        # Pre-expiry: BS mark using remaining time-to-maturity
        for i in range(expiry_idx):
            tau = max(T_exp - times[i], 1e-6)
            value = bs_price(paths[:, i], K, tau, rate, iv, call=is_call)
            portfolio_value[:, i] += w * (value - entry)

        # At and after expiry: terminal payoff at S(T_exp), held in cash
        S_T = paths[:, expiry_idx]
        payoff = np.maximum(S_T - K, 0) if is_call else np.maximum(K - S_T, 0)
        portfolio_value[:, expiry_idx:] += (w * (payoff - entry))[:, None]

    return portfolio_value


def run_backtest(
    scored: pd.DataFrame,
    spot: float,
    rate: float,
    realized_vol: float,
    decile: float = PORTFOLIO_DECILE,
    n_paths: int = N_PATHS,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> BacktestResult:
    book = select_portfolio(scored, decile)
    T_max = float(scored["T"].max())
    n_steps = max(20, int(round(T_max * 252)))
    times = np.linspace(0, T_max, n_steps + 1)

    paths = simulate_paths(spot, realized_vol, rate, T_max, n_steps, n_paths)
    pnl_paths = mark_book_to_model(paths, book, rate, times)

    # One-off transaction cost on gross notional at entry.
    gross_notional = float((book["weight"].abs() * book["market_price"]).sum())
    pnl_paths -= gross_notional * cost_bps / 1e4

    equity_curve = pnl_paths.mean(axis=0)
    final_pnl = pnl_paths[:, -1]

    return BacktestResult(
        equity_curve=equity_curve,
        pnl_paths=pnl_paths,
        final_pnl=final_pnl,
        metrics=compute_metrics(equity_curve, final_pnl),
        portfolio=book,
    )


# ============================================================
# 6.  PERFORMANCE METRICS
# ============================================================
def compute_metrics(equity: np.ndarray, final_pnl: np.ndarray) -> dict:
    """Sharpe of the avg equity curve, max drawdown, hit rate, PnL tails."""
    daily_ret = np.diff(equity)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) \
        if daily_ret.std() > 0 else float("nan")

    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    return {
        "terminal_pnl_mean": float(final_pnl.mean()),
        "terminal_pnl_std": float(final_pnl.std()),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "hit_rate": float((final_pnl > 0).mean()),
        "var_5pct": float(np.percentile(final_pnl, 5)),
        "pnl_95pct": float(np.percentile(final_pnl, 95)),
    }


# ============================================================
# 7.  PLOTTING
# ============================================================
def plot_surface(K, T, Z, title, zlabel):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(K, T, np.nan_to_num(Z), cmap="viridis", edgecolor="none")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (yrs)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()


def plot_equity(equity, metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity, lw=1.5)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title(
        f"Strategy Equity Curve  |  Sharpe {metrics['sharpe_ratio']:.2f}  "
        f"Max DD {metrics['max_drawdown']:.2f}"
    )
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Cumulative PnL")
    plt.tight_layout()


def plot_pnl_dist(final_pnl, metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(final_pnl, bins=60, alpha=0.85)
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(metrics["var_5pct"], color="red", ls=":", label="5% VaR")
    ax.set_title("Terminal PnL Distribution Across MC Paths")
    ax.set_xlabel("PnL")
    ax.legend()
    plt.tight_layout()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Loading {TICKER} option chain and history...")
    ticker = yf.Ticker(TICKER)
    opt, spot = load_chain(ticker, N_EXPIRIES)
    hist = ticker.history(period=f"{HISTORY_DAYS}d")["Close"]
    rv = realized_volatility(hist)

    print(f"  Spot:               {spot:.2f}")
    print(f"  Realised vol (30d): {rv:.2%}")
    print(f"  Contracts loaded:   {len(opt)}")

    # --- Surfaces ---
    grid = build_grid(opt, GRID_RES)
    K_grid, T_grid = grid
    iv_call = build_surface(opt, "call", "iv", grid)
    bs_call_surf = bs_price(spot, K_grid, T_grid, RISK_FREE_RATE, iv_call, call=True)

    # --- Mispricing + backtest ---
    scored = compute_mispricing(opt, spot, RISK_FREE_RATE)
    result = run_backtest(scored, spot, RISK_FREE_RATE, rv)

    # --- Report ---
    print("\nPortfolio")
    print("---------")
    print(f"  Long  legs: {(result.portfolio['side'] > 0).sum()}")
    print(f"  Short legs: {(result.portfolio['side'] < 0).sum()}")

    print("\nBacktest Diagnostics")
    print("--------------------")
    for k, v in result.metrics.items():
        print(f"  {k:20s} {v:>10.4f}")

    # --- Plots ---
    plot_surface(K_grid, T_grid, iv_call, "Implied Vol Surface (Calls)", "Implied Vol")
    plot_surface(K_grid, T_grid, bs_call_surf, "Black-Scholes Call Surface", "Price")
    plot_equity(result.equity_curve, result.metrics)
    plot_pnl_dist(result.final_pnl, result.metrics)
    plt.show()


if __name__ == "__main__":
    main()
