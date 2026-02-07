# ============================================================
# AAPL Options: Black–Scholes Volatility & Mispricing Diagnostics
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ============================================================
# 1. Black–Scholes Pricing
# ============================================================
def bs_price(spot, strike, maturity, rate, vol, call=True):
    eps = 1e-12
    vol = np.maximum(vol, eps)
    sqrtT = np.sqrt(np.maximum(maturity, eps))

    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * maturity) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    if call:
        return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    else:
        return strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)

# ============================================================
# 2. Interpolation
# ============================================================
def interpolate_surface(points, values, grid):
    surface = griddata(points, values, grid, method="linear")
    if np.isnan(surface).any():
        nearest = griddata(points, values, grid, method="nearest")
        surface = np.where(np.isnan(surface), nearest, surface)
    return surface

# ============================================================
# 3. Download Data
# ============================================================
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="1d")
spot_price = hist["Close"].iloc[-1]

rows = []
for exp in ticker.options[:6]:
    chain = ticker.option_chain(exp)

    T_val = max(
        0.0,
        (pd.to_datetime(exp).normalize() - pd.Timestamp.now().normalize()).days
    ) / 365.0

    calls = chain.calls.copy()
    calls["type"] = "call"

    puts = chain.puts.copy()
    puts["type"] = "put"

    df = pd.concat([calls, puts], ignore_index=True)
    df["T"] = T_val
    rows.append(df)

opt = pd.concat(rows, ignore_index=True)

# ============================================================
# 4. Clean
# ============================================================
opt = opt.rename(columns={
    "strike": "K",
    "lastPrice": "market_price",
    "impliedVolatility": "iv"
})

opt = opt[["K", "T", "market_price", "iv", "type"]]
opt = opt.dropna(subset=["K", "T", "market_price"])
opt = opt[opt["market_price"] > 0]

# ============================================================
# 5. Grid
# ============================================================
K_min, K_max = opt["K"].min(), opt["K"].max()
T_min, T_max = opt["T"].min(), opt["T"].max()

K_grid, T_grid = np.meshgrid(
    np.linspace(K_min, K_max, 80),
    np.linspace(T_min, T_max, 60)
)

# ============================================================
# 6. Surfaces
# ============================================================
def market_surface(df, option_type):
    subset = df[df["type"] == option_type]
    points = np.column_stack([subset["K"], subset["T"]])
    values = subset["market_price"]
    return interpolate_surface(points, values, (K_grid, T_grid))

def iv_surface(df, option_type):
    subset = df[df["type"] == option_type]
    points = np.column_stack([subset["K"], subset["T"]])
    values = subset["iv"]
    return interpolate_surface(points, values, (K_grid, T_grid))

market_call = market_surface(opt, "call")
market_put  = market_surface(opt, "put")

iv_call = iv_surface(opt, "call")
iv_put  = iv_surface(opt, "put")

iv_mean = opt["iv"].dropna().mean()
iv_call = np.where(np.isnan(iv_call), iv_mean, iv_call)
iv_put  = np.where(np.isnan(iv_put), iv_mean, iv_put)

# ============================================================
# 7. Black–Scholes
# ============================================================
r = 0.04
bs_call = bs_price(spot_price, K_grid, T_grid, r, iv_call, True)
bs_put  = bs_price(spot_price, K_grid, T_grid, r, iv_put, False)

mispricing_call = market_call - bs_call
mispricing_put  = market_put - bs_put

# ============================================================
# 8. Surface Plot Helper
# ============================================================
def plot_surface(Z, title, zlabel):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(K_grid, T_grid, np.nan_to_num(Z), cmap="viridis", edgecolor="none")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

# ============================================================
# 9. Mispricing Surfaces
# ============================================================
plot_surface(mispricing_call, "CALL Mispricing Surface", "Mispricing")
plot_surface(mispricing_put,  "PUT Mispricing Surface", "Mispricing")

# ============================================================
# 10. BS Surface with Market Dots
# ============================================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection="3d")

ax.plot_surface(
    K_grid,
    T_grid,
    bs_call,
    color="lightblue",
    alpha=0.7,
    edgecolor="none",
    shade=False
)

real_calls = opt[opt["type"] == "call"]

ax.scatter(
    real_calls["K"],
    real_calls["T"],
    real_calls["market_price"],
    color="red",
    s=20,
    label="Market quotes"
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity")
ax.set_zlabel("Option Price")
ax.set_title("Black–Scholes Surface with Market CALL Prices")

ax.legend()

plt.show()


# ============================================================
# 11. Implied Volatility Surface
# ============================================================
plot_surface(iv_call, "Implied Volatility Surface (CALL)", "Implied Vol")

# ============================================================
# 12. Key Metrics
# ============================================================
calls = real_calls.copy()

calls["bs_price"] = bs_price(
    spot_price,
    calls["K"].values,
    calls["T"].values,
    r,
    calls["iv"].fillna(iv_mean).values,
    True
)

calls["error"] = calls["market_price"] - calls["bs_price"]

rmse = np.sqrt(np.mean(calls["error"]**2))
mae  = np.mean(np.abs(calls["error"]))
bias = np.mean(calls["error"])

print("\nModel Diagnostics")
print("-----------------")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Bias: {bias:.4f}")
print(f"Contracts analyzed: {len(calls)}")
