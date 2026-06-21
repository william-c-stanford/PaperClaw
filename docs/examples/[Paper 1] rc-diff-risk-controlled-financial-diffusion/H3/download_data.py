"""Download S&P 500 ticker daily OHLCV from Yahoo Finance 2010-2024."""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

OUT = "prices.pkl"

with open("tickers.txt") as f:
    tickers = f.read().split()
print(f"Will fetch {len(tickers)} tickers")

START = "2010-01-01"
END = "2024-12-31"

# Use the batch download API
print("Batch downloading...")
df = yf.download(
    tickers,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
    threads=True,
    group_by="ticker",
)
print("Shape:", df.shape)
print("Columns sample:", list(df.columns)[:6])

# Build a DataFrame of adjusted closes
closes = {}
for tk in tickers:
    try:
        if (tk, "Close") in df.columns:
            s = df[(tk, "Close")].dropna()
            if len(s) > 2500:  # require enough data
                closes[tk] = s
    except Exception as e:
        print(f"skip {tk}: {e}")

print(f"Got {len(closes)} usable tickers")
close_df = pd.DataFrame(closes).dropna(how="all")
print("Combined close shape:", close_df.shape)
print("Date range:", close_df.index.min(), "->", close_df.index.max())

# Keep tickers with at least 90% coverage
coverage = close_df.notna().mean(axis=0)
keep = coverage[coverage > 0.9].index.tolist()
close_df = close_df[keep].ffill().dropna(how="all")
print(f"After 90% coverage filter: {close_df.shape}")
close_df.to_pickle(OUT)
print(f"Wrote {OUT}")
