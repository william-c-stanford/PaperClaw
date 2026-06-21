"""Download real S&P 500 / SPY data via yfinance and build rolling-window log-return tensors."""
import os, time, json, numpy as np, pandas as pd, yfinance as yf
import pickle

OUT = os.path.dirname(os.path.abspath(__file__))

# A representative cross-section of S&P 500 large-cap constituents
# (~40 liquid names spanning all GICS sectors)
SP500_SAMPLE = [
    # Tech
    "AAPL","MSFT","GOOGL","NVDA","META","AVGO","ORCL","ADBE","CSCO","CRM",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SPGI",
    # Healthcare
    "JNJ","UNH","PFE","MRK","ABBV","LLY","TMO",
    # Consumer
    "AMZN","WMT","HD","MCD","KO","PEP","COST","PG","NKE","SBUX",
    # Industrials/Energy/Materials
    "XOM","CVX","CAT","BA","GE","HON","UNP",
]
INDEX = ["SPY"]
ALL = SP500_SAMPLE + INDEX
START = "2010-01-01"
END   = "2024-12-31"
WIN   = 64  # daily log-return window length (computationally cheaper than 252)

def fetch():
    closes = {}
    failed = []
    for sym in ALL:
        for attempt in range(3):
            try:
                df = yf.Ticker(sym).history(start=START, end=END, auto_adjust=True)
                if df is None or len(df) < 2000:
                    raise RuntimeError(f"too short: {0 if df is None else len(df)}")
                closes[sym] = df["Close"].astype(float)
                break
            except Exception as e:
                if attempt == 2:
                    failed.append((sym, str(e)))
                else:
                    time.sleep(1.5)
    print(f"fetched {len(closes)} / {len(ALL)} tickers; failed={failed[:5]}")
    return closes, failed

def build_windows(closes):
    # Align all on common business-day index, then per-asset log-returns
    panel = pd.concat(closes, axis=1).sort_index()
    panel = panel.dropna(how="all").ffill(limit=2)
    # Log returns
    logret = np.log(panel / panel.shift(1)).iloc[1:]
    # Rolling windows per asset: (T, A) -> (N, WIN, A) too large; instead per-asset
    per_asset_windows = {}
    syms = list(logret.columns)
    for sym in syms:
        s = logret[sym].dropna().values.astype(np.float32)
        if len(s) < WIN + 50:
            continue
        # Slide step 1 (heavy) or step 5 (compact); use step 4 for speed
        idx = np.arange(0, len(s) - WIN + 1, 4)
        W = np.stack([s[i:i+WIN] for i in idx], axis=0)
        per_asset_windows[sym] = W  # (N_a, WIN)

    # Cross-asset correlation panel windows (for correlation-distance evaluation only)
    rets = logret[SP500_SAMPLE].dropna().values.astype(np.float32)  # (T, A_kept)
    A_keep = [s for s in SP500_SAMPLE if s in logret.columns]
    rets_aligned = logret[A_keep].dropna().values.astype(np.float32)

    return per_asset_windows, rets_aligned, A_keep, logret.index

def main():
    closes, failed = fetch()
    per_asset, rets_aligned, A_keep, idx = build_windows(closes)
    out_path = os.path.join(OUT, "data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dict(
            per_asset=per_asset,
            rets_aligned=rets_aligned,
            assets=A_keep,
            n_windows={k: v.shape for k, v in per_asset.items()},
            failed=failed,
            win=WIN,
        ), f)
    total = sum(v.shape[0] for v in per_asset.values())
    print(f"saved {out_path}; total windows={total}; assets={len(per_asset)}; rets_aligned={rets_aligned.shape}")

if __name__ == "__main__":
    main()
