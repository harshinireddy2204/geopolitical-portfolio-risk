import os
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf
from config import CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(tickers, start, end):
    key = "_".join(sorted(tickers)) + start + end
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"returns_{h}.parquet")

def get_returns(tickers, start, end):
    path = _cache_path(tickers, start, end)
    if os.path.exists(path):
        return pd.read_parquet(path)

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.ffill().dropna(how="all")
    returns = np.log(prices / prices.shift(1)).dropna()

    # Drop any column with more than 10% missing after differencing
    thresh = int(0.9 * len(returns))
    returns = returns.dropna(axis=1, thresh=thresh).dropna()

    returns.to_parquet(path)
    return returns