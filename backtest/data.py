import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(os.getenv("BACKTEST_CACHE_DIR", "backtest_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_ohlcv(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """
    Load OHLCV with caching. Returns DataFrame indexed by timestamp.
    Columns: Open, High, Low, Close, Volume
    """
    cache_key = f"{symbol}_{start}_{end}_{interval}.parquet"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        df = yf.Ticker(symbol).history(
            start=start, end=end, interval=interval, auto_adjust=True
        )
        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        df = df.dropna()

        try:
            df.to_parquet(cache_path)
        except Exception:
            pass

        return df
    except Exception as e:
        print(f"⚠️ data[{symbol}]: {e}")
        return None


def load_multi(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load multiple symbols, aligned to common dates."""
    data = {}
    for sym in symbols:
        df = load_ohlcv(sym, start, end)
        if df is not None and len(df) >= 60:
            data[sym] = df
    return data


def get_spy_for_period(start: str, end: str) -> pd.DataFrame | None:
    """Load SPY data extended backward for regime warmup."""
    start_dt = datetime.fromisoformat(start)
    extended = (start_dt - timedelta(days=400)).strftime("%Y-%m-%d")
    return load_ohlcv("SPY", extended, end)
