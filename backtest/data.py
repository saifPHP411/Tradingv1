import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(os.getenv("BACKTEST_CACHE_DIR", "backtest_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical OHLCV form shared by the live cache and fresh downloads.

    - Keep only the OHLCV columns.
    - Drop duplicated timestamps (keep first).
    - Sort ascending by timestamp.
    - Convert the index to naive UTC so bar timestamps compare consistently
      with the naive timestamps produced by walk-forward window boundaries.
      yfinance returns tz-aware (exchange-local) timestamps; leaving them
      tz-aware makes naive comparisons raise TypeError.
    - Drop rows with missing values.
    - Drop rows whose OHLC prices are non-positive or internally inconsistent
      (High < Low, or a price outside [Low, High]). Such bars would corrupt
      stop/target and gap logic.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out = out[~out.index.duplicated(keep="first")]
    out = out.sort_index()

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)

    out = out.dropna()

    o, h, l, c = (out[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    valid = (
        (o > 0) & (h > 0) & (l > 0) & (c > 0)
        & (h >= l)
        & (h >= o) & (h >= c)
        & (l <= o) & (l <= c)
    )
    out = out[valid]
    return out


def load_ohlcv(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """
    Load OHLCV with caching. Returns a DataFrame indexed by naive UTC timestamps.
    Columns: Open, High, Low, Close, Volume

    Data is normalized and validated on BOTH the cache-read path and the
    download path, so a cache written by an older version cannot reintroduce
    timezone/ordering/duplicate-timestamp issues.
    """
    cache_key = f"{symbol}_{start}_{end}_{interval}.parquet"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            normalized = _normalize(cached)
            if not normalized.empty:
                return normalized
        except Exception:
            pass

    try:
        df = yf.Ticker(symbol).history(
            start=start, end=end, interval=interval, auto_adjust=True
        )
        if df is None or df.empty:
            return None

        df = _normalize(df)
        if df.empty:
            print(f"⚠️ data[{symbol}]: no valid OHLCV rows after validation")
            return None

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
