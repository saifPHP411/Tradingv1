import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_ohlcv(symbol: str, period: str = "6mo"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None


def compute_rsi_wilder(closes: np.ndarray, period: int = 14):
    """Proper Wilder RSI using EMA-style smoothing."""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
    """True ATR using Wilder smoothing."""
    if len(closes) < period + 1:
        return None
    tr = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    tr = np.array(tr, dtype=float)
    if len(tr) < period:
        return None
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def compute_sma(closes, period):
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def compute_ema(closes, period):
    if len(closes) < period:
        return None
    return float(pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1])


def validate_ohlcv(df) -> tuple[bool, str]:
    if df is None or len(df) < 60:
        return False, "insufficient bars"
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    if np.any(closes <= 0) or np.any(highs <= 0) or np.any(lows <= 0):
        return False, "non-positive prices"
    if np.any(highs < lows):
        return False, "high < low"
    if np.any(np.abs(np.diff(closes)) / closes[:-1] > 0.5):
        return False, "absurd price jump >50%"
    last_ts = df.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_dt = last_ts.to_pydatetime()
    else:
        last_dt = last_ts
    try:
        now = datetime.now(last_dt.tzinfo) if last_dt.tzinfo else datetime.utcnow()
        if (now - last_dt) > timedelta(days=5):
            return False, "stale data >5d"
    except Exception:
        pass
    return True, "ok"


def compute_all_indicators(symbol: str):
    """Returns complete indicator dict or None if data invalid."""
    df = fetch_ohlcv(symbol)
    ok, reason = validate_ohlcv(df)
    if not ok:
        print(f"⚠️ signals[{symbol}]: {reason}")
        return None

    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    rsi = compute_rsi_wilder(closes)
    atr = compute_atr(highs, lows, closes)
    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, 50)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)

    if any(v is None for v in [rsi, atr, sma20, sma50, ema20, ema50]):
        return None

    vol_mean = float(np.mean(volumes[-20:]))
    vol_ratio = float(volumes[-1] / vol_mean) if vol_mean > 0 else 1.0

    mom5 = float((closes[-1] / closes[-6] - 1) * 100) if len(closes) >= 6 else 0.0
    mom10 = float((closes[-1] / closes[-11] - 1) * 100) if len(closes) >= 11 else 0.0

    return {
        "symbol": symbol,
        "last_close": float(closes[-1]),
        "atr14": round(float(atr), 4),
        "rsi14": round(float(rsi), 2),
        "sma20": round(float(sma20), 2),
        "sma50": round(float(sma50), 2),
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "volume_ratio": round(float(vol_ratio), 2),
        "momentum_5d_pct": round(mom5, 2),
        "momentum_10d_pct": round(mom10, 2),
        "high_20d": round(float(np.max(highs[-20:])), 2),
        "low_20d": round(float(np.min(lows[-20:])), 2),
        "bars": int(len(closes)),
    }


def compute_signal_score(ind: dict) -> dict:
    """0-100 technical score. Phase 3 will expand."""
    trend = 0
    if ind["ema20"] > ind["ema50"]:
        trend += 15
    if ind["last_close"] > ind["ema20"]:
        trend += 10
    if ind["last_close"] > ind["sma50"]:
        trend += 5

    momentum = 0
    if ind["momentum_5d_pct"] > 0:
        momentum += 12
    if ind["momentum_10d_pct"] > 0:
        momentum += 8
    if ind["momentum_5d_pct"] > 2:
        momentum += 5

    rsi = ind["rsi14"]
    if 40 <= rsi <= 70:
        rsi_score = 25
    elif 30 <= rsi < 40:
        rsi_score = 15
    elif 70 < rsi <= 80:
        rsi_score = 10
    else:
        rsi_score = 0

    if ind["volume_ratio"] > 1.2:
        vol = 20
    elif ind["volume_ratio"] > 0.8:
        vol = 12
    else:
        vol = 5

    return {
        "score": trend + momentum + rsi_score + vol,
        "trend": trend,
        "momentum": momentum,
        "rsi": rsi_score,
        "volume": vol,
  }
