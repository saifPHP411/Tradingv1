import numpy as np
import yfinance as yf
from signals import compute_sma, compute_atr, compute_rsi_wilder


def _fetch_spy():
    try:
        df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 200:
            return None
        return df
    except Exception:
        return None


def get_market_regime() -> dict:
    """
    Returns:
      {
        "regime": "BULL" | "BEAR" | "SIDEWAYS" | "HIGH_VOLATILITY" | "UNKNOWN",
        "spy_price": float,
        "spy_sma50": float,
        "spy_sma200": float,
        "atr_pct": float,       # ATR14 / price * 100
        "rsi": float,
        "size_multiplier": float  # 1.0 normal, 0.5 in high vol, 0.0 in extreme
      }
    """
    df = _fetch_spy()
    if df is None:
        return _unknown()

    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)

    price = float(closes[-1])
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, 200)
    atr = compute_atr(highs, lows, closes)
    rsi = compute_rsi_wilder(closes)

    if sma50 is None or sma200 is None or atr is None or rsi is None:
        return _unknown()

    atr_pct = (atr / price) * 100 if price > 0 else 0

    # ---- Classification ----
    # High volatility takes precedence
    if atr_pct >= 3.0:
        regime = "HIGH_VOLATILITY"
        size_mult = 0.4
    elif atr_pct >= 2.2 and rsi < 40:
        regime = "HIGH_VOLATILITY"
        size_mult = 0.5
    # Strong trend
    elif price > sma50 and sma50 > sma200 and rsi > 50:
        regime = "BULL"
        size_mult = 1.0
    elif price < sma50 and sma50 < sma200:
        regime = "BEAR"
        size_mult = 0.0  # no new longs in bear
    else:
        regime = "SIDEWAYS"
        size_mult = 0.7

    return {
        "regime": regime,
        "spy_price": round(price, 2),
        "spy_sma50": round(float(sma50), 2),
        "spy_sma200": round(float(sma200), 2),
        "atr_pct": round(atr_pct, 2),
        "rsi": round(float(rsi), 2),
        "size_multiplier": size_mult,
    }


def _unknown() -> dict:
    return {
        "regime": "UNKNOWN",
        "spy_price": 0.0,
        "spy_sma50": 0.0,
        "spy_sma200": 0.0,
        "atr_pct": 0.0,
        "rsi": 0.0,
        "size_multiplier": 0.5,  # be conservative when unknown
    }
