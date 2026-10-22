import numpy as np
import pandas as pd
import math
import yfinance as yf
from datetime import datetime, timedelta, timezone


def fetch_ohlcv(symbol: str, period: str = "6mo"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None


def compute_rsi_wilder(closes: np.ndarray, period: int = 14):
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


def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])


def compute_bollinger(closes, period=20, num_std=2):
    if len(closes) < period:
        return None, None
    s = pd.Series(closes)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    last_price = closes[-1]
    u = upper.iloc[-1]
    l = lower.iloc[-1]
    if u == l or pd.isna(u) or pd.isna(l):
        return None, None
    pct_b = (last_price - l) / (u - l)
    bandwidth = ((u - l) / mid.iloc[-1]) * 100 if mid.iloc[-1] > 0 else 0
    return float(pct_b), float(bandwidth)


def compute_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2:
        return None
    plus_dm = []
    minus_dm = []
    tr_list = []
    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0)
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

    plus_dm = np.array(plus_dm, dtype=float)
    minus_dm = np.array(minus_dm, dtype=float)
    tr_arr = np.array(tr_list, dtype=float)

    def smooth(arr):
        s = np.zeros_like(arr)
        s[period - 1] = np.sum(arr[:period])
        for i in range(period, len(arr)):
            s[i] = s[i - 1] - (s[i - 1] / period) + arr[i]
        return s

    tr_s = smooth(tr_arr)
    plus_di = 100 * smooth(plus_dm) / np.where(tr_s > 0, tr_s, 1)
    minus_di = 100 * smooth(minus_dm) / np.where(tr_s > 0, tr_s, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)

    adx_arr = np.zeros_like(dx)
    adx_arr[period - 1] = np.mean(dx[:period])
    for i in range(period, len(dx)):
        adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return float(adx_arr[-1])


def compute_stochastic(highs, lows, closes, period=14, smooth=3):
    if len(closes) < period + smooth:
        return None, None
    s_h = pd.Series(highs)
    s_l = pd.Series(lows)
    s_c = pd.Series(closes)
    hh = s_h.rolling(period).max()
    ll = s_l.rolling(period).min()
    k_raw = 100 * (s_c - ll) / (hh - ll).replace(0, np.nan)
    k = k_raw.rolling(smooth).mean()
    d = k.rolling(smooth).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])


def validate_ohlcv(df) -> tuple[bool, str]:
    if df is None or len(df) < 60:
        return False, "insufficient bars"
    for col in ("Close", "High", "Low", "Volume"):
        if col not in df.columns:
            return False, f"missing column {col}"
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    if not (np.all(np.isfinite(closes)) and np.all(np.isfinite(highs))
            and np.all(np.isfinite(lows))):
        return False, "non-finite prices (NaN/Inf)"
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
        now = datetime.now(last_dt.tzinfo) if last_dt.tzinfo else datetime.now(timezone.utc).replace(tzinfo=None)
        if (now - last_dt) > timedelta(days=5):
            return False, "stale data >5d"
    except Exception:
        pass
    return True, "ok"


def compute_indicators_from_ohlcv(df) -> dict | None:
    """Pure: compute the live indicator contract from an OHLCV DataFrame.

    This is the SINGLE implementation used by both the live path
    (``data_layer.fetch_market_data``) and ``compute_all_indicators`` so the
    ordering, formulas and bar window can never drift between them. Returns
    ``None`` when the data is invalid/stale or any required indicator is not a
    finite number, so callers fail closed.
    """
    ok, reason = validate_ohlcv(df)
    if not ok:
        print(f"⚠️ signals: {reason}")
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
    macd_line, macd_sig, macd_hist = compute_macd(closes)
    pct_b, bb_bandwidth = compute_bollinger(closes)
    adx = compute_adx(highs, lows, closes)
    k, d = compute_stochastic(highs, lows, closes)

    if any(v is None for v in [rsi, atr, sma20, sma50, ema20, ema50, adx]):
        return None

    vol_mean = float(np.mean(volumes[-20:]))
    vol_ratio = float(volumes[-1] / vol_mean) if vol_mean > 0 else 1.0
    mom5 = float((closes[-1] / closes[-6] - 1) * 100) if len(closes) >= 6 else 0.0
    mom10 = float((closes[-1] / closes[-11] - 1) * 100) if len(closes) >= 11 else 0.0

    ind = {
        "last_close": float(closes[-1]),
        "atr14": round(float(atr), 4),
        "rsi14": round(float(rsi), 2),
        "sma20": round(float(sma20), 2),
        "sma50": round(float(sma50), 2),
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "macd_line": round(macd_line, 4) if macd_line is not None else 0.0,
        "macd_signal": round(macd_sig, 4) if macd_sig is not None else 0.0,
        "macd_hist": round(macd_hist, 4) if macd_hist is not None else 0.0,
        "bb_pct_b": round(pct_b, 3) if pct_b is not None else 0.5,
        "bb_bandwidth": round(bb_bandwidth, 2) if bb_bandwidth is not None else 0.0,
        "adx": round(float(adx), 2),
        "stoch_k": round(k, 2) if k is not None else 50.0,
        "stoch_d": round(d, 2) if d is not None else 50.0,
        "volume_ratio": round(float(vol_ratio), 2),
        "momentum_5d_pct": round(mom5, 2),
        "momentum_10d_pct": round(mom10, 2),
        "high_20d": round(float(np.max(highs[-20:])), 2),
        "low_20d": round(float(np.min(lows[-20:])), 2),
        "bars": int(len(closes)),
    }

    ok, reason = validate_indicators(ind)
    if not ok:
        print(f"⚠️ signals: {reason}")
        return None
    return ind


# Keys the analyst/risk agents and the signal scorer require, with the value
# each must be a finite number. Missing or non-finite ⇒ fail closed.
REQUIRED_INDICATOR_KEYS = (
    "last_close", "atr14", "rsi14", "sma20", "sma50", "ema20", "ema50",
    "macd_hist", "bb_pct_b", "bb_bandwidth", "adx", "stoch_k", "stoch_d",
    "volume_ratio", "momentum_5d_pct", "momentum_10d_pct",
)


def validate_indicators(ind) -> tuple[bool, str]:
    """Reject a missing / non-finite indicator contract."""
    if not isinstance(ind, dict):
        return False, "indicators not a mapping"
    for k in REQUIRED_INDICATOR_KEYS:
        if k not in ind:
            return False, f"missing indicator {k}"
        v = ind[k]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False, f"non-numeric indicator {k}={v!r}"
        if not math.isfinite(fv):
            return False, f"non-finite indicator {k}={v!r}"
    if ind["last_close"] <= 0:
        return False, "last_close <= 0"
    if ind["atr14"] <= 0:
        return False, "atr14 <= 0"
    return True, "ok"


def compute_all_indicators(symbol: str):
    df = fetch_ohlcv(symbol)
    return compute_indicators_from_ohlcv(df)


def compute_signal_score(ind: dict, regime: dict | None = None, news: dict | None = None) -> dict:
    """Composite 0-100 score for LONG. Regime-aware."""
    trend = 0
    if ind["ema20"] > ind["ema50"]:
        trend += 12
    if ind["last_close"] > ind["ema20"]:
        trend += 8
    if ind["last_close"] > ind["sma50"]:
        trend += 5
    if ind["adx"] > 25:
        trend += 5

    momentum = 0
    if ind["momentum_5d_pct"] > 0:
        momentum += 8
    if ind["momentum_10d_pct"] > 0:
        momentum += 5
    if ind["macd_hist"] > 0:
        momentum += 7

    rsi = ind["rsi14"]
    if 40 <= rsi <= 70:
        rsi_score = 15
    elif 30 <= rsi < 40:
        rsi_score = 10
    elif 70 < rsi <= 78:
        rsi_score = 8
    else:
        rsi_score = 0

    bb = 0
    if 0.2 <= ind["bb_pct_b"] <= 0.8:
        bb += 8
    if ind["bb_pct_b"] > 0.9:
        bb -= 5
    if ind["bb_pct_b"] < 0.1:
        bb += 3
    bb = max(0, bb)

    if ind["volume_ratio"] > 1.2:
        vol = 12
    elif ind["volume_ratio"] > 0.8:
        vol = 8
    else:
        vol = 3

    news_component = 0
    if news and news.get("available"):
        news_component = int(news["score"] * 10)

    raw = trend + momentum + rsi_score + bb + vol + news_component
    raw = max(0, min(100, raw))

    if regime:
        r = regime.get("regime", "UNKNOWN")
        if r == "BEAR":
            raw = int(raw * 0.5)
        elif r == "HIGH_VOLATILITY":
            raw = int(raw * 0.75)
        elif r == "SIDEWAYS":
            raw = int(raw * 0.9)

    return {
        "score": raw, "trend": trend, "momentum": momentum,
        "rsi": rsi_score, "bb": bb, "volume": vol, "news": news_component,
    }


def compute_short_score(ind: dict, regime: dict | None = None) -> dict:
    """
    Mirror of compute_signal_score for SHORT. Used ONLY when
    settings['enable_short'] is set. This is a symmetric extension of the
    existing indicators, not a new strategy.
    """
    trend = 0
    if ind["ema20"] < ind["ema50"]:
        trend += 12
    if ind["last_close"] < ind["ema20"]:
        trend += 8
    if ind["last_close"] < ind["sma50"]:
        trend += 5
    if ind["adx"] > 25:
        trend += 5

    momentum = 0
    if ind["momentum_5d_pct"] < 0:
        momentum += 8
    if ind["momentum_10d_pct"] < 0:
        momentum += 5
    if ind["macd_hist"] < 0:
        momentum += 7

    rsi = ind["rsi14"]
    # Avoid shorting into deep oversold — mirror of long's overbought guard.
    if 30 <= rsi <= 60:
        rsi_score = 15
    elif 60 < rsi <= 70:
        rsi_score = 8
    elif 20 <= rsi < 30:
        rsi_score = 10
    else:
        rsi_score = 0

    bb = 0
    if 0.2 <= ind["bb_pct_b"] <= 0.8:
        bb += 8
    if ind["bb_pct_b"] < 0.1:
        bb -= 5
    if ind["bb_pct_b"] > 0.9:
        bb += 3
    bb = max(0, bb)

    if ind["volume_ratio"] > 1.2:
        vol = 12
    elif ind["volume_ratio"] > 0.8:
        vol = 8
    else:
        vol = 3

    raw = trend + momentum + rsi_score + bb + vol
    raw = max(0, min(100, raw))

    if regime:
        r = regime.get("regime", "UNKNOWN")
        if r == "BULL":
            raw = int(raw * 0.5)
        elif r == "HIGH_VOLATILITY":
            raw = int(raw * 0.75)
        elif r == "SIDEWAYS":
            raw = int(raw * 0.9)

    return {
        "score": raw, "trend": trend, "momentum": momentum,
        "rsi": rsi_score, "bb": bb, "volume": vol, "news": 0,
    }
