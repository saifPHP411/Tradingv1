import yfinance as yf
import finnhub
import pandas as pd
import numpy as np
import requests
from config import FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY


def fetch_ohlcv(symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    """Daily OHLCV bars, oldest→newest. Single source for the live path."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def fetch_finnhub_sentiment(symbol: str) -> dict:
    if not FINNHUB_API_KEY:
        return {}
    try:
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        news = client.company_news(symbol, _from="2025-01-01", to="2026-12-31")
        # نأخذ آخر 10 أخبار ونحسب متوسط المشاعر إن وُجد
        return {"news_count": len(news), "latest_headline": news[0]["headline"] if news else ""}
    except Exception as e:
        return {"error": str(e)}

def fetch_alpha_vantage_news(symbol: str) -> dict:
    """Alpha Vantage تقدم NEWS_SENTIMENT مجانيًا (بحدود)."""
    if not ALPHA_VANTAGE_API_KEY:
        return {}
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": ALPHA_VANTAGE_API_KEY, "limit": 10}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        feed = data.get("feed", [])
        if not feed:
            return {}
        sentiments = [float(item.get("overall_sentiment_score", 0)) for item in feed]
        return {"alpha_sentiment_avg": float(np.mean(sentiments)), "alpha_news_count": len(feed)}
    except Exception:
        return {}

def fetch_market_data(symbol: str) -> dict:
    """Assemble the live market_data contract for one symbol.

    Indicators come from ``signals.compute_indicators_from_ohlcv`` — the same
    implementation the analyst/risk agents expect — over one correctly ordered
    OHLCV window. Invalid, stale, short or non-finite data fails closed with an
    ``error`` key (callers must treat that as HOLD / no new entry).
    """
    from signals import compute_indicators_from_ohlcv

    df = fetch_ohlcv(symbol)
    if df is None or df.empty:
        return {"error": "no market data", "symbol": symbol}

    ind = compute_indicators_from_ohlcv(df)
    if not ind:
        return {"error": "invalid or stale market data", "symbol": symbol}

    ind["symbol"] = symbol
    ind.update(fetch_finnhub_sentiment(symbol))
    ind.update(fetch_alpha_vantage_news(symbol))
    ind["sources_used"] = [k for k in ["last_close", "alpha_sentiment_avg", "news_count"] if k in ind]
    return ind

def _compute_rsi(prices: np.ndarray) -> float:
    deltas = np.diff(prices)
    gains = deltas[deltas > 0].sum() / 14 if (deltas > 0).any() else 0
    losses = -deltas[deltas < 0].sum() / 14 if (deltas < 0).any() else 0
    if losses == 0:
        return 100.0
    return float(100 - (100 / (1 + gains / losses)))