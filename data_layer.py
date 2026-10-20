import yfinance as yf
import finnhub
import pandas as pd
import numpy as np
import requests
from config import FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY

def fetch_yfinance(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="3mo")
    if hist.empty:
        return {}
    closes = hist["Close"].values
    volumes = hist["Volume"].values
    return {
        "last_close": float(closes[-1]),
        "sma20": float(np.mean(closes[-20:])),
        "sma50": float(np.mean(closes[-50:])),
        "rsi14": _compute_rsi(closes[-15:]),
        "avg_volume": float(np.mean(volumes[-20:])),
        "price_change_5d_pct": float((closes[-1] / closes[-6] - 1) * 100),
    }

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
    """يجمع البيانات من كل المصادر المتاحة."""
    base = fetch_yfinance(symbol)
    if not base:
        return {"error": "لا توجد بيانات أساسية"}
    base["symbol"] = symbol
    base.update(fetch_finnhub_sentiment(symbol))
    base.update(fetch_alpha_vantage_news(symbol))
    base["sources_used"] = [k for k in ["last_close", "alpha_sentiment_avg", "news_count"] if k in base]
    return base

def _compute_rsi(prices: np.ndarray) -> float:
    deltas = np.diff(prices)
    gains = deltas[deltas > 0].sum() / 14 if (deltas > 0).any() else 0
    losses = -deltas[deltas < 0].sum() / 14 if (deltas < 0).any() else 0
    if losses == 0:
        return 100.0
    return float(100 - (100 / (1 + gains / losses)))