import os
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import finnhub
except ImportError:
    finnhub = None


def get_news_sentiment(symbol: str, lookback_hours: int = 48) -> dict:
    """
    Aggregates recent news sentiment for a symbol.
    Returns:
      {
        "available": bool,
        "score": float,       # -1.0 (very negative) to +1.0 (very positive)
        "news_count": int,
        "recent_headline": str,
        "has_major_news": bool  # any news in last 24h
      }
    """
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key or finnhub is None:
        return _neutral()

    try:
        client = finnhub.Client(api_key=api_key)
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        news = client.company_news(symbol, _from=start, to=end)
        if not news:
            return _neutral()

        # Filter to lookback window strictly
        cutoff = now - timedelta(hours=lookback_hours)
        recent = []
        for item in news:
            ts = item.get("datetime")
            if not ts:
                continue
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                continue
            if dt >= cutoff:
                recent.append(item)

        if not recent:
            return _neutral()

        # Simple keyword-based sentiment (Finnhub doesn't provide sentiment in free tier)
        positive_words = {"surge", "soar", "jump", "beat", "upgrade", "record",
                          "growth", "profit", "strong", "gain", "rally", "bullish"}
        negative_words = {"fall", "drop", "plunge", "miss", "downgrade", "loss",
                          "weak", "decline", "crash", "bearish", "lawsuit", "cut",
                          "warning", "concern"}

        scores = []
        for item in recent:
            headline = (item.get("headline") or "").lower()
            if not headline:
                continue
            pos = sum(1 for w in positive_words if w in headline)
            neg = sum(1 for w in negative_words if w in headline)
            if pos + neg == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / (pos + neg))

        if not scores:
            return _neutral()

        avg = float(np.mean(scores))
        # Determine if any news in last 24h (potential red flag before entry)
        recent_24 = any(
            (now - datetime.fromtimestamp(item["datetime"], tz=timezone.utc))
            < timedelta(hours=24)
            for item in recent if item.get("datetime")
        )

        return {
            "available": True,
            "score": round(avg, 3),
            "news_count": len(scores),
            "recent_headline": (recent[0].get("headline") or "")[:120],
            "has_major_news": recent_24,
        }
    except Exception as e:
        print(f"⚠️ news[{symbol}]: {e}")
        return _neutral()


def _neutral() -> dict:
    return {
        "available": False,
        "score": 0.0,
        "news_count": 0,
        "recent_headline": "",
        "has_major_news": False,
          }
