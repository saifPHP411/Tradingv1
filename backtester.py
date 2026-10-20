import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from graph import trading_graph
from state import TradingState

def fetch_historical_slice(symbol: str, as_of: datetime, lookback_days: int = 90) -> dict:
    """يجلب البيانات كما كانت في تاريخ معيّن (بدون تسرب مستقبلي)."""
    start = as_of - timedelta(days=lookback_days + 10)
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=as_of)
    if hist.empty or len(hist) < 55:
        return {}
    closes = hist["Close"].values
    volumes = hist["Volume"].values
    return {
        "symbol": symbol,
        "last_close": float(closes[-1]),
        "sma20": float(np.mean(closes[-20:])),
        "sma50": float(np.mean(closes[-50:])),
        "rsi14": _rsi(closes[-15:]),
        "avg_volume": float(np.mean(volumes[-20:])),
        "price_change_5d_pct": float((closes[-1] / closes[-6] - 1) * 100),
        "sources_used": ["yfinance_historical"],
    }

def _rsi(prices):
    d = np.diff(prices)
    g = d[d > 0].sum() / 14 if (d > 0).any() else 0
    l = -d[d < 0].sum() / 14 if (d < 0).any() else 0
    return 100.0 if l == 0 else float(100 - 100 / (1 + g / l))

def simulate_trade(entry_price: float, exit_price: float, side: str) -> float:
    """يرجع P&L% (بدون احتساب العمولات)."""
    if side == "buy":
        return (exit_price / entry_price - 1) * 100
    return (1 - exit_price / entry_price) * 100

def run_backtest(symbol: str, start: datetime, end: datetime, step_days: int = 5):
    """اختبار أمامي: يقرر كل N أيام وينتظر النتيجة على الأيام التالية."""
    print(f"\n🔬 اختبار خلفي: {symbol} من {start.date()} إلى {end.date()}")
    results = []
    current = start
    while current < end:
        data = fetch_historical_slice(symbol, current)
        if not data:
            current += timedelta(days=step_days)
            continue

        state: TradingState = {
            "symbol": symbol, "market_data": data, "memory_context": "",
            "analyst_report": None, "risk_report": None, "final_decision": None,
            "order_details": None, "broker_result": None,
            "consensus_reached": False, "logs": [],
        }
        try:
            result = trading_graph.invoke(
                state,
                config={"configurable": {"thread_id": f"bt_{symbol}_{current.date()}"}}
            )
        except Exception as e:
            print(f"  ⚠️ فشل الدورة في {current.date()}: {e}")
            current += timedelta(days=step_days)
            continue

        decision = result.get("final_decision")
        if decision and decision.startswith("EXECUTE"):
            # ننظر إلى السعر بعد أسبوعين (10 أيام تداول تقريبًا)
            future = current + timedelta(days=14)
            future_data = fetch_historical_slice(symbol, future, lookback_days=5)
            if future_data:
                pnl = simulate_trade(
                    entry_price=data["last_close"],
                    exit_price=future_data["last_close"],
                    side="buy" if decision == "EXECUTE_BUY" else "sell",
                )
                results.append({
                    "date": current.date(), "decision": decision, "pnl_pct": pnl,
                    "entry": data["last_close"], "exit": future_data["last_close"],
                })
                print(f"  {current.date()} | {decision} | P&L: {pnl:+.2f}%")
        current += timedelta(days=step_days)

    if not results:
        print("  لا صفقات في هذه الفترة.")
        return
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    print(f"\n📊 ملخص {symbol}:")
    print(f"  صفقات: {len(results)} | رابحة: {len(wins)} | خاسرة: {len(pnls) - len(wins)}")
    print(f"  متوسط P&L: {np.mean(pnls):+.2f}% | أفضل: {max(pnls):+.2f}% | أسوأ: {min(pnls):+.2f}%")
    print(f"  نسبة النجاح: {len(wins) / len(pnls):.1%}")

if __name__ == "__main__":
    from config import SYMBOLS
    for sym in SYMBOLS:
        run_backtest(
            sym,
            start=datetime(2025, 1, 1),
            end=datetime(2025, 9, 1),
            step_days=7,
        )