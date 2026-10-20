from datetime import datetime, time
from zoneinfo import ZoneInfo
from config import ACCOUNT_EQUITY_DEFAULT, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import get_all_settings

def is_market_open() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() <= time(16, 0)

def get_account_snapshot() -> dict:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"equity": ACCOUNT_EQUITY_DEFAULT, "last_equity": ACCOUNT_EQUITY_DEFAULT,
                "buying_power": ACCOUNT_EQUITY_DEFAULT,
                "open_positions": 0, "trades_today": 0}
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        acct = client.get_account()
        positions = client.get_all_positions()
        today_start = datetime.now(ZoneInfo("UTC")).replace(hour=0, minute=0, second=0, microsecond=0)
        today_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100, after=today_start))
        filled_today = [o for o in today_orders if o.filled_at is not None]
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "buying_power": float(acct.buying_power),
            "open_positions": len(positions),
            "trades_today": len(filled_today),
        }
    except Exception as e:
        print(f"⚠️ تعذر جلب بيانات الحساب: {e}")
        return {"equity": ACCOUNT_EQUITY_DEFAULT, "last_equity": ACCOUNT_EQUITY_DEFAULT,
                "buying_power": ACCOUNT_EQUITY_DEFAULT,
                "open_positions": 0, "trades_today": 0}

def check_safety_gates() -> tuple[bool, str]:
    if not is_market_open():
        return False, "السوق مغلق حاليًا"
    settings = get_all_settings()
    snap = get_account_snapshot()
    if snap["open_positions"] >= settings["max_open_positions"]:
        return False, f"تجاوز الحد ({settings['max_open_positions']})"
    if snap["trades_today"] >= settings["max_trades_per_day"]:
        return False, f"تجاوز الصفقات اليومية ({settings['max_trades_per_day']})"
    if snap["last_equity"] > 0:
        pnl_pct = (snap["equity"] - snap["last_equity"]) / snap["last_equity"]
        if pnl_pct <= -settings["max_daily_loss_pct"]:
            return False, f"خسارة يومية ({pnl_pct:.2%})"
    return True, "سليم"
