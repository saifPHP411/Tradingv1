from datetime import datetime, time
from zoneinfo import ZoneInfo
from config import (
    MAX_DAILY_LOSS_PCT, MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY,
    ONLY_MARKET_HOURS, ACCOUNT_EQUITY_DEFAULT, ALPACA_API_KEY, ALPACA_SECRET_KEY
)

def _get_client():
    from alpaca.trading.client import TradingClient
    from config import ALPACA_PAPER
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

def is_market_open() -> bool:
    """يتحقق من ساعات السوق الأمريكي (9:30–16:00 ET، أيام العمل)."""
    if not ONLY_MARKET_HOURS:
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() <= time(16, 0)

def get_account_snapshot() -> dict:
    """يرجع رأس المال، القيمة اليومية، وعدد المراكز المفتوحة."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"equity": ACCOUNT_EQUITY_DEFAULT, "last_equity": ACCOUNT_EQUITY_DEFAULT,
                "open_positions": 0, "trades_today": 0}
    try:
        client = _get_client()
        acct = client.get_account()
        positions = client.get_all_positions()

        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        today_orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=100,
            after=datetime.now(ZoneInfo("UTC")).replace(hour=0, minute=0, second=0)
        ))
        filled_today = [o for o in today_orders if o.filled_at is not None]

        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "open_positions": len(positions),
            "trades_today": len(filled_today),
        }
    except Exception as e:
        print(f"⚠️ تعذر جلب بيانات الحساب: {e}")
        return {"equity": ACCOUNT_EQUITY_DEFAULT, "last_equity": ACCOUNT_EQUITY_DEFAULT,
                "open_positions": 0, "trades_today": 0}

def check_safety_gates() -> tuple[bool, str]:
    """يرجع (مسموح؟, سبب الرفض). كل التنفيذ يمر من هنا."""
    if not is_market_open():
        return False, "السوق مغلق حاليًا"

    snap = get_account_snapshot()

    if snap["open_positions"] >= MAX_OPEN_POSITIONS:
        return False, f"وصلنا الحد الأقصى للمراكز المفتوحة ({MAX_OPEN_POSITIONS})"

    if snap["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, f"وصلنا الحد الأقصى للصفقات اليومية ({MAX_TRADES_PER_DAY})"

    if snap["last_equity"] > 0:
        daily_pnl_pct = (snap["equity"] - snap["last_equity"]) / snap["last_equity"]
        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            return False, f"تجاوزنا حد الخسارة اليومية ({daily_pnl_pct:.2%})"

    return True, "كل الفحوصات سليمة"