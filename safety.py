from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER,
    ACCOUNT_EQUITY_DEFAULT, MAX_DAILY_LOSS_PCT,
    MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY,
    ONLY_MARKET_HOURS,
)
from decision_log import (
    get_day_start_equity, set_day_start_equity,
    get_consecutive_losses, get_cooldown_until,
)

# US Market Holidays (NYSE closed) — includes 2026 and 2027
US_MARKET_HOLIDAYS = {
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth observed
    date(2027, 7, 5),    # Independence Day observed
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # Christmas observed
}


def is_market_holiday(dt: date | None = None) -> bool:
    if dt is None:
        dt = datetime.now(ZoneInfo("America/New_York")).date()
    return dt in US_MARKET_HOLIDAYS


def is_market_open() -> bool:
    if not ONLY_MARKET_HOURS:
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    if is_market_holiday(now.date()):
        return False
    t = now.time()
    return time(9, 30) <= t <= time(16, 0)


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
        today_orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200, after=today_start))
        filled_today = [o for o in today_orders if o.filled_at is not None]
        return {"equity": float(acct.equity), "last_equity": float(acct.last_equity),
                "buying_power": float(acct.buying_power),
                "open_positions": len(positions), "trades_today": len(filled_today)}
    except Exception as e:
        print(f"⚠️ safety: {e}")
        return {"equity": ACCOUNT_EQUITY_DEFAULT, "last_equity": ACCOUNT_EQUITY_DEFAULT,
                "buying_power": ACCOUNT_EQUITY_DEFAULT,
                "open_positions": 0, "trades_today": 0}


def check_safety_gates() -> tuple[bool, str]:
    if not is_market_open():
        return False, "market closed"
    cd = get_cooldown_until()
    if cd:
        try:
            if datetime.fromisoformat(cd) > datetime.utcnow():
                return False, f"cooldown until {cd}"
        except Exception:
            pass
    snap = get_account_snapshot()
    dse = get_day_start_equity()
    if dse is None:
        set_day_start_equity(snap["equity"])
        dse = snap["equity"]
    if dse > 0:
        daily_pnl_pct = (snap["equity"] - dse) / dse
        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            return False, f"daily loss limit ({daily_pnl_pct:.2%})"
    if snap["open_positions"] >= MAX_OPEN_POSITIONS:
        return False, f"max open positions ({MAX_OPEN_POSITIONS})"
    if snap["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, f"max trades per day ({MAX_TRADES_PER_DAY})"
    if get_consecutive_losses() >= 3:
        return False, "consecutive losses lock"
    return True, "ok"
