from datetime import datetime, time, timedelta
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


def is_market_open() -> bool:
    if not ONLY_MARKET_HOURS:
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    return time(9, 30) <= t <= time(16, 0)


def get_account_snapshot() -> dict:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {
            "equity": ACCOUNT_EQUITY_DEFAULT,
            "last_equity": ACCOUNT_EQUITY_DEFAULT,
            "buying_power": ACCOUNT_EQUITY_DEFAULT,
            "open_positions": 0,
            "trades_today": 0,
        }
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        acct = client.get_account()
        positions = client.get_all_positions()
        today_start = datetime.now(ZoneInfo("UTC")).replace(hour=0, minute=0, second=0, microsecond=0)
        today_orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200, after=today_start)
        )
        filled_today = [o for o in today_orders if o.filled_at is not None]
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "buying_power": float(acct.buying_power),
            "open_positions": len(positions),
            "trades_today": len(filled_today),
        }
    except Exception as e:
        print(f"⚠️ safety: get_account_snapshot failed: {e}")
        return {
            "equity": ACCOUNT_EQUITY_DEFAULT,
            "last_equity": ACCOUNT_EQUITY_DEFAULT,
            "buying_power": ACCOUNT_EQUITY_DEFAULT,
            "open_positions": 0,
            "trades_today": 0,
        }


def check_safety_gates() -> tuple[bool, str]:
    if not is_market_open():
        return False, "market closed"

    now_iso = datetime.utcnow().isoformat()
    cd = get_cooldown_until()
    if cd:
        try:
            if datetime.fromisoformat(cd) > datetime.utcnow():
                return False, f"cooldown until {cd}"
        except Exception:
            pass

    snap = get_account_snapshot()

    # Initialize day_start_equity lazily
    dse = get_day_start_equity()
    if dse is None:
        set_day_start_equity(snap["equity"])
        dse = snap["equity"]

    if dse > 0:
        daily_pnl_pct = (snap["equity"] - dse) / dse
        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
            return False, f"daily loss limit hit ({daily_pnl_pct:.2%})"

    if snap["open_positions"] >= MAX_OPEN_POSITIONS:
        return False, f"max open positions ({MAX_OPEN_POSITIONS})"

    if snap["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, f"max trades per day ({MAX_TRADES_PER_DAY})"

    if get_consecutive_losses() >= 3:
        return False, "consecutive losses lock active"

    return True, "ok"
