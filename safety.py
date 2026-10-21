from datetime import datetime, time, date
from zoneinfo import ZoneInfo

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER,
    ONLY_MARKET_HOURS,
)
from settings import get_all_settings
from decision_log import (
    get_day_start_equity, set_day_start_equity,
    get_consecutive_losses, get_cooldown_until,
)

US_MARKET_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}


def is_market_holiday(dt: date | None = None) -> bool:
    if dt is None:
        dt = datetime.now(ZoneInfo("America/New_York")).date()
    return dt in US_MARKET_HOLIDAYS


def is_market_open() -> bool:
    if not ONLY_MARKET_HOURS:
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5 or is_market_holiday(now.date()):
        return False
    t = now.time()
    return time(9, 30) <= t <= time(16, 0)


def get_account_snapshot() -> dict:
    """
    FAIL-CLOSED:
      - If API keys missing or API fails → available=False.
      - Never return a fake equity that could authorize a trade.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"available": False, "reason": "missing_alpaca_credentials"}

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        acct = client.get_account()
        positions = client.get_all_positions()

        today_start = datetime.now(ZoneInfo("UTC")).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=200, after=today_start
        ))

        # Count only ENTRY fills (BUY side of a SIMPLE order) as "trades"
        entries_today = [
            o for o in today_orders
            if o.filled_at is not None
            and str(o.side).lower().endswith("buy")
            and str(o.order_class).lower().endswith("simple")
        ]

        return {
            "available": True,
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "buying_power": float(acct.buying_power),
            "open_positions": len(positions),
            "trades_today": len(entries_today),
        }
    except Exception as e:
        print(f"⚠️ safety: snapshot failed: {e}")
        return {"available": False, "reason": f"api_error:{e}"}

def check_safety_gates() -> tuple[bool, str]:
    from decision_log import is_kill_switch_active

    # 1) Kill switch FIRST — highest priority
    active, reason = is_kill_switch_active()
    if active:
        return False, f"kill_switch_active: {reason or 'manual'}"

    # 2) Market hours
    if not is_market_open():
        return False, "market closed"

    # 3) Cooldown
    cd = get_cooldown_until()
    if cd:
        try:
            if datetime.fromisoformat(cd) > datetime.now(ZoneInfo("UTC")):
                return False, f"cooldown until {cd}"
        except Exception:
            pass

    # 4) Account data (FAIL-CLOSED)
    snap = get_account_snapshot()
    if not snap.get("available"):
        return False, f"account data unavailable ({snap.get('reason')})"

    settings = get_all_settings()

    # 5) Daily loss
    dse = get_day_start_equity()
    if dse is None:
        set_day_start_equity(snap["equity"])
        dse = snap["equity"]

    if dse > 0:
        daily_pnl_pct = (snap["equity"] - dse) / dse
        if daily_pnl_pct <= -settings["max_daily_loss_pct"]:
            return False, f"daily loss limit ({daily_pnl_pct:.2%})"

    # 6) Max open positions
    if snap["open_positions"] >= settings["max_open_positions"]:
        return False, f"max open positions ({settings['max_open_positions']})"

    # 7) Max trades per day
    if snap["trades_today"] >= settings["max_trades_per_day"]:
        return False, f"max trades per day ({settings['max_trades_per_day']})"

    # 8) Consecutive losses
    if get_consecutive_losses() >= settings["max_consecutive_losses"]:
        return False, "consecutive losses lock"

    return True, "ok"
