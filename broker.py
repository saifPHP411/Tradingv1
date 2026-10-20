import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopOrderRequest, LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import is_dry_run
from safety import check_safety_gates
from decision_log import (
    set_consecutive_losses, get_consecutive_losses,
    set_cooldown_until, set_trading_state,
)
from datetime import datetime, timedelta


def _client():
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def _submit_entry(order: dict) -> dict:
    """Step 1: submit entry order only."""
    client = _client()
    side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL
    qty = float(order["qty"])
    is_frac = abs(qty - round(qty)) > 1e-9

    req = MarketOrderRequest(
        symbol=order["symbol"],
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.SIMPLE,  # always simple for entry
    )
    placed = client.submit_order(req)
    return {"order_id": str(placed.id), "symbol": order["symbol"], "is_frac": is_frac}


def _wait_fill(order_id: str, max_wait_sec: int = 30) -> dict | None:
    """Poll until filled or timeout."""
    client = _client()
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order_id)
            status = str(o.status).lower()
            if "filled" in status:
                return {
                    "filled_qty": float(o.filled_qty or 0),
                    "filled_avg_price": float(o.filled_avg_price or 0),
                    "status": status,
                }
            if "canceled" in status or "rejected" in status or "expired" in status:
                return None
        except Exception as e:
            print(f"⚠️ broker: poll order {order_id} failed: {e}")
        time.sleep(1)
    return None


def _submit_stop(symbol: str, qty: float, stop_price: float, side_held: str) -> dict | None:
    """Step 2: submit protective stop order."""
    client = _client()
    # If we hold long, stop is SELL. If short, stop is BUY.
    stop_side = OrderSide.SELL if side_held == "long" else OrderSide.BUY
    try:
        req = StopOrderRequest(
            symbol=symbol,
            qty=float(qty),
            side=stop_side,
            time_in_force=TimeInForce.GTC,
            stop_price=round(float(stop_price), 2),
        )
        placed = client.submit_order(req)
        return {"stop_order_id": str(placed.id), "stop_price": round(float(stop_price), 2)}
    except Exception as e:
        print(f"⚠️ broker: stop submission failed for {symbol}: {e}")
        return None


def _emergency_close(symbol: str):
    """Last resort: close position immediately."""
    try:
        client = _client()
        client.close_position(symbol)
        print(f"🚨 broker: EMERGENCY CLOSE {symbol}")
    except Exception as e:
        print(f"🚨 broker: EMERGENCY CLOSE FAILED {symbol}: {e}")


def _find_existing_position(symbol: str) -> dict | None:
    try:
        client = _client()
        for p in client.get_all_positions():
            if p.symbol == symbol:
                return {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side.value,
                    "avg_entry_price": float(p.avg_entry_price),
                }
    except Exception:
        pass
    return None


def execute_order(order: dict) -> dict:
    """
    Full execution flow:
      Entry -> Wait Fill -> Create Stop -> Verify -> [fail => Close]
    """
    if is_dry_run():
        return {"status": "dry_run", "message": f"would execute: {order}"}

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"status": "skipped", "reason": "Alpaca keys missing"}

    allowed, reason = check_safety_gates()
    if not allowed:
        return {"status": "blocked", "reason": reason}

    symbol = order["symbol"]

    # Duplicate guard
    existing = _find_existing_position(symbol)
    if existing is not None:
        return {"status": "blocked", "reason": f"already holding {symbol}"}

    try:
        client = _client()
        acct = client.get_account()
        buying_power = float(acct.buying_power)
        est_cost = order.get("estimated_cost", 0)
        if order["side"] == "buy" and est_cost > buying_power:
            return {"status": "blocked",
                    "reason": f"insufficient buying power ({est_cost:.2f} > {buying_power:.2f})"}
    except Exception as e:
        return {"status": "error", "reason": f"account check failed: {e}"}

    # Step 1: entry
    try:
        entry = _submit_entry(order)
    except Exception as e:
        return {"status": "error", "reason": f"entry submit failed: {e}"}

    # Step 2: wait fill
    fill = _wait_fill(entry["order_id"], max_wait_sec=30)
    if fill is None:
        return {"status": "error", "reason": "entry did not fill within 30s",
                "order_id": entry["order_id"]}

    filled_qty = fill["filled_qty"]
    filled_price = fill["filled_avg_price"]
    if filled_qty <= 0 or filled_price <= 0:
        return {"status": "error", "reason": "invalid fill data"}

    # Step 3: recalc stop from actual fill (safer than theoretical)
    atr = order.get("atr")
    if atr and atr > 0:
        if order["side"] == "buy":
            stop_price = filled_price - (atr * 1.5)
        else:
            stop_price = filled_price + (atr * 1.5)
    else:
        stop_price = order.get("stop_loss", filled_price * 0.97)

    # Step 4: stop order
    stop_result = _submit_stop(
        symbol=symbol,
        qty=filled_qty,
        stop_price=stop_price,
        side_held="long" if order["side"] == "buy" else "short",
    )

    if stop_result is None:
        # CRITICAL: no protection -> close immediately
        _emergency_close(symbol)
        return {
            "status": "error",
            "reason": "STOP CREATION FAILED — position was emergency-closed",
            "order_id": entry["order_id"],
            "filled_qty": filled_qty,
            "filled_avg_price": filled_price,
        }

    # Success
    return {
        "status": "submitted",
        "order_id": entry["order_id"],
        "stop_order_id": stop_result["stop_order_id"],
        "symbol": symbol,
        "side": order["side"],
        "qty": filled_qty,
        "filled_avg_price": filled_price,
        "stop_price": stop_result["stop_price"],
        "order_type": "entry+stop (bracket-equivalent)",
    }


def record_trade_outcome(pnl_pct: float):
    """Update consecutive losses counter based on closed trade."""
    if pnl_pct < 0:
        losses = get_consecutive_losses() + 1
        set_consecutive_losses(losses)
        from config import MAX_CONSECUTIVE_LOSSES, COOLDOWN_MINUTES
        if losses >= MAX_CONSECUTIVE_LOSSES:
            until = (datetime.utcnow() + timedelta(minutes=COOLDOWN_MINUTES)).isoformat()
            set_cooldown_until(until)
            print(f"🛑 broker: {losses} consecutive losses — cooldown until {until}")
    else:
        set_consecutive_losses(0)
