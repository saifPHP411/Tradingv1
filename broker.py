import time
from datetime import datetime, timezone, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import is_dry_run, get_all_settings
from safety import check_safety_gates
from decision_log import (
    set_consecutive_losses, get_consecutive_losses,
    set_cooldown_until,
)
from risk_utils import normalize_price_for_symbol


def _client():
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def _submit_entry(order: dict) -> dict:
    client = _client()
    side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL
    qty = float(order["qty"])
    req = MarketOrderRequest(
        symbol=order["symbol"], qty=qty, side=side,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.SIMPLE,
    )
    placed = client.submit_order(req)
    return {"order_id": str(placed.id), "symbol": order["symbol"]}


def _wait_fill(order_id: str, max_wait_sec: int = 30) -> dict | None:
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
            print(f"⚠️ broker: poll failed: {e}")
        time.sleep(1)
    return None


def _check_partial_fill(order_id: str) -> dict | None:
    try:
        o = _client().get_order_by_id(order_id)
        qty = float(o.filled_qty or 0)
        if qty > 0:
            return {
                "filled_qty": qty,
                "filled_avg_price": float(o.filled_avg_price or 0),
                "status": str(o.status).lower(),
            }
    except Exception as e:
        print(f"⚠️ broker: partial check failed: {e}")
    return None


def _submit_stop(symbol: str, qty: float, stop_price: float, side_held: str) -> dict | None:
    client = _client()
    stop_side = OrderSide.SELL if side_held == "long" else OrderSide.BUY
    try:
        price = normalize_price_for_symbol(symbol, float(stop_price))
        req = StopOrderRequest(
            symbol=symbol, qty=float(qty), side=stop_side,
            time_in_force=TimeInForce.GTC, stop_price=price,
        )
        placed = client.submit_order(req)
        return {"stop_order_id": str(placed.id), "stop_price": price}
    except Exception as e:
        print(f"⚠️ broker: stop submission failed for {symbol}: {e}")
        return None


def _verify_stop_exists(symbol: str, expected_qty: float) -> bool:
    try:
        client = _client()
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol], limit=20
        ))
        for o in orders:
            otype = str(o.order_type).lower() if o.order_type else ""
            if "stop" in otype:
                q = float(o.qty or 0)
                if q + 1e-6 >= expected_qty:
                    return True
    except Exception as e:
        print(f"⚠️ broker: verify_stop failed: {e}")
    return False


def _emergency_close_and_verify(symbol: str) -> bool:
    """
    Close position and verify. Handles market-closed gracefully.
    Returns True only if broker confirms flat.
    """
    try:
        client = _client()

        from safety import is_market_open
        market_open = is_market_open()

        # Cancel all open orders on symbol first
        try:
            for o in client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[symbol], limit=50)):
                try:
                    client.cancel_order_by_id(o.id)
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt close (may fail outside market hours)
        close_error = None
        try:
            client.close_position(symbol)
        except Exception as e:
            close_error = str(e)

        # Verify — up to ~15s
        for _ in range(8):
            time.sleep(2)
            try:
                positions = client.get_all_positions()
                if not any(p.symbol == symbol for p in positions):
                    print(f"✅ EMERGENCY CLOSE VERIFIED: {symbol}")
                    return True
            except Exception:
                pass

        if close_error:
            print(f"🚨 CRITICAL: {symbol} close failed: {close_error} "
                  f"(market_open={market_open})")
        else:
            print(f"🚨 CRITICAL: {symbol} still open after close "
                  f"(market_open={market_open})")
        return False
    except Exception as e:
        print(f"🚨 CRITICAL: EMERGENCY CLOSE EXCEPTION: {symbol}: {e}")
        return False

def _find_existing_position(symbol: str) -> dict | None:
    try:
        for p in _client().get_all_positions():
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
    Entry → fill → stop at the EXACT price approved by risk_manager.
    Broker does NOT recalculate stop.
    """
    if is_dry_run():
        return {"status": "dry_run", "message": f"would execute: {order}"}

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"status": "skipped", "reason": "Alpaca keys missing"}

    allowed, reason = check_safety_gates()
    if not allowed:
        return {"status": "blocked", "reason": reason}

    symbol = order["symbol"]

    if _find_existing_position(symbol) is not None:
        return {"status": "blocked", "reason": f"already holding {symbol}"}

    try:
        acct = _client().get_account()
        bp = float(acct.buying_power)
        est = order.get("estimated_cost", 0)
        if order["side"] == "buy" and est > bp:
            return {"status": "blocked",
                    "reason": f"insufficient buying power ({est:.2f} > {bp:.2f})"}
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
        # Check partial fill before declaring failure
        partial = _check_partial_fill(entry["order_id"])
        if partial and partial["filled_qty"] > 0:
            # Cancel remainder, protect what filled
            try:
                _client().cancel_order_by_id(entry["order_id"])
            except Exception:
                pass
            stop = _submit_stop(
                symbol=symbol,
                qty=partial["filled_qty"],
                stop_price=order["stop_loss"],
                side_held="long" if order["side"] == "buy" else "short",
            )
            if stop is None:
                _emergency_close_and_verify(symbol)
                return {"status": "error",
                        "reason": "PARTIAL FILL — STOP FAILED — EMERGENCY CLOSED",
                        "filled_qty": partial["filled_qty"]}
            return {"status": "submitted_partial",
                    "order_id": entry["order_id"],
                    "stop_order_id": stop["stop_order_id"],
                    "filled_qty": partial["filled_qty"],
                    "filled_avg_price": partial["filled_avg_price"],
                    "stop_price": stop["stop_price"]}

        return {"status": "error",
                "reason": "entry did not fill within 30s (no partial)",
                "order_id": entry["order_id"]}

    filled_qty = fill["filled_qty"]
    filled_price = fill["filled_avg_price"]
    if filled_qty <= 0 or filled_price <= 0:
        return {"status": "error", "reason": "invalid fill data"}

    # Step 3: USE THE APPROVED STOP — no recalculation
    stop_price = order["stop_loss"]

    # Safety check for side-consistency
    if order["side"] == "buy" and stop_price >= filled_price:
        _emergency_close_and_verify(symbol)
        return {"status": "error",
                "reason": f"INVALID STOP: stop={stop_price} >= fill={filled_price}",
                "order_id": entry["order_id"]}
    if order["side"] == "sell" and stop_price <= filled_price:
        _emergency_close_and_verify(symbol)
        return {"status": "error",
                "reason": f"INVALID STOP: stop={stop_price} <= fill={filled_price}",
                "order_id": entry["order_id"]}

    # Step 4: place stop
    stop = _submit_stop(
        symbol=symbol,
        qty=filled_qty,
        stop_price=stop_price,
        side_held="long" if order["side"] == "buy" else "short",
    )

    if stop is None:
        _emergency_close_and_verify(symbol)
        return {"status": "error",
                "reason": "STOP CREATION FAILED — position emergency-closed",
                "order_id": entry["order_id"],
                "filled_qty": filled_qty,
                "filled_avg_price": filled_price}

    # Step 5: verify protection
    if not _verify_stop_exists(symbol, filled_qty):
        _emergency_close_and_verify(symbol)
        return {"status": "error",
                "reason": "STOP VERIFICATION FAILED — position emergency-closed",
                "order_id": entry["order_id"]}

    return {
        "status": "submitted",
        "order_id": entry["order_id"],
        "stop_order_id": stop["stop_order_id"],
        "symbol": symbol,
        "side": order["side"],
        "qty": filled_qty,
        "filled_avg_price": filled_price,
        "stop_price": stop_price,   # EXACT approved stop
        "order_type": "entry+stop (protected)",
    }


def move_broker_stop(symbol: str, new_stop_price: float, qty: float, side_held: str) -> dict | None:
    """Submit new stop first, then cancel old one. Never leaves position unprotected."""
    client = _client()
    new_stop = _submit_stop(symbol, qty, new_stop_price, side_held)
    if new_stop is None:
        return None
    # Cancel old stops (except the new one)
    try:
        for o in client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[symbol], limit=50)):
            if str(o.id) == new_stop["stop_order_id"]:
                continue
            otype = str(o.order_type).lower() if o.order_type else ""
            if "stop" in otype:
                try:
                    client.cancel_order_by_id(o.id)
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ move_stop cleanup: {e}")
    return new_stop


def record_trade_outcome(pnl_pct: float):
    settings = get_all_settings()
    if pnl_pct < 0:
        losses = get_consecutive_losses() + 1
        set_consecutive_losses(losses)
        if losses >= settings["max_consecutive_losses"]:
            until = (datetime.now(timezone.utc) + timedelta(
                minutes=settings["cooldown_minutes"]
            )).isoformat()
            set_cooldown_until(until)
            print(f"🛑 broker: {losses} consecutive losses — cooldown until {until}")
    else:
        set_consecutive_losses(0)
