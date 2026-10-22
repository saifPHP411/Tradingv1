import time
import uuid
from datetime import datetime, timezone, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import is_dry_run, broker_submission_allowed, get_all_settings
from safety import check_safety_gates
from decision_log import (
    set_consecutive_losses, get_consecutive_losses,
    set_cooldown_until,
)

# Safe import — emit_event may not exist yet in decision_log.
# If missing, becomes a no-op so broker never crashes on logging.
try:
    from decision_log import emit_event
except ImportError:
    def emit_event(*args, **kwargs):
        pass

# Safe import — kill switch lives in decision_log.
try:
    from decision_log import is_kill_switch_active
except ImportError:
    def is_kill_switch_active():
        return False, ""

from risk_utils import normalize_price_for_symbol, whole_share_qty


def _client():
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def _submit_entry(order: dict, client_order_id: str) -> dict:
    client = _client()
    side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL
    qty = whole_share_qty(order["qty"])
    if qty < 1:
        raise ValueError(f"refusing to submit non-whole share qty {order['qty']!r}")
    req = MarketOrderRequest(
        symbol=order["symbol"], qty=qty, side=side,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.SIMPLE,
        client_order_id=client_order_id,
    )
    placed = client.submit_order(req)
    return {"order_id": str(placed.id), "symbol": order["symbol"], "qty": qty}


def _get_order_state(order_id: str) -> dict | None:
    """Fetch the broker's authoritative view of one order.

    Returns ``None`` only when the order genuinely cannot be fetched (API
    uncertainty) — never as a stand-in for "not filled".
    """
    try:
        o = _client().get_order_by_id(order_id)
        return {
            "order_id": str(o.id),
            "status": str(o.status).lower(),
            "filled_qty": float(o.filled_qty or 0),
            "filled_avg_price": float(o.filled_avg_price or 0),
        }
    except Exception as e:
        print(f"⚠️ broker: order state fetch failed ({order_id}): {e}")
        return None


def _reconcile_order(order_id: str, max_wait_sec: int = 30) -> dict:
    """Poll an order until its final state is known, then classify it.

    A timeout is NOT proof that the order did not fill. The returned dict is
    one of:
      {"state": "filled",      "filled_qty", "filled_avg_price", "status"}
      {"state": "dead",        "status"}        terminal cancel/reject/expire
      {"state": "unresolved",  "status"}        still live/partial/unknown
    Callers must treat "unresolved" as fail-closed (an order may still fill).
    """
    client = _client()
    deadline = time.time() + max_wait_sec
    last = None
    while time.time() < deadline:
        st = _get_order_state(order_id)
        if st is None:
            return {"state": "unresolved", "status": "unknown"}
        last = st
        status = st["status"]
        if "filled" in status:
            return {"state": "filled",
                    "filled_qty": st["filled_qty"],
                    "filled_avg_price": st["filled_avg_price"],
                    "status": status}
        if any(k in status for k in ("canceled", "cancelled", "rejected", "expired")):
            return {"state": "dead", "status": status}
        time.sleep(1)

    # Timed out — fetch one final time; a late fill must not be misread as none.
    st = _get_order_state(order_id) or last
    if st is None:
        return {"state": "unresolved", "status": "unknown"}
    status = st["status"]
    if "filled" in status and st["filled_qty"] > 0:
        return {"state": "filled",
                "filled_qty": st["filled_qty"],
                "filled_avg_price": st["filled_avg_price"],
                "status": status}
    if any(k in status for k in ("canceled", "cancelled", "rejected", "expired")):
        return {"state": "dead", "status": status}
    return {"state": "unresolved", "status": status}


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
        try:
            emit_event("STOP_REJECTED", symbol=symbol, reason=str(e)[:200])
        except Exception:
            pass
        return None


def _verify_stop_exists(symbol: str, expected_qty: float,
                        expected_side: str | None = None,
                        expected_stop_price: float | None = None) -> bool:
    """Verify a working protective stop: qty, side, status and price.

    ``expected_side`` is the position side ("long"/"short"); the stop's order
    side must be the opposite. Returns False when no order satisfies ALL checks.
    """
    try:
        client = _client()
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol], limit=20
        ))
        for o in orders:
            otype = str(o.order_type).lower() if o.order_type else ""
            if "stop" not in otype:
                continue
            q = float(o.qty or 0)
            if q + 1e-6 < expected_qty:
                continue
            if expected_side:
                want = "sell" if expected_side in ("long", "buy") else "buy"
                if str(o.side).lower() != want:
                    continue
            if expected_stop_price is not None:
                sp = float(o.stop_price or 0)
                if sp <= 0 or abs(sp - float(expected_stop_price)) > 0.01:
                    continue
            return True
    except Exception as e:
        print(f"⚠️ broker: verify_stop failed: {e}")
    return False


def _emergency_close_and_verify(symbol: str) -> bool:
    """
    Close position and verify. Handles market-closed gracefully.
    Does NOT loop when market is closed (prevents log spam).
    Returns True only if broker confirms flat.
    """
    try:
        client = _client()

        # Runtime import avoids any circular at module load
        try:
            from safety import is_market_open
            market_open = is_market_open()
        except Exception:
            market_open = True

        # Cancel all open orders on symbol
        try:
            for o in client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[symbol], limit=50)):
                try:
                    client.cancel_order_by_id(o.id)
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt close
        close_error = None
        try:
            client.close_position(symbol)
            try:
                emit_event("EMERGENCY_CLOSE", symbol=symbol,
                           reason="protection_verification_failed")
            except Exception:
                pass
        except Exception as e:
            close_error = str(e)

        # Verify — up to ~16s
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


def _attach_protection(symbol: str, order: dict, entry_order_id: str,
                       filled_qty: float, filled_price: float) -> dict:
    """Place and verify the approved protective stop for a filled entry.

    Returns the final result dict. Emergency-closes and returns an error when
    the position cannot be protected. Never recalculates the stop price.
    """
    side_held = "long" if order["side"] == "buy" else "short"
    stop_price = order["stop_loss"]

    # Side consistency check — refuse to hold an unprotected position.
    if order["side"] == "buy" and stop_price >= filled_price:
        _emergency_close_and_verify(symbol)
        emit_event("STOP_REJECTED", symbol=symbol,
                   reason=f"invalid_stop: stop={stop_price} >= fill={filled_price}")
        return {"status": "error",
                "reason": f"INVALID STOP: stop={stop_price} >= fill={filled_price}",
                "order_id": entry_order_id}
    if order["side"] == "sell" and stop_price <= filled_price:
        _emergency_close_and_verify(symbol)
        emit_event("STOP_REJECTED", symbol=symbol,
                   reason=f"invalid_stop: stop={stop_price} <= fill={filled_price}")
        return {"status": "error",
                "reason": f"INVALID STOP: stop={stop_price} <= fill={filled_price}",
                "order_id": entry_order_id}

    stop = _submit_stop(symbol, filled_qty, stop_price, side_held)
    if stop is None:
        _emergency_close_and_verify(symbol)
        return {"status": "error",
                "reason": "STOP CREATION FAILED — position emergency-closed",
                "order_id": entry_order_id,
                "filled_qty": filled_qty,
                "filled_avg_price": filled_price}

    emit_event("STOP_SUBMITTED", symbol=symbol,
               stop_order_id=stop["stop_order_id"],
               stop_price=stop["stop_price"], qty=filled_qty)

    # Verify protection: qty, side, status and price — not just existence.
    if not _verify_stop_exists(symbol, filled_qty, side_held, stop["stop_price"]):
        _emergency_close_and_verify(symbol)
        emit_event("STOP_VERIFICATION_FAILED", symbol=symbol,
                   reason="stop missing/mismatched after submit")
        return {"status": "error",
                "reason": "STOP VERIFICATION FAILED — position emergency-closed",
                "order_id": entry_order_id}

    return {
        "status": "submitted",
        "order_id": entry_order_id,
        "stop_order_id": stop["stop_order_id"],
        "symbol": symbol,
        "side": order["side"],
        "qty": filled_qty,
        "filled_avg_price": filled_price,
        "stop_price": stop_price,   # exact approved stop
        "order_type": "entry+stop (protected)",
    }


def execute_order(order: dict) -> dict:
    """
    Entry → reconcile the ACTUAL fill → stop at the EXACT price approved by
    risk_manager. The broker never recalculates the stop.

    Fail-closed rules:
      - BACKTEST never submits (``is_dry_run``).
      - PAPER/LIVE may submit only if a valid protective stop can be placed.
      - A fill timeout is reconciled against the broker; a timeout is NOT proof
        that the order did not fill. If the final state cannot be confirmed the
        entry is failed closed and surfaced for operator intervention, and no
        retry is attempted (a retry could create a duplicate position).
    """
    symbol = order.get("symbol")

    if is_dry_run():
        emit_event("NO_TRADE", symbol=symbol, reason="dry_run")
        return {"status": "dry_run", "message": f"would execute: {order}"}

    if not broker_submission_allowed():
        emit_event("NO_TRADE", symbol=symbol, reason="submission_not_allowed")
        return {"status": "blocked",
                "reason": "broker submission not allowed in this mode"}

    # Never open a position we could not protect with a valid stop.
    qty = whole_share_qty(order.get("qty"))
    stop_price = order.get("stop_loss")
    if qty < 1 or not stop_price or float(stop_price) <= 0:
        emit_event("NO_TRADE", symbol=symbol,
                   reason=f"unprotectable_qty ({order.get('qty')!r})")
        return {"status": "blocked",
                "reason": f"cannot place a valid protective stop for "
                          f"qty {order.get('qty')!r}"}

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        emit_event("NO_TRADE", symbol=symbol, reason="missing_alpaca_keys")
        return {"status": "skipped", "reason": "Alpaca keys missing"}

    # Kill switch blocks new entries immediately.
    try:
        active, kreason = is_kill_switch_active()
    except Exception:
        active, kreason = False, ""
    if active:
        emit_event("SAFETY_BLOCK", symbol=symbol,
                   reason=f"kill_switch_active: {kreason or 'manual'}")
        return {"status": "blocked",
                "reason": f"kill switch active: {kreason or 'manual'}"}

    allowed, reason = check_safety_gates()
    if not allowed:
        emit_event("SAFETY_BLOCK", symbol=symbol, reason=reason)
        return {"status": "blocked", "reason": reason}

    if _find_existing_position(symbol) is not None:
        emit_event("NO_TRADE", symbol=symbol, reason="already_holding")
        return {"status": "blocked", "reason": f"already holding {symbol}"}

    # Pre-check buying power
    try:
        acct = _client().get_account()
        bp = float(acct.buying_power)
        est = order.get("estimated_cost", 0)
        if order["side"] == "buy" and est > bp:
            emit_event("NO_TRADE", symbol=symbol,
                       reason=f"insufficient_buying_power ({est:.2f} > {bp:.2f})")
            return {"status": "blocked",
                    "reason": f"insufficient buying power ({est:.2f} > {bp:.2f})"}
    except Exception as e:
        emit_event("API_ERROR", symbol=symbol, reason=f"account_check:{e}")
        return {"status": "error", "reason": f"account check failed: {e}"}

    # Step 1: entry — idempotency key so a retry cannot duplicate the order.
    client_order_id = f"tv1-{symbol}-{uuid.uuid4().hex[:16]}"
    try:
        entry = _submit_entry(order, client_order_id)
        emit_event("ORDER_SUBMITTED", symbol=symbol,
                   order_id=entry["order_id"], side=order["side"],
                   qty=entry["qty"], client_order_id=client_order_id)
    except Exception as e:
        emit_event("API_ERROR", symbol=symbol, reason=f"entry_submit:{e}")
        return {"status": "error", "reason": f"entry submit failed: {e}"}

    # Step 2: reconcile the broker's actual order state.
    rec = _reconcile_order(entry["order_id"], max_wait_sec=30)

    if rec["state"] == "unresolved":
        # The order may still fill. Cancel, then confirm the final state.
        # NO retry is attempted — that could create a duplicate position.
        try:
            _client().cancel_order_by_id(entry["order_id"])
        except Exception:
            pass
        final = _get_order_state(entry["order_id"])
        fq = float(final["filled_qty"]) if final else 0.0
        if fq > 0:
            filled_qty = fq
            filled_price = float(final["filled_avg_price"] or 0)
            emit_event("ORDER_PARTIAL_FILL", symbol=symbol,
                       order_id=entry["order_id"], filled_qty=filled_qty,
                       status=final["status"])
            if filled_price <= 0:
                _emergency_close_and_verify(symbol)
                return {"status": "error",
                        "reason": "partial fill without price — emergency-closed",
                        "order_id": entry["order_id"]}
            return _attach_protection(symbol, order, entry["order_id"],
                                      filled_qty, filled_price)
        if final is not None and any(
                k in final["status"] for k in
                ("canceled", "cancelled", "rejected", "expired")):
            emit_event("API_ERROR", symbol=symbol,
                       reason=f"entry_{final['status']}_no_fill")
            return {"status": "error",
                    "reason": f"entry order {final['status']} without fill",
                    "order_id": entry["order_id"]}
        emit_event("ORDER_UNRESOLVED", symbol=symbol,
                   order_id=entry["order_id"],
                   reason=f"final state unknown ({rec.get('status')})")
        return {"status": "unresolved",
                "reason": "entry order state could not be confirmed; "
                          "no retry attempted — operator must verify",
                "order_id": entry["order_id"]}

    if rec["state"] == "dead":
        emit_event("API_ERROR", symbol=symbol,
                   reason=f"entry_{rec.get('status')}_no_fill")
        return {"status": "error",
                "reason": f"entry order {rec.get('status')} without fill",
                "order_id": entry["order_id"]}

    filled_qty = rec["filled_qty"]
    filled_price = rec["filled_avg_price"]
    if filled_qty <= 0 or filled_price <= 0:
        emit_event("API_ERROR", symbol=symbol, reason="invalid_fill_data")
        return {"status": "error", "reason": "invalid fill data"}

    emit_event("ORDER_FILLED", symbol=symbol,
               order_id=entry["order_id"],
               qty=filled_qty, price=filled_price)

    # Steps 3-5: place and verify the approved protective stop.
    return _attach_protection(symbol, order, entry["order_id"],
                              filled_qty, filled_price)


def move_broker_stop(symbol: str, new_stop_price: float,
                     qty: float, side_held: str) -> dict | None:
    """
    Submit new stop FIRST, then cancel old one.
    Never leaves position unprotected during the move.
    """
    client = _client()
    new_stop = _submit_stop(symbol, qty, new_stop_price, side_held)
    if new_stop is None:
        return None

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

    emit_event("STOP_MOVED", symbol=symbol,
               new_stop=new_stop["stop_price"],
               stop_order_id=new_stop["stop_order_id"])
    return new_stop


def kill_switch_report() -> dict:
    """Read-only snapshot for the operator when the kill switch trips.

    Reports open positions and outstanding orders. Does NOT cancel or flatten
    anything — flattening is deliberately not automated here.
    """
    report = {"positions": [], "open_orders": [], "error": None}
    try:
        client = _client()
        for p in client.get_all_positions():
            report["positions"].append({
                "symbol": p.symbol, "qty": float(p.qty),
                "side": p.side.value,
                "unrealized_plpc": float(p.unrealized_plpc or 0),
            })
        for o in client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, limit=200)):
            report["open_orders"].append({
                "id": str(o.id), "symbol": o.symbol,
                "type": str(o.order_type), "side": str(o.side),
                "qty": float(o.qty or 0),
            })
    except Exception as e:
        report["error"] = str(e)
    return report


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
