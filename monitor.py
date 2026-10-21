import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from datetime import time as dtime

from langchain_ollama import ChatOllama
import json, re

from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER,
)
from settings import get_all_settings
from decision_log import (
    upsert_position_state, get_position_state, all_position_states,
    update_position_state, delete_position_state, log_exit_decision,
    recent_lessons, set_position_initial_stop,
    update_stop_order_id, set_trailing_active,
)
from broker import move_broker_stop
from risk_utils import normalize_price_for_symbol, validate_position_protection

_llm_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
_llm = ChatOllama(
    model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
    temperature=0.1, headers=_llm_headers, format="json",
)


def _alpaca():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def _market_open(allow_extended: bool) -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    if dtime(9, 30) <= t <= dtime(16, 0):
        return True
    if allow_extended:
        return (dtime(4, 0) <= t < dtime(9, 30)) or (dtime(16, 0) < t <= dtime(20, 0))
    return False


def _sync_positions() -> list:
    try:
        client = _alpaca()
        positions = client.get_all_positions()
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        open_orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
        )
    except Exception as e:
        print(f"⚠️ monitor: fetch failed: {e}")
        return []

    settings = get_all_settings()
    result = []
    live_symbols = set()

    stops_by_symbol = {}
    for o in open_orders:
        try:
            otype = str(o.order_type).lower() if o.order_type else ""
            if "stop" in otype:
                stops_by_symbol.setdefault(o.symbol, []).append({
                    "id": str(o.id),
                    "stop_price": float(o.stop_price) if o.stop_price else None,
                    "qty": float(o.qty) if o.qty else 0,
                    "side": str(o.side).lower() if o.side else None,
                })
        except Exception:
            continue

    for p in positions:
        sym = p.symbol
        live_symbols.add(sym)
        entry = float(p.avg_entry_price)
        qty = float(p.qty)
        side = p.side.value
        current = float(p.current_price)
        pnl_pct = float(p.unrealized_plpc) * 100

        state = get_position_state(sym)

        if not state:
            # No stored plan — reconstruct conservatively from ATR
            from signals import compute_all_indicators
            ind = compute_all_indicators(sym)
            atr = ind["atr14"] if ind else None
            if atr and atr > 0:
                stop_dist = atr * settings["stop_atr_multiplier"]
                if side == "long":
                    stop = entry - stop_dist
                    target = entry + stop_dist * settings["take_profit_rr"]
                else:
                    stop = entry + stop_dist
                    target = entry - stop_dist * settings["take_profit_rr"]
            else:
                # Last resort: percentage fallback but log it
                print(f"🚨 monitor: {sym} no ATR — cannot reconstruct protection")
                continue

            upsert_position_state(sym, entry, qty, side)
            update_position_state(
                sym,
                current_stop=normalize_price_for_symbol(sym, stop),
                current_target=normalize_price_for_symbol(sym, target),
            )
            set_position_initial_stop(sym, stop, abs(entry - stop))
            state = get_position_state(sym)
            print(f"📌 monitor: reconstructed {sym} SL={stop:.2f} TP={target:.2f}")

        # Verify broker-side stop
        broker_stops = stops_by_symbol.get(sym, [])
        position_view = {"symbol": sym, "qty": qty, "side": side}
        stop_view = broker_stops[0] if broker_stops else None
        ok, why = validate_position_protection(position_view, stop_view)
if not ok:
    from safety import is_market_open
    market_open = is_market_open()
    log_exit_decision(sym, "EMERGENCY_NO_STOP", why, pnl_pct, current, entry)

    # Try to attach a stop order first (works even outside market hours)
    try:
        from broker import _submit_stop
        from risk_utils import normalize_price_for_symbol
        state = get_position_state(sym)
        stop_price = state.get("current_stop") if state else None
        if stop_price:
            stop_result = _submit_stop(
                symbol=sym,
                qty=qty,
                stop_price=normalize_price_for_symbol(sym, float(stop_price)),
                side_held="long" if side == "long" else "short",
            )
            if stop_result:
                print(f"✅ monitor: attached stop to {sym} at {stop_price}")
                update_position_state(
                    sym, stop_order_id=stop_result["stop_order_id"],
                    current_stop=stop_result["stop_price"],
                )
                continue  # protected — carry on
    except Exception as e:
        print(f"⚠️ monitor: stop attach for {sym} failed: {e}")

    # Fall back to close only when market is open
    if market_open:
        from broker import _emergency_close_and_verify
        _emergency_close_and_verify(sym)
        delete_position_state(sym)
    else:
        print(f"⏸️ monitor: {sym} unprotected but market CLOSED — "
              f"will retry at open. Close manually in Alpaca.")
    continue

        live_stop_id = broker_stops[0]["id"]
        if state.get("stop_order_id") != live_stop_id:
            update_stop_order_id(sym, live_stop_id)

        high = max(state.get("highest_price") or entry, current, entry)
        low = min(state.get("lowest_price") or entry, current, entry)
        update_position_state(
            sym, highest_price=high, lowest_price=low,
            qty=qty, entry_price=entry, side=side,
        )

        result.append({
            "symbol": sym, "entry": entry, "qty": qty, "side": side,
            "current": current, "pnl_pct": pnl_pct,
            "high": high, "low": low,
            "state": get_position_state(sym),
        })

    for s in all_position_states():
        if s["symbol"] not in live_symbols:
            delete_position_state(s["symbol"])
            print(f"🧹 monitor: cleaned {s['symbol']}")

    return result


def _hard_gate(pos: dict):
    """
    Deterministic gate using the STORED risk plan.
    NO legacy percentages.
    """
    current = pos["current"]
    initial_stop = pos["state"].get("initial_stop")
    target = pos["state"].get("current_target")

    if initial_stop and target:
        if pos["side"] == "long":
            if current <= initial_stop:
                return ("CLOSE_NOW", f"initial stop hit ({current:.2f} <= {initial_stop:.2f})")
            if current >= target:
                return ("CLOSE_NOW", f"target hit ({current:.2f} >= {target:.2f})")
        else:
            if current >= initial_stop:
                return ("CLOSE_NOW", f"initial stop hit ({current:.2f} >= {initial_stop:.2f})")
            if current <= target:
                return ("CLOSE_NOW", f"target hit ({current:.2f} <= {target:.2f})")

    # Dynamic stop (raised by trailing)
    dyn_stop = pos["state"].get("current_stop")
    if dyn_stop:
        if pos["side"] == "long" and current <= dyn_stop:
            return ("CLOSE_NOW", f"dynamic stop ({current:.2f} <= {dyn_stop:.2f})")
        if pos["side"] == "short" and current >= dyn_stop:
            return ("CLOSE_NOW", f"dynamic stop ({current:.2f} >= {dyn_stop:.2f})")

    return None


def _trailing_update(pos: dict) -> float | None:
    """
    R-based trailing. Returns new stop price or None.
    """
    settings = get_all_settings()
    state = pos["state"]
    rps = state.get("initial_risk_per_share") or 0
    if rps <= 0:
        return None

    if pos["side"] != "long":
        return None  # short trailing handled separately if needed

    pnl_r = (pos["current"] - pos["entry"]) / rps
    if pnl_r < settings["trailing_activation_r"]:
        return None

    old_stop = state.get("current_stop") or 0
    distance_r = settings["trailing_distance_r"]
    new_stop = pos["high"] - (rps * distance_r)

    # Must move stop forward, and must not exceed current price
    if new_stop <= old_stop:
        return None
    if new_stop >= pos["current"]:
        return None

    # Minimum move
    min_move = pos["current"] * (settings["min_stop_move_pct"] / 100)
    if new_stop - old_stop < min_move:
        return None

    return normalize_price_for_symbol(pos["symbol"], new_stop)


def _exit_agent(pos: dict) -> dict:
    settings = get_all_settings()
    memory_lines = [r["content"] for r in recent_lessons(limit=3)]
    memory = "\n".join(f"- {m}" for m in memory_lines) if memory_lines else "no history"

    hold_min = 0.0
    try:
        opened = datetime.fromisoformat(pos["state"]["opened_at"])
        hold_min = (datetime.now(timezone.utc) - opened.replace(tzinfo=timezone.utc)).total_seconds() / 60
    except Exception:
        pass

    prompt = f"""أنت مدير خروج. قرارك: HOLD | CLOSE_NOW | TIGHTEN_STOP.

المركز:
- {pos['symbol']} | entry {pos['entry']:.2f} | now {pos['current']:.2f}
- P&L: {pos['pnl_pct']:+.2f}%
- initial_stop: {pos['state'].get('initial_stop')}
- current_stop: {pos['state'].get('current_stop')}
- target: {pos['state'].get('current_target')}
- hold: {hold_min:.0f} min

Memory:
{memory}

قواعد صارمة:
1. إذا hold < {settings['min_hold_minutes']} min AND loss < {settings['min_loss_to_close_pct']}% → HOLD (ضوضاء).
2. إذا loss >= {settings['min_loss_to_close_pct']}% AND اتجاه هابط → CLOSE_NOW.
3. إذا gain >= {settings['min_gain_to_trim_pct'] if 'min_gain_to_trim_pct' in settings else 2.0}% AND تراجع → TIGHTEN_STOP.
4. غير ذلك → HOLD.

JSON فقط:
{{"action":"HOLD"|"CLOSE_NOW"|"TIGHTEN_STOP","new_stop":null|number,"confidence":0-1,"reason":"سطر"}}"""

    for _ in range(2):
        try:
            resp = _llm.invoke(prompt)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group()) if m else json.loads(text)
            action = str(data.get("action", "HOLD")).upper()
            if action not in ("HOLD", "CLOSE_NOW", "TIGHTEN_STOP"):
                action = "HOLD"
            return {
                "action": action,
                "new_stop": data.get("new_stop"),
                "confidence": float(data.get("confidence", 0) or 0),
                "reason": str(data.get("reason", ""))[:200],
            }
        except Exception as e:
            print(f"⚠️ exit agent err: {e}")
    return {"action": "HOLD", "new_stop": None, "confidence": 0, "reason": "LLM failed"}


def _execute_close(pos: dict, reason: str):
    try:
        client = _alpaca()
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            for o in client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[pos["symbol"]], limit=50)):
                try:
                    client.cancel_order_by_id(o.id)
                except Exception:
                    pass
        except Exception:
            pass
        client.close_position(pos["symbol"])
        log_exit_decision(
            pos["symbol"], "CLOSE_NOW", reason,
            pos["pnl_pct"], pos["current"], pos["entry"],
        )
        try:
            from broker import record_trade_outcome
            record_trade_outcome(pos["pnl_pct"])
        except Exception as e:
            print(f"⚠️ monitor: outcome record failed: {e}")
        print(f"🚪 monitor: closed {pos['symbol']} — {reason}")
        return True
    except Exception as e:
        print(f"⚠️ monitor: close failed {pos['symbol']}: {e}")
        return False


class PositionMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._last_llm = {}
        self._last_tick = None
        self._last_error = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "last_tick": self._last_tick.isoformat() if self._last_tick else None,
            "last_error": self._last_error,
        }

    def start(self):
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="position-monitor")
        self._thread.start()
        print("👁️ monitor: started")

    def stop(self):
        if not self.is_running():
            return
        self._stop.set()
        print("⏹️ monitor: stopped")

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                print(f"⚠️ monitor tick failed: {e}")
            settings = get_all_settings()
            interval = max(15, settings["exit_check_seconds"])
            self._stop.wait(timeout=interval)

    def _tick(self):
        self._last_tick = datetime.now(timezone.utc)
        settings = get_all_settings()
        positions = _sync_positions()
        if not positions:
            return

        market_ok = _market_open(settings["allow_extended_hours"])

        for pos in positions:
            # 1) Hard gate (stored stop/target)
            gate = _hard_gate(pos)
            if gate:
                action, reason = gate
                if not market_ok:
                    print(f"⏸️ monitor: {pos['symbol']} gate hit but market closed")
                    continue
                _execute_close(pos, f"[GATE] {reason}")
                continue

            if not market_ok:
                continue

            # 2) R-based trailing
            new_stop = _trailing_update(pos)
            if new_stop is not None:
                side_held = "long" if pos["side"] == "long" else "short"
                moved = move_broker_stop(
                    pos["symbol"], new_stop, pos["qty"], side_held
                )
                if moved:
                    update_position_state(
                        pos["symbol"],
                        current_stop=new_stop,
                        stop_order_id=moved["stop_order_id"],
                    )
                    print(f"🔒 monitor: moved stop {pos['symbol']} → {new_stop:.2f}")

            # 3) LLM exit decision (throttled)
            last = self._last_llm.get(pos["symbol"], 0)
            if time.time() - last < settings["exit_llm_interval_seconds"]:
                continue

            decision = _exit_agent(pos)
            self._last_llm[pos["symbol"]] = time.time()
            update_position_state(
                pos["symbol"],
                last_exit_decision=decision["action"],
                last_exit_reason=decision["reason"],
                last_exit_at=datetime.now(timezone.utc).isoformat(),
            )
            print(f"🧠 exit[{pos['symbol']}] {decision['action']} — {decision['reason']}")

            if decision["action"] == "CLOSE_NOW" and decision["confidence"] >= settings["llm_exit_min_confidence"]:
                _execute_close(pos, f"[LLM] {decision['reason']}")
            elif decision["action"] == "TIGHTEN_STOP" and decision["new_stop"]:
                try:
                    new_stop_val = float(decision["new_stop"])
                    old_stop = pos["state"].get("current_stop") or 0
                    if pos["side"] == "long" and old_stop < new_stop_val < pos["current"]:
                        moved = move_broker_stop(
                            pos["symbol"], new_stop_val, pos["qty"], "long"
                        )
                        if moved:
                            update_position_state(
                                pos["symbol"],
                                current_stop=new_stop_val,
                                stop_order_id=moved["stop_order_id"],
                            )
                except Exception:
                    pass


monitor = PositionMonitor()
