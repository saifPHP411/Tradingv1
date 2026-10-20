import threading
import time
from datetime import datetime
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
    update_position_state, delete_position_state,
    log_exit_decision, recent_lessons,
    set_position_initial_stop, update_stop_order_id, set_trailing_active,
)

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
    """Fetch positions, ensure state + broker stop exist."""
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

    # Group stop orders by symbol
    stops_by_symbol = {}
    for o in open_orders:
        try:
            otype = str(o.order_type).lower() if o.order_type else ""
            if "stop" in otype:
                stops_by_symbol.setdefault(o.symbol, []).append({
                    "id": str(o.id),
                    "stop_price": float(o.stop_price) if o.stop_price else None,
                    "qty": float(o.qty) if o.qty else 0,
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
            if side == "long":
                stop = entry * (1 - settings["hard_stop_loss_pct"] / 100)
                target = entry * (1 + settings["hard_take_profit_pct"] / 100)
            else:
                stop = entry * (1 + settings["hard_stop_loss_pct"] / 100)
                target = entry * (1 - settings["hard_take_profit_pct"] / 100)
            upsert_position_state(sym, entry, qty, side)
            update_position_state(sym, current_stop=round(stop, 2),
                                  current_target=round(target, 2))
            set_position_initial_stop(sym, stop, abs(entry - stop))
            state = get_position_state(sym)
            print(f"📌 monitor: new position {sym} SL={stop:.2f} TP={target:.2f}")

        # CRITICAL: verify broker stop exists
        broker_stops = stops_by_symbol.get(sym, [])
        if not broker_stops:
            print(f"🚨 monitor: {sym} NO broker stop — emergency close")
            try:
                _alpaca().close_position(sym)
            except Exception as e:
                print(f"🚨 monitor: close failed {sym}: {e}")
            log_exit_decision(sym, "EMERGENCY_NO_STOP",
                              "no broker stop", pnl_pct, current, entry)
            delete_position_state(sym)
            continue

        # Persist broker stop id
        live_stop_id = broker_stops[0]["id"]
        if state.get("stop_order_id") != live_stop_id:
            update_stop_order_id(sym, live_stop_id)

        high = max(state.get("highest_price") or entry, current, entry)
        low = min(state.get("lowest_price") or entry, current, entry)
        update_position_state(sym, highest_price=high, lowest_price=low,
                              qty=qty, entry_price=entry, side=side)

        result.append({
            "symbol": sym, "entry": entry, "qty": qty, "side": side,
            "current": current, "pnl_pct": pnl_pct,
            "high": high, "low": low,
            "state": get_position_state(sym),
            "broker_stops": broker_stops,
        })

    for s in all_position_states():
        if s["symbol"] not in live_symbols:
            delete_position_state(s["symbol"])
            print(f"🧹 monitor: cleaned {s['symbol']}")

    return result


def _hard_gate(pos: dict, settings: dict):
    """Deterministic safety — closes if breached."""
    pnl = pos["pnl_pct"]
    if pnl <= -settings["hard_stop_loss_pct"]:
        return ("CLOSE_NOW", f"hard stop {pnl:.2f}%")
    if pnl >= settings["hard_take_profit_pct"]:
        return ("CLOSE_NOW", f"hard target {pnl:.2f}%")

    dyn_stop = pos["state"].get("current_stop")
    if dyn_stop and pos["current"] <= dyn_stop:
        return ("CLOSE_NOW", f"price {pos['current']:.2f} <= stop {dyn_stop:.2f}")
    return None


def _trailing_update(pos: dict, settings: dict):
    """
    Phase 2: computes new trailing stop that MOVES broker stop.
    Returns new stop price if should move, else None.
    """
    if pos["side"] != "long":
        return None  # only for longs in current setup
    if pos["pnl_pct"] < settings["trailing_activation_pct"]:
        return None

    trail_price = pos["high"] * (1 - settings["trailing_stop_pct"] / 100)
    old_stop = pos["state"].get("current_stop") or 0

    # Must move UP by at least min_stop_move_pct%
    min_move = pos["current"] * (settings["min_stop_move_pct"] / 100)
    if trail_price <= old_stop + min_move:
        return None

    # Must be below current price (safety)
    if trail_price >= pos["current"]:
        return None

    return trail_price


def _exit_agent(pos: dict, settings: dict) -> dict:
    memory_lines = [r["content"] for r in recent_lessons(limit=3)]
    memory = "\n".join(f"- {m}" for m in memory_lines) if memory_lines else "no history"

    hold_min = 0.0
    try:
        opened = datetime.fromisoformat(pos["state"]["opened_at"])
        hold_min = (datetime.utcnow() - opened).total_seconds() / 60
    except Exception:
        pass

    min_hold = settings.get("min_hold_minutes", 30)
    min_loss = settings.get("min_loss_to_close_pct", 1.5)
    min_gain = settings.get("min_gain_to_trim_pct", 2.0)

    prompt = f"""مدير خروج. قرارك: HOLD | CLOSE_NOW | TIGHTEN_STOP.

المركز:
- {pos['symbol']} | entry {pos['entry']:.2f} | now {pos['current']:.2f}
- P&L: {pos['pnl_pct']:+.2f}%
- highest: {pos['high']:.2f} | lowest: {pos['low']:.2f}
- hold: {hold_min:.0f} min
- stop: {pos['state'].get('current_stop') or 'n/a'}
- target: {pos['state'].get('current_target') or 'n/a'}

Memory:
{memory}

قواعد صارمة:
1. إذا hold < {min_hold} min AND loss < {min_loss}% → HOLD (ضوضاء)
2. إذا loss >= {min_loss}% AND اتجاه هابط → CLOSE_NOW
3. إذا gain >= {min_gain}% AND تراجع > 40% من أعلى ربح → TIGHTEN_STOP
4. غير ذلك → HOLD

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
        log_exit_decision(pos["symbol"], "CLOSE_NOW", reason,
                          pos["pnl_pct"], pos["current"], pos["entry"])

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


def _move_stop_if_better(pos: dict, new_price: float):
    """Moves broker stop safely. Returns True on success."""
    try:
        from broker import move_broker_stop
        side_held = "long" if pos["side"] == "long" else "short"
        result = move_broker_stop(pos["symbol"], new_price, pos["qty"], side_held)
        if result:
            update_position_state(
                pos["symbol"],
                current_stop=round(new_price, 2),
                stop_order_id=result["stop_order_id"],
            )
            print(f"🔒 monitor: moved stop {pos['symbol']} → {new_price:.2f}")
            return True
        else:
            print(f"⚠️ monitor: stop move failed {pos['symbol']}")
            return False
    except Exception as e:
        print(f"⚠️ monitor: stop move error {pos['symbol']}: {e}")
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
            interval = max(15, settings.get("exit_check_seconds", 60))
            self._stop.wait(timeout=interval)

    def _tick(self):
        self._last_tick = datetime.utcnow()
        settings = get_all_settings()
        positions = _sync_positions()
        if not positions:
            return

        market_ok = _market_open(settings.get("allow_extended_hours", False))

        for pos in positions:
            # 1) Hard gate (deterministic — always runs)
            gate = _hard_gate(pos, settings)
            if gate:
                action, reason = gate
                if not market_ok:
                    print(f"⏸️ monitor: {pos['symbol']} gate but market closed")
                    continue
                _execute_close(pos, f"[GATE] {reason}")
                continue

            if not market_ok:
                continue

            # 2) Trailing stop update (Phase 2 — moves broker stop)
            new_trail = _trailing_update(pos, settings)
            if new_trail is not None:
                _move_stop_if_better(pos, new_trail)
                # refresh state after move
                pos["state"] = get_position_state(pos["symbol"]) or pos["state"]

            # 3) LLM exit decision (throttled)
            last = self._last_llm.get(pos["symbol"], 0)
            if time.time() - last < settings.get("exit_llm_interval_seconds", 300):
                continue

            decision = _exit_agent(pos, settings)
            self._last_llm[pos["symbol"]] = time.time()
            update_position_state(
                pos["symbol"],
                last_exit_decision=decision["action"],
                last_exit_reason=decision["reason"],
                last_exit_at=datetime.utcnow().isoformat(),
            )
            print(f"🧠 exit[{pos['symbol']}] {decision['action']} — {decision['reason']}")

            min_conf = settings.get("llm_exit_min_confidence", 0.70)
            if decision["action"] == "CLOSE_NOW" and decision["confidence"] >= min_conf:
                _execute_close(pos, f"[LLM] {decision['reason']}")
            elif decision["action"] == "TIGHTEN_STOP" and decision["new_stop"]:
                try:
                    new_stop = float(decision["new_stop"])
                    old_stop = pos["state"].get("current_stop") or 0
                    if pos["side"] == "long" and old_stop < new_stop < pos["current"]:
                        _move_stop_if_better(pos, new_stop)
                except Exception:
                    pass


monitor = PositionMonitor()
