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
    """Fetch positions, ensure SL/TP exist in state, verify broker protection."""
    try:
        client = _alpaca()
        positions = client.get_all_positions()
        open_orders = client.get_orders(
            __import__("alpaca.trading.requests", fromlist=["GetOrdersRequest"]).GetOrdersRequest(
                status=__import__("alpaca.trading.enums", fromlist=["QueryOrderStatus"]).QueryOrderStatus.OPEN,
                limit=200,
            )
        )
    except Exception as e:
        print(f"⚠️ monitor: fetch failed: {e}")
        return []

    settings = get_all_settings()
    result = []
    live_symbols = set()

    # Map stop orders by symbol
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
            state = get_position_state(sym)
            print(f"📌 monitor: new position {sym} SL={stop:.2f} TP={target:.2f}")

        # CRITICAL: verify broker-side protection
        broker_stops = stops_by_symbol.get(sym, [])
        if not broker_stops:
            print(f"🚨 monitor: {sym} has NO stop order at broker — emergency close")
            try:
                _alpaca().close_position(sym)
            except Exception as e:
                print(f"🚨 monitor: emergency close failed {sym}: {e}")
            log_exit_decision(sym, "EMERGENCY_NO_STOP",
                              "broker had no stop order",
                              pnl_pct, current, entry)
            delete_position_state(sym)
            continue

        high = max(state.get("highest_price") or entry, current, entry)
        low = min(state.get("lowest_price") or entry, current, entry)
        update_position_state(sym, highest_price=high, lowest_price=low,
                              qty=qty, entry_price=entry, side=side)

        result.append({
            "symbol": sym, "entry": entry, "qty": qty, "side": side,
            "current": current, "pnl_pct": pnl_pct,
            "high": high, "low": low,
            "state": get_position_state(sym),
            "has_broker_stop": True,
        })

    for s in all_position_states():
        if s["symbol"] not in live_symbols:
            delete_position_state(s["symbol"])
            print(f"🧹 monitor: cleaned {s['symbol']}")

    return result


def _hard_gate(pos: dict, settings: dict):
    """Deterministic safety gate."""
    pnl = pos["pnl_pct"]
    if pnl <= -settings["hard_stop_loss_pct"]:
        return ("CLOSE_NOW", f"hard stop {pnl:.2f}%")
    if pnl >= settings["hard_take_profit_pct"]:
        return ("CLOSE_NOW", f"hard target {pnl:.2f}%")

    if pnl >= settings["trailing_activation_pct"]:
        trail = pos["high"] * (1 - settings["trailing_stop_pct"] / 100)
        if pos["current"] <= trail:
            return ("CLOSE_NOW",
                    f"trailing {pos['current']:.2f}<= {trail:.2f} (high {pos['high']:.2f})")

    dyn_stop = pos["state"].get("current_stop")
    if dyn_stop and pos["current"] <= dyn_stop:
        return ("CLOSE_NOW", f"dynamic stop {pos['current']:.2f}<= {dyn_stop:.2f}")
    return None


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

    prompt = f"""أنت مدير خروج. قرارك: HOLD | CLOSE_NOW | TIGHTEN_STOP.

المركز:
- {pos['symbol']} | entry {pos['entry']:.2f} | now {pos['current']:.2f}
- P&L: {pos['pnl_pct']:+.2f}%
- highest: {pos['high']:.2f} | lowest: {pos['low']:.2f}
- hold: {hold_min:.0f} min
- stop: {pos['state'].get('current_stop') or 'n/a'}
- target: {pos['state'].get('current_target') or 'n/a'}

Memory:
{memory}

قواعد صارمة (لا تتجاوزها):
1. إذا hold < {min_hold} min AND loss < {min_loss}% → HOLD (ضوضاء).
2. إذا loss >= {min_loss}% AND السعر أدنى من lowest + (highest-lowest)*0.5 → CLOSE_NOW.
3. إذا gain >= {min_gain}% AND السعر تراجع > 40% من أعلى ربح → TIGHTEN_STOP.
4. غير ذلك → HOLD.

أجب JSON فقط:
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
        # Cancel all open orders on symbol first
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

        # Update consecutive loss counter
        try:
            from broker import record_trade_outcome
            record_trade_outcome(pos["pnl_pct"])
        except Exception as e:
            print(f"⚠️ monitor: could not record trade outcome: {e}")

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
            gate = _hard_gate(pos, settings)
            if gate:
                action, reason = gate
                if not market_ok:
                    print(f"⏸️ monitor: {pos['symbol']} gate hit but market closed")
                    continue
                _execute_close(pos, f"[GATE] {reason}")
                continue

            if not market_ok:
                continue

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
                        update_position_state(pos["symbol"], current_stop=new_stop)
                        print(f"🔒 monitor: raised stop {pos['symbol']} -> {new_stop:.2f}")
                except Exception:
                    pass


monitor = PositionMonitor()
