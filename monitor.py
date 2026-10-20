import threading
import time
from datetime import datetime

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
    recent_lessons,
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
    from zoneinfo import ZoneInfo
    from datetime import time as dtime
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    if dtime(9, 30) <= t <= dtime(16, 0):
        return True
    if allow_extended:
        return (dtime(4, 0) <= t < dtime(9, 30)) or (dtime(16, 0) < t <= dtime(20, 0)) or (t >= dtime(20, 0) or t < dtime(4, 0))
    return False


def _sync_positions() -> list:
    """يجلب المراكز من Alpaca ويسجل الجديدة في position_state."""
    try:
        client = _alpaca()
        positions = client.get_all_positions()
    except Exception as e:
        print(f"⚠️ monitor: تعذر جلب المراكز: {e}")
        return []

    result = []
    live_symbols = set()
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
            upsert_position_state(sym, entry, qty, side)
            state = get_position_state(sym)

        # تحديث أعلى/أدنى سعر
        high = max(state.get("highest_price") or entry, current, entry)
        low = min(state.get("lowest_price") or entry, current, entry)
        update_position_state(sym, highest_price=high, lowest_price=low,
                              qty=qty, entry_price=entry, side=side)

        result.append({
            "symbol": sym, "entry": entry, "qty": qty, "side": side,
            "current": current, "pnl_pct": pnl_pct,
            "high": high, "low": low,
            "state": get_position_state(sym),
        })

    # حذف المراكز التي أُغلقت
    for s in all_position_states():
        if s["symbol"] not in live_symbols:
            delete_position_state(s["symbol"])
            print(f"🧹 monitor: حذف حالة المركز المغلق {s['symbol']}")

    return result


def _hard_gate(pos: dict, settings: dict) -> tuple[str, str] | None:
    """بوابة الأمان الحتمية — تعمل حتى لو تعطل LLM."""
    pnl = pos["pnl_pct"]
    if pnl <= -settings["hard_stop_loss_pct"]:
        return ("CLOSE_NOW", f"hard stop: {pnl:.2f}% ≤ -{settings['hard_stop_loss_pct']}%")
    if pnl >= settings["hard_take_profit_pct"]:
        return ("CLOSE_NOW", f"hard target: {pnl:.2f}% ≥ {settings['hard_take_profit_pct']}%")

    # تريلينغ ستوب
    if pnl >= settings["trailing_activation_pct"]:
        trail_price = pos["high"] * (1 - settings["trailing_stop_pct"] / 100)
        if pos["current"] <= trail_price:
            return ("CLOSE_NOW", f"trailing stop: {pos['current']:.2f} ≤ {trail_price:.2f} (high {pos['high']:.2f})")

    # وقف الخسارة الديناميكي (إن رفعه الوكيل سابقاً)
    dyn_stop = pos["state"].get("current_stop")
    if dyn_stop and pos["current"] <= dyn_stop:
        return ("CLOSE_NOW", f"dynamic stop: {pos['current']:.2f} ≤ {dyn_stop:.2f}")

    return None


def _exit_agent(pos: dict, settings: dict) -> dict:
    """وكيل الخروج — LLM يقرر HOLD / CLOSE_NOW / TIGHTEN_STOP."""
    memory_lines = [r["content"] for r in recent_lessons(limit=3)]
    memory = "\n".join(f"- {m}" for m in memory_lines) if memory_lines else "لا دروس سابقة."

    hold_hours = 0.0
    try:
        opened = datetime.fromisoformat(pos["state"]["opened_at"])
        hold_hours = (datetime.utcnow() - opened).total_seconds() / 3600
    except Exception:
        pass

    prompt = f"""أنت مدير خروج محترف. لديك مركز مفتوح وتقرر: هل نستمر أم نخرج الآن؟

المركز:
- الرمز: {pos['symbol']}
- الاتجاه: {pos['side']}
- سعر الدخول: {pos['entry']:.2f}
- السعر الحالي: {pos['current']:.2f}
- الربح/الخسارة: {pos['pnl_pct']:+.2f}%
- أعلى سعر منذ الدخول: {pos['high']:.2f}
- أدنى سعر منذ الدخول: {pos['low']:.2f}
- مدة الاحتفاظ: {hold_hours:.1f} ساعة
- وقف الخسارة الحالي: {pos['state'].get('current_stop') or 'غير محدد'}
- الهدف الحالي: {pos['state'].get('current_target') or 'غير محدد'}

سياق من الذاكرة:
{memory}

المبادئ:
1. الخسائر الصغيرة أفضل من الخسائر الكبيرة. إذا كانت الخسارة تتسع والاتجاه ضدك، اخرج.
2. لا تترك ربحاً كبيراً يتحول إلى خسارة. إذا اقترب السعر من أعلى قمة ثم بدأ يتراجع، فكر في TIGHTEN_STOP.
3. TIGHTEN_STOP يعني رفع وقف الخسارة إلى سعر أعلى (للشراء) لحماية الربح، دون إغلاق المركز.
4. HOLD فقط إذا كانت الأدلة تدعم استمرار الاتجاه.

أجب بـ JSON فقط:
{{"action":"HOLD"|"CLOSE_NOW"|"TIGHTEN_STOP","new_stop":<رقم أو null>,"confidence":<0-1>,"reason":"<سطر واحد>"}}
"""

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
            print(f"⚠️ exit agent خطأ: {e}")
    return {"action": "HOLD", "new_stop": None, "confidence": 0, "reason": "فشل الاستدعاء"}


def _execute_close(pos: dict, reason: str):
    try:
        client = _alpaca()
        # إلغاء أي أوامر مفتوحة على الرمز ثم إغلاق المركز
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[pos["symbol"]], limit=50))
            for o in open_orders:
                try:
                    client.cancel_order_by_id(o.id)
                except Exception:
                    pass
        except Exception:
            pass
        client.close_position(pos["symbol"])
        log_exit_decision(pos["symbol"], "CLOSE_NOW", reason,
                          pos["pnl_pct"], pos["current"], pos["entry"])
        print(f"🚪 monitor: إغلاق {pos['symbol']} — {reason}")
        return True
    except Exception as e:
        print(f"⚠️ monitor: فشل إغلاق {pos['symbol']}: {e}")
        log_exit_decision(pos["symbol"], "CLOSE_FAILED", str(e),
                          pos["pnl_pct"], pos["current"], pos["entry"])
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
        print("👁️ monitor: بدأ المراقب")

    def stop(self):
        if not self.is_running():
            return
        self._stop.set()
        print("⏹️ monitor: إيقاف المراقب")

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                print(f"⚠️ monitor: خطأ في الدورة: {e}")
            settings = get_all_settings()
            interval = max(15, settings["exit_check_seconds"])
            self._stop.wait(timeout=interval)

    def _tick(self):
        self._last_tick = datetime.utcnow()
        settings = get_all_settings()

        positions = _sync_positions()
        if not positions:
            return

        market_ok = _market_open(settings["allow_extended_hours"])

        for pos in positions:
            # 1) البوابة الحتمية — تعمل دائماً
            gate = _hard_gate(pos, settings)
            if gate:
                action, reason = gate
                if not market_ok:
                    print(f"⏸️ monitor: {pos['symbol']} يستحق إغلاق ({reason}) لكن السوق مغلق")
                    continue
                _execute_close(pos, f"[GATE] {reason}")
                continue

            # 2) وكيل الخروج — كل X ثوانٍ فقط
            if not market_ok:
                continue
            last = self._last_llm.get(pos["symbol"], 0)
            if time.time() - last < settings["exit_llm_interval_seconds"]:
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

            if decision["action"] == "CLOSE_NOW" and decision["confidence"] >= 0.55:
                _execute_close(pos, f"[LLM] {decision['reason']}")

            elif decision["action"] == "TIGHTEN_STOP" and decision["new_stop"]:
                try:
                    new_stop = float(decision["new_stop"])
                    # لا نرفع الوقف فوق السعر الحالي
                    if pos["side"] == "long" and new_stop < pos["current"]:
                        update_position_state(pos["symbol"], current_stop=new_stop)
                        print(f"🔒 monitor: رفع وقف {pos['symbol']} إلى {new_stop:.2f}")
                except Exception:
                    pass


monitor = PositionMonitor()
