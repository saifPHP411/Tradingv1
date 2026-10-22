from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import threading
import time
from datetime import datetime, timezone

from safety import get_account_snapshot
from settings import get_all_settings, update_settings, is_dry_run, get_symbols
from decision_log import (
    recent_decisions, equity_history, log_equity,
    recent_exit_decisions,
)
from main import run_cycle
from learner import learn_from_closed_trades
from monitor import monitor

templates = Jinja2Templates(directory="templates")


class AgentController:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._current = None
        self._started_at = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "current_symbol": self._current,
            "started_at": self._started_at.isoformat() if self._started_at else None,
        }

    def start(self) -> bool:
        with self._lock:
            if self.is_running():
                return False
            self._stop_event.clear()
            self._started_at = datetime.now(timezone.utc)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_running():
                return False
            self._stop_event.set()
            return True

    def _loop(self):
        try:
            while not self._stop_event.is_set():
                for sym in get_symbols():
                    if self._stop_event.is_set():
                        break
                    try:
                        self._current = sym
                        run_cycle(sym)
                    except Exception as e:
                        print(f"⚠️ فشل دورة {sym}: {e}")
                    time.sleep(2)

                # Snapshot equity after each full pass (fail-closed logging)
                try:
                    snap = get_account_snapshot()
                    if snap.get("available"):
                        log_equity(snap["equity"], snap["open_positions"])
                except Exception as e:
                    print(f"⚠️ equity log failed: {e}")

                try:
                    learn_from_closed_trades()
                except Exception as e:
                    print(f"⚠️ learner failed: {e}")

                interval_sec = max(1, get_all_settings()["cycle_interval_minutes"]) * 60
                self._stop_event.wait(timeout=interval_sec)
        finally:
            self._current = None
            self._started_at = None


controller = AgentController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ بدأ التطبيق")
    try:
        monitor.start()
    except Exception as e:
        print(f"⚠️ monitor start failed: {e}")
    yield
    controller.stop()
    monitor.stop()
    print("⏹️ إيقاف")


app = FastAPI(lifespan=lifespan)


# ============ Dashboard ============
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ============ Status (FAIL-CLOSED) ============
@app.get("/api/status")
def api_status():
    snap = get_account_snapshot()
    from decision_log import (
        get_protection_unverified, get_protection_unverified_reason,
    )
    base = {
        "dry_run": is_dry_run(),
        "trading_mode": get_all_settings().get("trading_mode"),
        "protection_unverified": get_protection_unverified(),
        "protection_unverified_reason": get_protection_unverified_reason(),
        **controller.status(),
    }

    if snap.get("available"):
        base.update({
            "available": True,
            "equity": snap["equity"],
            "last_equity": snap["last_equity"],
            "buying_power": snap["buying_power"],
            "open_positions": snap["open_positions"],
            "trades_today": snap["trades_today"],
        })
    else:
        base.update({
            "available": False,
            "reason": snap.get("reason", "unknown"),
            "equity": None,
            "last_equity": None,
            "buying_power": None,
            "open_positions": None,
            "trades_today": None,
        })

    return base


# ============ Positions ============
@app.get("/api/positions")
def api_positions():
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return []
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            }
            for p in client.get_all_positions()
        ]
    except Exception as e:
        return {"error": str(e)}


# ============ Decisions / Equity / Exits ============
@app.get("/api/decisions")
def api_decisions(limit: int = 30):
    return recent_decisions(limit)


@app.get("/api/equity")
def api_equity(limit: int = 200):
    return equity_history(limit)


@app.get("/api/exits")
def api_exits(limit: int = 30):
    return recent_exit_decisions(limit)


# ============ Settings ============
@app.get("/api/settings")
def api_get_settings():
    return get_all_settings()


@app.post("/api/settings")
async def api_update_settings(payload: dict):
    update_settings(payload)
    return get_all_settings()


# ============ Run control ============
@app.post("/api/run/toggle")
def api_toggle_run():
    if controller.is_running():
        controller.stop()
        return {"running": False}
    controller.start()
    return {"running": True}


@app.post("/api/run/start")
def api_start():
    controller.start()
    return {"running": controller.is_running()}


@app.post("/api/run/stop")
def api_stop():
    controller.stop()
    return {"running": controller.is_running()}


# ============ Monitor control ============
@app.get("/api/monitor/status")
def api_monitor_status():
    return monitor.status()


@app.post("/api/monitor/toggle")
def api_monitor_toggle():
    if monitor.is_running():
        monitor.stop()
        return {"running": False}
    monitor.start()
    return {"running": True}

# ============ Kill Switch ============
@app.get("/api/kill-switch")
def api_kill_switch_status():
    from decision_log import is_kill_switch_active
    active, reason = is_kill_switch_active()
    return {"active": active, "reason": reason}


@app.post("/api/kill-switch")
async def api_kill_switch(payload: dict):
    from decision_log import set_kill_switch, is_kill_switch_active
    active = bool(payload.get("active", True))
    reason = str(payload.get("reason", ""))[:200]
    set_kill_switch(active, reason)
    current, current_reason = is_kill_switch_active()
    result = {"active": current, "reason": current_reason}
    # Read-only operator report — does NOT cancel or flatten anything.
    if current:
        try:
            from broker import kill_switch_report
            result["report"] = kill_switch_report()
        except Exception as e:
            result["report"] = {"error": str(e)}
    return result


@app.get("/api/protection")
def api_protection_status():
    """Whether new entries are currently blocked by unverified protection."""
    from decision_log import (
        get_protection_unverified, get_protection_unverified_reason,
    )
    return {
        "protection_unverified": get_protection_unverified(),
        "reason": get_protection_unverified_reason(),
    }
