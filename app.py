from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import threading, time
from datetime import datetime

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
            self._started_at = datetime.utcnow()
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
                try:
                    snap = get_account_snapshot()
                    log_equity(snap["equity"], snap["open_positions"])
                except Exception:
                    pass
                try:
                    learn_from_closed_trades()
                except Exception:
                    pass
                interval = max(1, get_all_settings()["cycle_interval_minutes"]) * 60
                self._stop_event.wait(timeout=interval)
        finally:
            self._current = None
            self._started_at = None


controller = AgentController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ بدأ التطبيق")
    monitor.start()
    yield
    controller.stop()
    monitor.stop()
    print("⏹️ إيقاف")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/api/status")
def api_status():
    snap = get_account_snapshot()
    return {**snap, "dry_run": is_dry_run(), **controller.status()}


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


@app.get("/api/decisions")
def api_decisions(limit: int = 30):
    return recent_decisions(limit)


@app.get("/api/equity")
def api_equity(limit: int = 200):
    return equity_history(limit)


@app.get("/api/settings")
def api_get_settings():
    return get_all_settings()


@app.post("/api/settings")
async def api_update_settings(payload: dict):
    update_settings(payload)
    return get_all_settings()


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


@app.get("/api/exits")
def api_exits(limit: int = 30):
    return recent_exit_decisions(limit)
