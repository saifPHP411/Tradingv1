from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
import threading, time

from config import SYMBOLS
from safety import get_account_snapshot
from settings import is_dry_run, set_dry_run
from decision_log import recent_decisions, equity_history, log_equity
from main import run_cycle
from learner import learn_from_closed_trades
from memory_layer import store
from broker import execute_order  # not used directly, kept for imports

scheduler = BackgroundScheduler()
templates = Jinja2Templates(directory="templates")

def scheduled_cycle():
    for sym in SYMBOLS:
        try:
            run_cycle(sym)
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ فشل دورة {sym}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(scheduled_cycle, "interval", minutes=30, id="cycle")
    scheduler.add_job(learn_from_closed_trades, "interval", hours=6, id="learn")
    scheduler.start()
    print("✅ المجدول بدأ العمل")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/status")
def api_status():
    snap = get_account_snapshot()
    return {
        **snap,
        "dry_run": is_dry_run(),
        "symbols": SYMBOLS,
    }

@app.get("/api/decisions")
def api_decisions(limit: int = 30):
    return recent_decisions(limit)

@app.get("/api/equity")
def api_equity(limit: int = 200):
    return equity_history(limit)

@app.get("/api/memory")
def api_memory(limit: int = 20):
    try:
        items = store.search(("trading", "lessons"), query="الدرس", limit=limit)
        return [{"content": getattr(i, "value", {}).get("content", str(i))} for i in items]
    except Exception as e:
        return []

@app.post("/api/settings/dry_run")
async def api_set_dry_run(payload: dict):
    value = bool(payload.get("dry_run", True))
    set_dry_run(value)
    return {"dry_run": is_dry_run()}

@app.post("/api/run-cycle")
def api_run_cycle():
    def _run():
        for sym in SYMBOLS:
            try:
                run_cycle(sym)
            except Exception as e:
                print(f"⚠️ {e}")
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "symbols": SYMBOLS}