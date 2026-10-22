import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DB = Path(os.getenv("BACKTEST_DB_PATH", "backtest_results.db"))


def init():
    with sqlite3.connect(DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            run_type TEXT,
            start_date TEXT,
            end_date TEXT,
            params TEXT,
            metrics TEXT,
            num_trades INTEGER)""")
        conn.commit()


def save_run(symbol: str, start_date: str, end_date: str,
             params: dict, metrics: dict, run_type: str = "single",
             num_rejections: int = 0):
    """Persist a run. ``num_rejections`` (rejection EVENTS) is stored inside the
    existing ``params`` JSON so no schema change or migration is needed.

    A copy is merged so the caller's ``params`` dict is never mutated.
    """
    init()
    params = {**params, "num_rejections": int(num_rejections)}
    with sqlite3.connect(DB) as conn:
        conn.execute("""INSERT INTO runs
            (timestamp, symbol, run_type, start_date, end_date,
             params, metrics, num_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
            datetime.now(timezone.utc).isoformat(), symbol, run_type,
            start_date, end_date,
            json.dumps(params), json.dumps(metrics),
            metrics.get("num_trades", 0),
        ))
        conn.commit()


def list_runs(limit: int = 20) -> list:
    init()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"])
            d["metrics"] = json.loads(d["metrics"])
            result.append(d)
        return result
