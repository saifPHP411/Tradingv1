import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB = Path(os.getenv("BACKTEST_DB_PATH", "backtest_results.db"))


def init():
    with sqlite3.connect(DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            start_date TEXT,
            end_date TEXT,
            params TEXT,
            metrics TEXT,
            num_trades INTEGER)""")
        conn.commit()


def save_run(symbol: str, start_date: str, end_date: str,
             params: dict, metrics: dict):
    init()
    with sqlite3.connect(DB) as conn:
        conn.execute("""INSERT INTO runs
            (timestamp, symbol, start_date, end_date, params, metrics, num_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            datetime.utcnow().isoformat(), symbol, start_date, end_date,
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
