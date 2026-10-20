import sqlite3
import json
from datetime import datetime
from pathlib import Path

import os
DB = Path(os.getenv("DB_PATH", "decisions.db"))

def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                analyst_signal TEXT,
                analyst_confidence REAL,
                analyst_reason TEXT,
                risk_approved INTEGER,
                risk_veto INTEGER,
                risk_reason TEXT,
                final_decision TEXT,
                consensus_reached INTEGER,
                order_details TEXT,
                broker_result TEXT,
                market_data TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                open_positions INTEGER
            )
        """)

def log_decision(state: dict, broker_result: dict | None):
    analyst = state.get("analyst_report") or {}
    risk = state.get("risk_report") or {}
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            INSERT INTO decisions (timestamp, symbol, analyst_signal, analyst_confidence,
                analyst_reason, risk_approved, risk_veto, risk_reason, final_decision,
                consensus_reached, order_details, broker_result, market_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            state.get("symbol"),
            analyst.get("signal"),
            float(analyst.get("confidence", 0)),
            analyst.get("reason", ""),
            1 if risk.get("approved") else 0,
            1 if risk.get("veto") else 0,
            risk.get("reason", ""),
            state.get("final_decision"),
            1 if state.get("consensus_reached") else 0,
            json.dumps(state.get("order_details"), ensure_ascii=False) if state.get("order_details") else None,
            json.dumps(broker_result, ensure_ascii=False) if broker_result else None,
            json.dumps(state.get("market_data"), ensure_ascii=False),
        ))

def log_equity(equity: float, open_positions: int):
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, open_positions) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), equity, open_positions),
        )

def recent_decisions(limit: int = 50) -> list:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

def equity_history(limit: int = 200) -> list:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, equity, open_positions FROM equity_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

init_db()
