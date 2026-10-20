import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB = Path(os.getenv("DB_PATH", "decisions.db"))
DB.parent.mkdir(parents=True, exist_ok=True)


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
            analyst_signal TEXT, analyst_confidence REAL, analyst_reason TEXT,
            risk_approved INTEGER, risk_veto INTEGER, risk_reason TEXT,
            final_decision TEXT, consensus_reached INTEGER,
            order_details TEXT, broker_result TEXT, market_data TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            equity REAL, open_positions INTEGER)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS position_state (
            symbol TEXT PRIMARY KEY,
            entry_price REAL,
            qty REAL,
            side TEXT,
            highest_price REAL,
            lowest_price REAL,
            current_stop REAL,
            current_target REAL,
            opened_at TEXT,
            last_exit_decision TEXT,
            last_exit_reason TEXT,
            last_exit_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS exit_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, decision TEXT, reason TEXT,
            pnl_pct REAL, price REAL, entry_price REAL)""")
        conn.commit()


def log_decision(state: dict, broker_result: dict | None):
    analyst = state.get("analyst_report") or {}
    risk = state.get("risk_report") or {}
    with _conn() as conn:
        conn.execute("""INSERT INTO decisions (timestamp, symbol, analyst_signal,
            analyst_confidence, analyst_reason, risk_approved, risk_veto,
            risk_reason, final_decision, consensus_reached, order_details,
            broker_result, market_data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            datetime.utcnow().isoformat(), state.get("symbol"),
            analyst.get("signal"), float(analyst.get("confidence", 0)),
            analyst.get("reason", ""), 1 if risk.get("approved") else 0,
            1 if risk.get("veto") else 0, risk.get("reason", ""),
            state.get("final_decision"), 1 if state.get("consensus_reached") else 0,
            json.dumps(state.get("order_details"), ensure_ascii=False) if state.get("order_details") else None,
            json.dumps(broker_result, ensure_ascii=False) if broker_result else None,
            json.dumps(state.get("market_data"), ensure_ascii=False, default=str)))
        conn.commit()


def log_equity(equity: float, open_positions: int):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity, open_positions) VALUES (?,?,?)",
            (datetime.utcnow().isoformat(), equity, open_positions))
        conn.commit()


def recent_decisions(limit: int = 50) -> list:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def equity_history(limit: int = 200) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, equity, open_positions FROM equity_snapshots ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]


def get_setting(key: str, default=None):
    with _conn() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO app_settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))
        conn.commit()


def upsert_position_state(symbol: str, entry_price: float, qty: float, side: str):
    with _conn() as conn:
        conn.execute("""INSERT INTO position_state
            (symbol, entry_price, qty, side, highest_price, lowest_price, opened_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty""",
            (symbol, entry_price, qty, side, entry_price, entry_price,
             datetime.utcnow().isoformat()))
        conn.commit()


def get_position_state(symbol: str):
    with _conn() as conn:
        row = conn.execute("SELECT * FROM position_state WHERE symbol=?", (symbol,)).fetchone()
        return dict(row) if row else None


def all_position_states() -> list:
    with _conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM position_state").fetchall()]


def update_position_state(symbol: str, **fields):
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [symbol]
    with _conn() as conn:
        conn.execute(f"UPDATE position_state SET {sets} WHERE symbol=?", vals)
        conn.commit()


def delete_position_state(symbol: str):
    with _conn() as conn:
        conn.execute("DELETE FROM position_state WHERE symbol=?", (symbol,))
        conn.commit()


def log_exit_decision(symbol: str, decision: str, reason: str,
                      pnl_pct: float, price: float, entry_price: float):
    with _conn() as conn:
        conn.execute("""INSERT INTO exit_decisions
            (timestamp, symbol, decision, reason, pnl_pct, price, entry_price)
            VALUES (?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(), symbol, decision, reason,
             pnl_pct, price, entry_price))
        conn.commit()


def recent_exit_decisions(limit: int = 50) -> list:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM exit_decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


init_db()
