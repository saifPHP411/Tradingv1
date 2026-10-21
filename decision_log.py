import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
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
            entry_price REAL, qty REAL, side TEXT,
            highest_price REAL, lowest_price REAL,
            current_stop REAL, current_target REAL, opened_at TEXT,
            last_exit_decision TEXT, last_exit_reason TEXT, last_exit_at TEXT)""")

        conn.execute("""CREATE TABLE IF NOT EXISTS exit_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, symbol TEXT, decision TEXT, reason TEXT,
            pnl_pct REAL, price REAL, entry_price REAL)""")

        conn.execute("""CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, content TEXT, metadata TEXT)""")

        # Trading state (for settings version, day-start equity, cooldown, etc.)
        conn.execute("""CREATE TABLE IF NOT EXISTS trading_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT)""")

        # Trade-level state (used by broker/monitor for protection tracking)
        conn.execute("""CREATE TABLE IF NOT EXISTS trade_state (
            trade_id TEXT PRIMARY KEY,
            symbol TEXT, side TEXT,
            entry_order_id TEXT, stop_order_id TEXT,
            entry_price REAL, qty REAL,
            initial_stop REAL, take_profit REAL,
            risk_per_share REAL, risk_amount REAL,
            opened_at TEXT, status TEXT)""")

        # Migrations: add missing columns to position_state safely
        for col, ctype in [
            ("stop_order_id", "TEXT"),
            ("initial_stop", "REAL"),
            ("initial_risk_per_share", "REAL"),
            ("trailing_active", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE position_state ADD COLUMN {col} {ctype}")
            except Exception:
                pass  # already exists

        conn.execute("""CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event_type TEXT,
    symbol TEXT,
    trade_id TEXT,
    reason TEXT,
    data TEXT)""")
        conn.commit()


# ============ Decisions ============
def log_decision(state: dict, broker_result: dict | None):
    analyst = state.get("analyst_report") or {}
    risk = state.get("risk_report") or {}
    with _conn() as conn:
        conn.execute("""INSERT INTO decisions (timestamp, symbol, analyst_signal,
            analyst_confidence, analyst_reason, risk_approved, risk_veto,
            risk_reason, final_decision, consensus_reached, order_details,
            broker_result, market_data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            datetime.now(timezone.utc).isoformat(), state.get("symbol"),
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
            (datetime.now(timezone.utc).isoformat(), equity, open_positions))
        conn.commit()


def recent_decisions(limit: int = 50) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def equity_history(limit: int = 200) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, equity, open_positions FROM equity_snapshots "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ============ Settings (persistent) ============
def get_setting(key: str, default=None):
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO app_settings (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))
        conn.commit()


# ============ Trading state (for settings version, day-start, cooldown) ============
def get_trading_state(key: str, default=None):
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM trading_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_trading_state(key: str, value: str):
    with _conn() as conn:
        conn.execute(
            """INSERT INTO trading_state (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE
               SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, str(value), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


# ============ Day-start equity ============
def get_day_start_equity() -> float | None:
    raw = get_trading_state("day_start_equity")
    raw_date = get_trading_state("day_start_date")
    if not raw or not raw_date:
        return None
    today = datetime.now(timezone.utc).date().isoformat()
    if raw_date != today:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def set_day_start_equity(equity: float):
    today = datetime.now(timezone.utc).date().isoformat()
    set_trading_state("day_start_equity", str(equity))
    set_trading_state("day_start_date", today)


# ============ Consecutive losses & cooldown ============
def get_consecutive_losses() -> int:
    try:
        return int(get_trading_state("consecutive_losses", "0"))
    except Exception:
        return 0


def set_consecutive_losses(n: int):
    set_trading_state("consecutive_losses", str(int(n)))


def get_cooldown_until() -> str | None:
    return get_trading_state("cooldown_until")


def set_cooldown_until(iso_ts: str):
    set_trading_state("cooldown_until", iso_ts)


def clear_cooldown():
    set_trading_state("cooldown_until", "")


# ============ Position state ============
def upsert_position_state(symbol: str, entry_price: float, qty: float, side: str):
    with _conn() as conn:
        conn.execute("""INSERT INTO position_state
            (symbol, entry_price, qty, side, highest_price, lowest_price, opened_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty""",
            (symbol, entry_price, qty, side, entry_price, entry_price,
             datetime.now(timezone.utc).isoformat()))
        conn.commit()


def get_position_state(symbol: str):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM position_state WHERE symbol=?", (symbol,)
        ).fetchone()
        return dict(row) if row else None


def all_position_states() -> list:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM position_state"
        ).fetchall()]


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


def set_position_initial_stop(symbol: str, initial_stop: float, risk_per_share: float):
    """Sets initial_stop only once — used for R calculations."""
    state = get_position_state(symbol)
    if not state:
        return
    if state.get("initial_stop"):
        return
    update_position_state(
        symbol,
        initial_stop=round(initial_stop, 4),
        initial_risk_per_share=round(risk_per_share, 4),
    )


def update_stop_order_id(symbol: str, stop_order_id: str):
    update_position_state(symbol, stop_order_id=str(stop_order_id))


def set_trailing_active(symbol: str, active: bool):
    update_position_state(symbol, trailing_active=1 if active else 0)


def get_r_multiple(symbol: str, current_price: float) -> float | None:
    s = get_position_state(symbol)
    if not s or not s.get("initial_stop") or not s.get("initial_risk_per_share"):
        return None
    entry = s["entry_price"]
    rps = s["initial_risk_per_share"]
    if rps <= 0:
        return None
    if s["side"] == "long":
        return (current_price - entry) / rps
    return (entry - current_price) / rps


# ============ Exit decisions ============
def log_exit_decision(symbol: str, decision: str, reason: str,
                      pnl_pct: float, price: float, entry_price: float):
    with _conn() as conn:
        conn.execute("""INSERT INTO exit_decisions
            (timestamp, symbol, decision, reason, pnl_pct, price, entry_price)
            VALUES (?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(), symbol, decision, reason,
             pnl_pct, price, entry_price))
        conn.commit()


def recent_exit_decisions(limit: int = 50) -> list:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM exit_decisions ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()]


# ============ Lessons (for learner/reflect) ============
def log_lesson(content: str, metadata: dict | None = None):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO lessons (timestamp, content, metadata) VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), content,
             json.dumps(metadata or {}, ensure_ascii=False)))
        conn.commit()


def recent_lessons(limit: int = 20) -> list:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT content, timestamp FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()]


# ============ Trade state (used by broker/monitor) ============
def upsert_trade_state(trade_id: str, **fields):
    if not fields:
        return
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields.keys())
    vals = list(fields.values()) + [trade_id]
    with _conn() as conn:
        conn.execute(
            f"INSERT INTO trade_state (trade_id, {cols}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(trade_id) DO UPDATE SET {updates}",
            vals,
        )
        conn.commit()


def get_trade_state(trade_id: str):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM trade_state WHERE trade_id=?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_trade_state(trade_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM trade_state WHERE trade_id=?", (trade_id,))
        conn.commit()

# ============ Kill Switch ============
def is_kill_switch_active() -> tuple[bool, str]:
    val = get_trading_state("kill_switch_active", "false")
    reason = get_trading_state("kill_switch_reason", "") or ""
    return val == "true", reason


def set_kill_switch(active: bool, reason: str = ""):
    set_trading_state("kill_switch_active", "true" if active else "false")
    set_trading_state("kill_switch_reason", reason if active else "")
init_db()
