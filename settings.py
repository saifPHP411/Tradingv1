import json
from decision_log import get_setting, set_setting
from config import (
    SYMBOLS as DEFAULT_SYMBOLS, MIN_CONFIDENCE as DEFAULT_MIN_CONFIDENCE,
    MAX_RISK_PER_TRADE as DEFAULT_MAX_RISK, MAX_OPEN_POSITIONS as DEFAULT_MAX_POS,
    MAX_TRADES_PER_DAY as DEFAULT_MAX_TRADES, MAX_DAILY_LOSS_PCT as DEFAULT_MAX_LOSS,
    DRY_RUN as DEFAULT_DRY_RUN,
)

DEFAULTS = {
    "dry_run": "true" if DEFAULT_DRY_RUN else "false",
    "min_confidence": str(DEFAULT_MIN_CONFIDENCE),
    "max_risk_per_trade": str(DEFAULT_MAX_RISK),
    "max_open_positions": str(DEFAULT_MAX_POS),
    "max_trades_per_day": str(DEFAULT_MAX_TRADES),
    "max_daily_loss_pct": str(DEFAULT_MAX_LOSS),
    "symbols": json.dumps(DEFAULT_SYMBOLS),
    "cycle_interval_minutes": "5",
}

def _ensure_defaults():
    for k, v in DEFAULTS.items():
        if get_setting(k) is None:
            set_setting(k, v)

def get_all_settings() -> dict:
    _ensure_defaults()
    return {
        "dry_run": get_setting("dry_run") == "true",
        "min_confidence": float(get_setting("min_confidence")),
        "max_risk_per_trade": float(get_setting("max_risk_per_trade")),
        "max_open_positions": int(get_setting("max_open_positions")),
        "max_trades_per_day": int(get_setting("max_trades_per_day")),
        "max_daily_loss_pct": float(get_setting("max_daily_loss_pct")),
        "symbols": json.loads(get_setting("symbols")),
        "cycle_interval_minutes": int(get_setting("cycle_interval_minutes")),
    }

def update_settings(payload: dict):
    for k, v in payload.items():
        if k not in DEFAULTS:
            continue
        if k == "symbols" and isinstance(v, list):
            set_setting(k, json.dumps(v))
        elif k == "dry_run":
            set_setting(k, "true" if v else "false")
        else:
            set_setting(k, str(v))

def is_dry_run() -> bool:
    return get_setting("dry_run", DEFAULTS["dry_run"]) == "true"

def set_dry_run(v: bool):
    set_setting("dry_run", "true" if v else "false")

def get_symbols() -> list:
    _ensure_defaults()
    return json.loads(get_setting("symbols"))

def get_min_confidence() -> float:
    _ensure_defaults()
    return float(get_setting("min_confidence"))
