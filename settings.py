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
    "trading_capital": "0",   # ← جديد: 0 = استخدم رصيد الحساب الحقيقي
    "exit_check_seconds": "60",
"exit_llm_interval_seconds": "300",
"hard_stop_loss_pct": "8.0",
"hard_take_profit_pct": "15.0",
"trailing_activation_pct": "5.0",
"trailing_stop_pct": "3.0",
"allow_extended_hours": "false",
    "stop_atr_multiplier": "1.5",
"take_profit_rr": "2.0",
"trailing_activation_r": "2.0",   # بدل 1.5
"trailing_distance_r": "0.5",     # بدل 1.0 — مسافة أقرب، لكن بعد مسافة أبعد 1R مسافة
"max_position_pct": "0.15",
"max_portfolio_exposure_pct": "0.45",
"max_consecutive_losses": "3",
"cooldown_minutes": "120",
"min_signal_score": "60",
"min_hold_minutes": "30",
"min_loss_to_close_pct": "1.5",
"llm_exit_min_confidence": "0.70",
    "min_stop_move_pct": "0.3",
    "regime_enabled": "1",
"news_enabled": "1",
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
        "trading_capital": float(get_setting("trading_capital")),   # ← جديد
        "exit_check_seconds": int(get_setting("exit_check_seconds")),
"exit_llm_interval_seconds": int(get_setting("exit_llm_interval_seconds")),
"hard_stop_loss_pct": float(get_setting("hard_stop_loss_pct")),
"hard_take_profit_pct": float(get_setting("hard_take_profit_pct")),
"trailing_activation_pct": float(get_setting("trailing_activation_pct")),
"trailing_stop_pct": float(get_setting("trailing_stop_pct")),
"allow_extended_hours": get_setting("allow_extended_hours") == "true",
        "stop_atr_multiplier": float(get_setting("stop_atr_multiplier")),
"take_profit_rr": float(get_setting("take_profit_rr")),
"trailing_activation_r": float(get_setting("trailing_activation_r")),
"trailing_distance_r": float(get_setting("trailing_distance_r")),
"max_position_pct": float(get_setting("max_position_pct")),
"max_portfolio_exposure_pct": float(get_setting("max_portfolio_exposure_pct")),
"max_consecutive_losses": int(get_setting("max_consecutive_losses")),
"cooldown_minutes": int(get_setting("cooldown_minutes")),
"min_signal_score": int(get_setting("min_signal_score")),
"min_hold_minutes": int(get_setting("min_hold_minutes")),
"min_loss_to_close_pct": float(get_setting("min_loss_to_close_pct")),
"llm_exit_min_confidence": float(get_setting("llm_exit_min_confidence")),
        "min_stop_move_pct": float(get_setting("min_stop_move_pct")),
        "regime_enabled": int(get_setting("regime_enabled")),
"news_enabled": int(get_setting("news_enabled")),
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
