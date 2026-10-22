import json
from decision_log import (
    get_setting, set_setting, get_trading_state, set_trading_state
)
from config import (
    SYMBOLS as C_SYMBOLS,
    MIN_CONFIDENCE as C_MIN_CONFIDENCE,
    MAX_RISK_PER_TRADE as C_MAX_RISK,
    MAX_OPEN_POSITIONS as C_MAX_POS,
    MAX_TRADES_PER_DAY as C_MAX_TRADES,
    MAX_DAILY_LOSS_PCT as C_MAX_LOSS,
    MAX_POSITION_PCT as C_MAX_POS_PCT,
    MAX_PORTFOLIO_EXPOSURE_PCT as C_MAX_EXPO,
    MAX_CONSECUTIVE_LOSSES as C_MAX_CONSEC,
    COOLDOWN_MINUTES as C_COOLDOWN,
    STOP_ATR_MULTIPLIER as C_STOP_ATR,
    TAKE_PROFIT_RR as C_TP_RR,
    TRAILING_ACTIVATION_R as C_TRAIL_ACT,
    TRAILING_DISTANCE_R as C_TRAIL_DIST,
    MIN_STOP_MOVE_PCT as C_MIN_STOP_MOVE,
    MIN_SIGNAL_SCORE as C_MIN_SCORE,
    MIN_HOLD_MINUTES as C_MIN_HOLD,
    MIN_LOSS_TO_CLOSE_PCT as C_MIN_LOSS_CLOSE,
    LLM_EXIT_MIN_CONFIDENCE as C_LLM_EXIT_CONF,
    EXIT_CHECK_SECONDS as C_EXIT_CHECK,
    EXIT_LLM_INTERVAL_SECONDS as C_EXIT_LLM,
    ALLOW_EXTENDED_HOURS as C_ALLOW_EXT,
    CYCLE_INTERVAL_MINUTES as C_CYCLE,
    FEES_PER_SHARE as C_FEES_PER_SHARE,
    ENTRY_SLIPPAGE_BPS as C_ENTRY_SLIPPAGE,
    EXIT_SLIPPAGE_BPS as C_EXIT_SLIPPAGE,
    MAX_HOLD_BARS as C_MAX_HOLD,
    ENABLE_SHORT as C_ENABLE_SHORT,
    get_trading_mode,
)

SETTINGS_VERSION = 3
_VERSION_KEY = "settings_version"

# All defaults come from config. This is the single source of truth.
DEFAULTS = {
    "min_confidence": str(C_MIN_CONFIDENCE),
    "max_risk_per_trade": str(C_MAX_RISK),
    "max_open_positions": str(C_MAX_POS),
    "max_trades_per_day": str(C_MAX_TRADES),
    "max_daily_loss_pct": str(C_MAX_LOSS),
    "max_position_pct": str(C_MAX_POS_PCT),
    "max_portfolio_exposure_pct": str(C_MAX_EXPO),
    "max_consecutive_losses": str(C_MAX_CONSEC),
    "cooldown_minutes": str(C_COOLDOWN),
    "stop_atr_multiplier": str(C_STOP_ATR),
    "take_profit_rr": str(C_TP_RR),
    "trailing_activation_r": str(C_TRAIL_ACT),
    "trailing_distance_r": str(C_TRAIL_DIST),
    "min_stop_move_pct": str(C_MIN_STOP_MOVE),
    "min_signal_score": str(C_MIN_SCORE),
    "min_hold_minutes": str(C_MIN_HOLD),
    "min_loss_to_close_pct": str(C_MIN_LOSS_CLOSE),
    "llm_exit_min_confidence": str(C_LLM_EXIT_CONF),
    "exit_check_seconds": str(C_EXIT_CHECK),
    "exit_llm_interval_seconds": str(C_EXIT_LLM),
    "allow_extended_hours": "true" if C_ALLOW_EXT else "false",
    "cycle_interval_minutes": str(C_CYCLE),
    "fees_per_share": str(C_FEES_PER_SHARE),
    "entry_slippage_bps": str(C_ENTRY_SLIPPAGE),
    "exit_slippage_bps": str(C_EXIT_SLIPPAGE),
    "max_hold_bars": str(C_MAX_HOLD),
    "enable_short": str(C_ENABLE_SHORT),
    "symbols": json.dumps(C_SYMBOLS),
    "trading_capital": "0",
    "regime_enabled": "1",
    "news_enabled": "1",
}

# Keys from old versions that MUST NOT influence trading anymore.
_LEGACY_RISK_KEYS = (
    "hard_stop_loss_pct",
    "hard_take_profit_pct",
    "trailing_activation_pct",
    "trailing_stop_pct",
)


def _ensure_defaults():
    for k, v in DEFAULTS.items():
        if get_setting(k) is None:
            set_setting(k, v)


def _migrate_if_needed():
    """
    If persistent settings came from SETTINGS_VERSION < 2, overwrite
    every risk-critical key with the new config defaults and drop legacy keys.
    This prevents old DB values from silently overriding new risk model.
    """
    raw = get_trading_state(_VERSION_KEY)
    current = int(raw) if raw and raw.isdigit() else 1
    if current >= SETTINGS_VERSION:
        _ensure_defaults()
        return

    print(f"⚙️ settings: migrating v{current} → v{SETTINGS_VERSION}")

    # 1) Force risk-critical values to the new config defaults
    for k in (
        "stop_atr_multiplier", "take_profit_rr",
        "trailing_activation_r", "trailing_distance_r",
        "min_stop_move_pct", "min_signal_score",
        "max_risk_per_trade", "max_position_pct",
        "max_portfolio_exposure_pct", "max_consecutive_losses",
        "cooldown_minutes", "min_confidence",
        "min_hold_minutes", "min_loss_to_close_pct",
        "llm_exit_min_confidence",
    ):
        set_setting(k, DEFAULTS[k])

    # 2) Remove legacy percentage keys from active settings
    for k in _LEGACY_RISK_KEYS:
        set_setting(k, "")  # blank = unused, monitor must not read it

    # 3) Fill any missing keys
    _ensure_defaults()

    set_trading_state(_VERSION_KEY, str(SETTINGS_VERSION))
    print(f"⚙️ settings: migration complete (v{SETTINGS_VERSION})")


def get_all_settings() -> dict:
    _migrate_if_needed()
    return {
        "min_confidence": float(get_setting("min_confidence")),
        "max_risk_per_trade": float(get_setting("max_risk_per_trade")),
        "max_open_positions": int(get_setting("max_open_positions")),
        "max_trades_per_day": int(get_setting("max_trades_per_day")),
        "max_daily_loss_pct": float(get_setting("max_daily_loss_pct")),
        "max_position_pct": float(get_setting("max_position_pct")),
        "max_portfolio_exposure_pct": float(get_setting("max_portfolio_exposure_pct")),
        "max_consecutive_losses": int(get_setting("max_consecutive_losses")),
        "cooldown_minutes": int(get_setting("cooldown_minutes")),
        "stop_atr_multiplier": float(get_setting("stop_atr_multiplier")),
        "take_profit_rr": float(get_setting("take_profit_rr")),
        "trailing_activation_r": float(get_setting("trailing_activation_r")),
        "trailing_distance_r": float(get_setting("trailing_distance_r")),
        "min_stop_move_pct": float(get_setting("min_stop_move_pct")),
        "min_signal_score": int(get_setting("min_signal_score")),
        "min_hold_minutes": int(get_setting("min_hold_minutes")),
        "min_loss_to_close_pct": float(get_setting("min_loss_to_close_pct")),
        "llm_exit_min_confidence": float(get_setting("llm_exit_min_confidence")),
        "exit_check_seconds": int(get_setting("exit_check_seconds")),
        "exit_llm_interval_seconds": int(get_setting("exit_llm_interval_seconds")),
        "allow_extended_hours": get_setting("allow_extended_hours") == "true",
        "cycle_interval_minutes": int(get_setting("cycle_interval_minutes")),
        "fees_per_share": float(get_setting("fees_per_share")),
        "entry_slippage_bps": float(get_setting("entry_slippage_bps")),
        "exit_slippage_bps": float(get_setting("exit_slippage_bps")),
        "max_hold_bars": int(get_setting("max_hold_bars")),
        "enable_short": int(get_setting("enable_short")),
        "symbols": json.loads(get_setting("symbols")),
        "trading_capital": float(get_setting("trading_capital")),
        "regime_enabled": int(get_setting("regime_enabled")),
        "news_enabled": int(get_setting("news_enabled")),
        # explicit flags
        "dry_run": get_trading_mode() != "LIVE",
        "trading_mode": get_trading_mode(),
        "settings_version": SETTINGS_VERSION,
    }


def update_settings(payload: dict):
    # Reject legacy keys — they must never be written again
    for k, v in payload.items():
        if k in _LEGACY_RISK_KEYS:
            continue
        if k not in DEFAULTS:
            continue
        if k == "symbols" and isinstance(v, list):
            set_setting(k, json.dumps(v))
        else:
            set_setting(k, str(v))


def is_dry_run() -> bool:
    return get_trading_mode() != "LIVE"


def set_dry_run(v: bool):
    """Kept for API compatibility. Dry run is derived from trading mode now."""
    # No-op; trading mode is controlled by env vars.
    return


def get_symbols() -> list:
    _migrate_if_needed()
    return json.loads(get_setting("symbols"))


def get_min_confidence() -> float:
    _migrate_if_needed()
    return float(get_setting("min_confidence"))
