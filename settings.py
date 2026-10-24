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
    AI_CONTEXT_ENABLED as C_AI_CONTEXT_ENABLED,
    GEMINI_MODEL as C_GEMINI_MODEL,
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
    # AI market context is ADVISORY-ONLY and OFF by default. It is not read by
    # the strategy, execution engine or risk sizing.
    "ai_context_enabled": "1" if C_AI_CONTEXT_ENABLED else "0",
    "ai_context_model": C_GEMINI_MODEL,
}

# Keys from old versions that MUST NOT influence trading anymore.
_LEGACY_RISK_KEYS = (
    "hard_stop_loss_pct",
    "hard_take_profit_pct",
    "trailing_activation_pct",
    "trailing_stop_pct",
)


def _to_int(value, default: int) -> int:
    """Coerce a stored setting to int without ever raising.

    Handles the JSON-boolean case: the settings API stores ``str(True)`` as
    ``"True"``, and ``int("True")`` would otherwise crash every caller of
    ``get_all_settings`` (including the monitor loop). Unparseable values fall
    back to the current config default so a bad write cannot change behaviour
    silently and cannot take the process down.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in ("true", "yes", "on"):
            return 1
        if lowered in ("false", "no", "off", ""):
            return 0
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _to_float(value, default: float) -> float:
    """Coerce a stored setting to float without ever raising."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in ("true", "yes", "on"):
            return 1.0
        if lowered in ("false", "no", "off", ""):
            return 0.0
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _store_bool(value) -> str:
    """Render a settings value as the canonical ``"true"``/``"false"`` string.

    Real booleans and their textual spellings become lowercase strings so the
    stored form is unambiguous, whatever the caller sent.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on"):
            return "true"
        if lowered in ("false", "no", "off"):
            return "false"
    return str(value)


def _to_bool(value) -> bool:
    """Coerce a stored value to bool without ever raising."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return False


def _load_symbols(value) -> list:
    """Coerce a stored ``symbols`` value to a list without ever raising."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


# Settings that are conceptually boolean. Stored canonically as "true"/"false"
# by ``update_settings`` and read back through ``_to_bool``/``_to_int``.
_BOOL_SETTING_KEYS = frozenset({
    "allow_extended_hours",
    "regime_enabled",
    "news_enabled",
    "ai_context_enabled",
})


def _ensure_defaults():
    for k, v in DEFAULTS.items():
        if get_setting(k) is None:
            set_setting(k, v)


def _migrate_if_needed():
    """
    If persistent settings came from SETTINGS_VERSION < 2, overwrite
    every risk-critical key with the new config defaults and drop legacy keys.
    This prevents old DB values from silently overriding new risk model.

    Migration semantics (intentional, tested in tests/test_phase1_6_remediation.py):
      - The risk-critical keys listed below are FORCED back to the current
        config defaults on a version bump, so an operator-tuned value is
        deliberately discarded when the risk model changes. The keys that are
        NOT in that list (max_open_positions, max_trades_per_day,
        max_daily_loss_pct) are preserved.
      - Legacy percentage keys are blanked so the monitor cannot read them.
      - The migration is idempotent: once the stored version equals
        SETTINGS_VERSION, only missing defaults are filled.
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
    """Read every setting as a typed value.

    Coercion is fail-safe: a stored value that cannot be parsed falls back to
    the current config default (``DEFAULTS``) instead of raising. This keeps a
    single bad write from taking down every caller, including the monitor loop.
    """
    _migrate_if_needed()

    def _int(key: str) -> int:
        return _to_int(get_setting(key), int(float(DEFAULTS[key])))

    def _flt(key: str) -> float:
        return _to_float(get_setting(key), float(DEFAULTS[key]))

    return {
        "min_confidence": _flt("min_confidence"),
        "max_risk_per_trade": _flt("max_risk_per_trade"),
        "max_open_positions": _int("max_open_positions"),
        "max_trades_per_day": _int("max_trades_per_day"),
        "max_daily_loss_pct": _flt("max_daily_loss_pct"),
        "max_position_pct": _flt("max_position_pct"),
        "max_portfolio_exposure_pct": _flt("max_portfolio_exposure_pct"),
        "max_consecutive_losses": _int("max_consecutive_losses"),
        "cooldown_minutes": _int("cooldown_minutes"),
        "stop_atr_multiplier": _flt("stop_atr_multiplier"),
        "take_profit_rr": _flt("take_profit_rr"),
        "trailing_activation_r": _flt("trailing_activation_r"),
        "trailing_distance_r": _flt("trailing_distance_r"),
        "min_stop_move_pct": _flt("min_stop_move_pct"),
        "min_signal_score": _int("min_signal_score"),
        "min_hold_minutes": _int("min_hold_minutes"),
        "min_loss_to_close_pct": _flt("min_loss_to_close_pct"),
        "llm_exit_min_confidence": _flt("llm_exit_min_confidence"),
        "exit_check_seconds": _int("exit_check_seconds"),
        "exit_llm_interval_seconds": _int("exit_llm_interval_seconds"),
        "allow_extended_hours": _to_bool(get_setting("allow_extended_hours")),
        "cycle_interval_minutes": _int("cycle_interval_minutes"),
        "fees_per_share": _flt("fees_per_share"),
        "entry_slippage_bps": _flt("entry_slippage_bps"),
        "exit_slippage_bps": _flt("exit_slippage_bps"),
        "max_hold_bars": _int("max_hold_bars"),
        "enable_short": _int("enable_short"),
        "symbols": _load_symbols(get_setting("symbols")),
        "trading_capital": _flt("trading_capital"),
        "regime_enabled": _int("regime_enabled"),
        "news_enabled": _int("news_enabled"),
        # Advisory-only AI context. Not consumed by strategy/execution/risk.
        "ai_context_enabled": _int("ai_context_enabled"),
        "ai_context_model": get_setting("ai_context_model"),
        # explicit flags
        "dry_run": is_dry_run(),
        "trading_mode": get_trading_mode(),
        "settings_version": SETTINGS_VERSION,
    }


def update_settings(payload: dict):
    """Persist a settings payload.

    Only keys present in ``DEFAULTS`` are accepted; legacy risk keys and unknown
    keys are ignored. Boolean-ish keys are stored in the canonical
    ``"true"``/``"false"`` form so a later ``int()`` read cannot fail on
    ``"True"``. ``symbols`` accepts a list or a JSON string.
    """
    if not isinstance(payload, dict):
        return
    # Reject legacy keys — they must never be written again
    for k, v in payload.items():
        if k in _LEGACY_RISK_KEYS:
            continue
        if k not in DEFAULTS:
            continue
        if k == "symbols" and isinstance(v, list):
            set_setting(k, json.dumps(v))
        elif k in _BOOL_SETTING_KEYS:
            set_setting(k, _store_bool(v))
        else:
            set_setting(k, str(v))


def is_dry_run() -> bool:
    """True only when the process must never submit a broker order.

    BACKTEST is the only dry-run mode. PAPER targets Alpaca's paper endpoint
    and is allowed to submit (gated by safety + risk checks); LIVE is allowed
    only when the explicit live safeguards are satisfied.
    """
    return get_trading_mode() == "BACKTEST"


def is_paper() -> bool:
    """True when orders (if any) go to Alpaca's PAPER endpoint."""
    return get_trading_mode() == "PAPER"


def is_live() -> bool:
    """True only under the explicit live authorization safeguards."""
    return get_trading_mode() == "LIVE"


def broker_submission_allowed() -> bool:
    """Whether this process may ever call the broker to submit an order.

    BACKTEST must never submit. PAPER and LIVE may submit, but only after the
    safety and risk gates in broker.execute_order approve the order.
    """
    return get_trading_mode() in ("PAPER", "LIVE")


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
