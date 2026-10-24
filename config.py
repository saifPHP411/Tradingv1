import os
from dotenv import load_dotenv

load_dotenv()

# ============ LLM ============
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# ============ Alpaca ============
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ============ Data Providers (optional) ============
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ============ Gemini AI market context (OPTIONAL, ADVISORY, OFF BY DEFAULT) ============
# The API key is read from the environment only; it is never stored in code.
# The feature is disabled unless explicitly enabled here AND a key is present.
# It is advisory-only and is NOT wired into strategy/execution/risk.
AI_CONTEXT_ENABLED = os.getenv("AI_CONTEXT_ENABLED", "false").lower() == "true"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ============ Trading Mode — SINGLE SOURCE ============
# BACKTEST: no broker calls (used by backtester/engine)
# PAPER:    Alpaca Paper Trading — targets the paper endpoint, CAN submit
# LIVE:     requires TRADING_MODE=live + ENABLE_LIVE_TRADING=true + confirmation
#
# Unknown/typo values fail SAFE to PAPER (never LIVE).
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
LIVE_TRADING_CONFIRMATION = os.getenv("LIVE_TRADING_CONFIRMATION", "")

_KNOWN_MODES = ("backtest", "paper", "live")

def get_trading_mode() -> str:
    """The ONLY function the project should use to know trading mode.

    Resolution order:
      - "backtest"                 → BACKTEST (no broker orders, ever)
      - "live" + live safeguards   → LIVE
      - "live" without safeguards  → PAPER (fail safe)
      - "paper" or anything else   → PAPER (unknown values never select LIVE)
    """
    if TRADING_MODE == "backtest":
        return "BACKTEST"
    live_ok = (
        ENABLE_LIVE_TRADING
        and LIVE_TRADING_CONFIRMATION == "I_UNDERSTAND_LIVE_RISK"
        and TRADING_MODE == "live"
    )
    return "LIVE" if live_ok else "PAPER"

ALPACA_PAPER = get_trading_mode() != "LIVE"
if TRADING_MODE not in _KNOWN_MODES:
    print(f"⚠️ config: unknown TRADING_MODE={TRADING_MODE!r} — "
          f"failing safe to {get_trading_mode()}")
if get_trading_mode() == "LIVE":
    print("🚨 WARNING: LIVE TRADING IS ENABLED — REAL MONEY AT RISK 🚨")

# ============ Risk Defaults (CONSERVATIVE) ============
MAX_RISK_PER_TRADE = 0.005
MAX_POSITION_PCT = 0.15
MAX_PORTFOLIO_EXPOSURE_PCT = 0.45
MAX_OPEN_POSITIONS = 3
MAX_TRADES_PER_DAY = 6
MAX_DAILY_LOSS_PCT = 0.02
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_MINUTES = 120

# ============ ATR / R-based Risk Model (THE ONLY MODEL) ============
STOP_ATR_MULTIPLIER = 2.5
TAKE_PROFIT_RR = 2.0
TRAILING_ACTIVATION_R = 1.5
TRAILING_DISTANCE_R = 1.0
MIN_STOP_MOVE_PCT = 0.3

# ============ Signal Thresholds ============
MIN_CONFIDENCE = 0.70
MIN_SIGNAL_SCORE = 70

# ============ Fees / Slippage (configurable; backtest uses these) ============
FEES_PER_SHARE = 0.0
ENTRY_SLIPPAGE_BPS = 5
EXIT_SLIPPAGE_BPS = 5

# ============ Execution ============
# Bars after which an open position is force-closed ("MAX_HOLD").
# 60 daily bars ≈ 3 trading months. Overridable per run (CLI / walk-forward).
MAX_HOLD_BARS = 60

# ============ Strategy direction ============
# Strategy is LONG-only by design. The execution engine is SHORT-capable;
# set ENABLE_SHORT=1 (env) or pass settings["enable_short"]=1 to exercise it.
ENABLE_SHORT = 1 if os.getenv("ENABLE_SHORT", "0").strip() in ("1", "true", "True") else 0

# ============ LLM Exit Sensitivity ============
MIN_HOLD_MINUTES = 30
MIN_LOSS_TO_CLOSE_PCT = 1.5
MIN_GAIN_TO_TRIM_PCT = 2.0
LLM_EXIT_MIN_CONFIDENCE = 0.70

# ============ Data / Symbols ============
SYMBOLS = ["AAPL", "NVDA", "AMD", "ASML", "CRM", "PANW", "TSLA", "JNJ", "LLY", "AMZN"]
CYCLE_INTERVAL_MINUTES = 15

# ============ Exits monitor cadence ============
EXIT_CHECK_SECONDS = 60
EXIT_LLM_INTERVAL_SECONDS = 300
ALLOW_EXTENDED_HOURS = False

# ============ Data paths ============
DB_PATH = os.getenv("DB_PATH", "decisions.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "trading_memory.db")
GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "graph_checkpoints.db")

# ============ Account fallback (NEVER used for trade decisions) ============
# Kept only so UI can show "N/A". safety.py must FAIL CLOSED on API errors.
ACCOUNT_EQUITY_DEFAULT = 10000

# ============ Legacy aliases — DO NOT USE IN NEW CODE ============
ONLY_MARKET_HOURS = True
# DRY_RUN means "never submit a broker order" — true ONLY for BACKTEST.
# PAPER targets the paper endpoint and is allowed to submit. Use
# settings.is_dry_run() / is_paper() / is_live() instead of this constant.
DRY_RUN = get_trading_mode() == "BACKTEST"
