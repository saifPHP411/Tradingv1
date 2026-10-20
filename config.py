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

# ============ Trading Mode — LIVE LOCKED BY DEFAULT ============
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
LIVE_TRADING_CONFIRMATION = os.getenv("LIVE_TRADING_CONFIRMATION", "")

_allow_live = (
    ENABLE_LIVE_TRADING
    and LIVE_TRADING_CONFIRMATION == "I_UNDERSTAND_LIVE_RISK"
    and TRADING_MODE == "live"
)
ALPACA_PAPER = not _allow_live

if not ALPACA_PAPER:
    print("🚨🚨🚨 WARNING: LIVE TRADING IS ENABLED — REAL MONEY AT RISK 🚨🚨🚨")

# ============ Risk Defaults (conservative) ============
# ⚠️ تعديل الخيار ب: رفعنا ATR multiplier لتوسيع Stop، وخفّضنا المخاطرة للتعويض
MAX_RISK_PER_TRADE = 0.005          # ← كان 0.0075
MAX_POSITION_PCT = 0.15
MAX_PORTFOLIO_EXPOSURE_PCT = 0.45
MAX_OPEN_POSITIONS = 3
MAX_TRADES_PER_DAY = 6
MAX_DAILY_LOSS_PCT = 0.02
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_MINUTES = 120

# ============ Stop / Target ============
# ⚠️ تعديل الخيار ب: Stop أوسع = صفقات أقل لكن اتجاهات حقيقية
STOP_ATR_MULTIPLIER = 2.5           # ← كان 1.5
TAKE_PROFIT_RR = 2.0
TRAILING_ACTIVATION_R = 1.0
TRAILING_DISTANCE_R = 0.75

# ============ Signal Thresholds ============
MIN_CONFIDENCE = 0.70
MIN_SIGNAL_SCORE = 70               # ← كان 60

# ============ Fees / Slippage ============
FEES_PER_SHARE = 0.0
ENTRY_SLIPPAGE_BPS = 5
EXIT_SLIPPAGE_BPS = 5

# ============ LLM Exit Sensitivity ============
MIN_HOLD_MINUTES = 30
MIN_LOSS_TO_CLOSE_PCT = 1.5
MIN_GAIN_TO_TRIM_PCT = 2.0
LLM_EXIT_MIN_CONFIDENCE = 0.70

# ============ Data / Symbols ============
SYMBOLS = ["AAPL", "NVDA", "AMD", "ASML", "CRM", "PANW", "TSLA", "JNJ", "LLY", "AMZN"]
CYCLE_INTERVAL_MINUTES = 15

# ============ Legacy fallback percentages ============
HARD_STOP_LOSS_PCT = 3.0
HARD_TAKE_PROFIT_PCT = 6.0
TRAILING_ACTIVATION_PCT = 2.0
TRAILING_STOP_PCT = 1.5

# ============ Exits monitor cadence ============
EXIT_CHECK_SECONDS = 60
EXIT_LLM_INTERVAL_SECONDS = 300
ALLOW_EXTENDED_HOURS = False

# ============ Data paths ============
DB_PATH = os.getenv("DB_PATH", "decisions.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "trading_memory.db")
GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "graph_checkpoints.db")

# ============ Legacy aliases ============
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
ACCOUNT_EQUITY_DEFAULT = 10000
ONLY_MARKET_HOURS = True
