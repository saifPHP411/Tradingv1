import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = True

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

SYMBOLS = ["AAPL", "MSFT", "NVDA"]
MAX_RISK_PER_TRADE = 0.02
MIN_CONFIDENCE = 0.65

# ===== طبقات الأمان (جديد) =====
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"  # افتراضيًا: لا تنفيذ فعلي
MAX_DAILY_LOSS_PCT = 0.05      # إيقاف التنفيذ إذا خسرنا 5% من رأس المال اليوم
MAX_OPEN_POSITIONS = 3          # لا نفتح صفقة إذا كان لدينا 3 مراكز مفتوحة
MAX_TRADES_PER_DAY = 5          # حد أقصى لعدد الصفقات في اليوم
ONLY_MARKET_HOURS = True        # لا تنفيذ خارج ساعات السوق الأمريكي
ACCOUNT_EQUITY_DEFAULT = 10000  # رأس مال افتراضي إذا فشل جلب القيمة الحقيقية