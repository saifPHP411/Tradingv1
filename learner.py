from memory_layer import reflect_and_store
from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

def learn_from_closed_trades():
    if not ALPACA_API_KEY:
        return
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    # جلب آخر الأوامر المغلقة
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=20))
    for o in orders:
        # هنا نبني سجل الصفقة من بيانات الأمر (يمكن إثراؤه بالبيانات الفنية وقت الدخول)
        record = {
            "symbol": o.symbol,
            "decision": o.side.value,
            "analyst_reason": "غير محفوظ في الأمر",
            "pnl_pct": "غير متوفر مباشرة",
        }
        reflect_and_store(record)