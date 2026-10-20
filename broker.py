from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import is_dry_run
from safety import check_safety_gates

def execute_order(order: dict) -> dict:
    if is_dry_run():
        return {"status": "dry_run", "message": f"كان سيتم تنفيذ: {order}"}

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"status": "skipped", "reason": "مفاتيح Alpaca غير موجودة"}

    allowed, reason = check_safety_gates()
    if not allowed:
        return {"status": "blocked", "reason": reason}

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

    # فحص القوة الشرائية قبل الإرسال
    try:
        acct = client.get_account()
        buying_power = float(acct.buying_power)
        est_cost = order.get("estimated_cost", order["qty"] * order.get("entry_price", 0))
        if order["side"] == "buy" and est_cost > buying_power:
            return {"status": "blocked",
                    "reason": f"قوة شرائية غير كافية (تحتاج ~${est_cost:.0f}, متاح ${buying_power:.0f})"}
    except Exception as e:
        print(f"⚠️ تعذر التحقق من القوة الشرائية: {e}")

    side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL

    try:
        req = MarketOrderRequest(
            symbol=order["symbol"],
            qty=float(order["qty"]),  # ← كسري
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=order["stop_loss"]),
            take_profit=TakeProfitRequest(limit_price=order["take_profit"]),
        )
        placed = client.submit_order(req)
        return {
            "status": "submitted",
            "order_id": str(placed.id),
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": order["qty"],
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}
