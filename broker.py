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
    side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL

    try:
        req = MarketOrderRequest(
            symbol=order["symbol"],
            qty=order["qty"],
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
