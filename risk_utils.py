"""
Central risk helpers used by broker, monitor, backtest.
No component is allowed to invent stop/target logic outside these helpers.
"""

def normalize_price_for_symbol(symbol: str, price: float, tick_size: float = 0.01) -> float:
    if price is None or price <= 0:
        raise ValueError(f"invalid price {price} for {symbol}")
    return round(round(price / tick_size) * tick_size, 6)


def compute_r(entry_price: float, stop_price: float, side: str) -> float:
    if side in ("buy", "long"):
        r = entry_price - stop_price
    else:
        r = stop_price - entry_price
    return abs(r)


def validate_risk_plan(plan: dict) -> tuple[bool, str]:
    """Return (ok, reason). Reject any plan that could lead to unprotected/oversized trade."""
    for k in ("symbol", "side", "entry_price", "stop_loss", "take_profit", "qty", "risk_amount"):
        if k not in plan:
            return False, f"missing {k}"

    if plan["qty"] <= 0:
        return False, "qty <= 0"
    if plan["entry_price"] <= 0:
        return False, "entry_price <= 0"
    if plan["stop_loss"] <= 0 or plan["take_profit"] <= 0:
        return False, "invalid stop/target"

    if plan["side"] in ("buy", "long"):
        if not (plan["stop_loss"] < plan["entry_price"] < plan["take_profit"]):
            return False, "long: stop < entry < target violated"
    else:
        if not (plan["take_profit"] < plan["entry_price"] < plan["stop_loss"]):
            return False, "short: target < entry < stop violated"

    if plan["risk_amount"] <= 0:
        return False, "risk_amount <= 0"
    return True, "ok"


def validate_position_protection(position: dict, stop_order: dict | None) -> tuple[bool, str]:
    if position is None:
        return False, "no position"
    if stop_order is None:
        return False, "no stop order"
    if stop_order.get("qty", 0) + 1e-6 < position.get("qty", 0):
        return False, f"stop qty {stop_order['qty']} < position qty {position['qty']}"
    stop_side = (stop_order.get("side") or "").lower()
    pos_side = (position.get("side") or "").lower()
    expected = "sell" if pos_side in ("long", "buy") else "buy"
    if stop_side != expected:
        return False, f"stop side {stop_side} != expected {expected}"
    if not stop_order.get("stop_price"):
        return False, "stop price missing"
    return True, "ok"
