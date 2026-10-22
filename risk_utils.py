"""
Central risk helpers used by broker, monitor, backtest, and portfolio simulation.
No component is allowed to invent stop/target/slippage logic outside these helpers.
"""
from __future__ import annotations

import math


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


# ============ Portfolio-level helpers (shared by live + backtest) ============

def compute_position_size(equity: float, price: float, stop_distance: float,
                          max_risk_per_trade: float, risk_multiplier: float,
                          max_position_pct: float) -> float:
    """
    Pure function: risk-based sizing + position cap. No side effects.
    Returns 0 if inputs are invalid.
    """
    if equity <= 0 or price <= 0 or stop_distance <= 0:
        return 0.0
    risk_amount = equity * max_risk_per_trade * risk_multiplier
    qty = risk_amount / stop_distance
    max_notional = equity * max_position_pct
    if qty * price > max_notional:
        qty = max_notional / price
    return max(0.0, qty)


def current_exposure(positions: dict) -> float:
    """Sum of absolute market value of all open positions."""
    return sum(abs(p.qty * p.entry_price) for p in positions.values())


def validate_portfolio_exposure(equity: float, positions: dict,
                                new_notional: float,
                                max_portfolio_exposure_pct: float) -> tuple[bool, str]:
    current = current_exposure(positions)
    cap = equity * max_portfolio_exposure_pct
    if current + new_notional > cap:
        return False, (f"exposure {current + new_notional:.2f} > cap {cap:.2f}")
    return True, "ok"


def normalize_slippage(price: float, bps: float, is_buy: bool) -> float:
    """
    Signed slippage in price terms:
      - Buy entry / Short exit → positive (worse: pay more)
      - Sell entry / Long exit → negative (worse: receive less)
    """
    slip = price * (bps / 10000.0)
    return slip if is_buy else -slip


# ============ Broker-compatible sizing ============

# Alpaca stock orders: a fractional quantity is only valid for a MARKET + DAY
# order. A protective STOP order must be whole-share. Therefore any entry that
# needs a protective stop must be sized in WHOLE shares.
MIN_WHOLE_SHARES = 1


def whole_share_qty(raw_qty: float) -> int:
    """Floor a risk-based quantity to a whole, broker-stoppable share count.

    Fractional sizes are intentionally not used: Alpaca cannot attach a valid
    stop order to a fractional position, and this project never opens an
    unprotected position.
    """
    if raw_qty is None or not math.isfinite(float(raw_qty)) or raw_qty <= 0:
        return 0
    return int(math.floor(float(raw_qty)))


def can_protect_with_stop(qty: int, price: float, stop_price: float,
                          side: str = "buy") -> tuple[bool, str]:
    """Whether ``qty`` can be entered AND protected with a valid stop order.

    A valid protected trade requires at least one whole share and a stop price
    strictly on the correct side of the entry. Sub-one-share risk budgets are
    rejected here (the caller must skip the trade, never inflate capital).
    """
    if qty is None or qty < MIN_WHOLE_SHARES:
        return False, (f"qty {qty} < {MIN_WHOLE_SHARES} whole share — "
                       f"cannot place a valid protective stop")
    if price is None or price <= 0:
        return False, f"invalid entry price {price}"
    if stop_price is None or stop_price <= 0:
        return False, f"invalid stop price {stop_price}"
    if side in ("buy", "long") and stop_price >= price:
        return False, f"long stop {stop_price} must be < entry {price}"
    if side in ("sell", "short") and stop_price <= price:
        return False, f"short stop {stop_price} must be > entry {price}"
    return True, "ok"
