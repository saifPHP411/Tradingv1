"""Single source of truth for execution-cost inputs and fill arithmetic.

WHY THIS MODULE EXISTS
    Two cost paths existed and they did not agree:

      backtest/engine.py        long exit: exit_price + slippage(..., is_buy=False)
      evaluation/ab_ai_context  long exit: exit_px   - slippage(..., is_buy=False)

    ``normalize_slippage`` already returns a NEGATIVE number for a sell, so
    subtracting it *added* money on exit. The engine is conservative and the
    simplified replay was optimistic by 2 x bps per round trip. A campaign
    must not inherit two different answers to "what does a trade cost?".

WHAT THIS MODULE IS NOT
    It is not a new cost model. It does not change any value, add a borrow
    cost, or invent a fee. It names the existing inputs once and implements
    the fill arithmetic once, so both callers can be shown to agree.

    The engine is deliberately NOT modified: it keeps its own inline calls,
    and a contract test proves those calls equal the helpers here.

FROZEN VALUES
    Nothing here is a decision. ``COST_SETTING_KEYS`` names the inputs; their
    values come from settings. Changing a value is a separate, reviewed act.
"""
from __future__ import annotations

from risk_utils import normalize_slippage

# The execution-cost inputs that change what a round trip costs. These are the
# only cost inputs the engine actually consumes (verified against
# backtest/engine.py); naming them here keeps them from drifting apart.
COST_SETTING_KEYS = (
    "fees_per_share",
    "entry_slippage_bps",
    "exit_slippage_bps",
)

# Neutral defaults, used only when a caller supplies no settings. These match
# the module-level fallbacks the callers already used.
COST_DEFAULTS = {
    "fees_per_share": 0.0,
    "entry_slippage_bps": 0.0,
    "exit_slippage_bps": 0.0,
}

# Short borrow (financing) is NOT modelled anywhere in this codebase. A
# campaign that trades shorts must record this and must not present short
# results as fully realistic.
SHORT_BORROW_COST_MODE = "NOT_MODELED"


def resolve_cost_inputs(settings: dict | None) -> dict:
    """Extract the three cost inputs from a settings mapping.

    Missing or malformed values fall back to ``COST_DEFAULTS``. Never raises:
    a bad settings object must not break a caller that is only trying to
    record what it would have used.
    """
    s = settings if isinstance(settings, dict) else {}
    out: dict = {}
    for key in COST_SETTING_KEYS:
        raw = s.get(key, COST_DEFAULTS[key])
        try:
            if isinstance(raw, bool):
                raise TypeError("bool is not a cost")
            out[key] = float(raw)
        except (TypeError, ValueError):
            out[key] = COST_DEFAULTS[key]
    return out


def canonical_cost_inputs(cost_inputs: dict | None) -> dict:
    """Normalise any cost mapping to the canonical three keys, sorted."""
    resolved = resolve_cost_inputs(cost_inputs)
    return {key: resolved[key] for key in COST_SETTING_KEYS}


# --------------------------------------------------------------------------
# Fill arithmetic — the single implementation both callers must match
# --------------------------------------------------------------------------

def entry_fill_price(price: float, is_buy: bool,
                     cost_inputs: dict) -> float:
    """Entry fill: pay the slippage. A buy pays more; a short-sell gets less."""
    inputs = canonical_cost_inputs(cost_inputs)
    return float(price) + normalize_slippage(
        float(price), inputs["entry_slippage_bps"], bool(is_buy))


def exit_fill_price(price: float, is_buy_exit: bool,
                    cost_inputs: dict) -> float:
    """Exit fill: give up the slippage.

    ``is_buy_exit`` is True when the exit is a BUY (i.e. closing a short), so
    the same sign convention as the engine applies: a closing buy pays more,
    a closing sell receives less.
    """
    inputs = canonical_cost_inputs(cost_inputs)
    return float(price) + normalize_slippage(
        float(price), inputs["exit_slippage_bps"], bool(is_buy_exit))


def entry_fee(qty: float, cost_inputs: dict) -> float:
    inputs = canonical_cost_inputs(cost_inputs)
    return float(qty) * inputs["fees_per_share"]


def exit_fee(qty: float, cost_inputs: dict) -> float:
    inputs = canonical_cost_inputs(cost_inputs)
    return float(qty) * inputs["fees_per_share"]


def round_trip_fee(qty: float, cost_inputs: dict) -> float:
    """Both legs' fees. Equal to entry_fee + exit_fee by construction."""
    return entry_fee(qty, cost_inputs) + exit_fee(qty, cost_inputs)


def is_short_borrow_modeled() -> bool:
    """Always False today. Exposed so a campaign can assert it explicitly."""
    return False
