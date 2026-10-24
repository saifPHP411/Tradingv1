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
    It is not a claim of production readiness, and it is not an approval of any
    fee rate. No rate is invented here. Every rate defaults to ``0.0`` and is
    supplied by settings; the operator must confirm the official values, the
    rounding rules and the account eligibility before a campaign.

    The engine keeps its own portfolio accounting and position-sizing
    semantics. Only the *fee amount* computation is delegated here.

COST MODEL VERSION
    ``COST_MODEL_VERSION`` is bumped whenever the meaning of a cost input
    changes. It is part of the configuration snapshot hash, so a run under an
    older model cannot be confused with a run under a newer one.

TRANSACTION-AWARE FEES
    A single symmetric per-share number cannot express the real fee shapes.
    Three of the components below are side-dependent or notional-based:

      broker per-share commission   both sides, per share
      SEC transaction fee           SELLS ONLY, percentage of sell notional
      FINRA TAF                     SELLS ONLY, per share, optionally capped
      FINRA CAT                     BOTH sides, per executed share
      slippage                      both sides, direction-signed
      short borrow                  NOT MODELLED (see below)

    ``side_fee`` composes them for one side of one transaction and takes the
    side and the fill price, so a sell-only fee can never be charged on a buy.

FROZEN VALUES
    Nothing here is a decision. ``COST_SETTING_KEYS`` and
    ``ALL_COST_SETTING_KEYS`` name the inputs; their values come from settings.
    Changing a value is a separate, reviewed act.

FAIL-CLOSED
    ``resolve_cost_inputs`` keeps its historical fail-open behaviour so old
    records still hash. ``resolve_cost_inputs(..., strict=True)`` fails closed
    and is the entry point campaign code must use. See
    ``cost_inputs_problems``.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING

from risk_utils import normalize_slippage

# Bump when the *meaning* of a cost input changes, not when a value changes.
COST_MODEL_VERSION = 2

# The legacy cost inputs. Kept exactly as they were: the engine reads these
# three keys directly, and old records hash only these. Do not extend this
# tuple -- the transaction-aware components live in FEE_SETTING_KEYS below.
COST_SETTING_KEYS = (
    "fees_per_share",
    "entry_slippage_bps",
    "exit_slippage_bps",
)

# Transaction-aware fee components. All rates are unapproved placeholders until
# the operator confirms them against the official schedule; every default is
# 0.0 so an unconfigured run reproduces the legacy cost exactly.
#
#   fees_per_share      broker per-share commission, charged on BOTH sides
#   sec_fee_rate        SELL only, multiplied by the sell notional
#   taf_fee_per_share   SELL only, per executed share
#   taf_max_per_trade   SELL only cap on the TAF charge; <= 0 means uncapped
#   cat_fee_per_share   BOTH sides, per executed share
FEE_SETTING_KEYS = (
    "fees_per_share",
    "sec_fee_rate",
    "taf_fee_per_share",
    "taf_max_per_trade",
    "cat_fee_per_share",
)

SLIPPAGE_SETTING_KEYS = (
    "entry_slippage_bps",
    "exit_slippage_bps",
)

# Every numeric cost input. Strict campaign resolution requires all of them.
ALL_COST_SETTING_KEYS = FEE_SETTING_KEYS + SLIPPAGE_SETTING_KEYS

# Regulatory fees are conventionally rounded up to the nearest cent. That is a
# documented *option*, not a silent default: legacy behaviour is no rounding.
ROUNDING_NONE = "NONE"
ROUNDING_UP_TO_CENT = "UP_TO_CENT"
ROUNDING_MODES = (ROUNDING_NONE, ROUNDING_UP_TO_CENT)
DEFAULT_ROUNDING = ROUNDING_NONE

# Non-numeric cost-model options, hashed alongside the numeric inputs.
COST_OPTION_KEYS = ("regulatory_rounding",)

# Neutral defaults, used only when a caller supplies no settings. Every value
# is zero, so an unconfigured caller cannot be charged an invented fee.
COST_DEFAULTS = {
    "fees_per_share": 0.0,
    "entry_slippage_bps": 0.0,
    "exit_slippage_bps": 0.0,
    "sec_fee_rate": 0.0,
    "taf_fee_per_share": 0.0,
    "taf_max_per_trade": 0.0,
    "cat_fee_per_share": 0.0,
}

COST_OPTION_DEFAULTS = {
    "regulatory_rounding": DEFAULT_ROUNDING,
}

# Short borrow (financing) is NOT modelled anywhere in this codebase. A
# campaign that trades shorts must record this and must not present short
# results as fully realistic.
SHORT_BORROW_COST_MODE = "NOT_MODELED"


class CostConfigurationError(ValueError):
    """Raised when cost settings are missing or malformed in strict mode."""


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def _coerce(raw, default: float) -> float:
    """Return a usable float, or ``default`` when the value is unusable."""
    try:
        if isinstance(raw, bool):
            raise TypeError("bool is not a cost")
        return float(raw)
    except (TypeError, ValueError):
        return default


def cost_inputs_problems(settings: dict | None) -> list[str]:
    """Describe every reason ``settings`` is unfit for campaign use.

    Never raises. An empty list means the configuration is complete and
    parseable. A missing or malformed value is reported, not silently defaulted
    -- that is the whole point of failing closed.
    """
    problems: list[str] = []
    if not isinstance(settings, dict):
        return ["settings is not a mapping; no cost inputs available"]

    for key in ALL_COST_SETTING_KEYS:
        if key not in settings:
            problems.append(f"missing cost setting: {key}")
            continue
        raw = settings[key]
        if isinstance(raw, bool):
            problems.append(f"malformed cost setting: {key}={raw!r}")
            continue
        if isinstance(raw, str) and not raw.strip():
            problems.append(f"malformed cost setting: {key}={raw!r}")
            continue
        try:
            float(raw)
        except (TypeError, ValueError):
            problems.append(f"malformed cost setting: {key}={raw!r}")

    mode = settings.get("regulatory_rounding", DEFAULT_ROUNDING)
    if mode not in ROUNDING_MODES:
        problems.append(
            f"unknown regulatory_rounding: {mode!r} (expected one of "
            f"{list(ROUNDING_MODES)})")
    return problems


def resolve_cost_inputs(settings: dict | None, *, strict: bool = False) -> dict:
    """Extract the cost inputs from a settings mapping.

    Missing or malformed values fall back to ``COST_DEFAULTS``. This fail-open
    behaviour is retained so an old record with no costs still hashes.

    With ``strict=True`` the function instead fails closed: an incomplete or
    malformed configuration raises :class:`CostConfigurationError` rather than
    quietly charging zero. Campaign code must use ``strict=True``.
    """
    if strict:
        problems = cost_inputs_problems(settings)
        if problems:
            raise CostConfigurationError(
                "cost settings are not campaign-ready: " + "; ".join(problems))

    s = settings if isinstance(settings, dict) else {}
    out: dict = {}
    for key in ALL_COST_SETTING_KEYS:
        out[key] = _coerce(s.get(key, COST_DEFAULTS[key]), COST_DEFAULTS[key])
    return out


def resolve_cost_options(settings: dict | None, *, strict: bool = False) -> dict:
    """Extract the non-numeric cost-model options (currently the rounding mode).

    An unknown rounding mode is rejected in strict mode; otherwise it falls
    back to ``DEFAULT_ROUNDING``.
    """
    s = settings if isinstance(settings, dict) else {}
    mode = s.get("regulatory_rounding", DEFAULT_ROUNDING)
    if mode not in ROUNDING_MODES:
        if strict:
            raise CostConfigurationError(
                f"unknown regulatory_rounding: {mode!r}")
        mode = DEFAULT_ROUNDING
    return {"regulatory_rounding": mode}


def canonical_cost_inputs(cost_inputs: dict | None, *,
                          strict: bool = False) -> dict:
    """Normalise any cost mapping to the canonical three legacy keys, sorted.

    Only the three keys the engine reads directly. The transaction-aware
    components are canonicalised separately by ``canonical_cost_model``.
    """
    resolved = resolve_cost_inputs(cost_inputs, strict=strict)
    return {key: resolved[key] for key in COST_SETTING_KEYS}


def canonical_cost_model(cost_inputs: dict | None, *,
                         strict: bool = False) -> dict:
    """Normalise any cost mapping to every numeric cost key, sorted.

    This is the mapping the configuration snapshot hash binds.
    """
    resolved = resolve_cost_inputs(cost_inputs, strict=strict)
    return {key: resolved[key] for key in ALL_COST_SETTING_KEYS}


def resolve_engine_costs(settings: dict | None) -> tuple[dict, str]:
    """Resolve the cost inputs and rounding mode for a backtest/replay run.

    Honours the ``cost_model_strict`` flag in ``settings``: when it is set, a
    missing or malformed cost configuration fails closed instead of silently
    charging zero. Legacy runs leave the flag unset and keep the documented
    zero defaults, so existing fixtures are unaffected.
    """
    s = settings if isinstance(settings, dict) else {}
    strict = bool(s.get("cost_model_strict", False))
    inputs = resolve_cost_inputs(s, strict=strict)
    options = resolve_cost_options(s, strict=strict)
    return inputs, options["regulatory_rounding"]


# --------------------------------------------------------------------------
# Rounding
# --------------------------------------------------------------------------

def round_up_to_cent(amount: float) -> float:
    """Round a fee amount UP to the nearest cent, as regulatory fees are.

    Uses ``Decimal`` so binary-float representation cannot round a value such
    as 0.206 *down* to 0.20 when the intent is up to 0.21. A non-positive
    amount stays exactly 0.0 -- never 0.01.
    """
    if amount <= 0:
        return 0.0
    return float(Decimal(repr(float(amount))).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING))


def _apply_rounding(amount: float, rounding: str) -> float:
    if rounding == ROUNDING_UP_TO_CENT:
        return round_up_to_cent(amount)
    return float(amount)


# --------------------------------------------------------------------------
# Fee components -- each takes the transaction side explicitly
# --------------------------------------------------------------------------

def broker_commission(qty: float, cost_inputs: dict) -> float:
    """Per-share broker commission. Charged on BOTH sides.

    Whether a commission applies at all is an account-eligibility question the
    operator must answer; this function only applies the configured rate.
    """
    inputs = canonical_cost_model(cost_inputs)
    return abs(float(qty)) * inputs["fees_per_share"]


def sec_fee(price: float, qty: float, is_buy: bool,
            cost_inputs: dict) -> float:
    """SEC transaction fee. SELLS ONLY, on the sell notional.

    Basis is a rate multiplied by the trade value, not by the share count, so
    it must never be expressed as a per-share constant.
    """
    if is_buy:
        return 0.0
    inputs = canonical_cost_model(cost_inputs)
    sell_notional = abs(float(qty)) * float(price)
    return sell_notional * inputs["sec_fee_rate"]


def taf_fee(qty: float, is_buy: bool, cost_inputs: dict, *,
            rounding: str = DEFAULT_ROUNDING) -> float:
    """FINRA Trading Activity Fee. SELLS ONLY, per executed share, capped.

    The cap is applied before rounding, which is the order the schedule
    describes (per-share basis, then capped, then rounded up to the cent). A
    cap that is absent or non-positive means uncapped.
    """
    if is_buy:
        return 0.0
    inputs = canonical_cost_model(cost_inputs)
    amount = abs(float(qty)) * inputs["taf_fee_per_share"]
    cap = inputs["taf_max_per_trade"]
    if cap > 0:
        amount = min(amount, cap)
    return _apply_rounding(amount, rounding)


def cat_fee(qty: float, is_buy: bool, cost_inputs: dict, *,
            rounding: str = DEFAULT_ROUNDING) -> float:
    """FINRA Consolidated Audit Trail fee. BOTH sides, per executed share.

    ``is_buy`` is accepted so the caller must state the side even though the
    charge is symmetric today; if the basis ever becomes side-dependent, only
    this function changes.
    """
    inputs = canonical_cost_model(cost_inputs)
    amount = abs(float(qty)) * inputs["cat_fee_per_share"]
    return _apply_rounding(amount, rounding)


def side_fee(qty: float, price: float, is_buy: bool, cost_inputs: dict, *,
             rounding: str = DEFAULT_ROUNDING) -> float:
    """Total transaction fee for ONE side of a trade.

    ``price`` is the actual fill price, so the sell-only, notional-based SEC
    fee is computed from the proceeds actually received.
    """
    return (broker_commission(qty, cost_inputs)
            + sec_fee(price, qty, is_buy, cost_inputs)
            + taf_fee(qty, is_buy, cost_inputs, rounding=rounding)
            + cat_fee(qty, is_buy, cost_inputs, rounding=rounding))


# --------------------------------------------------------------------------
# Legacy symmetric helpers -- behaviour preserved exactly
# --------------------------------------------------------------------------

def entry_fee(qty: float, cost_inputs: dict) -> float:
    """Legacy symmetric entry fee. Equals ``qty * fees_per_share``."""
    inputs = canonical_cost_inputs(cost_inputs)
    return float(qty) * inputs["fees_per_share"]


def exit_fee(qty: float, cost_inputs: dict) -> float:
    """Legacy symmetric exit fee. Equals ``qty * fees_per_share``."""
    inputs = canonical_cost_inputs(cost_inputs)
    return float(qty) * inputs["fees_per_share"]


def round_trip_fee(qty: float, cost_inputs: dict) -> float:
    """Both legs' legacy fees. Equal to entry_fee + exit_fee by construction."""
    return entry_fee(qty, cost_inputs) + exit_fee(qty, cost_inputs)


# --------------------------------------------------------------------------
# Fill arithmetic -- the single implementation both callers must match
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


def is_short_borrow_modeled() -> bool:
    """Always False today. Exposed so a campaign can assert it explicitly."""
    return False
