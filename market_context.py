"""
Optional AI market-context layer backed by the Google Gemini API.

WHAT THIS IS
    A *standalone, advisory-only* helper that asks an LLM to describe current
    market conditions (bias / confidence / risk level) from a small, fixed set
    of trailing numeric market features.

WHAT THIS IS NOT
    - It never issues, sizes, prices or executes a trade.
    - It never returns a quantity, a side, a stop or a target.
    - It is DISABLED by default and, when enabled, is not wired into the
      strategy, the execution engine or risk sizing. A future change that
      connects it to trading must be a separate, explicitly-reviewed step
      after it has passed independent evaluation.
    - Failure is always safe: any error, timeout, rate limit or malformed
      response degrades to ``available=False`` and the original strategy runs
      completely unaffected.

SECURITY / PRIVACY
    - The API key is read from the ``GEMINI_API_KEY`` environment variable
      only. It is never hard-coded, logged, persisted or committed.
    - Only allow-listed numeric market features are ever sent. Account data,
      balances, positions, PnL, broker ids, keys and other symbols are never
      sent.
    - The model's reply is treated as UNTRUSTED input: it must parse as JSON
      and pass a strict schema/enum/range check or it is rejected outright.
      Model text is never executed, eval'd or interpreted as an instruction.

USAGE LIMITS
    A conservative in-process budget (per minute and per day) caps calls. It is
    a local guard, not a substitute for the provider's own quota.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Model string is configurable. Verify availability and quota for your own
# project in Google AI Studio before relying on it.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_MAX_CALLS_PER_MINUTE = 6
DEFAULT_MAX_CALLS_PER_DAY = 200

BIAS_VALUES = ("BULLISH", "BEARISH", "NEUTRAL")
RISK_VALUES = ("LOW", "MEDIUM", "HIGH")

# The ONLY market features allowed to leave the process. Everything else is
# dropped before the prompt is built. These are the labels used in the prompt.
_ALLOWED_MARKET_FIELDS = (
    "last_close",
    "ema20",
    "ema50",
    "rsi",
    "atr14",
    "adx",
    "volume_ratio",
    "bb_bandwidth",
)

# Accepted input spellings for a prompt label. The engine's indicator dict
# exposes RSI as ``rsi14`` (see backtest/engine.py:compute_indicators_at),
# while the prompt uses the shorter ``rsi``. Both are accepted so a caller can
# pass either the raw engine dict or the canonical label, and the value still
# reaches the request body.
_FIELD_ALIASES = {
    "rsi": ("rsi14", "rsi"),
}

_MAX_RATIONALE_CHARS = 280

# Hard caps so a malformed or hostile value can never produce an absurd
# timeout or an unbounded budget. Values outside these ranges are clamped.
_MAX_TIMEOUT_SECONDS = 120.0
_MAX_CALLS_PER_MINUTE = 600
_MAX_CALLS_PER_DAY = 100_000


# --------------------------------------------------------------------------
# Safe coercion (never raises)
# --------------------------------------------------------------------------


def _safe_int(value, default: int, *, minimum: int | None = None,
              maximum: int | None = None) -> int:
    """Coerce to int, never raising. Bad input yields ``default``.

    Booleans are rejected (``True`` is not a count). Text is accepted only if
    it parses as a whole number. The result is clamped to the given bounds.
    """
    if isinstance(value, bool):
        return default
    result = default
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if value == value and value not in (float("inf"), float("-inf")):
            result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            try:
                result = int(text, 10)
            except ValueError:
                result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_float(value, default: float, *, minimum: float | None = None,
                maximum: float | None = None) -> float:
    """Coerce to float, never raising. Bad input yields ``default``."""
    if isinstance(value, bool):
        return default
    result = default
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            try:
                result = float(text)
            except ValueError:
                result = default
    if result != result or result in (float("inf"), float("-inf")):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_flag(value, default: bool = False) -> bool:
    """Coerce a truthy flag, never raising.

    Accepts real booleans, numbers (``1``/``0``), and the textual forms
    ``"1"``/``"0"``/``"true"``/``"false"`` (any case). Anything else is
    ``default`` — so a malformed flag fails closed rather than enabling a
    feature.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return default
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    return default

# --------------------------------------------------------------------------
# Local usage budget
# --------------------------------------------------------------------------


class UsageBudget:
    """Thread-safe in-process call budget (per minute and per day).

    Purely a local guard so a runaway loop cannot hammer the provider. The
    provider enforces its own quota independently.
    """

    def __init__(self, max_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
                 max_per_day: int = DEFAULT_MAX_CALLS_PER_DAY):
        self.max_per_minute = max(0, int(max_per_minute))
        self.max_per_day = max(0, int(max_per_day))
        self._lock = threading.Lock()
        self._minute_stamp = 0.0
        self._minute_count = 0
        self._day_stamp = ""
        self._day_count = 0

    def _now(self) -> float:
        return time.monotonic()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def configure(self, max_per_minute: int, max_per_day: int) -> None:
        """Apply resolved limits (they may come from env/settings)."""
        with self._lock:
            self.max_per_minute = max(0, int(max_per_minute))
            self.max_per_day = max(0, int(max_per_day))

    def check(self) -> tuple[bool, str]:
        """Return (allowed, reason). Does not consume budget."""
        with self._lock:
            if self.max_per_minute <= 0 or self.max_per_day <= 0:
                return False, "budget_disabled"
            now = self._now()
            if now - self._minute_stamp >= 60.0:
                self._minute_stamp = now
                self._minute_count = 0
            today = self._today()
            if today != self._day_stamp:
                self._day_stamp = today
                self._day_count = 0
            if self._minute_count >= self.max_per_minute:
                return False, "rate_limited_per_minute"
            if self._day_count >= self.max_per_day:
                return False, "rate_limited_per_day"
            return True, "ok"

    def consume(self) -> None:
        with self._lock:
            now = self._now()
            if now - self._minute_stamp >= 60.0:
                self._minute_stamp = now
                self._minute_count = 0
            today = self._today()
            if today != self._day_stamp:
                self._day_stamp = today
                self._day_count = 0
            self._minute_count += 1
            self._day_count += 1


_BUDGET = UsageBudget()


def _reset_budget_for_tests() -> None:
    """Test hook: reset the module-level budget."""
    global _BUDGET
    _BUDGET = UsageBudget()


# --------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------


def _unavailable(reason: str, *, enabled: bool = False,
                 model: str = "") -> dict:
    """The single shape returned for every non-answer. Never raises."""
    return {
        "available": False,
        "enabled": bool(enabled),
        "reason": reason,
        "bias": "NEUTRAL",
        "confidence": 0.0,
        "risk_level": "MEDIUM",
        "rationale": "",
        "model": model,
        "advisory_only": True,
        "affects_trading": False,
    }


def _ok(*, bias: str, confidence: float, risk_level: str, rationale: str,
        model: str) -> dict:
    return {
        "available": True,
        "enabled": True,
        "reason": "ok",
        "bias": bias,
        "confidence": confidence,
        "risk_level": risk_level,
        "rationale": rationale,
        "model": model,
        "advisory_only": True,
        "affects_trading": False,
    }


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _cfg(settings: dict | None = None) -> dict:
    """Resolve configuration from env (key) and optional settings dict.

    The key comes from the environment ONLY. A settings dict can enable/disable
    and choose the model, but can never supply credentials. This function never
    raises: a malformed settings object or environment value falls back to a
    documented default, so a bad value cannot break the caller.
    """
    s = settings if isinstance(settings, dict) else {}
    raw_flag = s.get("ai_context_enabled")
    if raw_flag is None:
        raw_flag = os.getenv("AI_CONTEXT_ENABLED", "")
    return {
        "api_key": os.getenv("GEMINI_API_KEY", "") or "",
        "enabled": _safe_flag(raw_flag, default=False),
        "model": str(s.get("ai_context_model") or
                     os.getenv("GEMINI_MODEL", "") or DEFAULT_MODEL),
        "timeout": _safe_float(os.getenv("GEMINI_TIMEOUT_SECONDS", ""),
                               DEFAULT_TIMEOUT_SECONDS,
                               minimum=1.0, maximum=_MAX_TIMEOUT_SECONDS),
        "max_per_minute": _safe_int(os.getenv("GEMINI_MAX_CALLS_PER_MINUTE", ""),
                                    DEFAULT_MAX_CALLS_PER_MINUTE,
                                    minimum=1, maximum=_MAX_CALLS_PER_MINUTE),
        "max_per_day": _safe_int(os.getenv("GEMINI_MAX_CALLS_PER_DAY", ""),
                                 DEFAULT_MAX_CALLS_PER_DAY,
                                 minimum=1, maximum=_MAX_CALLS_PER_DAY),
    }


def is_enabled(settings: dict | None = None) -> bool:
    """True only when the feature flag is on AND a key is present.

    Never raises: a malformed settings object or flag reads as disabled.
    """
    cfg = _cfg(settings)
    return bool(cfg["enabled"] and cfg["api_key"])


# --------------------------------------------------------------------------
# Prompt construction (allow-list only)
# --------------------------------------------------------------------------


def sanitize_market_features(market_data: dict | None) -> dict:
    """Keep only allow-listed numeric, finite features.

    Anything that is not a finite real number is dropped. This is the single
    choke point that guarantees no account, position, PnL or credential data
    can reach the prompt even if a caller passes a richer dict.

    Input spellings are normalised to the prompt label, so the engine's
    ``rsi14`` becomes ``rsi`` and is not silently dropped.
    """
    out: dict = {}
    if not isinstance(market_data, dict):
        return out
    for label in _ALLOWED_MARKET_FIELDS:
        for key in _FIELD_ALIASES.get(label, (label,)):
            if key not in market_data:
                continue
            val = market_data[key]
            if isinstance(val, bool):
                continue
            if not isinstance(val, (int, float)):
                continue
            fval = float(val)
            if fval != fval or fval in (float("inf"), float("-inf")):  # NaN/Inf
                continue
            out[label] = round(fval, 6)
            break
    return out


def build_prompt(symbol: str, features: dict) -> str:
    """Deterministic prompt from sanitized features only.

    The symbol is restricted to a short uppercase ticker so it cannot smuggle
    free text into the prompt.
    """
    ticker = re.sub(r"[^A-Z0-9.\-]", "", str(symbol).upper())[:12]
    payload = json.dumps(features, sort_keys=True)
    return (
        "You are a market-conditions summarizer. Use ONLY the numeric "
        "features below; do not invent data and do not give trading advice.\n"
        f"Ticker: {ticker}\n"
        f"Features (JSON): {payload}\n"
        "Reply with ONLY a JSON object, no prose, matching exactly:\n"
        '{"bias": "BULLISH"|"BEARISH"|"NEUTRAL", '
        '"confidence": <number 0..1>, '
        '"risk_level": "LOW"|"MEDIUM"|"HIGH", '
        '"rationale": "<short reason, max 200 chars>"}'
    )


# --------------------------------------------------------------------------
# Response validation (untrusted input)
# --------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict | None:
    """Parse the first balanced JSON object in the text, or None."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except Exception:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def validate_context_payload(payload: object) -> dict | None:
    """Strictly validate the model's reply. Return None if not acceptable.

    Enforced:
      - must be a mapping
      - ``bias`` must be one of BIAS_VALUES (exact, case-normalized)
      - ``confidence`` must be a real number in [0, 1]
      - ``risk_level`` must be one of RISK_VALUES (exact, case-normalized)
      - ``rationale`` is coerced to a short, control-character-free string
    Unknown extra keys are ignored. Nothing from the payload is executed.
    """
    if not isinstance(payload, dict):
        return None

    bias = str(payload.get("bias", "")).strip().upper()
    if bias not in BIAS_VALUES:
        return None

    risk = str(payload.get("risk_level", "")).strip().upper()
    if risk not in RISK_VALUES:
        return None

    conf_raw = payload.get("confidence")
    if isinstance(conf_raw, bool) or not isinstance(conf_raw, (int, float)):
        return None
    conf = float(conf_raw)
    if conf != conf or conf in (float("inf"), float("-inf")):
        return None
    if not (0.0 <= conf <= 1.0):
        return None

    rationale = payload.get("rationale", "")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        rationale = str(rationale)
    rationale = re.sub(r"[\x00-\x1f\x7f]", " ", rationale).strip()
    rationale = rationale[:_MAX_RATIONALE_CHARS]

    return {
        "bias": bias,
        "confidence": round(conf, 4),
        "risk_level": risk,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# HTTP transport (stdlib only; no new dependency)
# --------------------------------------------------------------------------


def _post_generate_content(url: str, body: bytes, timeout: float) -> dict:
    """POST JSON and return the parsed body. Raises on transport error."""
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("non-object response")
    return parsed


def _extract_text(response: dict) -> str:
    """Pull candidate text out of a generateContent response, or ''."""
    try:
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(c for c in chunks if isinstance(c, str))
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def get_market_context(symbol: str, market_data: dict | None,
                       settings: dict | None = None,
                       *, _transport=None) -> dict:
    """Return an advisory market-context dict. NEVER raises.

    Args:
        symbol: ticker (sanitized to [A-Z0-9.-]).
        market_data: indicator dict; only allow-listed numeric fields are used.
        settings: optional settings dict; ``ai_context_enabled`` gates the call.
        _transport: test seam. Signature
            ``(url, body_bytes, timeout) -> dict``.

    Returns:
        A dict with ``available``, ``enabled``, ``reason``, ``bias``,
        ``confidence``, ``risk_level``, ``rationale``, ``model``,
        ``advisory_only`` and ``affects_trading``. On any failure,
        ``available`` is False and ``affects_trading`` stays False.
    """
    try:
        cfg = _cfg(settings)
    except Exception:  # _cfg is already safe; this is defence in depth
        cfg = {"api_key": "", "enabled": False, "model": DEFAULT_MODEL,
               "timeout": DEFAULT_TIMEOUT_SECONDS,
               "max_per_minute": DEFAULT_MAX_CALLS_PER_MINUTE,
               "max_per_day": DEFAULT_MAX_CALLS_PER_DAY}
    model = cfg["model"]

    try:
        if not cfg["enabled"]:
            return _unavailable("disabled", enabled=False, model=model)
        if not cfg["api_key"]:
            return _unavailable("no_api_key", enabled=True, model=model)

        features = sanitize_market_features(market_data)
        if not features:
            return _unavailable("no_market_features", enabled=True, model=model)

        _BUDGET.configure(cfg["max_per_minute"], cfg["max_per_day"])
        allowed, why = _BUDGET.check()
        if not allowed:
            return _unavailable(why, enabled=True, model=model)

        prompt = build_prompt(symbol, features)
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
        }).encode("utf-8")
        url = f"{_API_BASE}/{model}:generateContent?key={cfg['api_key']}"

        _BUDGET.consume()
        transport = _transport or _post_generate_content
        try:
            response = transport(url, body, cfg["timeout"])
        except Exception as exc:  # network, timeout, HTTP error, bad JSON
            return _unavailable(f"request_failed:{type(exc).__name__}",
                                enabled=True, model=model)

        text = _extract_text(response)
        if not text:
            return _unavailable("empty_response", enabled=True, model=model)

        parsed = _extract_json_object(text)
        if parsed is None:
            return _unavailable("unparseable_response", enabled=True,
                                model=model)

        validated = validate_context_payload(parsed)
        if validated is None:
            return _unavailable("schema_rejected", enabled=True, model=model)

        return _ok(bias=validated["bias"],
                   confidence=validated["confidence"],
                   risk_level=validated["risk_level"],
                   rationale=validated["rationale"],
                   model=model)
    except Exception as exc:  # absolute fail-safe: never break the caller
        return _unavailable(f"internal_error:{type(exc).__name__}",
                            enabled=cfg["enabled"], model=model)
