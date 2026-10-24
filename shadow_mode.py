"""Shadow Mode: auditable, advisory-only AI market-context logging.

WHAT THIS IS
    An append-only, tamper-evident record of what the advisory AI layer *would*
    have said at each decision point, alongside a *theoretical* veto. It is the
    first phase of the approved design: **Shadow Mode only**.

WHAT THIS IS NOT
    - It NEVER creates, sizes, prices or executes a trade.
    - Its output NEVER influences trading. ``veto_applied`` is hard-coded
      ``False``. There is no code path from this module into the engine.
    - It is not wired into ``backtest/engine.py``, the strategy, the risk
      engine, the broker or the monitor.

DESIGN DECISIONS (frozen before the campaign; see docs/AI_SHADOW_MODE.md)
    - ``C_MIN = 0.80``: the pre-registered confidence threshold at which a
      ``UNFAVORABLE`` recommendation becomes a *veto candidate*. It is
      reported for later analysis only; it changes nothing.
    - ``recommendation`` is DELETIONAL: it can only ever mark an entry the
      engine had already chosen as a veto candidate. It cannot create entries.
    - No ``regime`` or ``size_multiplier`` is sent to the model.
    - The schema is closed and has NO free-text field: every value is an enum
      or a bounded number, so no prose can enter the record or the prompt.
    - The API key is never logged; only ``api_key_present``.
    - The module never performs a network call by itself: the caller injects
      the transport, and tests always inject a fake one.

INTEGRATION SHAPE (not implemented here; future phase)
    The intended wiring is an external wrapper around the engine's
    ``compute_signal_score`` hook, exactly like
    ``evaluation/ab_ai_context.run_engine_arm``. The engine source is not
    modified. Because this module returns a record and nothing else, and the
    wrapper ignores it, arms A and C are identical by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import market_context as mc
import execution_costs as ec

# --------------------------------------------------------------------------
# Pre-registered constants (frozen; do NOT tune after a campaign starts)
# --------------------------------------------------------------------------

# Confidence at or above which an UNFAVORABLE recommendation is a veto
# candidate. Pre-registered before the forward-only campaign.
C_MIN = 0.80

# The only schema version accepted in the first release.
SCHEMA_VERSION = 1

# Causal-ordering violations are raised, never silently repaired. The
# exception is a ValueError subclass so a caller that already catches
# ValueError keeps working.
ORDERING_VIOLATION_PREFIX = "shadow_ordering_violation"


class ShadowOrderingError(ValueError):
    """Raised when timestamps are not causal. Never repaired, only rejected."""


# Closed enums. A reply containing anything else is rejected outright.
TREND_QUALITY_VALUES = (
    "STRONG_UP", "WEAK_UP", "FLAT", "WEAK_DOWN", "STRONG_DOWN", "UNKNOWN",
)
VOLATILITY_REGIME_VALUES = ("LOW", "NORMAL", "HIGH", "EXTREME", "UNKNOWN")
MOMENTUM_QUALITY_VALUES = (
    "ACCELERATING", "STEADY", "DECELERATING", "REVERSING", "UNKNOWN",
)
MARKET_RISK_VALUES = ("LOW", "MEDIUM", "HIGH", "EXTREME", "UNKNOWN")
RECOMMENDATION_VALUES = (
    "FAVORABLE", "NEUTRAL", "UNFAVORABLE", "INSUFFICIENT_DATA",
)

# A reply must contain exactly these keys: no more, no fewer. There is
# deliberately no free-text field.
REQUIRED_FIELDS = (
    "trend_quality",
    "volatility_regime",
    "momentum_quality",
    "market_risk",
    "confidence",
    "recommendation",
)

# Every field in the returned record, in order. Used by tests to assert the
# record shape is stable.
RECORD_FIELDS = (
    "seq", "schema_version", "run_id", "prev_hash", "entry_hash",
    "symbol", "decision_ts", "feature_asof_ts", "execution_ts",
    "features", "feature_hash",
    "model", "prompt_hash", "request_body_hash", "config_snapshot_hash",
    "git_sha",
    "request_ts", "response_ts", "latency_ms", "attempts", "http_status",
    "raw_response", "parse_status", "reason",
    "trend_quality", "volatility_regime", "momentum_quality", "market_risk",
    "confidence", "recommendation",
    "veto_candidate", "veto_applied", "c_min",
    "api_key_present", "budget",
)

# Schema text embedded in the prompt. Kept here so the prompt is derived from
# the same closed enums the validator enforces.
PROMPT_SCHEMA = (
    '{"trend_quality": "STRONG_UP"|"WEAK_UP"|"FLAT"|"WEAK_DOWN"|"STRONG_DOWN"'
    '|"UNKNOWN", '
    '"volatility_regime": "LOW"|"NORMAL"|"HIGH"|"EXTREME"|"UNKNOWN", '
    '"momentum_quality": "ACCELERATING"|"STEADY"|"DECELERATING"|"REVERSING"'
    '|"UNKNOWN", '
    '"market_risk": "LOW"|"MEDIUM"|"HIGH"|"EXTREME"|"UNKNOWN", '
    '"confidence": <number 0..1>, '
    '"recommendation": "FAVORABLE"|"NEUTRAL"|"UNFAVORABLE"'
    '|"INSUFFICIENT_DATA"}'
)


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------

def _canonical(obj) -> bytes:
    """Deterministic JSON bytes: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def sha256_hex(data) -> str:
    """SHA-256 of bytes, str or a JSON-serialisable object."""
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = _canonical(data)
    return hashlib.sha256(raw).hexdigest()


def feature_hash(features: dict) -> str:
    """Stable hash of the sanitized feature vector."""
    return sha256_hex(features)


def prompt_hash(prompt: str) -> str:
    return sha256_hex(prompt)


def request_body_hash(body: bytes) -> str:
    return sha256_hex(body)


def config_snapshot_hash(*, model: str, c_min: float, thinking_level: str,
                         max_output_tokens: int, schema_version: int,
                         cost_inputs: dict | None = None,
                         short_borrow_cost_mode: str | None = None) -> str:
    """Hash of everything that must stay frozen for the campaign.

    If any of these change, the hash changes, so a later reviewer can prove the
    campaign ran under one configuration.

    ``cost_inputs`` covers the execution-cost inputs that change what a round
    trip costs (fees per share, entry and exit slippage). They belong here
    because two runs with different costs are different experiments even when
    the model, prompt and ``C_MIN`` are identical. Passing None yields the
    neutral defaults, so an old record with no costs still hashes.

    ``short_borrow_cost_mode`` records whether short financing was modelled.
    It is part of the snapshot so a campaign cannot quietly claim a cost basis
    it did not use.
    """
    costs = ec.canonical_cost_inputs(cost_inputs)
    borrow = (short_borrow_cost_mode
              if short_borrow_cost_mode is not None
              else ec.SHORT_BORROW_COST_MODE)
    return sha256_hex({
        "model": model,
        "c_min": c_min,
        "thinking_level": thinking_level,
        "max_output_tokens": max_output_tokens,
        "schema_version": schema_version,
        "prompt_schema": PROMPT_SCHEMA,
        "required_fields": list(REQUIRED_FIELDS),
        "cost_inputs": costs,
        "short_borrow_cost_mode": borrow,
    })


def _require_ordered(*, feature_asof_ts, decision_ts, execution_ts) -> None:
    """Enforce ``feature_asof_ts <= decision_ts <= execution_ts``.

    Raises :class:`ShadowOrderingError` on violation. Nothing is repaired: no
    min/max, no clamping, no substitution. A non-causal observation must not
    be written at all, because a record that looks authoritative while
    describing future data is worse than no record.

    A ``None`` bound is treated as "not supplied" and is skipped. A bound that
    IS supplied must be mutually comparable, else that is itself a violation.
    """
    checks = (
        ("feature_asof_ts", feature_asof_ts, "decision_ts", decision_ts),
        ("decision_ts", decision_ts, "execution_ts", execution_ts),
    )
    for a_name, a_val, b_name, b_val in checks:
        if a_val is None or b_val is None:
            continue
        try:
            violates = a_val > b_val
        except TypeError as exc:
            raise ShadowOrderingError(
                f"{ORDERING_VIOLATION_PREFIX}: {a_name} and {b_name} are not "
                f"comparable ({type(a_val).__name__} vs {type(b_val).__name__})"
            ) from exc
        if violates:
            raise ShadowOrderingError(
                f"{ORDERING_VIOLATION_PREFIX}: {a_name}={a_val!r} is after "
                f"{b_name}={b_val!r}; a causal observation requires "
                f"{a_name} <= {b_name}"
            )


def git_sha(repo_dir: str | os.PathLike | None = None) -> str:
    """Best-effort short git SHA. Never raises; returns '' on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------
# Closed-schema validation (no free text)
# --------------------------------------------------------------------------

def _enum(value, allowed: tuple) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text if text in allowed else None


def _confidence(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    conf = float(value)
    if conf != conf or conf in (float("inf"), float("-inf")):
        return None
    if not (0.0 <= conf <= 1.0):
        return None
    return round(conf, 4)


def validate_shadow_payload(payload: object) -> dict | None:
    """Strictly validate a Shadow Mode reply. Return None if unacceptable.

    Rejects: non-mappings, unknown keys, missing keys, wrong types, values
    outside the closed enums, out-of-range confidence. There is no free-text
    field, so no prose can be stored or interpreted. Nothing is executed.
    """
    if not isinstance(payload, dict):
        return None
    if set(payload.keys()) != set(REQUIRED_FIELDS):
        return None

    trend = _enum(payload.get("trend_quality"), TREND_QUALITY_VALUES)
    vol = _enum(payload.get("volatility_regime"), VOLATILITY_REGIME_VALUES)
    mom = _enum(payload.get("momentum_quality"), MOMENTUM_QUALITY_VALUES)
    risk = _enum(payload.get("market_risk"), MARKET_RISK_VALUES)
    rec = _enum(payload.get("recommendation"), RECOMMENDATION_VALUES)
    conf = _confidence(payload.get("confidence"))

    if None in (trend, vol, mom, risk, rec) or conf is None:
        return None

    return {
        "trend_quality": trend,
        "volatility_regime": vol,
        "momentum_quality": mom,
        "market_risk": risk,
        "confidence": conf,
        "recommendation": rec,
    }


def build_shadow_prompt(symbol: str, features: dict) -> str:
    """Deterministic prompt with the closed schema and no free-text field."""
    ticker = mc.build_prompt(symbol, features).split("\n")[1]
    payload = json.dumps(features, sort_keys=True)
    return (
        "You are a market-conditions classifier. Use ONLY the numeric "
        "features below; do not invent data and do not give trading advice.\n"
        f"{ticker}\n"
        f"Features (JSON): {payload}\n"
        "Reply with ONLY a JSON object, no prose, with EXACTLY these keys:\n"
        f"{PROMPT_SCHEMA}\n"
        "Use UNKNOWN / INSUFFICIENT_DATA when the data does not support a "
        "judgement. Never include any other key."
    )


# --------------------------------------------------------------------------
# Veto candidate rule (deletion-only, theoretical)
# --------------------------------------------------------------------------

def is_veto_candidate(recommendation: str, confidence: float,
                      *, c_min: float = C_MIN) -> bool:
    """True when the advisory layer *would* veto an entry the engine chose.

    Purely descriptive. The caller records this value and does nothing with
    it. It can never create an entry, only mark one for later analysis.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return False
    return recommendation == "UNFAVORABLE" and conf >= float(c_min)


# --------------------------------------------------------------------------
# Append-only, hash-chained audit log
# --------------------------------------------------------------------------

GENESIS_HASH = "0" * 64


class ShadowLog:
    """Append-only JSONL log with a SHA-256 hash chain.

    Each entry stores ``prev_hash`` and its own ``entry_hash`` computed over
    the entry content *including* ``prev_hash``. Editing, reordering, inserting
    or deleting any line breaks the chain and is detectable by
    :func:`verify_chain`.

    Thread-safe. Only appends; there is no update or delete method.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._last_hash = GENESIS_HASH
        if self.path.exists():
            self._recover()

    def _recover(self) -> None:
        """Resume the chain from an existing log without rewriting it."""
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        break
                    self._seq = int(entry.get("seq", self._seq))
                    self._last_hash = str(entry.get("entry_hash",
                                                    self._last_hash))
        except Exception:
            pass

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def append(self, entry: dict) -> dict:
        """Append one entry, returning it with seq/prev_hash/entry_hash set."""
        with self._lock:
            self._seq += 1
            record = dict(entry)
            record["seq"] = self._seq
            record["prev_hash"] = self._last_hash
            record.pop("entry_hash", None)
            record["entry_hash"] = self._chain_hash(record)
            line = json.dumps(record, sort_keys=True, default=str)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._last_hash = record["entry_hash"]
            return record

    @staticmethod
    def _chain_hash(record: dict) -> str:
        material = {k: v for k, v in record.items() if k != "entry_hash"}
        return sha256_hex(material)


def verify_chain(path: str | os.PathLike) -> tuple[bool, str]:
    """Verify the hash chain. Return (ok, reason)."""
    p = Path(path)
    if not p.exists():
        return True, "empty"
    prev = GENESIS_HASH
    expected_seq = 0
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            expected_seq += 1
            try:
                entry = json.loads(line)
            except Exception:
                return False, f"line {lineno}: unparseable"
            if entry.get("prev_hash") != prev:
                return False, f"line {lineno}: prev_hash mismatch"
            if entry.get("seq") != expected_seq:
                return False, f"line {lineno}: seq {entry.get('seq')} != {expected_seq}"
            if entry.get("entry_hash") != ShadowLog._chain_hash(entry):
                return False, f"line {lineno}: entry_hash mismatch"
            prev = entry["entry_hash"]
    return True, "ok"


# --------------------------------------------------------------------------
# Shadow runner
# --------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    return mc._safe_int(os.getenv(name, ""), default, minimum=1)


class ShadowRunner:
    """Records what the advisory layer would say. Changes nothing.

    The model call is delegated to :func:`market_context.call_model`, so the
    transport, retry, budget, generation config and parsing logic are shared
    with the existing layer rather than rebuilt.
    """

    def __init__(self, log: ShadowLog, *, run_id: str = "",
                 repo_dir: str | os.PathLike | None = None,
                 transport=None, clock=None, feature_source=None):
        self.log = log
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.repo_dir = repo_dir
        self._transport = transport
        self._clock = clock or (lambda: time.time())
        self._feature_source = feature_source
        self._git_sha = git_sha(repo_dir)

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(self._clock()))

    def _config_hash(self, cfg: dict, settings: dict | None = None) -> str:
        return config_snapshot_hash(
            model=cfg.get("model", mc.DEFAULT_MODEL),
            c_min=C_MIN,
            thinking_level=cfg.get("thinking_level", mc.DEFAULT_THINKING_LEVEL),
            max_output_tokens=cfg.get("max_output_tokens",
                                      mc.DEFAULT_MAX_OUTPUT_TOKENS),
            schema_version=SCHEMA_VERSION,
            cost_inputs=ec.resolve_cost_inputs(settings),
            short_borrow_cost_mode=ec.SHORT_BORROW_COST_MODE,
        )

    def _rebuild_features(self, market_data, feature_asof_ts) -> dict:
        """Re-derive the feature vector from an INDEPENDENT source.

        Contract for ``feature_source``
            A callable ``feature_source(feature_asof_ts) -> mapping`` that
            returns the *raw* indicator mapping available up to and including
            ``feature_asof_ts`` — that is, derived from OHLCV whose last bar is
            at or before that timestamp. It must NOT be handed, or return, the
            engine's ready-made indicator dict: re-hashing the engine's own
            output would prove nothing about causality.

            The mapping is passed through the same allow-list sanitizer as the
            sent vector, then hashed. ``matches`` is True only when the rebuilt
            hash equals the sent hash.

        A mismatch, an exception, or a missing source is recorded as such. It
        is never hidden and never silently upgraded to a match.
        """
        sent = mc.sanitize_market_features(market_data)
        sent_hash = feature_hash(sent)
        base = {"available": False, "matches": None, "rebuilt_hash": "",
                "source": "none"}
        if self._feature_source is None:
            return base
        if feature_asof_ts is None:
            return {**base, "error": "missing_feature_asof_ts"}
        try:
            rebuilt_raw = self._feature_source(feature_asof_ts)
        except Exception as exc:
            return {**base, "source": "injected",
                    "error": f"{type(exc).__name__}"}
        if not isinstance(rebuilt_raw, dict):
            return {**base, "source": "injected",
                    "error": f"non_mapping:{type(rebuilt_raw).__name__}"}
        rebuilt = mc.sanitize_market_features(rebuilt_raw)
        rebuilt_hash = feature_hash(rebuilt)
        out = {
            "available": True,
            "source": "injected",
            "rebuilt_hash": rebuilt_hash,
            "matches": rebuilt_hash == sent_hash,
        }
        if not out["matches"]:
            # Record WHICH fields differ, so a leak is diagnosable and not
            # merely flagged.
            differing = sorted(
                key for key in set(sent) | set(rebuilt)
                if sent.get(key) != rebuilt.get(key))
            out["differing_fields"] = differing
            out["sent_hash"] = sent_hash
        return out

    def observe(self, *, symbol: str, market_data: dict,
                decision_ts, feature_asof_ts=None, execution_ts=None,
                settings: dict | None = None,
                engine_context: dict | None = None) -> dict:
        """Make one advisory call and append an audit record.

        Raises :class:`ShadowOrderingError` when the supplied timestamps are
        not causal (``feature_asof_ts <= decision_ts <= execution_ts``). The
        check runs before anything else, so a non-causal call makes no network
        request and writes no record. Timestamps are never repaired.

        Apart from that ordering violation, never raises: any other failure is
        recorded in the returned record. The caller is expected to ignore the
        record.
        """
        # Fail closed BEFORE any call or write. A record that looks valid
        # while describing future data is worse than no record at all.
        _require_ordered(feature_asof_ts=feature_asof_ts,
                         decision_ts=decision_ts,
                         execution_ts=execution_ts)

        request_ts = self._now_iso()
        features = mc.sanitize_market_features(market_data)

        cfg: dict = {}
        try:
            cfg = mc._cfg(settings)
        except Exception:
            cfg = {"model": mc.DEFAULT_MODEL,
                   "thinking_level": mc.DEFAULT_THINKING_LEVEL,
                   "max_output_tokens": mc.DEFAULT_MAX_OUTPUT_TOKENS}

        # The prompt/body hashes must be computed from the exact bytes sent,
        # so the Shadow prompt and validator are passed into call_model and
        # the body it used is read back from the outcome.
        prompt = build_shadow_prompt(symbol, features)

        outcome = mc.call_model(
            symbol, market_data, settings,
            _transport=self._transport,
            _prompt_builder=build_shadow_prompt,
            _validator=validate_shadow_payload,
        )
        body = outcome.get("body_bytes") or b""
        raw_text = outcome.get("raw_text", "")

        parse_status = "ok" if outcome.get("available") else (
            outcome.get("reason", "unknown"))
        # call_model already ran validate_shadow_payload; reuse its result.
        parsed = outcome.get("parsed") if outcome.get("available") else None
        if outcome.get("available") and parsed is None:
            parse_status = "shadow_schema_rejected"

        recommendation = parsed["recommendation"] if parsed else "INSUFFICIENT_DATA"
        confidence = parsed["confidence"] if parsed else 0.0
        veto_candidate = bool(
            parsed and is_veto_candidate(recommendation, confidence))

        budget_snapshot = {}
        try:
            budget_snapshot = {
                "minute_count": getattr(mc._BUDGET, "_minute_count", None),
                "day_count": getattr(mc._BUDGET, "_day_count", None),
            }
        except Exception:
            budget_snapshot = {}

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "symbol": str(symbol).upper()[:12],
            "decision_ts": str(decision_ts) if decision_ts is not None else None,
            "feature_asof_ts": (str(feature_asof_ts)
                                if feature_asof_ts is not None else None),
            "execution_ts": (str(execution_ts)
                             if execution_ts is not None else None),
            "features": features,
            "feature_hash": feature_hash(features),
            "feature_rebuild": self._rebuild_features(market_data,
                                                      feature_asof_ts),
            "engine_context": dict(engine_context) if engine_context else {},
            "model": cfg.get("model", mc.DEFAULT_MODEL),
            "prompt_hash": prompt_hash(prompt),
            "request_body_hash": request_body_hash(body),
            "config_snapshot_hash": self._config_hash(cfg, settings),
            "cost_inputs": ec.canonical_cost_inputs(
                ec.resolve_cost_inputs(settings)),
            "short_borrow_cost_mode": ec.SHORT_BORROW_COST_MODE,
            "git_sha": self._git_sha,
            "request_ts": request_ts,
            "response_ts": self._now_iso(),
            "latency_ms": outcome.get("latency_ms", 0),
            "attempts": outcome.get("attempts", 0),
            "http_status": outcome.get("http_status"),
            "raw_response": raw_text,
            "parse_status": parse_status,
            "reason": outcome.get("reason", ""),
            "trend_quality": parsed["trend_quality"] if parsed else "UNKNOWN",
            "volatility_regime": (parsed["volatility_regime"]
                                  if parsed else "UNKNOWN"),
            "momentum_quality": (parsed["momentum_quality"]
                                 if parsed else "UNKNOWN"),
            "market_risk": parsed["market_risk"] if parsed else "UNKNOWN",
            "confidence": confidence,
            "recommendation": recommendation,
            "veto_candidate": veto_candidate,
            # Shadow Mode: never applied. Hard-coded, not configurable.
            "veto_applied": False,
            "c_min": C_MIN,
            "api_key_present": bool(outcome.get("api_key_present")),
            "budget": budget_snapshot,
        }
        return self.log.append(record)


# --------------------------------------------------------------------------
# Read-only analysis helpers (never touch trading)
# --------------------------------------------------------------------------

def read_records(path: str | os.PathLike) -> list[dict]:
    """Read all records. Read-only; used by tests and analysis."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(records: list[dict]) -> dict:
    """Descriptive counts over records. No trading implication."""
    total = len(records)
    vetoes = sum(1 for r in records if r.get("veto_candidate"))
    applied = sum(1 for r in records if r.get("veto_applied"))
    return {
        "total": total,
        "veto_candidates": vetoes,
        "veto_applied": applied,
        "c_min": C_MIN,
        "parse_status_counts": _counts(records, "parse_status"),
        "recommendation_counts": _counts(records, "recommendation"),
    }


def _counts(records: list[dict], key: str) -> dict:
    out: dict = {}
    for r in records:
        value = r.get(key)
        out[value] = out.get(value, 0) + 1
    return out
