# Shadow Mode (Phase 1 of the approved AI design)

Status: **infrastructure only**. Not wired into the engine, the strategy, the
risk engine, the broker or the monitor. Nothing in this document has been run
against the live API.

## What Shadow Mode is

An append-only, tamper-evident record of what the advisory AI layer *would*
have said at each decision point, plus a *theoretical* veto flag. It answers
one question for later review: "if we had let this advisory layer veto entries,
what would it have wanted to veto?"

## What Shadow Mode is not

- It never creates, sizes, prices or executes a trade.
- Its output never influences trading. `veto_applied` is hard-coded `False`.
- There is no code path from `shadow_mode.py` into the engine.

The design was approved on these terms: the AI layer is **deletional only**.
The strongest action available to it is to mark, for later analysis, an entry
the engine had *already* chosen. It cannot add an entry.

## Frozen decisions

These are pre-registered. They must not be tuned once a campaign starts.

| Item | Value | Meaning |
|---|---|---|
| `C_MIN` | `0.80` | Confidence at or above which `UNFAVORABLE` becomes a veto *candidate* |
| Schema | v1, closed, no free text | Every field is a bounded enum or number |
| Model | `gemini-3.5-flash-lite` | Unchanged from the existing layer |
| Regime input | not sent | `regime` and `size_multiplier` are never sent to the model |
| Default | AI off | No key or `ai_context_enabled != 1` means no call |

Because the veto is only ever a *candidate*, changing `C_MIN` cannot change
trading. It changes which entries appear in the review list.

## The closed schema

The reply must contain exactly these keys. There is deliberately **no free-text
field**, so no prose can be stored or interpreted.

```json
{
  "trend_quality":     "STRONG_UP | WEAK_UP | FLAT | WEAK_DOWN | STRONG_DOWN | UNKNOWN",
  "volatility_regime": "LOW | NORMAL | HIGH | EXTREME | UNKNOWN",
  "momentum_quality":  "ACCELERATING | STEADY | DECELERATING | REVERSING | UNKNOWN",
  "market_risk":       "LOW | MEDIUM | HIGH | EXTREME | UNKNOWN",
  "confidence":        0.0,
  "recommendation":    "FAVORABLE | NEUTRAL | UNFAVORABLE | INSUFFICIENT_DATA"
}
```

Rejection rules (any failure ⇒ treated as `INSUFFICIENT_DATA`, never as a
veto):

- non-mapping, or any unknown key, or any missing key;
- any value outside its enum;
- `confidence` not a finite number in `[0, 1]`.

`recommendation` can never express an order. `BUY`, `SELL`, `HOLD`, `SHORT`,
`LONG` are not in the enum, so an order cannot be represented at all.

## The veto rule

```
veto_candidate = (recommendation == "UNFAVORABLE") and (confidence >= C_MIN)
veto_applied   = False        # always, in Shadow Mode
```

`FAVORABLE` is inert: it is not an entry signal. Only `UNFAVORABLE` can produce
a candidate, and only as a marker.

## Data sent at the decision bar

Only the existing allow-list of causal numeric indicators, exactly as
`market_context.sanitize_market_features` already enforces:

`last_close, ema20, ema50, rsi, atr14, adx, volume_ratio, bb_bandwidth`

Anything else — account, position, PnL, engine regime, size multiplier,
credentials, or any non-finite number — is dropped at that single choke point.

## Record shape

One JSON object per line (JSONL). Every entry is appended; there is no update
or delete path.

```json
{
  "seq": 10432,
  "schema_version": 1,
  "run_id": "20261103T155900Z",
  "prev_hash": "9f2c...",
  "entry_hash": "a41b...",
  "symbol": "AAPL",
  "decision_ts": "2026-11-03T15:59:00Z",
  "feature_asof_ts": "2026-11-03T15:59:00Z",
  "execution_ts": "2026-11-03T16:00:00Z",
  "features": {"last_close": 187.4, "ema20": 185.1, "ema50": 180.9,
               "rsi": 58.2, "atr14": 2.31, "adx": 27.4,
               "volume_ratio": 1.08, "bb_bandwidth": 0.041},
  "feature_hash": "b7d1...",
  "feature_rebuild": {"available": false, "matches": null, "rebuilt_hash": ""},
  "engine_context": {"would_enter": true, "side": "buy", "score": 74},
  "model": "gemini-3.5-flash-lite",
  "prompt_hash": "c93e...",
  "request_body_hash": "51af...",
  "config_snapshot_hash": "8de0...",
  "cost_inputs": {"fees_per_share": 0.0, "entry_slippage_bps": 5.0,
                  "exit_slippage_bps": 5.0},
  "short_borrow_cost_mode": "NOT_MODELED",
  "git_sha": "05b4f749274aa7136e6b7cc75727e65968cceda2",
  "request_ts": "2026-11-03T15:59:02Z",
  "response_ts": "2026-11-03T15:59:04Z",
  "latency_ms": 1812,
  "attempts": 1,
  "http_status": 200,
  "raw_response": "{\"trend_quality\":\"STRONG_UP\", ...}",
  "parse_status": "ok",
  "reason": "ok",
  "trend_quality": "STRONG_UP",
  "volatility_regime": "NORMAL",
  "momentum_quality": "STEADY",
  "market_risk": "LOW",
  "confidence": 0.91,
  "recommendation": "UNFAVORABLE",
  "veto_candidate": true,
  "veto_applied": false,
  "c_min": 0.8,
  "api_key_present": true,
  "budget": {"minute_count": 3, "day_count": 41}
}
```

### The API key is never logged

Only `api_key_present` (a boolean) is stored. A test asserts the configured
key string does not appear anywhere in the log file.

### Hashes

| Hash | Covers | Purpose |
|---|---|---|
| `feature_hash` | the sanitized feature vector | prove which inputs were used |
| `prompt_hash` | the exact prompt text | prove the prompt was not changed |
| `request_body_hash` | the exact request bytes | prove what was sent |
| `config_snapshot_hash` | model, `C_MIN`, thinking level, token cap, schema, **cost inputs, short-borrow mode** | prove the campaign ran under one frozen config |
| `entry_hash` / `prev_hash` | the whole entry, chained | tamper detection |

`config_snapshot_hash` covers the model, `C_MIN`, the thinking level, the
output-token cap, the schema version, the prompt schema, the required fields,
the three execution-cost inputs (fees per share, entry and exit slippage) and
the short-borrow mode. If any of these change, the hash changes. Costs are
included because two runs with different costs are different experiments even
when the model and prompt are identical.

### Causal ordering is enforced, not merely recorded

`observe()` requires `feature_asof_ts <= decision_ts <= execution_ts`. A
violation raises `ShadowOrderingError` (a `ValueError` subclass) **before** any
model call or log write, so no non-causal record can exist. Timestamps are
never clamped or substituted: a bad timestamp is a failure, not a
normalisation.

### Hash chain

`entry_hash = SHA-256(entry without entry_hash)`, and each entry stores the
previous entry's hash. Editing, deleting, reordering or inserting a line breaks
the chain. `verify_chain(path)` returns `(ok, reason)`.

### Feature reconstruction

The feature vector is a plain dict of causal indicators. When a
`feature_source` callable is supplied, the runner re-derives the vector from
OHLCV that ends at or before `feature_asof_ts` and stores its hash in
`feature_rebuild`. A mismatch is recorded, not hidden, so any leakage becomes
visible in the record itself.

## Proving the AI does not affect trading

The claim is tested directly against the real engine, not inferred:

- **A vs C, trade-for-trade.** Arm A runs the engine with AI off. Arm C runs
  the engine with the Shadow observer hooked into `compute_signal_score`, and
  the observer's result is discarded. The trade lists, the equity curves and
  the end equity must be identical. The fixture produces 8 real trades, so the
  comparison is not vacuous.
- **Maximal veto.** Even when every observation is forced to
  `UNFAVORABLE` at `confidence = 1.0`, trading is unchanged.
- **Disabled restores the original.** With AI disabled there is no behaviour
  change at all.
- **Structural.** `shadow_mode.py` does not import `backtest`, does not
  reference `simulate_portfolio`, and contains no order path.

The hook is an external wrapper around the engine's own score function. The
engine source is not modified — the same technique already used and reviewed in
`evaluation/ab_ai_context.run_engine_arm`.

## Failure handling

Every failure degrades to "no advisory input" and never to a veto:

| Condition | Behaviour |
|---|---|
| timeout | one retry, then recorded as failed |
| `429` | `Retry-After` honoured (capped 5s), one retry, then recorded |
| `5xx` | one retry, then recorded |
| `400/401/403/404/422` | no retry (a bad key cannot succeed on repeat) |
| malformed / truncated / non-JSON reply | rejected, recorded as failed |
| reply outside the closed schema | rejected, recorded as `schema_rejected` |
| missing API key | no call, recorded as `no_api_key` |
| budget exhausted | no call, recorded as rate-limited |

Each attempt consumes the local budget, so a retry cannot exceed the
per-minute or per-day limits. Failures are still appended and still chained.

## Known limitations

- The record format is versioned (`schema_version`); a future change must bump
  it, and old and new records must not be compared as if identical.
- `feature_rebuild` is only populated when the caller supplies a
  `feature_source`. Without it, reconstruction is unavailable and is recorded
  as such rather than assumed.
- `engine_context` is supplied by the caller and is not independently verified
  by this module.
- Shadow Mode has never been run against the live API. The generation-config
  questions documented in `AI_MARKET_CONTEXT.md` still apply.
- The record is only as trustworthy as the filesystem it is written to; the
  hash chain detects tampering, it does not prevent it.
- No campaign has started. `C_MIN`, the prompt, the model, the symbol set and
  the costs remain to be frozen immediately before a real forward-only run.
