# AI Campaign Preflight Audit

Status: **audit + hardening**. No campaign has been started. No live Gemini
call has been made. No cost, threshold, `C_MIN` or strategy value has been
changed.

Audit base: branch `review/m4-m5-m7-followup`, commit
`beb779ea3fe89a3697c5193a48212c629604bdb6`.

## 0. Hardening status (this pass)

Four gaps found by the first audit have been closed. One remains by design.

| Blocker | Status | What changed |
|---|---|---|
| **B1** cost model not campaign-realistic | **BLOCKED** (by design) | Not touched. `FEES_PER_SHARE` is still `0.0`. Requires an explicit decision |
| **B2** costs outside `config_snapshot_hash` | **FIXED** | The hash now covers fees, both slippage legs and the short-borrow mode; the record also stores the cost inputs it used |
| **B3** ordering recorded, not enforced | **FIXED** | `observe()` now raises `ShadowOrderingError` before any call or write |
| **B4** no `feature_source` contract | **FIXED** | Contract documented and tested; a mismatching rebuild is flagged *and* diagnosed |
| **B7** no short borrow cost | **DOCUMENTED** | `short_borrow_cost_mode = "NOT_MODELED"` is recorded in every entry and in the hash. No borrow cost added |
| **B8** two different cost models | **FIXED** | A shared `execution_costs` module; the replay now delegates to it. A real sign bug was found and corrected |

### The B8 bug, precisely

`normalize_slippage` returns a **negative** number for a sell. The engine did
the right thing:

```
engine long exit:   exit_price + normalize_slippage(..., is_buy=False)   -> pays
```

The simplified replay did not:

```
replay long exit:   exit_px    - normalize_slippage(..., is_buy=False)   -> gains
```

Subtracting a negative *added* money, so the replay was optimistic by
`2 x bps` on every round trip. Measured end to end on the audit fixture:

| Replay | Final equity |
|---|---|
| old formula | 10574.92 |
| shared cost model | 10549.64 |
| **overstatement removed** | **$25.28** |

The engine's own numbers are unchanged (`10273.35`, 8 trades); only the
replay moved, toward the engine's convention. No cost value was altered.

## 1. Current status

Shadow Mode infrastructure exists and is reviewed. It records what the
advisory layer *would* say and marks a theoretical veto. It is **not wired
into any runtime path** and it changes nothing about trading.

| Item | State |
|---|---|
| `shadow_mode.py` | present, reviewed |
| `tests/test_shadow_mode.py` | present, 92 cases |
| `docs/AI_SHADOW_MODE.md` | present |
| `market_context.call_model()` | present, additive |
| Engine / strategy / risk / broker | **byte-identical** to the pre-audit base |
| Campaign runner | **does not exist** (intentional) |
| Live API key | not present |
| Live Gemini call | never made |

## 2. Frozen vs not frozen

Nothing is frozen yet. The table records what *must* be frozen immediately
before the first live run, and what currently blocks that.

| Item | Current value | Frozen? |
|---|---|---|
| `C_MIN` | `0.80` | pre-registered, **not yet frozen** |
| Model | `gemini-3.5-flash-lite` | not frozen |
| Prompt schema | v1, closed, no free text | not frozen |
| `maxOutputTokens` | `1024` | not frozen |
| Thinking level | `minimal` | not frozen |
| Symbol set | undecided | not frozen |
| Evaluation window `T0` | undecided | not frozen |
| Cost model | `FEES_PER_SHARE = 0.0` | **not fit for a campaign** |
| Feature source | not implemented | not frozen |

## 3. Invariant audit (PASS / FAIL)

Each row was checked empirically against the running code, not inferred from
documentation.

| # | Invariant | Result | Evidence |
|---|---|---|---|
| 1 | `veto_applied=False` always | **PASS** | Literal `"veto_applied": False` in source; no path assigns `True`; observed with a forced maximal veto |
| 2 | No import/call lets AI change an engine decision | **PASS** | `shadow_mode.py` does not import `backtest`, does not reference `simulate_portfolio`, has no order path; engine source does not mention `shadow_mode` |
| 3 | `trades(A) == trades(C)` trade-for-trade | **PASS** | Identical fingerprint (symbol, entry/exit time, entry/exit price, pnl); 8 trades, non-vacuous |
| 4 | `end_equity(A) == end_equity(C)` | **PASS** | `10273.35 == 10273.35`; equity curves identical too |
| 5 | Feature vector excludes engine-derived values | **PASS** | Sanitizer output is exactly the 8-field allow-list; `regime`, `size_multiplier`, `equity`, `future_close` all dropped |
| 6 | `feature_asof_ts` does not exceed `decision_ts` | **PASS** | Enforced in `observe()`: a violation raises `ShadowOrderingError` before any call or write. Was PARTIAL; see B3 |
| 7 | `feature_hash` reconstructible as-of | **PASS** | Rebuild from OHLCV truncated at `feature_asof_ts` produces an identical hash |
| 8 | `prompt_hash` / model / config hashes reproducible | **PASS** | Deterministic across repeats; `config_snapshot_hash` changes when any frozen input changes; key-order independent |
| 9 | `raw_response` contains no API key or secrets | **PASS** | Configured key string absent from the log; no `AIza` / private-key patterns; `raw_response` stored verbatim |
| 10 | Hash chain detects tampering | **PASS** | Edit, delete, reorder and forged insertion each detected; restored file verifies |
| 11 | No network call in the tests | **PASS** | Transports always injected; `socket.create_connection`/`getaddrinfo` counted 0 calls during `observe()` |

### Note on row 6 (now resolved)

The module used to record the three timestamps without rejecting an entry
whose `feature_asof_ts` was later than `decision_ts`. That gap is closed:
`observe()` now enforces `feature_asof_ts <= decision_ts <= execution_ts` and
raises before making a call or writing a record. Timestamps are never clamped,
substituted or normalised.

## 4. Cost-model audit

### Where cost is used

| Location | Role |
|---|---|
| `config.py:85-87` | the constants: `FEES_PER_SHARE`, `ENTRY_SLIPPAGE_BPS`, `EXIT_SLIPPAGE_BPS` |
| `settings.py:66-68, 282-284` | exposes them through `get_all_settings()` |
| `risk_utils.py:96` | `normalize_slippage()` — the single slippage helper |
| `backtest/engine.py:562` | entry fill = open + signed slippage |
| `backtest/engine.py:615` | `entry_fees = qty * fees_per_share` |
| `backtest/engine.py:681-684` | exit fill = exit + signed slippage; `exit_fees` |
| `backtest/engine.py:808-811` | force-close at end of data, same cost path |
| `backtest/engine.py:342, 355` | accounting: fees and slippage deducted |
| `backtest/metrics.py:139, 163` | aggregates `total_slippage` |
| `evaluation/ab_ai_context.py:114-190` | the simplified replay re-implements the same three inputs |

### Components of a trade's cost today

1. **Entry slippage** — `price * entry_slippage_bps / 10000`, signed so a buy
   pays more.
2. **Exit slippage** — same, signed the other way.
3. **Entry fee** — `qty * fees_per_share`.
4. **Exit fee** — `qty * fees_per_share`.

There is **no** commission-per-trade, no exchange fee, no SEC/TAF fee, no
borrow cost for shorts, and no spread model beyond the fixed-bps slippage.

### Answers to the specific questions

| Question | Answer |
|---|---|
| Is there a slippage model? | Yes — fixed bps, 5 in / 5 out, signed by direction. Applied on entry and on both exit paths. |
| Are fees applied to entry and exit? | Yes, both, as `qty * fees_per_share`. But `FEES_PER_SHARE = 0.0`, so the fee term is currently **zero**. |
| Do short and long pay the same? | Yes. The same `fees_per_share` and the same bps are used; only the slippage *sign* differs. There is no borrow cost, so shorts are in fact modelled as cheaper than reality. |
| Do backtester and forward/paper use the same model? | **No.** The backtester applies costs. The live/paper path (`monitor.py`, `broker.py`, `app.py`, `main.py`, `backtester.py`) contains **no** reference to fees or slippage at all — costs there are whatever the broker does. |

### Alpaca-specific confirmation (documentation only)

Two official statements sharpen the two rows above. See
`docs/AI_CAMPAIGN_FREEZE.md` §1A for the full audit and source URLs.

- **Commission is conditional.** Alpaca's disclosure scopes commission-free
  trading to *self-directed individual cash* accounts trading *US-listed
  securities through an API*; partner-established accounts and non-retail flow
  may differ. The user's account eligibility is unconfirmed, so a zero-commission
  assumption is `PENDING_USER_DECISION`, not a fact.
- **Paper does not reproduce these costs.** Alpaca states paper trading does not
  account for regulatory fees (along with market impact, latency slippage, queue
  position and price improvement). That is the forward/paper row above, confirmed
  by the broker.

### Measured impact on the A==C fixture (8 trades)

| Scenario | End equity | Delta |
|---|---|---|
| Current (`fees=0.0`, 5/5 bps) | 10273.35 | — |
| Fees at $0.005/share | 10272.99 | −0.36 |
| Fees at $0.01/share | 10272.63 | −0.72 |
| Slippage set to 0 | 10277.53 | +4.18 |

Total shares traded: 36. Slippage contributes $5.85 of cost; fees contribute
$0.00.

**Reading:** slippage is doing the work; the fee term is inert. Because the
advisory layer's only possible effect is to *remove* trades, a cost model that
understates per-trade cost systematically flatters the layer. At this fixture's
size the fee error is cents, but on a real multi-symbol campaign it scales with
share count and must be correct before any decision.

### Minimum change to make costs realistic

One line, and it must be a deliberate, reviewed change:

```
config.py:  FEES_PER_SHARE = 0.0   ->   the chosen per-share fee
```

Nothing else needs to change: both entry and exit already apply it, and the
slippage path already works. `FEES_PER_SHARE` was **not** changed by this audit.

## 5. Proposed campaign cost model (not applied)

> **Superseded in part by the Alpaca audit in
> `docs/AI_CAMPAIGN_FREEZE.md` §1A.** The earlier "$0.005 per share,
> Alpaca-style per-share commission" row below was a *placeholder guess* made
> before the broker's official schedule was read. It must not be used. Alpaca's
> commission is **conditional on account type**, and its regulatory fees
> (SEC / FINRA TAF / CAT) are **separate components with their own basis**, not
> a flat per-share commission. Treat the row below as withdrawn.

Explicit, freezable, and expressible in the existing settings keys so no code
change is needed beyond the constant above.

| Component | Proposed value | Where it lives | Rationale |
|---|---|---|---|
| Per-share fee | **withdrawn — see freeze doc §1A** | `FEES_PER_SHARE` | was a placeholder; real treatment is conditional commission plus separately-based regulatory fees |
| Entry slippage | 5 bps | `ENTRY_SLIPPAGE_BPS` | unchanged; conservative for liquid large caps |
| Exit slippage | 5 bps | `EXIT_SLIPPAGE_BPS` | unchanged; symmetric |
| Short borrow cost | **not modelled** | — | must be stated as a limitation, not hidden |
| Spread beyond slippage | **not modelled** | — | folded into the bps assumption |
| Regulatory fees | **not modelled** | — | must be recorded as fee *types* with their own basis, not as a flat per-share number |

Freeze requirements: the cost values, recorded in the campaign's
`config_snapshot_hash`. Costs are **now part of that hash** (blocker B2 was
fixed in the hardening pass: the hash covers `fees_per_share`,
`entry_slippage_bps`, `exit_slippage_bps` and `short_borrow_cost_mode`), and
each Shadow record also stores the `cost_inputs` it used.

Two honest caveats:

- A per-share fee is a *choice*, not a measurement. It must be justified
  against the broker's actual schedule before the run, and written down. The
  current placeholder `0.0` must not be carried forward unexamined.
- Shorts genuinely cost more (borrow). If the campaign trades shorts, the
  result is optimistic by an unquantified amount, and the report must say so.

## 6. Feature-source audit

All eight allow-listed features are produced by
`backtest.engine.compute_indicators_at(df, i)`, which uses `df.iloc[:i+1]` —
strictly the prefix ending at the decision bar.

| Feature | Derivation | Rebuildable exactly as-of from OHLCV? |
|---|---|---|
| `last_close` | `Close[i]` | Yes |
| `ema20` | EMA of `Close[0..i]` | Yes |
| `ema50` | EMA of `Close[0..i]` | Yes |
| `rsi14` | Wilder RSI over `Close[0..i]` | Yes |
| `atr14` | ATR over `High/Low/Close[0..i]` | Yes |
| `adx` | ADX over `High/Low/Close[0..i]` | Yes |
| `volume_ratio` | `Volume[i] / mean(Volume[i-19..i])` | Yes |
| `bb_bandwidth` | Bollinger bandwidth over `Close[0..i]` | Yes |

**Verified empirically:** rebuilding from a dataframe truncated at
`feature_asof_ts` yields a byte-identical `feature_hash`.

**No feature needs an extra source.** None requires SPY, fundamentals,
earnings, news or account state. The only precondition is `i >= 60`, below
which `compute_indicators_at` returns `None` and no observation can be made.

**Constraint to respect:** the source must be the raw OHLCV prefix, not the
engine's own indicator dict, if the rebuild is to be an *independent* check. If
the rebuild is fed the same dict the engine already produced, it proves nothing.
This is why `feature_source` is a caller-supplied callable rather than being
derived internally.

**Not implemented now:** no `feature_source` is wired anywhere. Supplying one
is a campaign-runner decision.

## 7. Reproducibility audit

| Artefact | Reproducible | Notes |
|---|---|---|
| `feature_hash` | Yes | Canonical JSON, sorted keys, 6-dp rounding |
| `prompt_hash` | Yes | Prompt is a pure function of symbol + sanitized features |
| `request_body_hash` | Yes | Hashes the exact bytes sent |
| `config_snapshot_hash` | Yes | Covers model, `C_MIN`, thinking level, token cap, schema version, prompt schema, required fields |
| `git_sha` | Yes | Best-effort; empty string if git is unavailable |
| `latency_ms` | No | Wall clock; expected to vary |
| `response_ts` / `request_ts` | No | Wall clock; expected to vary |

Excluded from the hash set by design: timestamps and latency. Everything that
determines *what was asked* is hashable and stable.

Gaps:

- Cost values are **not** in `config_snapshot_hash` (blocker B2).
- The prompt text itself is not stored, only its hash. Reproducing the prompt
  requires the feature vector and the builder, both of which are recoverable
  from the record.
- `git_sha` is empty if the run happens outside a git checkout.

## 8. Architecture check (forward run without touching the engine)

The required shape is:

```
Engine decision
  -> sanitized causal features
    -> Shadow AI observation
      -> append-only log
```

and **not** `AI -> Engine decision`.

This shape is already supported, and was demonstrated in this audit: an
external wrapper around `backtest.engine.compute_signal_score` calls
`original(...)` first, then passes the *result* of that call to
`ShadowRunner.observe()`, and returns the original result unchanged. The engine
source is not modified. `observe()` accepts a plain dict, so it cannot reach
later rows.

Verified in this audit: with the wrapper installed and 148 observations
recorded, arm A and arm C were identical trade-for-trade and in end equity.

The reverse direction is structurally impossible: `shadow_mode.py` has no
import of the engine, no order path, and returns a record the wrapper discards.

**Not built:** a campaign runner that installs this wrapper. That is a
deliberate omission.

## 9. Remaining blockers

| # | Blocker | Why it blocks | Suggested resolution (needs approval) |
|---|---|---|---|
| B1 | Cost model is not campaign-realistic (`FEES_PER_SHARE = 0.0`) | Understates per-trade cost, which flatters any filter that removes trades | Set the per-share fee deliberately; freeze it. **Still open — no value chosen.** Alpaca commission is conditional on account type and regulatory fees are separate components; see `docs/AI_CAMPAIGN_FREEZE.md` §1A |
| B2 | ~~Costs are outside `config_snapshot_hash`~~ | — | **FIXED.** The hash now covers fees, both slippage legs and the short-borrow mode; each record also stores its cost inputs |
| B3 | ~~Timestamp ordering is recorded, not enforced~~ | — | **FIXED.** `observe()` raises `ShadowOrderingError` before any call or write |
| B4 | ~~No `feature_source` wired~~ | — | **CONTRACT FIXED.** Documented and tested; mismatch is flagged and diagnosed. A real source is still the runner's job (B5) |
| B5 | No campaign runner | Nothing can start a run | Build it only after B1 and the freeze list are agreed |
| B6 | Live-API facts unverified | Whether the model accepts `thinkingConfig`, and whether 1024 tokens suffices, is unknown | One manual, watched smoke call before the campaign |
| B7 | Short borrow cost not modelled | Short results are optimistic by an unquantified amount | **DOCUMENTED, not modelled.** Every record carries `short_borrow_cost_mode = "NOT_MODELED"`. Alpaca charges $0 borrow on easy-to-borrow shares and does not currently offer shorting in hard-to-borrow names, which bounds the omission but does not remove it. See the long-only proposal below |
| B8 | ~~Backtester and live/paper cost models differ~~ | — | **FIXED for the two code paths that exist.** A shared `execution_costs` module; the replay delegates to it. The live/paper path still applies no cost of its own (B8b) |
| B8b | The live/paper path applies no cost model at all | A forward result is whatever the broker charges, not what the backtest assumed | Out of scope for this pass; must be documented, never silently equated with the backtest |
| B9 | Symbol set, window `T0`, primary metric not chosen | No pre-registration is possible yet | Decide and freeze before the run |
| B10 | `C_MIN`, prompt, model not frozen | Post-hoc tuning would invalidate the result | Freeze all three in one recorded step |

## 10. B7 — short borrow cost, and the long-only proposal

No borrow cost was added. `execution_costs.SHORT_BORROW_COST_MODE` is
`"NOT_MODELED"`, and every Shadow record and the `config_snapshot_hash` carry
that value, so a campaign cannot present short results as fully realistic.

**Proposal only — not frozen, not implemented.** The cleanest first campaign is
**long-only**, for three reasons:

1. Borrow cost is unmodelled, so short P&L is optimistic by an unknown amount.
2. Borrow availability and recall risk are not representable in this engine.
3. A long-only campaign has one fewer unquantified assumption, which makes the
   veto analysis easier to trust.

This is a recommendation for a decision, not a decision. If shorts are kept,
the campaign report must state the omission explicitly and must not quote short
results as though financing were included.

## 11. B8 — the shared cost contract

`execution_costs.py` is now the single place the cost inputs are named and the
fill arithmetic is implemented:

| Function | Meaning |
|---|---|
| `COST_SETTING_KEYS` | the three inputs the engine actually consumes |
| `resolve_cost_inputs(settings)` | extract them safely; never raises |
| `canonical_cost_inputs(...)` | the canonical, sorted form that gets hashed |
| `entry_fill_price(price, is_buy, inputs)` | entry fill |
| `exit_fill_price(price, is_buy_exit, inputs)` | exit fill |
| `entry_fee` / `exit_fee` / `round_trip_fee` | fees per leg and both legs |
| `is_short_borrow_modeled()` | always False today |

The engine is **not** modified. It keeps its inline calls, and a contract test
asserts those inline calls equal the helpers, so any future drift fails a test
rather than silently changing results.

`evaluation/ab_ai_context.py` now imports the helpers instead of re-deriving
them, which is what removed the exit-slippage sign bug.

A future campaign can therefore use the backtester's cost model by construction:
there is now one definition of it.

## 12. Checklist before the first live Gemini run

Every item must be true, and the outcome recorded, before any live call.

**Freeze**

- [ ] `C_MIN` frozen at its pre-registered value (currently `0.80`), unchanged
- [ ] Model string frozen
- [ ] Prompt schema v1 frozen; `config_snapshot_hash` recorded
- [ ] `maxOutputTokens` and thinking level frozen
- [ ] Symbol set frozen
- [ ] Evaluation window `T0` frozen, chosen to post-date model training
- [ ] Primary metric pre-registered (one metric, not a shopping list)
- [ ] Cost values frozen; `config_snapshot_hash` covers them (done)
- [ ] Short/long decision recorded, with `short_borrow_cost_mode` acknowledged

**Costs**

- [ ] `FEES_PER_SHARE` set to a justified, non-zero value
- [ ] Slippage bps reviewed and frozen
- [ ] Short borrow cost either modelled or explicitly declared as a limitation

**Correctness**

- [x] Timestamp ordering enforced, not merely recorded (B3 — done)
- [ ] `feature_source` supplied and proven to match on a sample (B4 contract done; source pending)
- [ ] `A == C` re-demonstrated on the real symbol set and window
- [ ] Hash chain verified over the whole run
- [ ] API key confirmed absent from every log line

**Operations**

- [ ] One watched smoke call confirms the request is accepted (B6)
- [ ] `maxOutputTokens` confirmed sufficient to avoid `incomplete`
- [ ] Rate-limit and budget behaviour confirmed under the campaign's cadence
- [ ] Kill switch tested
- [ ] Log path, rotation and retention decided

**Governance**

- [ ] Data-sharing terms for sending market data to the provider reviewed
- [ ] Independent review of the runner and the wrapper
- [ ] Rollback plan agreed
- [ ] Explicit go/no-go sign-off recorded

## 13. What this pass did not do

- No live Gemini call, no network request, no API key.
- No campaign started, no campaign runner created, no scheduler, no automatic
  veto, no automatic A/B deployment.
- No change to `C_MIN` (`0.80`), any threshold, any cost value, or any
  strategy, signal or execution rule.
- No change to `backtest/engine.py`, `signals.py`, `risk_engine.py`,
  `broker.py`, `agents.py`, `monitor.py`, `run_backtest.py`, `config.py`,
  `app.py`, `settings.py`, `state.py`, `regime.py`, `risk_utils.py`,
  `decision_log.py`, `backtest/metrics.py`, `backtest/monte_carlo.py` or
  `backtest/walkforward.py` — all byte-identical.
- No borrow cost added; no new fee value chosen.
- No change to the model, prompt, schema, thinking configuration or
  `maxOutputTokens`.
- No push, merge, PR or deploy.
