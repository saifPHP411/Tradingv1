# AI Campaign Freeze Specification (DRAFT)

Status: **DRAFT — nothing here is frozen yet.** No campaign has been started.
No live Gemini call has been made. No network request, no API key, no paper or
live trading, no campaign runner.

This document does one thing: it states exactly what must be decided before a
forward-only campaign can begin, and which decisions are already settled. It
does not choose any value that requires a decision by the operator.

Base: branch `review/m4-m5-m7-followup`, commit
`142d9eeccdc7ba428067f2c9a6b066318b6dcf53`.

Marker conventions used below:

| Marker | Meaning |
|---|---|
| **FROZEN** | already decided; changing it after the campaign starts makes it a new campaign |
| **PENDING_USER_DECISION** | the operator must choose; this document must not |
| **PROPOSED** | a recommendation awaiting a decision; not binding |
| **DERIVED** | fixed by the current implementation, not a free choice |

## 1. Configuration structure

| Parameter | Proposed value | Status | Frozen? |
|---|---|---|---|
| AI mode | Shadow Mode only (advisory, veto candidate) | FROZEN | Yes |
| `C_MIN` | `0.80` | FROZEN | Yes |
| Model | `gemini-3.5-flash-lite` | PENDING_USER_DECISION | No |
| Prompt version | `1` (embedded `PROMPT_SCHEMA`) | DERIVED | No — freeze by hash |
| Prompt hash | `PENDING_USER_DECISION` (recorded at run) | PENDING_USER_DECISION | No |
| Schema version | `1` | DERIVED | No — freeze by hash |
| Temperature | `0.0` | DERIVED (hard-coded in `_build_generation_config`) | Yes, as implemented |
| Thinking level | `minimal` | PENDING_USER_DECISION | No |
| Max output tokens | `1024` | PENDING_USER_DECISION | No |
| Symbols | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Timeframe | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Campaign start `T0` | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Campaign end | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Decision frequency | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| LONG/SHORT | `PROPOSED: LONG-only` | PROPOSED | No |
| Cost model | shared `execution_costs` | FROZEN (implementation) | Yes |
| `FEES_PER_SHARE` | `0.0` (unchanged) | **PENDING_USER_DECISION** | No |
| `ENTRY_SLIPPAGE_BPS` | `5` (unchanged) | PENDING_USER_DECISION | No |
| `EXIT_SLIPPAGE_BPS` | `5` (unchanged) | PENDING_USER_DECISION | No |
| `SHORT_BORROW_COST_MODE` | `NOT_MODELED` | FROZEN | Yes |
| Primary metric | `delta_total_return_pct(B − A)` | PROPOSED | No |
| Secondary metrics | see §5 | PROPOSED | No |
| Minimum observations | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Minimum vetoes | `PENDING_USER_DECISION` | PENDING_USER_DECISION | No |
| Failure policy | fail-open to baseline | FROZEN | Yes |
| Retry policy | max 2 attempts total (1 retry) | FROZEN (as implemented) | Yes |
| Feature source | `PENDING_USER_DECISION` (injected; no live feed) | PENDING_USER_DECISION | No |
| Feature reconstruction requirement | must match, else flagged | FROZEN (contract) | Yes |
| Logging format | JSONL, hash-chained, append-only | FROZEN | Yes |
| Git SHA | recorded per record | FROZEN (mechanism) | Yes |

`config_snapshot_hash` is the single value that binds the model, `C_MIN`,
thinking level, token cap, schema version, prompt schema, required fields,
the three cost inputs and the short-borrow mode. It must be recorded at the
start of the campaign and must not change during it.

At the current defaults the reference hash is:

```
2465a88ac2272616b7a3839f93614ebb06e1246bde21141b2fe9d378bf1c7708
```

This value is **illustrative only**. It reflects today's defaults
(`gemini-3.5-flash-lite`, `minimal`, `1024`, schema `1`, and the cost inputs
as currently configured). It must be recomputed and re-recorded once the model,
thinking level, token cap and cost values are decided. It is not a frozen
campaign hash.

## 2. Frozen decisions

These are settled. None of them may be changed after the campaign begins
without declaring a new campaign.

1. **`C_MIN = 0.80`.** Pre-registered. It only affects which entries appear in
   the veto-candidate list; it never changes trading.
2. **`veto_applied = False`, always.** Hard-coded, not configurable. There is
   no code path that can set it to True.
3. **The AI has no authority to create orders.** It cannot open, size, price,
   route or execute anything.
4. **The AI cannot express BUY / SELL / HOLD.** Those verbs are not in the
   closed `recommendation` enum (`FAVORABLE`, `NEUTRAL`, `UNFAVORABLE`,
   `INSUFFICIENT_DATA`), so an order cannot even be represented.
5. **The AI can only produce a `recommendation` from the closed schema.**
   Six fields, no free text, strict enums, `confidence` bounded to `[0, 1]`.
   Anything else is rejected as `INSUFFICIENT_DATA`.
6. **Failure behaviour is fail-open to baseline.** On any failure — timeout,
   `429`, `5xx`, malformed or schema-violating reply, missing key, exhausted
   budget — the advisory layer contributes nothing and the engine behaves
   exactly as it would with AI off. It never blocks a trade.
7. **Maximum attempts = 2.** `_MAX_RETRIES = 1`, so one retry after the first
   attempt, bounded by the same per-call deadline and budget.
8. **Temperature = `0.0`**, as the current implementation already sets.
9. **`SHORT_BORROW_COST_MODE = "NOT_MODELED"`.** No borrow cost is modelled,
   and every record states so.
10. **Causal ordering is enforced.** `feature_asof_ts <= decision_ts <=
    execution_ts`, or the observation is rejected before any call or write.
11. **The feature vector is the eight-field allow-list** and excludes
    `regime`, `size_multiplier`, `equity`, `future_close` and any other
    engine-derived value.
12. **Shadow Mode changes nothing.** Arms A and C are identical trade-for-trade
    and in end equity, proven against the real engine.

**LONG-only is PROPOSED, not frozen.** The current architecture does not force
an explicit decision from the operator: Shadow Mode works identically for long
and short, and the engine's `enable_short` setting is independent of this
layer. The long-only proposal rests on the unmodelled borrow cost (§3) and is
recorded as a recommendation only. It becomes frozen only if the operator
explicitly decides it.

## 3. Pending user decisions

These must be decided by the operator. This document deliberately leaves them
open.

### 3.1 `FEES_PER_SHARE`

```
Current value : 0.0
STATUS        : PENDING_USER_DECISION
```

No value is proposed here. `0.0` must **not** be used for the final campaign
unless the operator explicitly decides that the experiment represents
**zero-commission execution** and records that assumption. Using `0.0` without
that decision would understate per-trade cost, which systematically flatters
any filter that removes trades — the exact bias this layer is being tested for.

### 3.1a Cost fallback differs from the configured value

A trap worth stating. When `resolve_cost_inputs` is given no settings, or a
value it cannot read, it falls back to `COST_DEFAULTS`:

```
{'fees_per_share': 0.0, 'entry_slippage_bps': 0.0, 'exit_slippage_bps': 0.0}
```

But `config.py` currently sets `ENTRY_SLIPPAGE_BPS = 5` and
`EXIT_SLIPPAGE_BPS = 5`. So a campaign that forgets to pass settings would
silently hash and record **zero slippage**, not the configured 5 bps. The
campaign must pass the real settings explicitly, and the recorded
`cost_inputs` must be checked against `config.py` before the run is trusted.
This is a wiring requirement, not a code defect, and no value is changed here.

### 3.2 Symbols

```
symbols       : PENDING_USER_DECISION
```

No symbol set is chosen here. `AAPL` and `MSFT` were used only in tests and
must not be carried into a campaign as if they were selected. No data
collection has begun.

### 3.3 Time window

```
T0            : PENDING_USER_DECISION
campaign_end  : PENDING_USER_DECISION
```

`T0` must post-date the model's training data and must be chosen before the
first live call. No window is proposed here.

### 3.4 Other open items

| Item | Status |
|---|---|
| Model string | PENDING_USER_DECISION |
| Thinking level | PENDING_USER_DECISION |
| Max output tokens | PENDING_USER_DECISION |
| Timeframe | PENDING_USER_DECISION |
| Decision frequency | PENDING_USER_DECISION |
| Minimum observations | PENDING_USER_DECISION |
| Minimum vetoes | PENDING_USER_DECISION |
| Feature source implementation | PENDING_USER_DECISION |
| Primary metric (confirm) | PENDING_USER_DECISION |
| LONG-only vs LONG+SHORT | PENDING_USER_DECISION |

## 4. Proposed but not frozen

| Item | Proposal | Note |
|---|---|---|
| LONG-only first campaign | Recommended | Avoids one unquantified assumption (borrow cost). Not binding |
| Primary metric | `delta_total_return_pct(B − A)` | Must be confirmed before the run |
| Campaign length | 12–24 months | PROPOSED |
| Symbol count | `>= 5` | PROPOSED |
| Minimum vetoes | `>= 100` | PROPOSED |
| Minimum total trades | `>= 200` preferred | PROPOSED |
| Statistical plan | §6 | PROPOSED |

None of these are frozen. All require an explicit decision.

## 5. Metrics

### Primary (proposed, not frozen)

```
primary_metric = delta_total_return_pct(B - A)
```

where A is the engine baseline and B is the engine with the Shadow layer
observing. Because Shadow Mode cannot change trading, B and A are expected to
be identical on returns; the primary metric is recorded to *demonstrate* that,
not to claim an effect.

### Secondary (proposed, not frozen)

- `delta_end_equity`
- `delta_max_drawdown`
- `delta_trade_count`
- `delta_win_rate`
- `delta_profit_factor`
- `veto_rate` (veto candidates / observations)
- `AI failure rate` (non-`ok` parse status / observations)
- `AI agreement / stability` (how often the same input yields the same
  recommendation and a confidence within a stated band)

**The final metric set requires a decision before the campaign starts.** Once
chosen, it must not be changed after seeing results.

## 6. Statistical methodology (plan only — nothing implemented)

Documented as a plan. No test has been run and no code was written.

- **Pre-registered hypothesis.** Stated before the run: Shadow Mode does not
  change trading outcomes; the advisory layer's veto candidates are recorded
  for later analysis. The hypothesis and the decision rule are fixed in
  advance.
- **B vs A primary comparison.** The primary comparison is B against A on the
  pre-registered primary metric.
- **Permutation test.** For a difference in the primary metric, use a
  permutation test that respects the paired structure of the two arms.
- **Block / stationary bootstrap.** For serially dependent return series, use
  a block or stationary bootstrap rather than an i.i.d. bootstrap, so
  autocorrelation and volatility clustering are preserved.
- **Paired comparisons where applicable.** Where the same decision points are
  compared across arms, use paired methods rather than unpaired ones.
- **Deflated Sharpe ratio.** If more than one prompt or configuration is
  tried, apply a Deflated Sharpe correction so multiple-testing inflation is
  accounted for.
- **No implementation now.** The plan is documented only.

## 7. Critical rule after freeze

Once frozen, the following must not be changed after seeing campaign results:

- prompt
- model
- `C_MIN`
- symbols
- time window
- cost model
- primary metric
- statistical plan

**If any of these changes, the result is a new campaign.** The prior campaign's
records remain, but they cannot be pooled with the new run or presented as one
experiment.

## 8. B8b status

```
B8b = BLOCKED
```

**Reason:** the live/paper execution path applies **no cost model**. It contains
no reference to fees or slippage; costs there are whatever the broker charges.
The backtester, by contrast, applies fees and slippage, and the simplified
replay now shares the same model via `execution_costs`.

Consequence for the future runner: **any campaign runner must not allow A/B
results that are inconsistent with the cost model.** Because the two paths
disagree, a forward result and a backtest result are not directly comparable,
and no runner may present them as if they were. This is not fixed in this
phase and is not in scope here.

## 9. LIVE API PREFLIGHT

A separate checklist, to be run **independently of campaign data**. No item
here has been executed. This preflight must not write to the campaign log and
must not use campaign symbols or the campaign window.

**Endpoint and model**

- [ ] Endpoint reachable: `POST /v1beta/models/{model}:generateContent`
- [ ] Chosen model string is accepted by the endpoint
- [ ] Model name is confirmed current, not deprecated or renamed

**Authentication**

- [ ] A valid key is supplied via the environment, never in code or logs
- [ ] The key is confirmed absent from every log line after a call
- [ ] An invalid key produces a clean, non-retried failure (`401`/`403`)

**Request shape**

- [ ] `thinkingConfig.thinkingLevel` is accepted for this model
- [ ] If it is rejected, the fallback (omit `thinkingConfig`) is confirmed
- [ ] `maxOutputTokens = 1024` is accepted
- [ ] `maxOutputTokens = 1024` is sufficient to avoid `incomplete` replies
      under the real prompt
- [ ] `temperature = 0.0` is accepted

**Response behaviour**

- [ ] A well-formed reply parses and validates against the closed schema
- [ ] The closed schema holds in practice (exactly six fields, enums only)
- [ ] A deliberately malformed reply is rejected, not coerced
- [ ] A truncated reply (`finishReason` indicating incompleteness) is detected

**Timing and failure**

- [ ] Actual latency is measured and recorded
- [ ] The timeout value is adequate for observed latency
- [ ] Retry behaviour is confirmed: at most one retry, within the deadline
- [ ] `429` honours `Retry-After` (capped) and does not loop
- [ ] `5xx` retries once and then fails open
- [ ] Budget exhaustion fails open without a call

**Isolation**

- [ ] The preflight uses a throwaway log path, not the campaign log
- [ ] The preflight uses no campaign symbol or window
- [ ] The preflight result is recorded separately and labelled as preflight

## 10. What this document is not

- It is not a campaign runner and does not create one.
- It does not start data collection.
- It does not choose any value marked `PENDING_USER_DECISION`.
- It does not freeze any item marked `PROPOSED`.
- It makes no live API call and uses no network.
