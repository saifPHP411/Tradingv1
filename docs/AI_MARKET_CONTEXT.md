# AI Market Context (optional, advisory-only)

This document explains the experimental Gemini layer. Read the limitations
before drawing any conclusion from it.

## What it is

`market_context.py` is a standalone helper that asks Google Gemini to
summarise current market conditions (bias / confidence / risk level) from a
small set of trailing numeric indicators.

It is **advisory-only**:

- it never issues, sizes, prices or executes a trade;
- it never returns a quantity, side, stop or target;
- it is **disabled by default**;
- it is **not** wired into the strategy, execution engine or risk sizing.

A future change that connects it to trading must be a separate, explicitly
reviewed step, only after it passes independent evaluation.

## Setup

1. Get a key from Google AI Studio (https://aistudio.google.com/apikey).
2. Export it as an environment variable. Never commit it.

```bash
export GEMINI_API_KEY="your-key-here"
```

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | API key. Read from the environment only. |
| `AI_CONTEXT_ENABLED` | `false` | Master switch (also `ai_context_enabled` in settings). |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model string. |
| `GEMINI_TIMEOUT_SECONDS` | `12` | Per-request timeout. |
| `GEMINI_MAX_CALLS_PER_MINUTE` | `6` | Local per-minute budget. |
| `GEMINI_MAX_CALLS_PER_DAY` | `200` | Local per-day budget. |

The feature runs only when the flag is on **and** a key is present. Either
missing means it is a safe no-op.

## Verifying the API facts yourself

Model names, availability and free-tier quotas change. Do not assume. Check:

- Models and current names: https://ai.google.dev/gemini-api/docs/models
- Rate limits and free tier: https://ai.google.dev/gemini-api/docs/rate-limits
- Quickstart / SDK: https://ai.google.dev/gemini-api/docs/quickstart

Free-tier limits are applied per project, are measured in requests per minute
(RPM), tokens per minute (TPM) and requests per day (RPD), and are lower for
preview models. The numbers above are the module's own conservative local
budget, **not** a statement of the provider's quota.

This module talks to the REST endpoint
`POST https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`
using only the Python standard library, so no new dependency is required.
If you prefer the official SDK, `pip install -U google-genai` and note that the
current quickstart uses `client.interactions.create(...)`.

## Safety properties

- **Key handling**: read from `GEMINI_API_KEY` only. Never logged, stored or
  returned in any result dict.
- **Data minimisation**: only allow-listed numeric indicators are sent
  (`last_close`, `ema20`, `ema50`, `rsi`, `atr14`, `adx`, `volume_ratio`,
  `bb_bandwidth`). Account data, balances, positions, PnL, broker ids and other
  symbols are dropped before the prompt is built.
- **Untrusted reply**: the model's output must parse as JSON and pass a strict
  schema check (enum bias, enum risk, numeric confidence in `[0, 1]`). Anything
  else is rejected. Model text is never executed.
- **Fail-safe**: any error, timeout, rate limit or malformed reply degrades to
  `available=False` with a reason. The original strategy is unaffected.
- **Budgets**: a local per-minute and per-day cap prevents runaway calls.

## Running the tests

```bash
python -m pytest tests/test_ai_market_context.py -q
python -m pytest tests/test_ai_ab_evaluation.py -q
```

Both suites are fully offline. `conftest.py` replaces the real transport so no
test can reach the network, and every call in the tests injects a fake
transport.

## The A/B experiment

```bash
# Preferred: both arms through the REAL engine (fair comparison).
python -m evaluation.ab_ai_context --mode engine --symbols AAPL,MSFT \
    --start 2023-01-02 --end 2024-08-30

# Legacy: simplified approximate replay. Exploratory numbers only.
python -m evaluation.ab_ai_context --mode simplified --symbols AAPL,MSFT \
    --start 2023-01-02 --end 2024-08-30
```

### Important: what this experiment is not

The strategy's execution engine deliberately refuses to consume LLM output in
historical backtests, because the model is trained on data that postdates the
simulated period (see the header of `backtest/engine.py`). A genuine, leak-free
historical replay of the real Gemini model is therefore **not possible** with
this codebase.

To stay honest, the driver uses a **deterministic, causal stand-in** instead of
a live API call. The stand-in looks only at trailing indicators at the decision
bar, so it cannot leak future information. It measures the *gate mechanism*
("if an advisor with this behaviour existed, how would the numbers change?"),
**not** the real model's edge.

Both arms use the same symbols, the same period, the same costs and the same
exit rules. The only difference is the entry gate.

### `engine` mode vs `simplified` mode

`engine` mode runs **both** arms through `backtest.engine.simulate_portfolio`,
unmodified. The gate is injected at run time by wrapping
`compute_signal_score`; the engine source is never changed. This guarantees
that cooldown, consecutive-loss guard, regime gating, position sizing, exit
rules, capital management and costs are identical between arms.

`simplified` mode (`run_arm`) is a **re-implementation**, not the strategy. It
does not model cooldown, the consecutive-loss guard, the regime gate (BEAR
blocks longs), multi-symbol capital sharing, or the engine's `strong_uptrend`
filter. Its numbers must be treated as exploratory. The earlier tables below
came from this mode and are retained only for continuity.

### The stand-in gate is redundant with the engine (M7)

The engine's own long-entry rule already requires:

```
close > ema20 > ema50  and  adx > 25     # "strong_uptrend"
```

`ai_gate_decision` tests **exactly the same condition**. In `engine` mode the
gate therefore cannot change any entry: it is a plumbing/leak check, not a
measure of alpha. Measured on AAPL+MSFT (2023-01-02..2024-08-30):

| Arm | Return | Trades | End equity | Blocked |
|---|---|---|---|---|
| baseline | +2.24% | 18 | 10223.94 | 0 |
| ai-gated | +2.24% | 18 | 10223.94 | 316 |

316 gate rejections, zero change in outcome. This is the expected result while
the stand-in duplicates the engine's filter. A meaningful A/B requires a gate
whose decision differs from `strong_uptrend`; that is a design change, not a
fix, and is left to a future explicitly-reviewed step.

### Observed result (5 symbols, cached data, simplified mode)

| Period | Arm | Mean return | Trades | Win rate | Worst DD |
|---|---|---|---|---|---|
| In-sample 2019-2022 | baseline | +5.38% | 120 | 50.1% | -2.36% |
| In-sample 2019-2022 | ai-gated | +2.75% | 78 | 47.6% | -1.74% |
| Out-of-sample 2023-2024 | baseline | +2.19% | 63 | 46.1% | -1.94% |
| Out-of-sample 2023-2024 | ai-gated | +1.58% | 45 | 45.4% | -1.85% |

These numbers come from `simplified` mode and are therefore **approximate**;
they are not produced by the strategy engine and should not be quoted as such.
They are retained only to document what the earlier, less rigorous run showed.

## Generation config (M4)

`gemini-3.5-flash-lite` has thinking **on** by default (level `minimal`), and
Google's documentation states that `maxOutputTokens` is a *hard cutoff that
includes thought tokens*: if the model hits it while reasoning, generation
stops with status `incomplete` and returns truncated or empty output.

The request therefore sets the two budgets separately:

```json
"generationConfig": {
  "temperature": 0.0,
  "maxOutputTokens": 1024,
  "thinkingConfig": {"thinkingLevel": "minimal"}
}
```

- `GEMINI_MAX_OUTPUT_TOKENS` overrides the ceiling (clamped to 256..8192).
- `GEMINI_THINKING_LEVEL` overrides the level (`minimal|low|medium|high`);
  setting it to the empty string omits `thinkingConfig` entirely, for models or
  endpoints that reject the field.

1024 is a considered default, **not a measured optimum**. Neither raising the
ceiling nor lowering thinking is assumed to improve quality.

### Requires live verification (not testable offline)

- That the deployed model accepts `thinkingConfig.thinkingLevel` on the
  `v1beta` `generateContent` endpoint (a live call must confirm no 400).
- The actual thought-token consumption of a real call, and therefore whether
  1024 is generous enough to avoid `incomplete` in practice.
- Whether `minimal` degrades the quality of the JSON answer.

## Retry policy (M5)

At most **one** retry, only for transient statuses (`429, 500, 502, 503, 504`)
and for timeouts/connection errors. Client errors (`400, 401, 403, 404, 422`)
are never retried: a bad key or malformed request cannot succeed on a repeat.

- `Retry-After` is honoured when present, capped at 5 seconds.
- Each attempt consumes the local call budget, so a retry cannot slip past the
  per-minute or per-day limits.
- The whole call is bounded by `GEMINI_TIMEOUT_SECONDS`; if the required delay
  does not fit inside the remaining deadline, the retry is skipped rather than
  waiting.
- There is no unbounded loop: at most 1 + 1 attempts per call.

## Remaining limitations

- The stand-in is not the Gemini model; no real AI alpha is demonstrated.
- The stand-in duplicates the engine's `strong_uptrend` filter, so `engine`
  mode currently cannot show any gate effect (see above).
- The symbol set is small and survivorship-biased.
- A real evaluation of the model would require forward-only paper runs with
  frozen prompts and a logged decision trail.
- The M4 generation config has not been exercised against the live API.
