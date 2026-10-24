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
python -m evaluation.ab_ai_context --symbols AAPL,MSFT --start 2023-01-02 --end 2024-08-30
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

### Observed result (5 symbols, cached data)

| Period | Arm | Mean return | Trades | Win rate | Worst DD |
|---|---|---|---|---|---|
| In-sample 2019-2022 | baseline | +5.38% | 120 | 50.1% | -2.36% |
| In-sample 2019-2022 | ai-gated | +2.75% | 78 | 47.6% | -1.74% |
| Out-of-sample 2023-2024 | baseline | +2.19% | 63 | 46.1% | -1.94% |
| Out-of-sample 2023-2024 | ai-gated | +1.58% | 45 | 45.4% | -1.85% |

The gate reduced both the number of trades and the return in both periods. It
slightly reduced drawdown. **The stand-in did not improve performance.** This is
a mechanism result, not a verdict on Gemini, but it is the honest reading of
what was actually measured.

## Remaining limitations

- The stand-in is not the Gemini model; no real AI alpha is demonstrated.
- The symbol set is small and survivorship-biased.
- A real evaluation of the model would require forward-only paper runs with
  frozen prompts and a logged decision trail.
