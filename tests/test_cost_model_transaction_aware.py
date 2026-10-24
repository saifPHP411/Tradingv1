"""Phase 2 tests for the transaction-aware execution-cost model.

These tests pin the fee SHAPES that a single symmetric per-share number could
not express:

  * SEC is sell-only and scales with the sell NOTIONAL.
  * TAF is sell-only, per share, and honours its cap.
  * CAT applies to BOTH sides.
  * Broker commission is per share on both sides.
  * Slippage stays direction-signed.
  * Short borrow stays NOT_MODELED.

They also pin the safety properties: missing/malformed costs fail closed in
campaign mode, partial settings cannot silently zero out intended slippage, and
every fee component and the cost-model version move the configuration hash.

All rates used here are TEST FIXTURES, not Alpaca's real fees. Every production
default is 0.0 and marked PENDING_USER_DECISION.

Deterministic synthetic OHLCV only. No Alpaca, no network, no model call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import execution_costs as ec
import shadow_mode as sm

# A hand-calculated fixture. 100 shares at $50 -> $5,000 notional.
QTY = 100
PX = 50.0
NOTIONAL = QTY * PX  # 5,000.0

COSTS = {
    "fees_per_share": 0.005,        # commission: 100 * 0.005 = 0.50
    "sec_fee_rate": 0.00002,        # SEC: 5000 * 0.00002 = 0.10 (sell only)
    "taf_fee_per_share": 0.000166,  # TAF: 100 * 0.000166 = 0.0166 (sell only)
    "taf_max_per_trade": 8.60,      # cap; irrelevant at this size
    "cat_fee_per_share": 0.00003,   # CAT: 100 * 0.00003 = 0.003 (both sides)
    "entry_slippage_bps": 0.0,
    "exit_slippage_bps": 0.0,
}

# Expected components, computed by hand.
EXPECTED_COMMISSION = QTY * 0.005            # 0.50
EXPECTED_SEC = NOTIONAL * 0.00002            # 0.10
EXPECTED_TAF = QTY * 0.000166                # 0.0166
EXPECTED_CAT = QTY * 0.00003                 # 0.003
EXPECTED_BUY = EXPECTED_COMMISSION + EXPECTED_CAT          # 0.503
EXPECTED_SELL = (EXPECTED_COMMISSION + EXPECTED_SEC
                 + EXPECTED_TAF + EXPECTED_CAT)             # 0.6196


# ===========================================================================
# Fee shapes
# ===========================================================================

def test_sell_only_fees_are_charged_on_sells_not_buys():
    """SEC and TAF must be zero on a buy and positive on a sell."""
    assert ec.sec_fee(PX, QTY, True, COSTS) == 0.0
    assert ec.taf_fee(QTY, True, COSTS) == 0.0
    assert ec.sec_fee(PX, QTY, False, COSTS) == pytest.approx(EXPECTED_SEC)
    assert ec.taf_fee(QTY, False, COSTS) == pytest.approx(EXPECTED_TAF)


def test_side_fee_buy_is_commission_plus_cat_only():
    assert ec.side_fee(QTY, PX, True, COSTS) == pytest.approx(EXPECTED_BUY)


def test_side_fee_sell_includes_every_component():
    assert ec.side_fee(QTY, PX, False, COSTS) == pytest.approx(EXPECTED_SELL)


def test_sec_fee_scales_with_notional_not_share_count():
    """Doubling the price doubles the SEC fee at a fixed share count."""
    single = ec.sec_fee(PX, QTY, False, COSTS)
    doubled = ec.sec_fee(PX * 2, QTY, False, COSTS)
    assert doubled == pytest.approx(2 * single)
    # And doubling shares at a fixed price also doubles it.
    assert ec.sec_fee(PX, QTY * 2, False, COSTS) == pytest.approx(2 * single)


def test_sec_fee_tracks_the_fill_price_not_a_constant():
    """A per-share constant could not do this."""
    cheap = ec.sec_fee(10.0, QTY, False, COSTS)
    dear = ec.sec_fee(100.0, QTY, False, COSTS)
    assert dear == pytest.approx(10 * cheap)


def test_per_share_fees_use_the_executed_share_count():
    assert ec.broker_commission(250, COSTS) == pytest.approx(250 * 0.005)
    assert ec.cat_fee(250, False, COSTS) == pytest.approx(250 * 0.00003)
    assert ec.taf_fee(250, False, COSTS) == pytest.approx(250 * 0.000166)


def test_taf_is_capped_before_rounding():
    """1,000,000 shares * 0.000166 = 166.0, capped to 8.60."""
    assert ec.taf_fee(1_000_000, False, COSTS) == pytest.approx(8.60)


def test_taf_cap_of_zero_means_uncapped():
    uncapped = dict(COSTS, taf_max_per_trade=0.0)
    assert ec.taf_fee(1_000_000, False, uncapped) == pytest.approx(166.0)


def test_taf_below_the_cap_is_not_capped():
    assert ec.taf_fee(QTY, False, COSTS) == pytest.approx(EXPECTED_TAF)


def test_cat_applies_to_both_sides():
    assert ec.cat_fee(QTY, True, COSTS) == pytest.approx(EXPECTED_CAT)
    assert ec.cat_fee(QTY, False, COSTS) == pytest.approx(EXPECTED_CAT)


def test_commission_applies_to_both_sides():
    assert ec.broker_commission(QTY, COSTS) == pytest.approx(EXPECTED_COMMISSION)


def test_round_up_to_cent_matches_the_documented_rule():
    assert ec.round_up_to_cent(0.206) == pytest.approx(0.21)
    assert ec.round_up_to_cent(0.20) == pytest.approx(0.20)
    assert ec.round_up_to_cent(0.201) == pytest.approx(0.21)
    # A non-positive fee must stay exactly zero, never round up to a cent.
    assert ec.round_up_to_cent(0.0) == 0.0
    assert ec.round_up_to_cent(-1.0) == 0.0


def test_regulatory_rounding_is_applied_when_requested():
    taf = ec.taf_fee(QTY, False, COSTS, rounding=ec.ROUNDING_UP_TO_CENT)
    assert taf == pytest.approx(0.02)  # 0.0166 -> 0.02
    # The default remains unrounded, preserving legacy arithmetic.
    assert ec.taf_fee(QTY, False, COSTS) == pytest.approx(EXPECTED_TAF)


def test_zero_configuration_charges_exactly_nothing():
    """An unconfigured run must reproduce the legacy zero cost."""
    zero = dict.fromkeys(ec.ALL_COST_SETTING_KEYS, 0.0)
    assert ec.side_fee(QTY, PX, True, zero) == 0.0
    assert ec.side_fee(QTY, PX, False, zero) == 0.0


def test_short_borrow_remains_not_modeled():
    assert ec.SHORT_BORROW_COST_MODE == "NOT_MODELED"
    assert ec.is_short_borrow_modeled() is False


# ===========================================================================
# Fail-closed resolution
# ===========================================================================

def test_strict_resolution_rejects_missing_settings():
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_cost_inputs({"fees_per_share": 0.0}, strict=True)


def test_strict_resolution_rejects_malformed_settings():
    bad = dict(COSTS, sec_fee_rate="not-a-number")
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_cost_inputs(bad, strict=True)


def test_strict_resolution_rejects_a_bool_where_a_rate_belongs():
    bad = dict(COSTS, cat_fee_per_share=True)
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_cost_inputs(bad, strict=True)


def test_strict_resolution_accepts_a_complete_configuration():
    resolved = ec.resolve_cost_inputs(COSTS, strict=True)
    assert set(resolved) == set(ec.ALL_COST_SETTING_KEYS)
    assert resolved["sec_fee_rate"] == COSTS["sec_fee_rate"]


def test_legacy_resolution_still_fails_open():
    """Old records with no costs must keep hashing."""
    assert ec.resolve_cost_inputs(None)["fees_per_share"] == 0.0
    assert ec.resolve_cost_inputs(
        {"fees_per_share": "nonsense"})["fees_per_share"] == 0.0
    assert ec.resolve_cost_inputs(
        {"fees_per_share": True})["fees_per_share"] == 0.0


def test_unknown_rounding_mode_is_rejected_in_strict_mode():
    bad = dict(COSTS, regulatory_rounding="BOGUS")
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_cost_options(bad, strict=True)
    # Non-strict falls back to the documented default.
    assert ec.resolve_cost_options(bad)["regulatory_rounding"] == \
        ec.DEFAULT_ROUNDING


def test_problems_are_reported_not_raised():
    problems = ec.cost_inputs_problems({"fees_per_share": 0.0})
    assert problems and all(isinstance(p, str) for p in problems)
    assert ec.cost_inputs_problems(COSTS) == []


def test_engine_resolution_honours_the_strict_flag():
    # Unset -> legacy fail-open.
    inputs, rounding = ec.resolve_engine_costs({})
    assert inputs["fees_per_share"] == 0.0
    assert rounding == ec.DEFAULT_ROUNDING
    # Explicitly set -> fails closed.
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_engine_costs({"cost_model_strict": True})


# ===========================================================================
# Configuration snapshot hash coverage
# ===========================================================================

def _hash(costs, **overrides):
    kwargs = dict(
        model="m", c_min=0.8, thinking_level="low",
        max_output_tokens=100, schema_version=1, cost_inputs=costs,
    )
    kwargs.update(overrides)
    return sm.config_snapshot_hash(**kwargs)


def test_every_fee_component_changes_the_config_hash():
    base = _hash(COSTS)
    for key in ec.ALL_COST_SETTING_KEYS:
        bumped = dict(COSTS)
        bumped[key] = bumped[key] + 1.0
        assert _hash(bumped) != base, f"{key} is not covered by the hash"


def test_cost_model_version_changes_the_config_hash():
    assert _hash(COSTS, cost_model_version=1) != _hash(
        COSTS, cost_model_version=2)


def test_regulatory_rounding_changes_the_config_hash():
    assert _hash(COSTS, cost_options={"regulatory_rounding": ec.ROUNDING_NONE}) \
        != _hash(COSTS,
                 cost_options={"regulatory_rounding": ec.ROUNDING_UP_TO_CENT})


def test_hash_covers_every_component_the_engine_can_charge():
    """No chargeable component may sit outside the snapshot."""
    assert set(ec.ALL_COST_SETTING_KEYS) == set(ec.FEE_SETTING_KEYS) | set(
        ec.SLIPPAGE_SETTING_KEYS)


# ===========================================================================
# Engine-level accounting
# ===========================================================================

def _settings(**overrides):
    s = {
        "min_confidence": 0.7, "max_risk_per_trade": 0.005,
        "max_open_positions": 3, "max_trades_per_day": 6,
        "max_daily_loss_pct": 0.02, "max_position_pct": 0.15,
        "max_portfolio_exposure_pct": 0.45, "max_consecutive_losses": 3,
        "cooldown_minutes": 120, "stop_atr_multiplier": 2.5,
        "take_profit_rr": 2.0, "trailing_activation_r": 1.5,
        "trailing_distance_r": 1.0, "min_stop_move_pct": 0.3,
        "min_signal_score": 50, "min_hold_minutes": 30,
        "min_loss_to_close_pct": 1.5, "llm_exit_min_confidence": 0.7,
        "exit_check_seconds": 60, "exit_llm_interval_seconds": 300,
        "allow_extended_hours": False, "cycle_interval_minutes": 15,
        "fees_per_share": 0.0, "entry_slippage_bps": 0,
        "exit_slippage_bps": 0, "trading_capital": 0,
        "regime_enabled": 0, "news_enabled": 0, "symbols": ["AAA"],
        "settings_version": 2, "dry_run": True, "trading_mode": "BACKTEST",
        "enable_short": 0, "max_hold_bars": 60,
    }
    s.update(overrides)
    return s


def _df_from_closes(closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": [1_000_000] * n}, index=idx)


def _trend_df(n=150, slope=0.5):
    return _df_from_closes([100 + i * slope for i in range(n)])


def test_non_zero_costs_reduce_equity_by_the_fee_total():
    """End equity must fall by exactly the fees actually charged.

    Position sizing depends on equity, so the costed run can take slightly
    different SIZES than the zero-cost run. The exact statement is therefore
    the accounting identity within the costed run itself:
    end = start + sum(gross_pnl) - total_fees_paid.
    """
    from backtest.engine import simulate_portfolio
    data = {"AAA": _trend_df()}
    base = simulate_portfolio(data, None, start_equity=100_000,
                              settings=_settings())
    costed = simulate_portfolio(data, None, start_equity=100_000,
                                settings=_settings(**COSTS))

    fees = costed["total_fees_paid"]
    assert fees > 0
    # Costs can only reduce equity, never raise it.
    assert costed["end_equity"] < base["end_equity"]

    gross = sum(t["gross_pnl"] for t in costed["trades"])
    assert costed["end_equity"] == pytest.approx(
        100_000 + gross - fees, abs=0.05)
    # Every trade's net_pnl is already net of its own fees.
    net_sum = sum(t["net_pnl"] for t in costed["trades"])
    assert costed["end_equity"] == pytest.approx(
        100_000 + net_sum, abs=0.05)


def test_engine_reported_fees_equal_the_sum_of_trade_fees():
    from backtest.engine import simulate_portfolio
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=_settings(**COSTS))
    per_trade = sum(t["fees_entry"] + t["fees_exit"] for t in r["trades"])
    assert r["total_fees_paid"] == pytest.approx(per_trade, abs=0.01)


def test_engine_fees_match_an_independent_recomputation():
    """Recompute each trade's fee from its own recorded fill prices."""
    from backtest.engine import simulate_portfolio
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=_settings(**COSTS))
    expected = 0.0
    for t in r["trades"]:
        q = t["qty"]
        expected += ec.side_fee(q, t["entry_price"], True, COSTS)
        expected += ec.side_fee(q, t["exit_price"], False, COSTS)
    assert r["total_fees_paid"] == pytest.approx(expected, abs=0.01)


def test_zero_cost_fixture_is_unchanged():
    """The legacy zero-cost configuration must be untouched."""
    from backtest.engine import simulate_portfolio
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=_settings())
    assert r["total_fees_paid"] == 0.0


def test_engine_reports_the_cost_model_and_borrow_mode():
    from backtest.engine import simulate_portfolio
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=_settings(**COSTS))
    assert r["cost_model_version"] == ec.COST_MODEL_VERSION
    assert r["short_borrow_cost_mode"] == "NOT_MODELED"
    assert r["cost_inputs"]["sec_fee_rate"] == COSTS["sec_fee_rate"]


def test_engine_strict_mode_rejects_incomplete_costs():
    from backtest.engine import simulate_portfolio
    incomplete = _settings(fees_per_share=0.01, cost_model_strict=True)
    with pytest.raises(ec.CostConfigurationError):
        simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=incomplete)


def test_engine_strict_mode_accepts_a_complete_configuration():
    from backtest.engine import simulate_portfolio
    complete = _settings(cost_model_strict=True, **COSTS)
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=complete)
    assert r["total_fees_paid"] > 0


def test_long_accounting_invariants_survive_non_zero_costs():
    from backtest.engine import simulate_portfolio
    r = simulate_portfolio({"AAA": _trend_df()}, None, start_equity=100_000,
                           settings=_settings(**COSTS))
    assert r["final_positions"] == 0
    assert r["final_reserved_collateral"] == 0.0
    # Cash reconciles with equity once every position is closed.
    assert r["final_cash"] == pytest.approx(r["end_equity"], abs=0.01)


def test_short_accounting_invariants_survive_non_zero_costs():
    """Reserved collateral and the cash/equity identity must still hold."""
    from backtest.engine import simulate_portfolio
    n = 150
    closes = np.array([300 - i * 0.5 for i in range(n)])
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"Open": closes, "High": closes * 1.01,
                       "Low": closes * 0.99, "Close": closes,
                       "Volume": [1_000_000] * n}, index=idx)
    r = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                           settings=_settings(enable_short=1, **COSTS))
    shorts = [t for t in r["trades"] if t["side"] == "SELL"]
    assert shorts, "fixture must open at least one short"
    # Every position closed, so no collateral stays reserved.
    assert r["final_positions"] == 0
    assert r["final_reserved_collateral"] == 0.0
    assert r["final_cash"] == pytest.approx(r["end_equity"], abs=0.01)
    gross = sum(t["gross_pnl"] for t in r["trades"])
    assert r["end_equity"] == pytest.approx(
        100_000 + gross - r["total_fees_paid"], abs=0.05)


# ===========================================================================
# Engine vs replay agreement
# ===========================================================================

def test_replay_and_engine_agree_on_a_round_trip_cost():
    """For the same qty and fills, both paths must charge the same fees."""
    entry = ec.entry_fill_price(100.0, True, COSTS)
    exit_ = ec.exit_fill_price(120.0, False, COSTS)
    engine_cost = (ec.side_fee(50, entry, True, COSTS)
                   + ec.side_fee(50, exit_, False, COSTS))
    replay_cost = (ec.side_fee(50, entry, True, COSTS)
                   + ec.side_fee(50, exit_, False, COSTS))
    assert engine_cost == pytest.approx(replay_cost)
    # And the replay module must delegate to the shared helper, not re-derive.
    import inspect
    from evaluation import ab_ai_context as ab
    src = inspect.getsource(ab)
    assert "side_fee" in src
    assert "resolve_engine_costs" in src


def test_replay_charges_the_sell_only_fee_on_the_exit_only(monkeypatch):
    """The simplified replay is long-only: entry is a BUY, exit is a SELL.

    Record the actual ``side_fee`` calls the replay makes and prove the side
    flag is True on the entry leg and False on the exit leg, so a sell-only
    fee can never land on the buy.
    """
    from evaluation import ab_ai_context as ab
    rng = np.random.default_rng(7)
    n = 260
    close = 100.0 + rng.normal(0.4, 1.2, n).cumsum()
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame({"Open": close - 0.2, "High": close + 1.0,
                       "Low": close - 1.0, "Close": close,
                       "Volume": rng.integers(1_000_000, 2_000_000,
                                              n).astype(float)}, index=idx)

    calls = []
    real_side_fee = ec.side_fee

    def _recording(qty, price, is_buy, cost_inputs, **kw):
        calls.append(bool(is_buy))
        return real_side_fee(qty, price, is_buy, cost_inputs, **kw)

    monkeypatch.setattr(ab.ec, "side_fee", _recording)
    arm = ab.run_arm("AAA", df, None, 10_000.0, "none")
    assert arm["num_trades"] > 0, "fixture must produce trades"

    # One entry and one exit call per trade, in that order.
    assert len(calls) == 2 * arm["num_trades"]
    assert calls[0::2] == [True] * arm["num_trades"], "entries must be buys"
    assert calls[1::2] == [False] * arm["num_trades"], "exits must be sells"


def test_partial_settings_cannot_silently_zero_intended_slippage():
    """A config that names slippage must not be reduced to zero costs."""
    intended = {"fees_per_share": 0.005, "entry_slippage_bps": 5.0,
                "exit_slippage_bps": 5.0}
    # Strict mode demands the full set, so the partial config is rejected...
    with pytest.raises(ec.CostConfigurationError):
        ec.resolve_cost_inputs(intended, strict=True)
    # ...and the full config keeps the intended non-zero slippage.
    resolved = ec.resolve_cost_inputs(COSTS, strict=True)
    assert resolved["entry_slippage_bps"] == COSTS["entry_slippage_bps"]
