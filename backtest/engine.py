"""
Bar-driven Portfolio Backtester.

Design invariants:
  - All symbols share one cash/equity/positions portfolio.
  - Signals computed at bar i, executed at bar i+1 open (no lookahead).
  - Stop/Target honoured with gap and same-candle conservative rule.
  - Long and Short supported.
  - R-based trailing shared with monitor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import (
    compute_rsi_wilder, compute_atr, compute_sma, compute_ema,
    compute_macd, compute_bollinger, compute_adx, compute_stochastic,
)
from settings import get_all_settings
from risk_utils import (
    normalize_price_for_symbol, compute_position_size,
    validate_portfolio_exposure, normalize_slippage,
)


# ============ Indicators & Regime (shared with live) ============
def compute_indicators_at(df: pd.DataFrame, i: int) -> dict | None:
    if i < 60:
        return None
    window = df.iloc[: i + 1]
    closes = window["Close"].values.astype(float)
    highs = window["High"].values.astype(float)
    lows = window["Low"].values.astype(float)
    volumes = window["Volume"].values.astype(float)
    if len(closes) < 60:
        return None

    rsi = compute_rsi_wilder(closes)
    atr = compute_atr(highs, lows, closes)
    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, 50)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    macd_line, macd_sig, macd_hist = compute_macd(closes)
    pct_b, bb_bw = compute_bollinger(closes)
    adx = compute_adx(highs, lows, closes)
    k, d = compute_stochastic(highs, lows, closes)

    if any(v is None for v in [rsi, atr, sma20, sma50, ema20, ema50, adx]):
        return None

    vol_mean = float(np.mean(volumes[-20:]))
    vol_ratio = float(volumes[-1] / vol_mean) if vol_mean > 0 else 1.0
    mom5 = float((closes[-1] / closes[-6] - 1) * 100) if len(closes) >= 6 else 0.0
    mom10 = float((closes[-1] / closes[-11] - 1) * 100) if len(closes) >= 11 else 0.0

    return {
        "last_close": float(closes[-1]),
        "atr14": float(atr),
        "rsi14": float(rsi),
        "sma20": float(sma20), "sma50": float(sma50),
        "ema20": float(ema20), "ema50": float(ema50),
        "macd_hist": float(macd_hist) if macd_hist is not None else 0.0,
        "bb_pct_b": float(pct_b) if pct_b is not None else 0.5,
        "bb_bandwidth": float(bb_bw) if bb_bw is not None else 0.0,
        "adx": float(adx),
        "stoch_k": float(k) if k is not None else 50.0,
        "stoch_d": float(d) if d is not None else 50.0,
        "volume_ratio": float(vol_ratio),
        "momentum_5d_pct": float(mom5),
        "momentum_10d_pct": float(mom10),
    }


def compute_regime_at(spy_df: pd.DataFrame, ts) -> dict:
    if spy_df is None:
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}
    try:
        window = spy_df.loc[:ts]
    except Exception:
        window = spy_df.iloc[:0]
    if len(window) < 200:
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}
    closes = window["Close"].values.astype(float)
    highs = window["High"].values.astype(float)
    lows = window["Low"].values.astype(float)
    price = float(closes[-1])
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, 200)
    atr = compute_atr(highs, lows, closes)
    rsi = compute_rsi_wilder(closes)
    if any(v is None for v in [sma50, sma200, atr, rsi]):
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}
    atr_pct = (atr / price) * 100 if price > 0 else 0
    if atr_pct >= 3.0:
        return {"regime": "HIGH_VOLATILITY", "size_multiplier": 0.4}
    if atr_pct >= 2.2 and rsi < 40:
        return {"regime": "HIGH_VOLATILITY", "size_multiplier": 0.5}
    if price > sma50 and sma50 > sma200 and rsi > 50:
        return {"regime": "BULL", "size_multiplier": 1.0}
    if price < sma50 and sma50 < sma200:
        return {"regime": "BEAR", "size_multiplier": 0.0}
    return {"regime": "SIDEWAYS", "size_multiplier": 0.7}


def compute_score(indicators: dict, regime: dict) -> int:
    trend = 0
    if indicators["ema20"] > indicators["ema50"]:
        trend += 12
    if indicators["last_close"] > indicators["ema20"]:
        trend += 8
    if indicators["last_close"] > indicators["sma50"]:
        trend += 5
    if indicators["adx"] > 25:
        trend += 5

    momentum = 0
    if indicators["momentum_5d_pct"] > 0:
        momentum += 8
    if indicators["momentum_10d_pct"] > 0:
        momentum += 5
    if indicators["macd_hist"] > 0:
        momentum += 7

    rsi = indicators["rsi14"]
    if 40 <= rsi <= 70:
        rsi_score = 15
    elif 30 <= rsi < 40:
        rsi_score = 10
    elif 70 < rsi <= 78:
        rsi_score = 8
    else:
        rsi_score = 0

    bb = 0
    if 0.2 <= indicators["bb_pct_b"] <= 0.8:
        bb += 8
    if indicators["bb_pct_b"] > 0.9:
        bb -= 5
    if indicators["bb_pct_b"] < 0.1:
        bb += 3
    bb = max(0, bb)

    if indicators["volume_ratio"] > 1.2:
        vol = 12
    elif indicators["volume_ratio"] > 0.8:
        vol = 8
    else:
        vol = 3

    raw = trend + momentum + rsi_score + bb + vol
    raw = max(0, min(100, raw))
    r = regime.get("regime", "UNKNOWN")
    if r == "BEAR":
        raw = int(raw * 0.5)
    elif r == "HIGH_VOLATILITY":
        raw = int(raw * 0.75)
    elif r == "SIDEWAYS":
        raw = int(raw * 0.9)
    return raw


# ============ Position models ============
LONG = "long"
SHORT = "short"


class PortfolioPosition:
    __slots__ = (
        "symbol", "side", "entry_price", "qty", "stop", "initial_stop",
        "target", "entry_time", "entry_idx", "highest_price",
        "lowest_price", "bars_held",
    )

    def __init__(self, symbol: str, side: str, entry_price: float, qty: float,
                 stop: float, target: float, entry_time, entry_idx: int):
        self.symbol = symbol
        self.side = side          # LONG | SHORT
        self.entry_price = entry_price
        self.qty = qty
        self.stop = stop
        self.initial_stop = stop
        self.target = target
        self.entry_time = entry_time
        self.entry_idx = entry_idx
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.bars_held = 0

    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def market_value(self, current_price: float) -> float:
        return abs(self.qty * current_price)


# ============ Portfolio state ============
class PortfolioState:
    def __init__(self, start_equity: float, settings: dict):
        self.equity = start_equity
        self.cash = start_equity
        self.settings = settings
        self.positions: dict[str, PortfolioPosition] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.daily_start_equity = start_equity
        self.current_date: str | None = None
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until: str | None = None

    def new_day(self, date_iso: str):
        self.current_date = date_iso
        self.daily_start_equity = self.equity
        self.trades_today = 0
        if self.cooldown_until and self.cooldown_until <= date_iso:
            self.cooldown_until = None

    def can_trade(self, now_iso: str) -> tuple[bool, str]:
        s = self.settings
        if self.cooldown_until and now_iso < self.cooldown_until:
            return False, "cooldown"
        if self.consecutive_losses >= s["max_consecutive_losses"]:
            return False, "consecutive_losses"
        if self.trades_today >= s["max_trades_per_day"]:
            return False, "max_trades_per_day"
        if len(self.positions) >= s["max_open_positions"]:
            return False, "max_open_positions"
        if self.daily_start_equity > 0:
            daily_pnl = (self.equity - self.daily_start_equity) / self.daily_start_equity
            if daily_pnl <= -s["max_daily_loss_pct"]:
                return False, "daily_loss"
        return True, "ok"

    def record_trade(self, trade: dict):
        self.trades.append(trade)
        self.equity += trade["net_pnl"]
        self.cash += trade["net_pnl"]
        if trade["net_pnl"] < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.settings["max_consecutive_losses"]:
                # Block rest of the backtest (deterministic, conservative)
                self.cooldown_until = "9999-12-31"
        else:
            self.consecutive_losses = 0

    def snapshot(self, ts):
        self.equity_curve.append({"time": ts, "equity": round(self.equity, 2)})


# ============ Shared helpers ============
def _close_price(df: pd.DataFrame, ts) -> float | None:
    try:
        return float(df.loc[ts, "Close"])
    except Exception:
        return None


def _bar_row(df: pd.DataFrame, ts):
    try:
        return df.loc[ts]
    except Exception:
        return None


# ============ Main Portfolio Simulation ============
def simulate_portfolio(symbols_data: dict[str, pd.DataFrame],
                       spy_df: pd.DataFrame | None,
                       start_equity: float = 10000.0,
                       settings: dict | None = None) -> dict:
    """
    Portfolio-level backtest with shared cash, equity, exposure, limits.
    Supports LONG and SHORT. Executes next-bar open after signal.
    """
    if settings is None:
        settings = get_all_settings()
    state = PortfolioState(start_equity, settings)

    all_times = sorted({ts for df in symbols_data.values() for ts in df.index})
    cursors = {sym: 0 for sym in symbols_data}
    pending: dict[str, dict | None] = {sym: None for sym in symbols_data}
    last_ts = all_times[-1] if all_times else None

    for ts in all_times:
        date_iso = ts.date().isoformat()
        if state.current_date != date_iso:
            state.new_day(date_iso)

        # ---- 1) Execute pending entries at this bar's open
        for sym, df in symbols_data.items():
            p = pending.get(sym)
            if p is None:
                continue
            row = _bar_row(df, ts)
            if row is None:
                continue
            open_p = float(row["Open"])
            is_buy = p["side"] == "buy"
            slip = normalize_slippage(open_p, settings["entry_slippage_bps"], is_buy)
            fill_price = open_p + slip

            if fill_price <= 0:
                pending[sym] = None
                continue

            if p["side"] == "buy":
                stop_price = fill_price - p["stop_dist"]
                target_price = fill_price + p["target_dist"]
                side = LONG
            else:
                stop_price = fill_price + p["stop_dist"]
                target_price = fill_price - p["target_dist"]
                side = SHORT

            stop_price = normalize_price_for_symbol(sym, stop_price)
            target_price = normalize_price_for_symbol(sym, target_price)
            qty = p["qty"]
            notional = qty * fill_price

            ok, why = validate_portfolio_exposure(
                state.equity, state.positions, notional,
                settings["max_portfolio_exposure_pct"]
            )
            if not ok:
                pending[sym] = None
                continue
            if notional > state.cash:
                pending[sym] = None
                continue

            pos = PortfolioPosition(
                symbol=sym, side=side, entry_price=fill_price, qty=qty,
                stop=stop_price, target=target_price,
                entry_time=ts, entry_idx=cursors[sym],
            )
            state.positions[sym] = pos
            state.trades_today += 1
            pending[sym] = None

        # ---- 2) Manage open positions
        for sym in list(state.positions.keys()):
            pos = state.positions[sym]
            df = symbols_data[sym]
            row = _bar_row(df, ts)
            if row is None:
                continue
            open_p = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            pos.bars_held += 1

            exit_price = None
            exit_reason = None

            if pos.side == LONG:
                if open_p <= pos.stop:
                    exit_price, exit_reason = open_p, "STOP_GAP"
                elif open_p >= pos.target:
                    exit_price, exit_reason = open_p, "TARGET_GAP"
                elif low <= pos.stop:
                    exit_price, exit_reason = pos.stop, "STOP"
                elif high >= pos.target:
                    exit_price, exit_reason = pos.target, "TARGET"
                elif pos.bars_held >= 60:
                    exit_price, exit_reason = close, "MAX_HOLD"
            else:  # SHORT
                if open_p >= pos.stop:
                    exit_price, exit_reason = open_p, "STOP_GAP"
                elif open_p <= pos.target:
                    exit_price, exit_reason = open_p, "TARGET_GAP"
                elif high >= pos.stop:
                    exit_price, exit_reason = pos.stop, "STOP"
                elif low <= pos.target:
                    exit_price, exit_reason = pos.target, "TARGET"
                elif pos.bars_held >= 60:
                    exit_price, exit_reason = close, "MAX_HOLD"

            if exit_price is not None:
                is_buy_exit = pos.side == SHORT
                slip = normalize_slippage(exit_price, settings["exit_slippage_bps"], is_buy_exit)
                final_exit = exit_price + slip
                fees = pos.qty * settings["fees_per_share"]
                if pos.side == LONG:
                    gross = (final_exit - pos.entry_price) * pos.qty
                    rps = pos.risk_per_share()
                    r_multiple = (final_exit - pos.entry_price) / rps if rps > 0 else 0.0
                else:
                    gross = (pos.entry_price - final_exit) * pos.qty
                    rps = pos.risk_per_share()
                    r_multiple = (pos.entry_price - final_exit) / rps if rps > 0 else 0.0
                net = gross - fees

                state.record_trade({
                    "symbol": sym,
                    "side": "BUY" if pos.side == LONG else "SELL",
                    "direction": pos.side,
                    "entry_time": pos.entry_time,
                    "entry_price": pos.entry_price,
                    "exit_time": ts,
                    "exit_price": final_exit,
                    "qty": pos.qty,
                    "initial_stop": pos.initial_stop,
                    "final_stop": pos.stop,
                    "take_profit": pos.target,
                    "gross_pnl": round(gross, 4),
                    "fees_exit": round(fees, 4),
                    "slippage_exit": round(slip * pos.qty, 4),
                    "net_pnl": round(net, 4),
                    "r_multiple": round(r_multiple, 3),
                    "bars_held": pos.bars_held,
                    "exit_reason": exit_reason,
                    "portfolio_equity_after": round(state.equity, 2),
                })
                del state.positions[sym]
                continue

            # R-based trailing
            if pos.side == LONG:
                if close > pos.highest_price:
                    pos.highest_price = close
                rps = pos.risk_per_share()
                if rps > 0:
                    pnl_r = (close - pos.entry_price) / rps
                    if pnl_r >= settings["trailing_activation_r"]:
                        new_stop = pos.highest_price - (rps * settings["trailing_distance_r"])
                        if new_stop > pos.stop and new_stop < close:
                            pos.stop = normalize_price_for_symbol(sym, new_stop)
            else:
                if close < pos.lowest_price:
                    pos.lowest_price = close
                rps = pos.risk_per_share()
                if rps > 0:
                    pnl_r = (pos.entry_price - close) / rps
                    if pnl_r >= settings["trailing_activation_r"]:
                        new_stop = pos.lowest_price + (rps * settings["trailing_distance_r"])
                        if new_stop < pos.stop and new_stop > close:
                            pos.stop = normalize_price_for_symbol(sym, new_stop)

        # ---- 3) Generate signals (executed next bar)
        if state.can_trade(ts.isoformat())[0]:
            for sym, df in symbols_data.items():
                if sym in state.positions or pending.get(sym) is not None:
                    continue
                try:
                    idx = df.index.get_loc(ts)
                except KeyError:
                    continue
                if idx < 60:
                    continue
                ind = compute_indicators_at(df, idx)
                if ind is None:
                    continue
                regime = compute_regime_at(spy_df, ts)
                score = compute_score(ind, regime)
                if score < settings["min_signal_score"]:
                    continue
                if regime["regime"] == "BEAR":
                    continue
                strong_uptrend = (
                    ind["last_close"] > ind["ema20"]
                    and ind["ema20"] > ind["ema50"]
                    and ind["adx"] > 25
                )
                if not strong_uptrend:
                    continue

                atr = ind["atr14"]
                stop_dist = atr * settings["stop_atr_multiplier"]
                if stop_dist <= 0 or stop_dist > ind["last_close"] * 0.20:
                    continue
                target_dist = stop_dist * settings["take_profit_rr"]

                risk_mult = regime.get("size_multiplier", 1.0)
                qty = compute_position_size(
                    state.equity, ind["last_close"], stop_dist,
                    settings["max_risk_per_trade"], risk_mult,
                    settings["max_position_pct"],
                )
                if qty <= 0 or qty * ind["last_close"] < 1.0:
                    continue

                pending[sym] = {
                    "side": "buy",
                    "qty": qty,
                    "stop_dist": stop_dist,
                    "target_dist": target_dist,
                    "signal_bar_idx": idx,
                    "regime": regime["regime"],
                    "score": score,
                }

        state.snapshot(ts)

    # Force-close any open positions at end
    for sym, pos in list(state.positions.items()):
        df = symbols_data[sym]
        try:
            final_close = float(df.iloc[-1]["Close"])
        except Exception:
            continue
        is_buy_exit = pos.side == SHORT
        slip = normalize_slippage(final_close, settings["exit_slippage_bps"], is_buy_exit)
        final_exit = final_close + slip
        fees = pos.qty * settings["fees_per_share"]
        if pos.side == LONG:
            gross = (final_exit - pos.entry_price) * pos.qty
            rps = pos.risk_per_share()
            r_multiple = (final_exit - pos.entry_price) / rps if rps > 0 else 0.0
        else:
            gross = (pos.entry_price - final_exit) * pos.qty
            rps = pos.risk_per_share()
            r_multiple = (pos.entry_price - final_exit) / rps if rps > 0 else 0.0
        net = gross - fees
        state.record_trade({
            "symbol": sym,
            "side": "BUY" if pos.side == LONG else "SELL",
            "direction": pos.side,
            "entry_time": pos.entry_time,
            "entry_price": pos.entry_price,
            "exit_time": last_ts,
            "exit_price": final_exit,
            "qty": pos.qty,
            "initial_stop": pos.initial_stop,
            "final_stop": pos.stop,
            "take_profit": pos.target,
            "gross_pnl": round(gross, 4),
            "fees_exit": round(fees, 4),
            "net_pnl": round(net, 4),
            "r_multiple": round(r_multiple, 3),
            "bars_held": pos.bars_held,
            "exit_reason": "END_OF_DATA",
            "portfolio_equity_after": round(state.equity, 2),
        })
        del state.positions[sym]

    return {
        "start_equity": start_equity,
        "end_equity": round(state.equity, 2),
        "trades": state.trades,
        "equity_curve": state.equity_curve,
        "final_positions": len(state.positions),
    }


# ============ Legacy single-symbol entry (kept for compatibility) ============
def simulate(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame | None,
             start_equity: float = 10000.0, max_hold_bars: int = 60) -> dict:
    """Single-symbol convenience wrapper around simulate_portfolio."""
    r = simulate_portfolio({symbol: df}, spy_df, start_equity)
    # Adapt to old shape (adds 'symbol')
    r["symbol"] = symbol
    return r
