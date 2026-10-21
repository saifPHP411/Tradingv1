import numpy as np
import pandas as pd

from signals import (
    compute_rsi_wilder, compute_atr, compute_sma, compute_ema,
    compute_macd, compute_bollinger, compute_adx, compute_stochastic,
)
from settings import get_all_settings
from risk_utils import normalize_price_for_symbol


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
        "adx": float(adx),
        "stoch_k": float(k) if k is not None else 50.0,
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


class Position:
    def __init__(self, symbol, entry_price, qty, stop, target, entry_time, entry_idx):
        self.symbol = symbol
        self.entry_price = entry_price
        self.qty = qty
        self.stop = stop
        self.initial_stop = stop
        self.target = target
        self.entry_time = entry_time
        self.entry_idx = entry_idx
        self.highest_close = entry_price
        self.bars_held = 0

    def risk_per_share(self):
        return abs(self.entry_price - self.initial_stop)


def simulate(
    symbol: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None,
    start_equity: float = 10000.0,
    max_hold_bars: int = 60,
) -> dict:
    """
    Bar-driven simulation.
    - Decisions use data up to bar i only.
    - Entry fills at next bar's Open + slippage (no lookahead).
    - Gap through stop honored.
    - Same-candle SL+TP → conservative: stop first.
    """
    settings = get_all_settings()
    equity = start_equity
    equity_curve = []
    trades = []
    position: Position | None = None
    pending_entry = None

    n_bars = len(df)
    for i in range(60, n_bars):
        row = df.iloc[i]
        ts = df.index[i]
        open_p = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        # ---- Execute pending entry at THIS bar's open (from signal on i-1)
        if pending_entry is not None:
            slip = pending_entry["entry_ref"] * (settings["entry_slippage_bps"] / 10000.0)
            fill_price = open_p + slip
            stop_price = fill_price - pending_entry["stop_dist"]
            target_price = fill_price + pending_entry["target_dist"]
            qty = pending_entry["qty"]

            fees = qty * settings["fees_per_share"]
            equity -= fees

            position = Position(
                symbol=symbol, entry_price=fill_price, qty=qty,
                stop=normalize_price_for_symbol(symbol, stop_price),
                target=normalize_price_for_symbol(symbol, target_price),
                entry_time=ts, entry_idx=i,
            )
            trade_record = {
                "symbol": symbol, "side": "BUY",
                "entry_time": ts, "entry_price": fill_price,
                "qty": qty,
                "initial_stop": position.initial_stop,
                "take_profit": position.target,
                "signal_bar_idx": pending_entry["signal_bar_idx"],
                "regime": pending_entry.get("regime", ""),
                "signal_score": pending_entry.get("score", 0),
                "fees_entry": fees,
                "slippage_entry": slip * qty,
                "risk_amount": abs(fill_price - position.initial_stop) * qty,
            }
            pending_entry = None

        # ---- Manage open position
        if position is not None:
            position.bars_held += 1
            exit_reason = None
            exit_price = None

            # Gap handling: if OPEN is beyond stop, exit at open (worse than stop)
            if position.side if hasattr(position, "side") else True:
                if position.stop is not None:
                    # LONG
                    if open_p <= position.stop:
                        exit_price = open_p
                        exit_reason = "STOP_GAP"
                    elif open_p >= position.target:
                        exit_price = open_p
                        exit_reason = "TARGET_GAP"
                    elif low <= position.stop:
                        # Same-candle conflict: stop first (conservative)
                        exit_price = position.stop
                        exit_reason = "STOP"
                    elif high >= position.target:
                        exit_price = position.target
                        exit_reason = "TARGET"
                    elif position.bars_held >= max_hold_bars:
                        exit_price = close
                        exit_reason = "MAX_HOLD"

            if exit_price is not None:
                slip_out = exit_price * (settings["exit_slippage_bps"] / 10000.0)
                final_exit = exit_price - slip_out
                fees_out = position.qty * settings["fees_per_share"]
                gross_pnl = (final_exit - position.entry_price) * position.qty
                net_pnl = gross_pnl - fees_out
                equity += net_pnl

                trade_record.update({
                    "exit_time": ts,
                    "exit_price": final_exit,
                    "exit_reason": exit_reason,
                    "gross_pnl": round(gross_pnl, 4),
                    "fees_exit": round(fees_out, 4),
                    "slippage_exit": round(slip_out * position.qty, 4),
                    "net_pnl": round(net_pnl, 4),
                    "r_multiple": round(
                        (final_exit - position.entry_price) / position.risk_per_share(), 3
                    ) if position.risk_per_share() > 0 else 0,
                    "bars_held": position.bars_held,
                })
                trades.append(trade_record)
                position = None
            else:
                # R-based trailing
                if close > position.highest_close:
                    position.highest_close = close
                rps = position.risk_per_share()
                if rps > 0:
                    pnl_r = (close - position.entry_price) / rps
                    if pnl_r >= settings["trailing_activation_r"]:
                        new_stop = position.highest_close - (rps * settings["trailing_distance_r"])
                        if new_stop > position.stop and new_stop < close:
                            position.stop = normalize_price_for_symbol(symbol, new_stop)

        # ---- Look for new signal (execute on next bar's open)
        if position is None and pending_entry is None:
            ind = compute_indicators_at(df, i)
            if ind is None:
                equity_curve.append({"time": ts, "equity": round(equity, 2)})
                continue

            regime = compute_regime_at(spy_df, ts)
            score = compute_score(ind, regime)

            strong_uptrend = (
                ind["last_close"] > ind["ema20"] and
                ind["ema20"] > ind["ema50"] and
                ind["adx"] > 25
            )

            if score >= settings["min_signal_score"] and regime["regime"] != "BEAR" and strong_uptrend:
                atr = ind["atr14"]
                stop_dist = atr * settings["stop_atr_multiplier"]
                target_dist = stop_dist * settings["take_profit_rr"]

                if stop_dist > 0 and stop_dist < close * 0.20:
                    risk_mult = regime.get("size_multiplier", 1.0)
                    risk_amount = equity * settings["max_risk_per_trade"] * risk_mult
                    qty = risk_amount / stop_dist

                    max_notional = equity * settings["max_position_pct"]
                    if qty * close > max_notional:
                        qty = max_notional / close

                    if qty > 0 and qty * close >= 1.0:
                        pending_entry = {
                            "qty": qty,
                            "stop_dist": stop_dist,
                            "target_dist": target_dist,
                            "entry_ref": close,
                            "signal_bar_idx": i,
                            "regime": regime["regime"],
                            "score": score,
                        }

        equity_curve.append({"time": ts, "equity": round(equity, 2)})

    # Force close any open position at end
    if position is not None:
        final_close = float(df.iloc[-1]["Close"])
        slip_out = final_close * (settings["exit_slippage_bps"] / 10000.0)
        final_exit = final_close - slip_out
        fees_out = position.qty * settings["fees_per_share"]
        gross_pnl = (final_exit - position.entry_price) * position.qty
        net_pnl = gross_pnl - fees_out
        equity += net_pnl
        trade_record.update({
            "exit_time": df.index[-1],
            "exit_price": final_exit,
            "exit_reason": "END_OF_DATA",
            "gross_pnl": round(gross_pnl, 4),
            "fees_exit": round(fees_out, 4),
            "net_pnl": round(net_pnl, 4),
            "r_multiple": round(
                (final_exit - position.entry_price) / position.risk_per_share(), 3
            ) if position.risk_per_share() > 0 else 0,
            "bars_held": position.bars_held,
        })
        trades.append(trade_record)

    return {
        "symbol": symbol,
        "start_equity": start_equity,
        "end_equity": round(equity, 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }
