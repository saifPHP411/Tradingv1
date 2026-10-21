"""
Bar-driven Portfolio Backtester.

Accounting model: CASH-SECURED SPOT with RESERVED COLLATERAL for shorts.
No margin, no leverage.

Formulas (per position, per leg):

    LONG  entry : cash -= entry_notional + entry_fees
    LONG  exit  : cash += exit_notional  - exit_fees

    SHORT entry : cash += entry_notional - entry_fees
                  reserved_collateral += entry_notional
    SHORT exit  : cash -= exit_notional  + exit_fees
                  reserved_collateral -= entry_notional

    available_cash = cash - reserved_collateral
    equity(prices) = cash + sum(signed_market_value)
        long  => +qty * price
        short => -qty * price

    realized_pnl accumulates on close.
    unrealized_pnl is derived on demand from open positions and current prices.
    No double counting: equity = cash + signed_market_value always.

Capital constraints (explicit):
    - can_open() requires available_cash >= notional + fees for BOTH sides.
    - For SHORT, proceeds are credited to cash (so the equity identity holds),
      but the same notional is reserved, so the proceeds are NOT spendable.
      This is an exposure-based short model with reserved collateral, not
      margin accounting.
    - Total exposure is additionally capped by max_portfolio_exposure_pct
      via validate_portfolio_exposure() (which uses abs market value, so it
      includes shorts).
    - max_open_positions is enforced BOTH at signal-generation time (inside
      can_trade) AND at order-execution time (inside the pending-entry loop).
      The execution-time check is essential because multiple symbols can
      queue entries in the same bar; without it, all of them would open.

Other invariants:
    - Equity curve marked-to-market every bar (last Close per symbol).
    - A final snapshot is appended after force-closing open positions so
      equity_curve[-1].equity == end_equity.
    - Signals computed on bar i, executed on bar i+1 Open (no lookahead).
    - Gap handled at Open; same-candle conflict resolved STOP-first.
    - R-based trailing uses Close (matches monitor semantics).
    - Consecutive losses trigger a *timed* cooldown from
      settings['cooldown_minutes']. When the cooldown expires, both
      cooldown_until and consecutive_losses are reset so trading can resume.
      An active cooldown is never extended by subsequent losses.
    - Signal scoring is imported from signals.py (no duplication).
    - Historical news is unavailable in the backtest. We pass an explicit
      news={'available': False} dict to compute_signal_score(), which yields
      news_component = 0 deterministically. Live may add up to ±10 points
      from news, so backtest scores can differ from live by that amount.
      This is a documented input limitation, not full signal parity.

Strategy vs execution:
    - Strategy is LONG-only by design (enable_short defaults to 0).
    - Execution engine is LONG and SHORT capable.
    - SHORT signals (when enable_short=1) use compute_short_score() from
      signals.py, a symmetric mirror of compute_signal_score(). It is not
      a new strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import (
    compute_rsi_wilder, compute_atr, compute_sma, compute_ema,
    compute_macd, compute_bollinger, compute_adx, compute_stochastic,
    compute_signal_score, compute_short_score,
)
from settings import get_all_settings
from risk_utils import (
    normalize_price_for_symbol, compute_position_size,
    validate_portfolio_exposure, normalize_slippage,
)


# ---------------------------------------------------------------- News
# The backtester has no historical news feed. This constant documents the
# limitation explicitly so nobody later assumes backtest scores include news.
BACKTEST_NEWS_AVAILABLE = False
_NEWS_UNAVAILABLE = {"available": False, "score": 0.0, "news_count": 0}


# ============ Indicators & Regime ============
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


# ============ Position models ============
LONG = "long"
SHORT = "short"


class PortfolioPosition:
    __slots__ = (
        "symbol", "side", "entry_price", "qty", "stop", "initial_stop",
        "target", "entry_time", "entry_idx", "highest_price",
        "lowest_price", "bars_held", "entry_fees", "entry_slippage_cost",
    )

    def __init__(self, symbol, side, entry_price, qty, stop, target,
                 entry_time, entry_idx):
        self.symbol = symbol
        self.side = side
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
        self.entry_fees = 0.0
        self.entry_slippage_cost = 0.0

    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.initial_stop)


# ============ Portfolio state ============
class PortfolioState:
    def __init__(self, start_equity: float, settings: dict):
        self.start_equity = start_equity
        self.cash = start_equity
        self.reserved_collateral = 0.0
        self.settings = settings
        self.positions: dict[str, PortfolioPosition] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.daily_start_equity = start_equity
        self.current_date: str | None = None
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until: str | None = None
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self._last_prices: dict[str, float] = {}

    # ---- mark-to-market ----
    def signed_market_value(self, prices: dict[str, float]) -> float:
        mv = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            if pos.side == LONG:
                mv += price * pos.qty
            else:
                mv -= price * pos.qty
        return mv

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.signed_market_value(prices)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            if pos.side == LONG:
                total += (price - pos.entry_price) * pos.qty
            else:
                total += (pos.entry_price - price) * pos.qty
        return total

    @property
    def available_cash(self) -> float:
        """Cash that is not tied up as reserved collateral for shorts."""
        return self.cash - self.reserved_collateral

    # ---- timestamp helper ----
    @staticmethod
    def _parse_ts(x):
        """Parse any pandas-compatible timestamp into a naive UTC Timestamp."""
        try:
            t = pd.Timestamp(x)
        except Exception:
            return None
        if t.tz is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    # ---- day / risk gates ----
    def new_day(self, date_iso: str):
        self.current_date = date_iso
        self.daily_start_equity = self.equity(self._last_prices)
        self.trades_today = 0

    def can_trade(self, now_iso: str, prices: dict[str, float]) -> tuple[bool, str]:
        s = self.settings

        # ---- Cooldown handling ----
        if self.cooldown_until:
            cu = self._parse_ts(self.cooldown_until)
            now = self._parse_ts(now_iso)
            if cu is None or now is None:
                return False, "cooldown_unparseable"
            if now < cu:
                return False, "cooldown"
            self.cooldown_until = None
            self.consecutive_losses = 0

        # Defensive: if the loss threshold was reached but no cooldown was set
        # (e.g., timestamp conversion failed inside record_trade), start one
        # from now, or reset if cooldown_minutes <= 0. Never permanently lock.
        if self.consecutive_losses >= s["max_consecutive_losses"]:
            mins = int(s.get("cooldown_minutes", 120))
            if mins <= 0:
                self.consecutive_losses = 0
            else:
                now = self._parse_ts(now_iso)
                if now is None:
                    return False, "cooldown_unparseable"
                self.cooldown_until = (
                    now + pd.Timedelta(minutes=mins)
                ).isoformat()
                return False, "cooldown"

        if self.trades_today >= s["max_trades_per_day"]:
            return False, "max_trades_per_day"
        if len(self.positions) >= s["max_open_positions"]:
            return False, "max_open_positions"
        if self.daily_start_equity > 0:
            eq = self.equity(prices)
            daily_pnl = (eq - self.daily_start_equity) / self.daily_start_equity
            if daily_pnl <= -s["max_daily_loss_pct"]:
                return False, "daily_loss"
        return True, "ok"

    # ---- cash / position ops ----
    def can_open(self, entry_notional: float, entry_fees: float,
                 side: str = "long") -> tuple[bool, str]:
        """
        Cash-secured gate with reserved collateral (SHORT-safe).

            available = cash - reserved_collateral
            require available >= entry_notional + entry_fees

        For SHORT, proceeds are credited to cash on fill (so the equity
        identity holds), but the same notional is added to
        reserved_collateral. The proceeds are therefore NOT spendable:
        available_cash stays at the pre-trade level.

        `side` is accepted for API compatibility. The reserved-collateral
        rule applies to both LONG and SHORT. Total exposure is additionally
        capped by max_portfolio_exposure_pct via validate_portfolio_exposure.
        """
        available = self.available_cash
        required = entry_notional + entry_fees
        if available < required:
            return False, (
                f"insufficient_cash (available={available:.2f} < "
                f"required={required:.2f}; cash={self.cash:.2f}, "
                f"reserved={self.reserved_collateral:.2f})"
            )
        return True, "ok"

    def open_position(self, pos: PortfolioPosition, entry_fees: float,
                      entry_slippage_cost: float):
        notional = pos.entry_price * pos.qty
        if pos.side == LONG:
            self.cash -= notional + entry_fees
        else:
            self.cash += notional - entry_fees
            self.reserved_collateral += notional
        self.fees_paid += entry_fees
        pos.entry_fees = entry_fees
        pos.entry_slippage_cost = entry_slippage_cost
        self.positions[pos.symbol] = pos

    def close_position(self, sym: str, exit_price: float, exit_fees: float,
                       exit_slippage_cost: float, exit_reason: str, ts,
                       prices: dict[str, float]):
        pos = self.positions.pop(sym)
        exit_notional = exit_price * pos.qty

        if pos.side == LONG:
            gross = (exit_price - pos.entry_price) * pos.qty
            self.cash += exit_notional - exit_fees
        else:
            gross = (pos.entry_price - exit_price) * pos.qty
            self.cash -= exit_notional + exit_fees
            self.reserved_collateral -= pos.entry_price * pos.qty
            if self.reserved_collateral < 0:
                self.reserved_collateral = 0.0

        self.fees_paid += exit_fees
        total_net = gross - exit_fees - pos.entry_fees
        self.realized_pnl += total_net

        rps = pos.risk_per_share()
        risk_amount = rps * pos.qty
        r_multiple = (gross / risk_amount) if risk_amount > 0 else 0.0

        trade = {
            "symbol": sym,
            "side": "BUY" if pos.side == LONG else "SELL",
            "direction": pos.side,
            "entry_time": pos.entry_time,
            "entry_price": round(pos.entry_price, 6),
            "exit_time": ts,
            "exit_price": round(exit_price, 6),
            "qty": round(pos.qty, 6),
            "initial_stop": round(pos.initial_stop, 6),
            "final_stop": round(pos.stop, 6),
            "take_profit": round(pos.target, 6),
            "risk_amount": round(risk_amount, 4),
            "gross_pnl": round(gross, 4),
            "fees_entry": round(pos.entry_fees, 4),
            "fees_exit": round(exit_fees, 4),
            "slippage_entry": round(pos.entry_slippage_cost, 4),
            "slippage_exit": round(exit_slippage_cost, 4),
            "net_pnl": round(total_net, 4),
            "r_multiple": round(r_multiple, 3),
            "bars_held": pos.bars_held,
            "exit_reason": exit_reason,
            "portfolio_equity_after": round(self.equity(prices), 2),
            "portfolio_cash_after": round(self.cash, 2),
            "portfolio_reserved_after": round(self.reserved_collateral, 2),
        }
        self.trades.append(trade)

        if total_net < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.settings["max_consecutive_losses"]:
                mins = int(self.settings.get("cooldown_minutes", 120))
                if mins <= 0:
                    self.consecutive_losses = 0
                    self.cooldown_until = None
                elif self.cooldown_until is None:
                    try:
                        self.cooldown_until = (
                            pd.Timestamp(ts) + pd.Timedelta(minutes=mins)
                        ).isoformat()
                    except Exception:
                        self.cooldown_until = None
                print(f"🛑 cooldown: {self.consecutive_losses} consecutive losses "
                      f"→ active until {self.cooldown_until}")
        else:
            self.consecutive_losses = 0

    def snapshot(self, ts, prices: dict[str, float]):
        eq = self.equity(prices)
        self._last_prices = dict(prices)
        self.equity_curve.append({
            "time": ts,
            "equity": round(eq, 2),
            "cash": round(self.cash, 2),
            "reserved_collateral": round(self.reserved_collateral, 2),
            "available_cash": round(self.available_cash, 2),
            "unrealized": round(self.unrealized_pnl(prices), 2),
            "n_positions": len(self.positions),
        })


# ============ Helpers ============
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
    if settings is None:
        settings = get_all_settings()

    if not BACKTEST_NEWS_AVAILABLE:
        print("ℹ️ backtest: historical news unavailable — scoring with "
              "news={'available': False} (deterministic, no lookahead).")

    state = PortfolioState(start_equity, settings)
    all_times = sorted({ts for df in symbols_data.values() for ts in df.index})
    cursors = {sym: 0 for sym in symbols_data}
    pending: dict[str, dict | None] = {sym: None for sym in symbols_data}
    last_prices: dict[str, float] = {}
    last_ts = all_times[-1] if all_times else None
    enable_short = bool(int(settings.get("enable_short", 0) or 0))

    for ts in all_times:
        # ---- 0) Update last known closes (mark-to-market)
        for sym, df in symbols_data.items():
            row = _bar_row(df, ts)
            if row is not None:
                last_prices[sym] = float(row["Close"])

        date_iso = ts.date().isoformat()
        if state.current_date != date_iso:
            state.new_day(date_iso)

        # ---- 1) Execute pending entries at this bar's open
        for sym, df in symbols_data.items():
            p = pending.get(sym)
            if p is None:
                continue

            # Respect max_open_positions at execution time.
            # Multiple symbols may be pending in the same bar; the count
            # must be re-checked before each open, not only at signal time.
            # If the limit is missing or invalid, fail closed (0 opens).
            limit = settings.get("max_open_positions")
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):
                limit_int = 0
            if limit_int <= 0 or len(state.positions) >= limit_int:
                pending[sym] = None
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
            entry_notional = qty * fill_price
            entry_fees = qty * settings["fees_per_share"]

            ok, why = state.can_open(entry_notional, entry_fees, side)
            if not ok:
                pending[sym] = None
                continue

            exp_ok, exp_why = validate_portfolio_exposure(
                state.equity(last_prices), state.positions,
                entry_notional, settings["max_portfolio_exposure_pct"],
            )
            if not exp_ok:
                pending[sym] = None
                continue

            pos = PortfolioPosition(
                symbol=sym, side=side, entry_price=fill_price, qty=qty,
                stop=stop_price, target=target_price,
                entry_time=ts, entry_idx=cursors[sym],
            )
            state.open_position(pos, entry_fees, abs(slip) * qty)
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
                exit_slip = normalize_slippage(
                    exit_price, settings["exit_slippage_bps"], is_buy_exit)
                final_exit = exit_price + exit_slip
                exit_fees = pos.qty * settings["fees_per_share"]
                state.close_position(
                    sym, final_exit, exit_fees,
                    abs(exit_slip) * pos.qty, exit_reason, ts, last_prices,
                )
                continue

            # R-based trailing (Close-based, matches monitor)
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
        can, _ = state.can_trade(ts.isoformat(), last_prices)
        if can:
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

                long_score = compute_signal_score(
                    ind, regime, news=_NEWS_UNAVAILABLE)["score"]
                strong_uptrend = (
                    ind["last_close"] > ind["ema20"]
                    and ind["ema20"] > ind["ema50"]
                    and ind["adx"] > 25
                )
                long_ok = (
                    long_score >= settings["min_signal_score"]
                    and regime["regime"] != "BEAR"
                    and strong_uptrend
                )

                short_ok = False
                short_score = 0
                if enable_short:
                    short_score = compute_short_score(ind, regime)["score"]
                    strong_downtrend = (
                        ind["last_close"] < ind["ema20"]
                        and ind["ema20"] < ind["ema50"]
                        and ind["adx"] > 25
                    )
                    short_ok = (
                        short_score >= settings["min_signal_score"]
                        and regime["regime"] != "BULL"
                        and strong_downtrend
                    )

                if long_ok and not short_ok:
                    side, chosen_score = "buy", long_score
                elif short_ok and not long_ok:
                    side, chosen_score = "sell", short_score
                else:
                    continue

                atr = ind["atr14"]
                stop_dist = atr * settings["stop_atr_multiplier"]
                if stop_dist <= 0 or stop_dist > ind["last_close"] * 0.20:
                    continue
                target_dist = stop_dist * settings["take_profit_rr"]

                risk_mult = regime.get("size_multiplier", 1.0)
                qty = compute_position_size(
                    state.equity(last_prices), ind["last_close"], stop_dist,
                    settings["max_risk_per_trade"], risk_mult,
                    settings["max_position_pct"],
                )
                if qty <= 0 or qty * ind["last_close"] < 1.0:
                    continue

                pending[sym] = {
                    "side": side,
                    "qty": qty,
                    "stop_dist": stop_dist,
                    "target_dist": target_dist,
                    "signal_bar_idx": idx,
                    "regime": regime["regime"],
                    "score": chosen_score,
                }

        state.snapshot(ts, last_prices)

    # ---- Force-close remaining positions at end of data
    for sym, pos in list(state.positions.items()):
        df = symbols_data[sym]
        try:
            final_close = float(df.iloc[-1]["Close"])
        except Exception:
            continue
        is_buy_exit = pos.side == SHORT
        exit_slip = normalize_slippage(
            final_close, settings["exit_slippage_bps"], is_buy_exit)
        final_exit = final_close + exit_slip
        exit_fees = pos.qty * settings["fees_per_share"]
        last_prices[sym] = final_close
        state.close_position(
            sym, final_exit, exit_fees,
            abs(exit_slip) * pos.qty, "END_OF_DATA", last_ts, last_prices,
        )

    # Final snapshot: after force-close, equity_curve[-1] must match end_equity.
    if all_times:
        final_eq = state.equity(last_prices)
        state._last_prices = dict(last_prices)
        final_entry = {
            "time": last_ts,
            "equity": round(final_eq, 2),
            "cash": round(state.cash, 2),
            "reserved_collateral": round(state.reserved_collateral, 2),
            "available_cash": round(state.available_cash, 2),
            "unrealized": round(state.unrealized_pnl(last_prices), 2),
            "n_positions": len(state.positions),
        }
        if state.equity_curve and state.equity_curve[-1]["time"] == last_ts:
            state.equity_curve[-1] = final_entry
        else:
            state.equity_curve.append(final_entry)

    return {
        "start_equity": start_equity,
        "end_equity": round(state.equity(last_prices), 2),
        "final_cash": round(state.cash, 2),
        "final_reserved_collateral": round(state.reserved_collateral, 2),
        "trades": state.trades,
        "equity_curve": state.equity_curve,
        "final_positions": len(state.positions),
        "total_fees_paid": round(state.fees_paid, 2),
    }


# ============ Legacy single-symbol wrapper ============
def simulate(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame | None,
             start_equity: float = 10000.0, max_hold_bars: int = 60) -> dict:
    r = simulate_portfolio({symbol: df}, spy_df, start_equity)
    r["symbol"] = symbol
    return r
