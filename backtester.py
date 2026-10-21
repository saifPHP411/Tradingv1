"""
DEPRECATED — This module used the live graph and is not a realistic backtest.
Use `run_backtest.py` instead, which drives `backtest/engine.py`.

Kept only to avoid breaking any stale import.
"""
raise ImportError(
    "backtester.py is deprecated and unsafe for realistic simulation. "
    "Use `python run_backtest.py --symbol AAPL --start 2024-01-01 --end 2025-01-01` instead."
)
