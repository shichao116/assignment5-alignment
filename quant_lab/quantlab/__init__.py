"""quantlab — a small, honest backtesting lab for personal investors."""
from .backtest import BacktestConfig, BacktestResult, backtest, benchmark_returns, check_lookahead, run_weights
from .data import load_csv, load_prices, synthetic_prices
from .research import grid_search, verdict, walk_forward
from .strategies import REGISTRY, Strategy, VolTarget, get_strategy, register

__all__ = [
    "BacktestConfig", "BacktestResult", "backtest", "benchmark_returns", "check_lookahead", "run_weights",
    "load_csv", "load_prices", "synthetic_prices", "grid_search", "verdict", "walk_forward",
    "REGISTRY", "Strategy", "VolTarget", "get_strategy", "register",
]
