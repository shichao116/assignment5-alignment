"""Daily portfolio simulator with realistic frictions.

Timeline for each trading day t:
  1. Holdings earn day t's close-to-close return and drift with prices.
  2. On rebalance days, trade to the target decided `lag` days earlier
     (lag=1: signal from yesterday's close, traded at today's close),
     paying `cost_bps` on every dollar traded.

`lag=0` (trade at the same close the signal was computed from) is
optimistic; keep the default unless you really can execute that way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .strategies import Strategy


@dataclass
class BacktestConfig:
    cost_bps: float = 5.0  # commission + half-spread + slippage per unit traded, in basis points
    lag: int = 1  # days between signal and execution
    rebalance: str = "D"  # "D" daily, "W" weekly, "M" monthly, "Q" quarterly
    min_trade: float = 0.0  # skip trades smaller than this weight change (reduces churn)
    cash_rate: float = 0.0  # annual rate earned on cash (and paid on borrowing)
    initial_capital: float = 10_000.0


@dataclass
class BacktestResult:
    returns: pd.Series  # daily net returns
    equity: pd.Series
    weights: pd.DataFrame  # end-of-day holdings (after rebalancing)
    turnover: pd.Series  # sum |trade| per day, as fraction of equity
    costs: pd.Series  # cost per day, as fraction of equity
    label: str = ""
    config: BacktestConfig = field(default_factory=BacktestConfig)

    def stats(self, benchmark: pd.Series | None = None) -> dict:
        s = M.summary(self.returns)
        s["avg_annual_turnover"] = float(self.turnover.mean() * 252)
        s["total_costs"] = float(self.costs.sum())
        s["avg_exposure"] = float(self.weights.abs().sum(axis=1).mean())
        if benchmark is not None:
            s.update(M.relative(self.returns, benchmark))
        return s

    def slice(self, start=None, end=None) -> "BacktestResult":
        r = self.returns.loc[start:end]
        return BacktestResult(
            returns=r,
            equity=(1 + r).cumprod() * self.config.initial_capital,
            weights=self.weights.loc[start:end],
            turnover=self.turnover.loc[start:end],
            costs=self.costs.loc[start:end],
            label=self.label,
            config=self.config,
        )


def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each period (and the very first day)."""
    freq = freq.upper()
    if freq == "D":
        return np.ones(len(index), dtype=bool)
    key = {"W": index.to_period("W"), "M": index.to_period("M"), "Q": index.to_period("Q")}
    if freq not in key:
        raise ValueError("rebalance must be one of D, W, M, Q")
    p = pd.Series(key[freq], index=index)
    mask = (p != p.shift(-1)).to_numpy().copy()
    mask[0] = True
    return mask


def run_weights(prices: pd.DataFrame, target: pd.DataFrame, cfg: BacktestConfig | None = None,
                label: str = "") -> BacktestResult:
    """Simulate trading toward `target` weights (same shape as `prices`)."""
    cfg = cfg or BacktestConfig()
    prices = prices.sort_index()
    target = target.reindex(index=prices.index, columns=prices.columns)

    rets = prices.pct_change().to_numpy()
    rets = np.nan_to_num(rets, nan=0.0)
    tgt = target.shift(cfg.lag).to_numpy()
    # Can't hold an asset before it has a price: zero those targets.
    tradable = prices.notna().to_numpy()
    tgt = np.where(tradable, tgt, np.where(np.isnan(tgt), np.nan, 0.0))
    reb = rebalance_mask(prices.index, cfg.rebalance)

    n, k = rets.shape
    cost_rate = cfg.cost_bps / 1e4
    cash_daily = (1 + cfg.cash_rate) ** (1 / 252) - 1

    w = np.zeros(k)
    started = False  # enter on the first available signal, not the first scheduled rebalance
    out_ret = np.zeros(n)
    out_w = np.zeros((n, k))
    out_turn = np.zeros(n)
    out_cost = np.zeros(n)

    for t in range(n):
        # 1) mark to market
        if t > 0:
            cash = 1.0 - w.sum()
            asset_pnl = w * rets[t]
            r = asset_pnl.sum() + cash * cash_daily
            gross = 1.0 + r
            w = (w + asset_pnl) / gross if gross > 0 else np.zeros(k)
        else:
            r = 0.0
        # 2) rebalance
        turn = 0.0
        row = tgt[t]
        if (reb[t] or not started) and not np.all(np.isnan(row)):
            started = True
            desired = np.nan_to_num(row, nan=0.0)
            trade = desired - w
            if cfg.min_trade > 0:
                trade[np.abs(trade) < cfg.min_trade] = 0.0
            turn = np.abs(trade).sum()
            w = w + trade
        cost = turn * cost_rate
        out_ret[t] = (1 + r) * (1 - cost) - 1
        out_w[t] = w
        out_turn[t] = turn
        out_cost[t] = cost

    idx = prices.index
    ret = pd.Series(out_ret, index=idx, name=label or "strategy")
    return BacktestResult(
        returns=ret,
        equity=(1 + ret).cumprod() * cfg.initial_capital,
        weights=pd.DataFrame(out_w, index=idx, columns=prices.columns),
        turnover=pd.Series(out_turn, index=idx),
        costs=pd.Series(out_cost, index=idx),
        label=label,
        config=cfg,
    )


def backtest(strategy: Strategy, prices: pd.DataFrame, cfg: BacktestConfig | None = None,
             start: str | None = None, end: str | None = None) -> BacktestResult:
    """Run `strategy` on `prices`.

    Signals are computed on the full history (strategies only look backward,
    so early data serves as warm-up), then results are reported for
    [start, end].
    """
    target = strategy.weights(prices)
    res = run_weights(prices, target, cfg, label=strategy.label())
    if start or end:
        res = res.slice(start, end)
    return res


def benchmark_returns(prices: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    """Buy-and-hold returns of one ticker (default: the first column)."""
    col = ticker or prices.columns[0]
    return prices[col].pct_change().fillna(0.0).rename(f"{col} buy&hold")


def check_lookahead(strategy: Strategy, prices: pd.DataFrame, n_checks: int = 5,
                    seed: int = 0) -> list[str]:
    """Detect a strategy that peeks at the future.

    Recomputes weights on data truncated at several random dates and checks
    that the weights on the cut-off date are unchanged. Returns a list of
    problems (empty = passed).
    """
    full = strategy.weights(prices)
    rng = np.random.default_rng(seed)
    lo = len(prices) // 3
    problems = []
    for cut in sorted(rng.choice(np.arange(lo, len(prices) - 1), size=n_checks, replace=False)):
        part = strategy.weights(prices.iloc[: cut + 1])
        a = full.iloc[cut].fillna(0).to_numpy()
        b = part.iloc[-1].fillna(0).to_numpy()
        if not np.allclose(a, b, atol=1e-9):
            problems.append(f"weights on {prices.index[cut].date()} change when future data is removed")
    return problems
