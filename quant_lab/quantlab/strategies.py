"""Strategies turn prices into target portfolio weights.

Contract: `weights(prices)` returns a DataFrame shaped like `prices` where row t
holds the target weights decided using ONLY data up to and including day t's
close. The backtester handles execution delay, so never shift inside a strategy.

Weights are fractions of equity: 1.0 = fully invested, 0 = cash, negative =
short. Rows that are all NaN mean "no decision yet" (warm-up) and are held
in cash.

To write your own, subclass `Strategy`, declare parameters as dataclass
fields, implement `weights`, and decorate with `@register`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

REGISTRY: dict[str, type["Strategy"]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class Strategy:
    name = "base"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def label(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({p})"


def get_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}")
    cls = REGISTRY[name]
    types = {f.name: f.type for f in fields(cls)}
    unknown = set(params) - set(types)
    if unknown:
        raise TypeError(f"{name} has no parameter(s) {sorted(unknown)}; it takes {sorted(types)}")
    return cls(**params)


def _equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    """1/N across True cells in each row (rows with no True -> all zeros)."""
    m = mask.astype(float)
    n = m.sum(axis=1).replace(0, np.nan)
    return m.div(n, axis=0).fillna(0.0)


def _warmup(w: pd.DataFrame, n: int) -> pd.DataFrame:
    w = w.copy()
    w.iloc[:n] = np.nan
    return w


# --------------------------------------------------------------------------- #
# Baselines — every idea must beat these after costs, or it isn't an idea.
# --------------------------------------------------------------------------- #


@register
@dataclass
class BuyAndHold(Strategy):
    """Equal-weight every asset, rebalanced by the backtester's schedule."""

    name = "buy_hold"

    def weights(self, prices):
        return _equal_weight(prices.notna())


@register
@dataclass
class InverseVol(Strategy):
    """Weight assets by 1/volatility so each contributes similar risk."""

    name = "inverse_vol"
    lookback: int = 63

    def weights(self, prices):
        vol = prices.pct_change().rolling(self.lookback).std()
        inv = 1.0 / vol.replace(0, np.nan)
        return _warmup(inv.div(inv.sum(axis=1), axis=0).fillna(0.0), self.lookback)


# --------------------------------------------------------------------------- #
# Trend following
# --------------------------------------------------------------------------- #


@register
@dataclass
class SMACrossover(Strategy):
    """Hold an asset while its fast moving average is above the slow one."""

    name = "sma_cross"
    fast: int = 50
    slow: int = 200

    def weights(self, prices):
        on = prices.rolling(self.fast).mean() > prices.rolling(self.slow).mean()
        return _warmup(_equal_weight_of_universe(on, prices), self.slow)


@register
@dataclass
class PriceAboveSMA(Strategy):
    """Classic 'stay in while price > N-day average, else cash' timing rule."""

    name = "trend_filter"
    window: int = 200

    def weights(self, prices):
        on = prices > prices.rolling(self.window).mean()
        return _warmup(_equal_weight_of_universe(on, prices), self.window)


@register
@dataclass
class TimeSeriesMomentum(Strategy):
    """Long assets whose trailing return (skipping the last `skip` days) is positive."""

    name = "ts_momentum"
    lookback: int = 252
    skip: int = 21

    def weights(self, prices):
        mom = prices.shift(self.skip) / prices.shift(self.lookback) - 1
        return _warmup(_equal_weight_of_universe(mom > 0, prices), self.lookback)


@register
@dataclass
class CrossSectionalMomentum(Strategy):
    """Rank assets by trailing return and hold the top `top_n` (relative strength).

    If `absolute` is set, a winner is only held if its own return is also
    positive (the 'dual momentum' idea) — otherwise that slot sits in cash.
    """

    name = "xs_momentum"
    lookback: int = 126
    skip: int = 0
    top_n: int = 3
    absolute: bool = True

    def weights(self, prices):
        mom = prices.shift(self.skip) / prices.shift(self.lookback) - 1
        rank = mom.rank(axis=1, ascending=False)
        pick = rank <= self.top_n
        if self.absolute:
            pick &= mom > 0
        w = pick.astype(float) / self.top_n
        return _warmup(w, self.lookback + self.skip)


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #


@register
@dataclass
class RSIMeanReversion(Strategy):
    """Buy short-term oversold dips (low RSI), exit once RSI recovers.

    Optionally only buy dips while the long-term trend is up (`trend_window`>0).
    """

    name = "rsi_reversion"
    period: int = 2
    entry: float = 10.0
    exit: float = 70.0
    trend_window: int = 200

    def weights(self, prices):
        r = rsi(prices, self.period)
        allowed = prices > prices.rolling(self.trend_window).mean() if self.trend_window else True
        enter = (r < self.entry) & allowed
        leave = (r > self.exit) | ~allowed if self.trend_window else (r > self.exit)
        state = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        state[enter] = 1.0
        state[leave & ~enter] = 0.0
        holding = state.ffill().fillna(0.0).astype(bool)
        return _warmup(_equal_weight_of_universe(holding, prices), max(self.trend_window, self.period))


@register
@dataclass
class BollingerReversion(Strategy):
    """Long when price closes below the lower band, flat when it gets back to the mean."""

    name = "bollinger"
    window: int = 20
    n_std: float = 2.0

    def weights(self, prices):
        mid = prices.rolling(self.window).mean()
        sd = prices.rolling(self.window).std()
        state = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        state[prices < mid - self.n_std * sd] = 1.0
        state[prices >= mid] = 0.0
        holding = state.ffill().fillna(0.0).astype(bool)
        return _warmup(_equal_weight_of_universe(holding, prices), self.window)


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #


@dataclass
class VolTarget(Strategy):
    """Scale another strategy's exposure so its realised volatility ~ `target`.

    Not registered for the CLI directly; use it from Python:
        VolTarget(SMACrossover(), target=0.10)
    """

    name = "vol_target"
    inner: Strategy = None  # type: ignore[assignment]
    target: float = 0.10
    lookback: int = 63
    max_leverage: float = 1.0

    def weights(self, prices):
        w = self.inner.weights(prices)
        rets = prices.pct_change().fillna(0.0)
        port = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
        vol = port.rolling(self.lookback).std() * np.sqrt(252)
        scale = (self.target / vol).clip(upper=self.max_leverage).fillna(0.0)
        return w.mul(scale, axis=0)

    def label(self):
        return f"vol_target({self.inner.label()}, target={self.target})"


def _equal_weight_of_universe(on: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Each asset gets 1/N of equity when 'on' (N = assets with data), else cash.

    Unlike `_equal_weight`, switching one asset off moves its slice to cash
    instead of piling it into the others.
    """
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return on.astype(float).div(n, axis=0).fillna(0.0)


def rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)
