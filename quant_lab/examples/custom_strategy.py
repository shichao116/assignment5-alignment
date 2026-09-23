"""Write your own strategy, check it for look-ahead bias, and test it honestly.

Run from the quant_lab directory:  python examples/custom_strategy.py
"""
from dataclasses import dataclass

import pandas as pd

from quantlab import BacktestConfig, Strategy, backtest, check_lookahead, load_prices, register
from quantlab import metrics as M
from quantlab.research import format_verdict, verdict, walk_forward


@register  # makes it available to the CLI and to walk_forward by name
@dataclass
class StocksBondsTrend(Strategy):
    """Hold stocks when they're above their long-run average, otherwise bonds."""

    name = "stocks_bonds_trend"
    window: int = 200
    stock: str = "SPY"
    bond: str = "IEF"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Use only data up to each day's close. Never use .shift(-n) or future rows.
        up = prices[self.stock] > prices[self.stock].rolling(self.window).mean()
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w[self.stock] = up.astype(float)
        w[self.bond] = (~up).astype(float)
        w.iloc[: self.window] = float("nan")  # warm-up: no decision yet
        return w


prices = load_prices(["SPY", "IEF"], start="2003-01-01").dropna()
strat = StocksBondsTrend()
cfg = BacktestConfig(cost_bps=5, rebalance="W")

assert not check_lookahead(strat, prices), "strategy peeks at the future!"

res = backtest(strat, prices, cfg)
bench = prices["SPY"].pct_change().fillna(0).rename("SPY")
print(strat.label())
for k in ("cagr", "sharpe", "max_drawdown", "avg_annual_turnover"):
    print(f"  {k:<20} {M.fmt(k, res.stats()[k])}")
print("In-sample checks:\n" + format_verdict(verdict(res.returns, bench)))

# The honest version: let the window be chosen on past data only.
wf = walk_forward("stocks_bonds_trend", {"window": [50, 100, 150, 200, 250]}, prices, cfg,
                  train_years=5, test_years=1)
print("\nWalk-forward (out-of-sample) checks:\n"
      + format_verdict(verdict(wf.oos_returns, bench.reindex(wf.oos_returns.index), oos=True)))
