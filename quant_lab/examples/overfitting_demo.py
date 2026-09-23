"""Why you must distrust your best backtest.

We generate random-walk prices (no pattern exists by construction), search
hundreds of strategy variants, and look at the "best" one. Then we see how
it does on the next, unseen stretch of the same random process.

Run from the quant_lab directory:  python examples/overfitting_demo.py
"""
from quantlab import BacktestConfig, backtest, get_strategy, synthetic_prices
from quantlab import metrics as M
from quantlab.research import grid_search, walk_forward

prices = synthetic_prices(1, years=20, annual_return=0.05, seed=42)
cfg = BacktestConfig(cost_bps=5)
grid = {"period": [2, 3, 5, 10], "entry": [5, 10, 20, 30], "exit": [50, 60, 70, 80], "trend_window": [0, 100, 200]}

train_end, test_start = "2014-12-31", "2015-01-01"
table = grid_search("rsi_reversion", grid, prices, cfg, end=train_end)
best = table.iloc[0]
params = {k: type(grid[k][0])(best[k]) for k in grid}
print(f"Searched {table.attrs['n_trials']} variants on 10 years of pure noise.")
print(f"Best in-sample: {params}  Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']:.1%}")

oos = backtest(get_strategy("rsi_reversion", **params), prices, cfg, start=test_start)
print(f"Same params on the next 10 years: Sharpe {M.sharpe(oos.returns):.2f}, CAGR {M.cagr(oos.returns):.1%}")
dsr = M.deflated_sharpe(table.attrs["returns"][table.index[0]], table.attrs["n_trials"], table["sharpe"].tolist())
print(f"Deflated Sharpe of the in-sample winner: {dsr:.0%} (the tool was right to doubt it)")
