# quantlab: a personal quant research lab

A small, readable backtesting system for amateur investors who want to try
trading ideas on real market data and find out, honestly, whether they work.

It is built around one idea: **an easy way to find a strategy that "worked" in
the past is to try many variants until one fits the noise.** Most of this
toolkit exists to catch that. It gives you:

| Piece | What it does |
|---|---|
| `data.py` | Daily prices from Yahoo Finance, cached locally. Also loads your own CSVs and generates random-walk "control" data. |
| `strategies.py` | 9 built-in strategies (trend, momentum, mean-reversion, risk-parity, baselines) and a simple way to write your own. |
| `backtest.py` | Portfolio simulator: next-day execution, transaction costs, weekly/monthly rebalancing, weight drift, cash yield. Also a look-ahead-bias detector. |
| `metrics.py` | CAGR, Sharpe, Sortino, drawdowns, plus statistical tests: bootstrap confidence intervals, Probabilistic Sharpe and Deflated Sharpe. |
| `research.py` | Grid search, **walk-forward optimisation**, and a plain-language "is this edge real?" checklist. |
| `report.py` | Self-contained HTML reports with equity, drawdown and rolling-Sharpe charts. |
| `cli.py` | The `quantlab` command. |

## Install

```bash
cd quant_lab
pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
pytest                       # 21 tests, runs offline
```

## 10-minute tour

```bash
# What strategies exist, and their parameters
quantlab list

# Backtest the classic 50/200-day moving-average cross on the S&P 500
quantlab backtest sma_cross --tickers SPY -p fast=50 -p slow=200 --report reports/sma.html

# Multi-asset momentum: hold the 3 strongest of 8 asset classes, rebalance monthly
U=SPY,EFA,EEM,TLT,IEF,GLD,VNQ,DBC
quantlab backtest xs_momentum --tickers $U --require-all -p top_n=3 --rebalance M --benchmark equal

# Search parameters (in-sample: optimistic by design)
quantlab optimize xs_momentum --tickers $U --require-all --rebalance M \
    -g lookback=21,63,126,189,252 -g top_n=1,2,3,4 -g skip=0,21

# The honest test: pick parameters on 5 past years, trade the next year, repeat
quantlab walkforward xs_momentum --tickers $U --require-all --rebalance M --benchmark equal \
    -g lookback=21,63,126,189,252 -g top_n=1,2,3,4 -g skip=0,21 --report reports/wf.html

# What would the strategy hold today? (for paper trading by hand)
quantlab signal xs_momentum --tickers $U --require-all -p top_n=3 --capital 25000
```

Useful flags on every command: `--cost-bps` (default 5), `--lag` (default 1
day between signal and trade), `--rebalance D|W|M|Q`, `--start/--end`,
`--benchmark TICKER|equal`, `--csv myprices.csv`, `--synthetic N`.

## The research workflow

1. **Start with a hypothesis, not a parameter sweep.** "Assets that went up
   over the past year tend to keep going up for a while" is a hypothesis.
   "RSI(7) < 23 on Tuesdays" is not a hypothesis.
2. **Backtest with realistic frictions.** Keep `--lag 1` and a cost of at
   least 5 bps for liquid ETFs; use 20+ bps for small caps. Check
   `avg_annual_turnover`: a turnover of 20 at 10 bps costs 2% a year.
3. **Compare with the right benchmark.** For a single index, buy & hold.
   For a multi-asset rotation, `--benchmark equal` (the same assets, equal
   weights, rebalanced monthly). Many "great" strategies are just the
   benchmark with extra steps.
4. **Count your trials.** Every variant you try, including ones you dropped,
   raises the bar. Pass `--trials N` to `backtest` so the Deflated Sharpe
   accounts for it.
5. **Walk forward.** Only the stitched out-of-sample result of `walkforward`
   estimates what you would really have earned. Expect it to be much worse
   than the best in-sample number.
6. **Run the control.** Run the same search with `--synthetic 5` (random
   walks with no pattern). If it finds "edges" there that look as good as on
   real data, your real-data result is probably noise too.
   `python examples/overfitting_demo.py` shows this: the best of 192 variants
   earns a Sharpe of 0.97 on noise, then −0.25 on the next 10 years.
7. **Paper trade** with `quantlab signal` for a few months before risking
   money, and compare the live fills with the backtest.

### Reading the "Is this edge real?" checklist

| Check | Why it matters |
|---|---|
| ≥ 10 years of history | Shorter samples may not include a crash, a bear market or a rate cycle. |
| Sharpe CI excludes 0 | A block bootstrap of daily returns. If the interval includes 0, the result could be luck. |
| Deflated Sharpe > 95% | Probability the Sharpe is real after correcting for how many variants you tried (Bailey & López de Prado). |
| Beats the benchmark, *significantly* | A paired bootstrap on the Sharpe difference. This is the check most strategies fail. |
| Shallower drawdown | Many trend rules don't beat buy & hold on return, but cut the worst loss, which can help you stick with a plan. |
| Both halves positive | An edge that only existed in 2003–2012 is probably gone. |
| Out-of-sample | In-sample results are always optimistic. |

## What the built-in strategies show on real data (through Sept 2026, 5 bps costs)

These are the tool's actual outputs, not predictions:

- **SPY 50/200 SMA cross:** Sharpe 0.71 vs SPY's 0.69. Max drawdown −34% vs
  −55%, but CAGR is 2.6 points lower. The Sharpe advantage is **not**
  statistically significant (95% CI −0.25 to +0.33).
- **SPY/IEF 200-day trend switch** (see `examples/custom_strategy.py`):
  out-of-sample Sharpe 0.83 vs 0.64 with half the drawdown. It still falls
  short of significance (CI −0.30 to +0.56).
- **8-asset momentum rotation, walk-forward:** out-of-sample Sharpe 0.53 vs
  0.68 for simply holding all 8 equally. In-sample Sharpe averaged 1.05, but
  out-of-sample it averaged 0.64.

That is roughly what the academic literature suggests. Simple, public rules
on liquid ETFs mostly give you **risk control** (smaller crashes). They rarely
give provably higher returns. Treat any result that looks much better with
suspicion until it has passed walk-forward testing, the noise control and
paper trading.

## Writing your own strategy

```python
from dataclasses import dataclass
import pandas as pd
from quantlab import Strategy, register, backtest, check_lookahead, load_prices, BacktestConfig

@register
@dataclass
class MyIdea(Strategy):
    """One-line description shown by `quantlab list`."""
    name = "my_idea"
    window: int = 100

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Row t = target weights using data up to day t's close ONLY.
        # 1.0 = fully invested, 0 = cash, negative = short. NaN rows = warm-up.
        up = prices > prices.rolling(self.window).mean()
        return up.astype(float) / prices.shape[1]

prices = load_prices(["SPY", "QQQ", "IWM"], start="2003-01-01").dropna()
assert not check_lookahead(MyIdea(), prices)
print(backtest(MyIdea(), prices, BacktestConfig(rebalance="W")).stats())
```

Don't shift inside `weights()`; the backtester applies the execution delay.
`check_lookahead` recomputes your weights on truncated data and catches any
use of future rows (`shift(-1)`, centred windows, full-sample normalisation
and so on).

## How the simulation works

For each day *t*: holdings earn day *t*'s close-to-close return and drift
with prices. Then, on a rebalance day, the portfolio trades to the target
set at the close of day *t − lag*, paying `cost_bps` × traded notional. With
the default `lag=1`, a signal from Monday's close is traded at Tuesday's
close. This is conservative and achievable with market-on-close orders.

Known simplifications: close prices only (no intraday stops), no taxes, no
short-borrow fees beyond `cash_rate`, dividends via Yahoo's adjusted
closes, and survivorship bias if you backtest today's index members. Prefer
ETFs and indexes over hand-picked stocks for that last reason.

## Ideas for extending it

- A broker adapter (e.g. Alpaca's paper-trading API) that turns
  `quantlab signal` into orders.
- More data: fundamentals, FRED macro series, VIX as a regime filter.
- Combining strategies (a portfolio of trend + momentum + risk parity) and
  testing whether the blend is more robust than each part.
- A tax-aware mode for taxable accounts, where turnover is costly.

*This is research software, not investment advice. Backtested results are
hypothetical and don't guarantee future results.*
