"""Command line: `python -m quantlab <command> ...` (or `quantlab ...` once installed).

Examples
  quantlab list
  quantlab backtest sma_cross --tickers SPY -p fast=50 -p slow=200
  quantlab optimize sma_cross --tickers SPY -g fast=20,50,100 -g slow=100,150,200,250
  quantlab walkforward xs_momentum --tickers SPY,EFA,EEM,TLT,GLD,VNQ -g lookback=63,126,252 -g top_n=1,2,3
  quantlab signal xs_momentum --tickers SPY,EFA,EEM,TLT,GLD,VNQ -p top_n=2
"""
from __future__ import annotations

import argparse
import ast
import inspect
import sys
from dataclasses import fields

import pandas as pd

from . import metrics as M
from .backtest import BacktestConfig, backtest, benchmark_returns, check_lookahead
from .data import load_csv, load_prices, synthetic_prices
from .report import html_report
from .research import format_verdict, grid_search, verdict, walk_forward
from .strategies import REGISTRY, get_strategy


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {"true": True, "false": False}.get(s.lower(), s)


def _kv(items, multi=False):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = [_value(x) for x in v.split(",")] if multi else _value(v)
    return out


def _prices(a) -> pd.DataFrame:
    if a.csv:
        p = load_csv(a.csv)
    elif a.synthetic:
        p = synthetic_prices(a.synthetic, seed=a.seed)
    else:
        p = load_prices(a.tickers, start=a.data_start, refresh=a.refresh)
    if a.require_all:
        p = p.dropna()
    return p


def _cfg(a) -> BacktestConfig:
    return BacktestConfig(cost_bps=a.cost_bps, lag=a.lag, rebalance=a.rebalance, cash_rate=a.cash_rate)


def _bench(a, prices):
    if a.benchmark == "equal":
        from .strategies import BuyAndHold
        cfg = BacktestConfig(cost_bps=a.cost_bps, rebalance="M")
        return backtest(BuyAndHold(), prices, cfg).returns.rename("equal-weight universe")
    if a.benchmark and a.benchmark not in prices.columns:
        b = load_prices([a.benchmark], start=a.data_start)
        return b[a.benchmark].pct_change().fillna(0).rename(f"{a.benchmark} buy&hold")
    return benchmark_returns(prices, a.benchmark)


def _print_stats(title, stats):
    print(f"\n== {title} ==")
    for k, v in stats.items():
        print(f"  {k:<24} {M.fmt(k, v)}")


def cmd_list(a):
    for name, cls in sorted(REGISTRY.items()):
        doc = inspect.getdoc(cls) or ""
        params = ", ".join(f"{f.name}={f.default}" for f in fields(cls))
        print(f"{name:<16} {params}\n{'':<16} {doc.splitlines()[0] if doc else ''}")


def cmd_backtest(a):
    prices = _prices(a)
    strat = get_strategy(a.strategy, **_kv(a.param))
    problems = check_lookahead(strat, prices)
    if problems:
        print("WARNING — possible look-ahead bias:\n  " + "\n  ".join(problems))
    res = backtest(strat, prices, _cfg(a), a.start, a.end)
    bench = _bench(a, prices).reindex(res.returns.index).fillna(0)
    stats = res.stats(bench)
    _print_stats(res.label, stats)
    checks = verdict(res.returns, bench, n_trials=a.trials)
    print("\nReality check (in-sample):\n" + format_verdict(checks))
    last = res.weights.iloc[-1]
    print("\nCurrent holdings:", ", ".join(f"{k} {v:.0%}" for k, v in last.items() if abs(v) > 1e-4) or "cash")
    if a.report:
        path = html_report(a.report, res.label, res.returns, bench, stats, checks,
                           notes=f"cost {a.cost_bps} bps, lag {a.lag}d, rebalance {a.rebalance}")
        print(f"\nReport written to {path}")


def cmd_optimize(a):
    prices = _prices(a)
    grid = _kv(a.grid, multi=True)
    table = grid_search(a.strategy, grid, prices, _cfg(a), a.start, a.end, objective=a.objective,
                        constraint=_constraint(a.strategy))
    pd.set_option("display.width", 200)
    print(table.head(a.top).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n = table.attrs["n_trials"]
    best_rets = table.attrs["returns"][table.index[0]]
    dsr = M.deflated_sharpe(best_rets, n, table["sharpe"].tolist())
    print(f"\nTried {n} combinations. Best in-sample Sharpe {table['sharpe'].iloc[0]:.2f}. Even with zero "
          f"skill, the luckiest of {n} variants would typically beat a true Sharpe of 0 by "
          f"~{M.expected_max_sharpe(n, table['sharpe'].std()):.2f}.")
    print(f"Deflated Sharpe (probability the winner's edge is real): {dsr:.1%}")
    print("Tip: prefer a parameter region where neighbours also do well over a lone peak, "
          "then confirm with `walkforward`.")
    if a.csv_out:
        table.to_csv(a.csv_out, index=False)


def cmd_walkforward(a):
    prices = _prices(a)
    grid = _kv(a.grid, multi=True)
    wf = walk_forward(a.strategy, grid, prices, _cfg(a), a.train_years, a.test_years, a.anchored,
                      a.objective, _constraint(a.strategy), start=a.start)
    if wf.oos_returns.empty:
        sys.exit("Not enough data for even one fold; shorten --train-years.")
    print(wf.folds.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    bench = _bench(a, prices).reindex(wf.oos_returns.index).fillna(0)
    stats = wf.stats(bench)
    _print_stats("Out-of-sample (stitched test windows)", stats)
    print(f"\nIn-sample Sharpe averaged {stats['mean_is_sharpe']:.2f} but out-of-sample averaged "
          f"{stats['mean_oos_sharpe']:.2f}. A big gap means the optimiser is mostly fitting noise.")
    checks = verdict(wf.oos_returns, bench, n_trials=1, oos=True)
    print("\nReality check:\n" + format_verdict(checks))
    if a.report:
        path = html_report(a.report, f"{a.strategy} walk-forward", wf.oos_returns, bench, stats, checks,
                           extra_tables={"Folds": wf.folds},
                           notes=f"train {a.train_years}y / test {a.test_years}y, grid {grid}")
        print(f"\nReport written to {path}")


def cmd_signal(a):
    prices = _prices(a)
    strat = get_strategy(a.strategy, **_kv(a.param))
    w = strat.weights(prices).iloc[-1].fillna(0)
    print(f"Target weights from {strat.label()} using data through {prices.index[-1].date()}:")
    for k, v in w.items():
        dollars = f"  (${v * a.capital:,.0f})" if a.capital else ""
        print(f"  {k:<8} {v:7.1%}{dollars}")
    print(f"  {'cash':<8} {1 - w.sum():7.1%}")


def _constraint(name):
    if name == "sma_cross":
        return lambda p: p.get("fast", 0) < p.get("slow", 1e9)
    return None


def build_parser():
    ap = argparse.ArgumentParser(prog="quantlab", description="Personal quant research lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list built-in strategies").set_defaults(fn=cmd_list)

    def common(p):
        p.add_argument("strategy", choices=sorted(REGISTRY))
        g = p.add_argument_group("data")
        g.add_argument("--tickers", default="SPY", help="comma-separated, e.g. SPY,TLT,GLD")
        g.add_argument("--csv", help="use your own CSV instead of downloading")
        g.add_argument("--synthetic", type=int, metavar="N", help="use N random-walk assets (a control)")
        g.add_argument("--seed", type=int, default=0)
        g.add_argument("--data-start", default="2000-01-01", help="download history from here (warm-up)")
        g.add_argument("--start", help="first date to evaluate")
        g.add_argument("--end", help="last date to evaluate")
        g.add_argument("--refresh", action="store_true", help="re-download cached data")
        g.add_argument("--require-all", action="store_true", help="drop dates where any ticker is missing")
        g.add_argument("--benchmark", help="benchmark ticker, or 'equal' for an equal-weight monthly-rebalanced portfolio of --tickers (default: first ticker)")
        e = p.add_argument_group("execution")
        e.add_argument("--cost-bps", type=float, default=5.0)
        e.add_argument("--lag", type=int, default=1)
        e.add_argument("--rebalance", default="D", choices=list("DWMQ"))
        e.add_argument("--cash-rate", type=float, default=0.0)
        p.add_argument("--report", help="write an HTML report to this path")

    p = sub.add_parser("backtest", help="backtest one parameter set")
    common(p)
    p.add_argument("-p", "--param", action="append", help="name=value")
    p.add_argument("--trials", type=int, default=1,
                   help="how many variants you've tried so far (honesty input for deflated Sharpe)")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("optimize", help="grid-search parameters (in-sample)")
    common(p)
    p.add_argument("-g", "--grid", action="append", required=True, help="name=v1,v2,v3")
    p.add_argument("--objective", default="sharpe", choices=["sharpe", "cagr", "calmar"])
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv-out")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("walkforward", help="out-of-sample test of the optimisation process")
    common(p)
    p.add_argument("-g", "--grid", action="append", required=True, help="name=v1,v2,v3")
    p.add_argument("--objective", default="sharpe", choices=["sharpe", "cagr", "calmar", "sortino"])
    p.add_argument("--train-years", type=float, default=5)
    p.add_argument("--test-years", type=float, default=1)
    p.add_argument("--anchored", action="store_true", help="expanding instead of rolling training window")
    p.set_defaults(fn=cmd_walkforward)

    p = sub.add_parser("signal", help="today's target weights (for manual paper/real trading)")
    common(p)
    p.add_argument("-p", "--param", action="append", help="name=value")
    p.add_argument("--capital", type=float, help="show dollar amounts for this account size")
    p.set_defaults(fn=cmd_signal)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
