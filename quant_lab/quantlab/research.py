"""Tools for testing whether a strategy's edge is real rather than luck.

The core danger in amateur quant work is overfitting: try enough parameter
combinations and one will look brilliant on past data. Everything here is
built to expose that:

* `grid_search`     — try many parameter sets, but also record how many you tried.
* `walk_forward`    — pick parameters on past data only, then trade the next
                      unseen period with them; stitch those unseen periods together.
* `verdict`         — plain-language checklist combining the evidence.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .backtest import BacktestConfig, backtest, run_weights
from .strategies import get_strategy


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def grid_search(strategy_name: str, grid: dict[str, list], prices: pd.DataFrame,
                cfg: BacktestConfig | None = None, start=None, end=None,
                objective: str = "sharpe", constraint=None) -> pd.DataFrame:
    """Backtest every combination in `grid`; return a table sorted by `objective`.

    `constraint(params) -> bool` can skip invalid combos (e.g. fast >= slow).
    The table's attrs hold the daily returns of each run (for deflated Sharpe).
    """
    rows, rets = [], {}
    for i, p in enumerate(param_grid(grid)):
        if constraint and not constraint(p):
            continue
        res = backtest(get_strategy(strategy_name, **p), prices, cfg, start, end)
        s = res.stats()
        rows.append({**p, **{k: s[k] for k in ("sharpe", "cagr", "max_drawdown", "volatility",
                                                "calmar", "avg_annual_turnover")}})
        rets[len(rows) - 1] = res.returns
    table = pd.DataFrame(rows)
    table.attrs["returns"] = rets
    table.attrs["n_trials"] = len(rows)
    return table.sort_values(objective, ascending=False)


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series  # stitched out-of-sample daily returns
    folds: pd.DataFrame  # per-fold chosen params and IS/OOS Sharpe
    n_trials: int

    def stats(self, benchmark=None) -> dict:
        s = M.summary(self.oos_returns)
        if benchmark is not None:
            s.update(M.relative(self.oos_returns, benchmark))
        s["mean_is_sharpe"] = float(self.folds["is_sharpe"].mean())
        s["mean_oos_sharpe"] = float(self.folds["oos_sharpe"].mean())
        return s


def walk_forward(strategy_name: str, grid: dict[str, list], prices: pd.DataFrame,
                 cfg: BacktestConfig | None = None, train_years: float = 5, test_years: float = 1,
                 anchored: bool = False, objective: str = "sharpe", constraint=None,
                 start: str | None = None) -> WalkForwardResult:
    """Rolling (or anchored/expanding) walk-forward optimisation.

    For each fold: choose the best params on the training window, then record
    how those params do on the following test window. The stitched test
    windows are an honest estimate of what the *process* of optimising
    would have earned.
    """
    cfg = cfg or BacktestConfig()
    combos = [p for p in param_grid(grid) if not constraint or constraint(p)]
    # Pre-compute every combo's full-history returns once (strategies are causal,
    # so slicing afterwards introduces no look-ahead).
    all_rets = {}
    for i, p in enumerate(combos):
        strat = get_strategy(strategy_name, **p)
        all_rets[i] = run_weights(prices, strat.weights(prices), cfg).returns

    idx = prices.index
    t0 = pd.Timestamp(start) if start else idx[0]
    train = pd.DateOffset(days=int(train_years * 365.25))
    test = pd.DateOffset(days=int(test_years * 365.25))

    folds, pieces = [], []
    test_start = t0 + train
    while test_start < idx[-1]:
        train_start = t0 if anchored else test_start - train
        train_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + test - pd.Timedelta(days=1), idx[-1])

        scores = {i: _score(r.loc[train_start:train_end], objective) for i, r in all_rets.items()}
        best = max(scores, key=scores.get)
        oos = all_rets[best].loc[test_start:test_end]
        if len(oos):
            pieces.append(oos)
            folds.append({"train_start": train_start.date(), "test_start": test_start.date(),
                          "test_end": test_end.date(), **combos[best],
                          "is_sharpe": scores[best] if objective == "sharpe"
                          else M.sharpe(all_rets[best].loc[train_start:train_end]),
                          "oos_sharpe": M.sharpe(oos), "oos_return": float((1 + oos).prod() - 1)})
        test_start = test_start + test

    oos_returns = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return WalkForwardResult(oos_returns.rename(f"{strategy_name} walk-forward"),
                             pd.DataFrame(folds), len(combos))


def _score(r: pd.Series, objective: str) -> float:
    if len(r) < 20:
        return -np.inf
    fn = {"sharpe": M.sharpe, "cagr": M.cagr, "calmar": M.calmar, "sortino": M.sortino}[objective]
    v = fn(r)
    return -np.inf if v is None or np.isnan(v) else v


def verdict(returns: pd.Series, benchmark: pd.Series | None = None, n_trials: int = 1,
            trial_sharpes: list[float] | None = None, oos: bool = False) -> list[tuple[str, bool, str]]:
    """A checklist of (check, passed, detail). Honest, not encouraging."""
    s = M.summary(returns)
    checks = []
    years = s["years"]
    checks.append(("Enough history (>= 10 years)", years >= 10, f"{years:.1f} years"))
    lo, hi = M.bootstrap_sharpe_ci(returns)
    checks.append(("Sharpe 95% CI excludes zero", lo > 0, f"Sharpe {s['sharpe']:.2f}, CI [{lo:.2f}, {hi:.2f}]"))
    dsr = M.deflated_sharpe(returns, n_trials, trial_sharpes)
    checks.append((f"Deflated Sharpe > 95% (after {n_trials} trial(s))", dsr > 0.95, f"{dsr:.1%}"))
    if benchmark is not None:
        rel = M.relative(returns, benchmark)
        checks.append(("Beats the benchmark on risk-adjusted basis", s["sharpe"] > rel["bench_sharpe"],
                       f"Sharpe {s['sharpe']:.2f} vs {rel['bench_sharpe']:.2f}"))
        dlo, dhi = M.bootstrap_sharpe_ci(returns, benchmark=benchmark)
        checks.append(("...and the Sharpe advantage is statistically significant", dlo > 0,
                       f"difference 95% CI [{dlo:+.2f}, {dhi:+.2f}]"))
        checks.append(("Shallower worst drawdown than the benchmark",
                       s["max_drawdown"] > rel["bench_max_drawdown"],
                       f"{s['max_drawdown']:.1%} vs {rel['bench_max_drawdown']:.1%}"))
    checks.append(("Positive in >= 60% of calendar years", s["pct_positive_years"] >= 0.6,
                   f"{s['pct_positive_years']:.0%}"))
    halves = np.array_split(returns.to_numpy(), 2)
    h = [M.sharpe(pd.Series(x)) for x in halves]
    checks.append(("Works in both halves of the sample", min(h) > 0, f"Sharpe {h[0]:.2f} / {h[1]:.2f}"))
    checks.append(("Result is out-of-sample (walk-forward)", oos,
                   "yes" if oos else "no — in-sample results are optimistic"))
    return checks


def format_verdict(checks) -> str:
    lines = []
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    n_ok = sum(ok for _, ok, _ in checks)
    lines.append(f"  {n_ok}/{len(checks)} checks passed.")
    return "\n".join(lines)
