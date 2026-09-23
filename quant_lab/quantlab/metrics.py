"""Performance and statistical-significance metrics for daily return series."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649


def cagr(r: pd.Series) -> float:
    years = len(r) / TRADING_DAYS
    total = float((1 + r).prod())
    return total ** (1 / years) - 1 if years > 0 and total > 0 else float("nan")


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TRADING_DAYS
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(r: pd.Series) -> float:
    down = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / down * np.sqrt(TRADING_DAYS)) if down > 0 else 0.0


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def longest_drawdown_days(r: pd.Series) -> int:
    underwater = (drawdown(r) < 0).to_numpy()
    best = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return cagr(r) / abs(mdd) if mdd < 0 else float("nan")


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark_sr) given sample length, skew and kurtosis.

    Bailey & López de Prado (2012). Sharpe values are annualised.
    """
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()  # per-period
    sr_b = benchmark_sr / np.sqrt(TRADING_DAYS)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3  # pandas gives excess kurtosis
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    z = (sr - sr_b) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe you'd expect from the BEST of `n_trials` strategies with zero true skill."""
    if n_trials <= 1:
        return 0.0
    z1 = _norm_ppf(1 - 1 / n_trials)
    z2 = _norm_ppf(1 - 1 / (n_trials * math.e))
    return sharpe_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """Probability the strategy's Sharpe is real after trying `n_trials` variants.

    Bailey & López de Prado (2014). `trial_sharpes` (annualised) are used to
    estimate how much Sharpe varies across your attempts; without them a
    conservative default of 0.5 is used.
    """
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sd = float(np.nanstd(trial_sharpes, ddof=1))
    else:
        sd = 0.5
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sd))


def bootstrap_sharpe_ci(r: pd.Series, n_boot: int = 1000, block: int = 21, alpha: float = 0.05,
                        seed: int = 0, benchmark: pd.Series | None = None) -> tuple[float, float]:
    """Block-bootstrap confidence interval for the annualised Sharpe.

    With `benchmark`, the interval is for Sharpe(r) - Sharpe(benchmark),
    resampling the same days for both (a paired test).
    """
    x = r.to_numpy()
    y = benchmark.reindex(r.index).fillna(0.0).to_numpy() if benchmark is not None else None
    n = len(x)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)

    def sr(v):
        sd = v.std(ddof=1)
        return v.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0

    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        out[i] = sr(x[idx]) - (sr(y[idx]) if y is not None else 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def summary(r: pd.Series) -> dict:
    r = r.dropna()
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return {
        "start": str(r.index[0].date()) if len(r) else "",
        "end": str(r.index[-1].date()) if len(r) else "",
        "years": len(r) / TRADING_DAYS,
        "total_return": float((1 + r).prod() - 1),
        "cagr": cagr(r),
        "volatility": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "longest_drawdown_days": longest_drawdown_days(r),
        "calmar": calmar(r),
        "best_year": float(yearly.max()) if len(yearly) else float("nan"),
        "worst_year": float(yearly.min()) if len(yearly) else float("nan"),
        "pct_positive_years": float((yearly > 0).mean()) if len(yearly) else float("nan"),
        "prob_sharpe_gt_0": probabilistic_sharpe(r),
    }


def relative(r: pd.Series, bench: pd.Series) -> dict:
    b = bench.reindex(r.index).fillna(0.0)
    active = r - b
    te = active.std() * np.sqrt(TRADING_DAYS)
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else float("nan")
    return {
        "bench_cagr": cagr(b),
        "bench_sharpe": sharpe(b),
        "bench_max_drawdown": max_drawdown(b),
        "excess_cagr": cagr(r) - cagr(b),
        "beta": float(beta),
        "alpha_ann": float((r.mean() - beta * b.mean()) * TRADING_DAYS),
        "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan"),
    }


def monthly_table(r: pd.Series) -> pd.DataFrame:
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    t = m.unstack()
    t.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: t.shape[1]]
    t["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    return t


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation; accurate to ~1e-9, avoids a scipy dependency.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    rr = q * q
    return (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q / \
        (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1)


PCT_KEYS = {"total_return", "cagr", "volatility", "max_drawdown", "best_year", "worst_year",
            "pct_positive_years", "prob_sharpe_gt_0", "total_costs", "bench_cagr", "bench_max_drawdown",
            "excess_cagr", "alpha_ann", "oos_return"}


def fmt(key: str, v) -> str:
    if isinstance(v, float):
        return f"{v:.1%}" if key in PCT_KEYS else f"{v:.2f}"
    return str(v)
