import numpy as np
import pandas as pd
import pytest

from quantlab import BacktestConfig, backtest, check_lookahead, run_weights, synthetic_prices
from quantlab import metrics as M
from quantlab.backtest import rebalance_mask
from quantlab.research import grid_search, verdict, walk_forward
from quantlab.strategies import REGISTRY, SMACrossover, Strategy, VolTarget, get_strategy


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(4, years=12, seed=1)


def test_buy_and_hold_matches_asset_return_without_costs(prices):
    p = prices[["SYN0"]]
    res = run_weights(p, pd.DataFrame(1.0, index=p.index, columns=p.columns), BacktestConfig(cost_bps=0))
    asset = p["SYN0"].pct_change().fillna(0)
    # first day: no position yet (lag=1); thereafter identical
    assert np.allclose(res.returns.iloc[2:], asset.iloc[2:])
    assert res.returns.iloc[:2].abs().sum() == 0


def test_costs_charged_on_turnover(prices):
    p = prices[["SYN0"]]
    tgt = pd.DataFrame(1.0, index=p.index, columns=p.columns)
    free = run_weights(p, tgt, BacktestConfig(cost_bps=0))
    paid = run_weights(p, tgt, BacktestConfig(cost_bps=10))
    # one full buy on day 1 → equity lower by exactly 10bps
    assert paid.equity.iloc[-1] == pytest.approx(free.equity.iloc[-1] * (1 - 0.001), rel=1e-9)
    assert paid.turnover.sum() == pytest.approx(1.0)


def test_weights_drift_between_rebalances(prices):
    tgt = pd.DataFrame(0.5, index=prices.index, columns=prices.columns[:2])
    res = run_weights(prices[prices.columns[:2]], tgt, BacktestConfig(rebalance="M", cost_bps=0))
    w = res.weights.iloc[10:40]
    assert w.std().max() > 0  # drifted
    assert np.allclose(res.weights.sum(axis=1).iloc[1:], 1.0)  # invested from first signal


def test_rebalance_mask_monthly():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    m = rebalance_mask(idx, "M")
    assert m.sum() == 12 + 1  # month ends + first day


def test_execution_timing():
    """A target set at day t's close earns day t+1's return (lag=0), or t+2's (lag=1)."""
    p = synthetic_prices(1, years=5, seed=3)
    tomorrow_up = (p.shift(-1) > p).astype(float)  # a (cheating) perfect forecast
    at_close = run_weights(p, tomorrow_up, BacktestConfig(lag=0, cost_bps=0))
    next_day = run_weights(p, tomorrow_up, BacktestConfig(lag=1, cost_bps=0))
    assert M.sharpe(at_close.returns) > 5
    assert abs(M.sharpe(next_day.returns)) < 1.5


def test_lookahead_detector_flags_peeking_strategy(prices):
    class Peek(Strategy):
        name = "peek"

        def weights(self, px):
            return (px.shift(-1) > px).astype(float) / px.shape[1]

    assert check_lookahead(Peek(), prices)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_builtin_strategies_have_no_lookahead(name, prices):
    assert check_lookahead(get_strategy(name), prices) == []


def test_vol_target_scales_exposure(prices):
    base = backtest(SMACrossover(20, 100), prices, BacktestConfig(cost_bps=0))
    vt = backtest(VolTarget(SMACrossover(20, 100), target=0.05), prices, BacktestConfig(cost_bps=0))
    assert M.ann_vol(vt.returns.iloc[300:]) < M.ann_vol(base.returns.iloc[300:])
    assert check_lookahead(VolTarget(SMACrossover(20, 100), target=0.05), prices) == []


def test_get_strategy_rejects_unknown_param():
    with pytest.raises(TypeError):
        get_strategy("sma_cross", fats=10)


def test_metrics_known_values():
    r = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=3))
    assert M.max_drawdown(r) == pytest.approx(-0.5)
    assert M._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert M._norm_ppf(0.001) == pytest.approx(-3.090232, abs=1e-5)


def test_psr_and_deflation():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-01", periods=252 * 10)
    good = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)  # Sharpe ~1.6
    assert M.probabilistic_sharpe(good) > 0.99
    assert M.deflated_sharpe(good, 1000, None) < M.deflated_sharpe(good, 1, None)


def test_walk_forward_is_out_of_sample(prices):
    wf = walk_forward("sma_cross", {"fast": [10, 50], "slow": [100, 200]}, prices,
                      train_years=4, test_years=2, constraint=lambda p: p["fast"] < p["slow"])
    assert wf.oos_returns.index.is_unique and wf.oos_returns.index.is_monotonic_increasing
    first_test = pd.Timestamp(wf.folds["test_start"].iloc[0])
    assert wf.oos_returns.index[0] >= first_test
    assert first_test >= prices.index[0] + pd.DateOffset(years=4) - pd.Timedelta(days=2)


def test_optimizer_on_noise_is_not_trusted():
    """On pure random walks, the best of many variants should not pass the deflated-Sharpe bar."""
    noise = synthetic_prices(1, years=10, annual_return=0.0, seed=7)
    table = grid_search("sma_cross", {"fast": [5, 10, 20, 50], "slow": [50, 100, 150, 200, 250]}, noise,
                        constraint=lambda p: p["fast"] < p["slow"])
    best = table.attrs["returns"][table.index[0]]
    assert M.deflated_sharpe(best, table.attrs["n_trials"], table["sharpe"].tolist()) < 0.95


def test_verdict_shape(prices):
    res = backtest(SMACrossover(), prices)
    checks = verdict(res.returns, prices["SYN0"].pct_change().fillna(0))
    assert all(len(c) == 3 for c in checks)
