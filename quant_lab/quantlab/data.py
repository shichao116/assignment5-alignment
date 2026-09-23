"""Price data: download from Yahoo Finance (cached on disk), load CSVs, or simulate.

Every loader returns a "wide" DataFrame of adjusted close prices:
index = trading dates, columns = tickers.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE = Path(os.environ.get("QUANTLAB_CACHE", Path.cwd() / ".cache" / "prices"))


def load_prices(
    tickers: list[str] | str,
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted daily closes for `tickers`, cached per ticker as CSV.

    The cache stores the full history downloaded for a ticker; `start`/`end`
    slice it. Pass `refresh=True` to re-download (e.g. to pick up new days).
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    series = {}
    for t in tickers:
        path = cache / f"{t.replace('^', '_')}.csv" if cache else None
        s = None
        if path and path.exists() and not refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            if s.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7):
                s = None  # cache doesn't reach back far enough
        if s is None:
            s = _download_yahoo(t, start)
            if path:
                s.to_csv(path, header=[t])
        series[t] = s

    prices = pd.DataFrame(series).sort_index()
    prices = prices.loc[pd.Timestamp(start):]
    if end:
        prices = prices.loc[: pd.Timestamp(end)]
    return prices.dropna(how="all")


def _download_yahoo(ticker: str, start: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r} (bad ticker or no network?)")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker).dropna()


def load_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """Load your own data.

    Accepts either a wide file (Date, TICKER1, TICKER2, ...) or a long file
    with columns Date, Ticker, Close.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    cols = {c.lower(): c for c in df.columns}
    if "ticker" in cols and "close" in cols:
        df = df.pivot(index=date_col, columns=cols["ticker"], values=cols["close"])
    else:
        df = df.set_index(date_col)
    return df.sort_index().astype(float)


def synthetic_prices(
    tickers: list[str] | int = 5,
    years: float = 15,
    annual_return: float = 0.07,
    annual_vol: float = 0.18,
    correlation: float = 0.5,
    seed: int | None = 0,
    start: str = "2005-01-03",
) -> pd.DataFrame:
    """Random-walk prices with NO exploitable structure.

    Useful as a control: any strategy that looks great on this data is
    fitting noise. Run your optimizer here first and see what "edge" it finds.
    """
    if isinstance(tickers, int):
        tickers = [f"SYN{i}" for i in range(tickers)]
    n_assets = len(tickers)
    n_days = int(years * 252)
    rng = np.random.default_rng(seed)
    cov = np.full((n_assets, n_assets), correlation) + np.eye(n_assets) * (1 - correlation)
    chol = np.linalg.cholesky(cov)
    daily_vol = annual_vol / np.sqrt(252)
    drift = annual_return / 252 - 0.5 * daily_vol**2
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    log_ret = drift + daily_vol * shocks
    prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=tickers)
