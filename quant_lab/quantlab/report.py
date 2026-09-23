"""Self-contained HTML reports (charts embedded as images)."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402

STRAT = "#2a78d6"
BENCH = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def equity_chart(r: pd.Series, bench: pd.Series | None, title: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    eq = (1 + r).cumprod()
    ax.plot(eq.index, eq, color=STRAT, lw=2, label="Strategy")
    if bench is not None:
        be = (1 + bench.reindex(r.index).fillna(0)).cumprod()
        ax.plot(be.index, be, color=BENCH, lw=2, label=bench.name or "Benchmark")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_ylabel("Growth of $1 (log scale)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    return _png(fig)


def drawdown_chart(r: pd.Series, bench: pd.Series | None) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    dd = M.drawdown(r)
    ax.fill_between(dd.index, dd, 0, color=STRAT, alpha=0.35, lw=0)
    ax.plot(dd.index, dd, color=STRAT, lw=1.2, label="Strategy")
    if bench is not None:
        bd = M.drawdown(bench.reindex(r.index).fillna(0))
        ax.plot(bd.index, bd, color=BENCH, lw=1.2, label=bench.name or "Benchmark")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Drawdown from previous peak", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    return _png(fig)


def rolling_sharpe_chart(r: pd.Series, window: int = 252) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    rs = r.rolling(window).mean() / r.rolling(window).std() * (252 ** 0.5)
    ax.plot(rs.index, rs, color=STRAT, lw=1.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("Rolling 1-year Sharpe ratio (is the edge stable?)", loc="left", fontsize=11, color="#0b0b0b")
    _style(ax)
    return _png(fig)


def _fmt(k: str, v) -> str:
    return html.escape(M.fmt(k, v))


def html_report(path: str | Path, title: str, returns: pd.Series, bench: pd.Series | None = None,
                stats: dict | None = None, checks=None, extra_tables: dict[str, pd.DataFrame] | None = None,
                notes: str = "") -> Path:
    stats = stats or M.summary(returns)
    rows = "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{_fmt(k, v)}</td></tr>"
                   for k, v in stats.items())
    check_html = ""
    if checks:
        items = "".join(
            f"<li class='{'ok' if ok else 'bad'}'><b>{'✓ PASS' if ok else '✗ FAIL'}</b> "
            f"{html.escape(n)} <span>— {html.escape(d)}</span></li>" for n, ok, d in checks)
        check_html = f"<h2>Is this edge real?</h2><ul class='checks'>{items}</ul>"
    monthly = M.monthly_table(returns).map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    tables = {"Monthly returns": monthly, **(extra_tables or {})}
    table_html = "".join(f"<h2>{html.escape(t)}</h2><div class='scroll'>{df.round(3).to_html(border=0)}</div>"
                         for t, df in tables.items())
    imgs = [equity_chart(returns, bench, "Growth of $1"), drawdown_chart(returns, bench),
            rolling_sharpe_chart(returns)]
    img_html = "".join(f"<img src='data:image/png;base64,{b}' alt='chart'>" for b in imgs)
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{html.escape(title)}</title><style>
body{{font:14px/1.5 system-ui,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:1100px;margin:0 auto;padding:16px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}} img{{max-width:100%;display:block;margin:12px 0}}
table{{border-collapse:collapse;font-size:13px}} th,td{{padding:3px 10px;text-align:right;border-bottom:1px solid #e4e3df}}
th{{color:#52514e;font-weight:500;text-align:left}} .stats{{display:inline-block;vertical-align:top}}
.scroll{{overflow-x:auto}} .checks{{list-style:none;padding:0}} .checks li{{margin:4px 0}}
.ok b{{color:#008300}} .bad b{{color:#c62828}} .checks span{{color:#52514e}} .note{{color:#52514e}}
</style></head><body><h1>{html.escape(title)}</h1><p class='note'>{html.escape(notes)}</p>
{img_html}{check_html}<h2>Statistics</h2><table class='stats'>{rows}</table>{table_html}
<p class='note'>Past performance, especially backtested performance, does not guarantee future results.</p>
</body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
