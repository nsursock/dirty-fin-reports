"""PNG figure generation for the phase-1 simple report.

Ported from the dirty-trading-bot engine but decoupled: input is the ledger
rows, the equity curves and the training CSVs — no trading environment required.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .training import (
    PPO_KEYS,
    SAC_KEYS,
    detect_algorithm,
    load_training_csv,
    moving_average,
    series,
)
from .viz import Palette, base_layout, style_axes, write_png
from .plausibility import check_value
from .trades import (
    by_exit as by_exit_fn,
    by_side as by_side_fn,
)

_MIN_PNG = 32_000
_MAX_TRADE_MARKERS = 12_000
_FONT_FAMILY = "JetBrains Mono, Menlo, Monaco, Consolas, monospace"


def _subplot_box(fig, c, idx, text, size=9):
    """Subtle translucent info box pinned to the top-right of one subplot."""
    r, g, b = (int(c.panel[i : i + 2], 16) for i in (1, 3, 5))
    xref = "x domain" if idx == 1 else f"x{idx} domain"
    yref = "y domain" if idx == 1 else f"y{idx} domain"
    fig.add_annotation(
        x=0.99, y=0.99, xref=xref, yref=yref,
        xanchor="right", yanchor="top", showarrow=False, align="left",
        text=text, font=dict(size=size, color=c.ink, family=_FONT_FAMILY),
        bgcolor=f"rgba({r},{g},{b},0.66)", bordercolor=c.grid, borderwidth=1,
        borderpad=5,
    )


def _skew(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = float(x.std())
    if s == 0.0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (s ** 3))


def _underwater(dd: np.ndarray) -> tuple[int, int]:
    """Longest and current underwater streak (in bars) from a drawdown curve."""
    if dd.size == 0:
        return 0, 0
    u = dd < -1e-9
    longest = cur = 0
    for x in u:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    tail = 0
    for x in u[::-1]:
        if x:
            tail += 1
        else:
            break
    return int(longest), int(tail)


def _num(value, fmt, metric=None, check_val=None, c=None, bounds=None):
    """Format a stat; flag implausible values (from ``plausibility``) in color."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    text = fmt(value) if callable(fmt) else format(value, fmt)
    b = check_value(metric, check_val if check_val is not None else value, bounds)
    if b.ok:
        return text
    col = c.down if b.severity == "high" else c.accent
    mark = "!!" if b.severity == "high" else "!"
    return f"<span style='color:{col}'>{text}{mark}</span>"


def _card(c, rows, width=8):
    """Two-column monospace stats card from ``(label, value_html)`` pairs."""
    lines = [f"{str(label).ljust(width).replace(' ', '&nbsp;')}{text}"
             for label, text in rows]
    return "<br>".join(lines)


def figure1(
    net,
    gross,
    steps,
    ledger,
    path,
    pm,
    ts,
    per_symbol=None,
    theme="synthwave",
    title_prefix="",
    overlays: bool = False,
    verdict: str | None = None,
    periods_per_year: int = 252,
    freq: str = "daily",
    bounds: dict | None = None,
    start_date: str | datetime | None = None,
    bar_minutes: int = 5,
    tick_angle: float | None = 22.5,
):
    """Equity curve, per-trade returns, drawdown and return distribution.

    ``pm`` (portfolio metrics) and ``ts`` (trade stats) are the canonical,
    already-computed values from ``report_dict``/``assemble`` — this function is
    a pure renderer and does not recompute them (the old behavior of calling
    ``metrics``/``trade_stats`` here drifted from ``report.json`` when
    ``rf_annual``/``freq`` differed).

    ``overlays=False`` (default) keeps the equity panel clean — one net and one
    gross line. Set ``overlays=True`` to fade per-symbol curves underneath.
    Each subplot carries a compact stats card; any value that trips the
    ``plausibility`` bounds is colored + marked in place.

    Time axes (equity, trade returns, drawdown) are labeled with real dates
    anchored at ``start_date`` with ``bar_minutes`` bars; their tick labels are
    rotated by ``tick_angle`` degrees (``None`` disables the tilt).
    """
    c = Palette.of(theme)
    net = np.asarray(net, dtype=float)
    gross = np.asarray(gross, dtype=float)
    steps = np.asarray(steps, dtype=float)
    peak = np.maximum.accumulate(net)
    dd = (net - peak) / np.maximum(peak, 1e-8)

    date0 = np.datetime64(start_date) if start_date is not None else np.datetime64(
        datetime(2025, 1, 1))
    step_sec = np.timedelta64(bar_minutes * 60, "s")

    def _to_dates(x):
        return date0 + (np.rint(np.asarray(x, dtype=float) * bar_minutes * 60)
                        .astype("int64") * np.timedelta64(1, "s"))

    x_dates = _to_dates(steps)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Equity curve", "Trade returns", "Drawdown", "Return distribution"),
        horizontal_spacing=0.09, vertical_spacing=0.16,
    )

    fig.add_trace(go.Scatter(x=x_dates, y=gross, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=x_dates, y=net, mode="lines", fill="tonexty",
                             fillcolor=c.down_soft, line=dict(color=c.up, width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)

    if overlays and per_symbol is not None and np.asarray(per_symbol).size:
        for row in np.asarray(per_symbol):
            fig.add_trace(go.Scatter(x=x_dates, y=row, mode="lines",
                                     line=dict(color=c.accent, width=1.2), opacity=0.35,
                                     showlegend=False, hoverinfo="skip"), 1, 1)

    fig.add_trace(go.Scatter(x=x_dates, y=net, mode="lines", line=dict(color=c.up, width=7),
                             opacity=0.2, showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=x_dates, y=net, mode="lines", line=dict(color=c.up, width=2.6),
                             name="Net"), 1, 1)
    fig.add_trace(go.Scatter(x=x_dates, y=gross, mode="lines",
                             line=dict(color=c.accent, width=1.8, dash="dot"),
                             name="Gross"), 1, 1)
    fig.add_hline(y=float(net[0]), line=dict(color=c.muted, width=1, dash="dash"), row=1, col=1)

    xs = np.array([float(t.get("closed_at", 0.0) or 0.0) for t in ledger], dtype=float)
    coll = np.array([float(t.get("collateral", 0.0) or 0.0) for t in ledger], dtype=float)
    ys = np.where(
        coll > 1e-9,
        np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in ledger], dtype=float)
        / np.maximum(coll, 1e-9),
        0.0,
    )
    liq = np.array([t.get("exit_type") == "liquidation" for t in ledger], dtype=bool)
    trade_rets = ys
    if ys.size > _MAX_TRADE_MARKERS:
        idx = np.unique(np.linspace(0, ys.size - 1, _MAX_TRADE_MARKERS).round().astype(int))
        xs, ys, liq, coll = xs[idx], ys[idx], liq[idx], coll[idx]
    win = ys > 0
    loss = ys <= 0
    close_dates = _to_dates(xs)
    fig.add_trace(go.Scatter(x=close_dates[win], y=ys[win], mode="markers",
                             marker=dict(size=7, color=c.down, opacity=0.75, line=dict(width=0)),
                             name="Win"), 1, 2)
    fig.add_trace(go.Scatter(x=close_dates[loss & ~liq], y=ys[loss & ~liq], mode="markers",
                             marker=dict(size=7, color=c.up, opacity=0.75, line=dict(width=0)),
                             name="Loss"), 1, 2)
    fig.add_trace(go.Scatter(x=close_dates[liq], y=ys[liq], mode="markers",
                             marker=dict(size=9, symbol="x", color=c.up, opacity=0.9,
                                         line=dict(width=0)),
                             name="Liquidation"), 1, 2)
    fig.add_hline(y=0, line=dict(color=c.grid, width=1), row=1, col=2)
    fig.add_hline(y=-1.0, line=dict(color=c.down, width=1, dash="dash"), row=1, col=2)

    fig.add_trace(go.Scatter(x=x_dates, y=dd, mode="lines", fill="tozeroy",
                             fillcolor=c.down_soft, line=dict(color=c.down, width=2.4),
                             name="Drawdown", showlegend=False), 2, 1)
    fig.add_trace(go.Histogram(x=trade_rets, nbinsx=36,
                               marker=dict(color=c.up, line=dict(color=c.bg, width=0.6),
                                           opacity=0.9),
                               showlegend=False), 2, 2)
    fig.add_vline(x=0, line=dict(color=c.muted, width=1, dash="dash"), row=2, col=2)
    if trade_rets.size:
        fig.add_vline(x=float(np.mean(trade_rets)), line=dict(color=c.accent, width=1.8),
                      row=2, col=2)

    num = partial(_num, c=c, bounds=bounds)
    n_close = len(ledger)
    liq_count = int(liq.sum())
    max_dd = float(pm.get("max_drawdown") or 0.0)
    avg_dd = float(dd.mean()) if dd.size else 0.0
    longest_uw, cur_uw = _underwater(dd)
    if trade_rets.size:
        mean_r = float(trade_rets.mean())
        med_r = float(np.median(trade_rets))
        std_r = float(trade_rets.std())
        skew_r = _skew(trade_rets)
        p5 = float(np.percentile(trade_rets, 5))
        p95 = float(np.percentile(trade_rets, 95))
        pos_pct = float((trade_rets > 0).mean())
    else:
        mean_r = med_r = std_r = skew_r = p5 = p95 = pos_pct = 0.0
    years = net.size / max(periods_per_year, 1) if net.size else 0.0
    cagr_val = pm.get("cagr")
    if not check_value("cagr", cagr_val, bounds).ok or years < 0.05:
        cagr_val = None

    title_text = f"{title_prefix}BOT PERFORMANCE"
    if verdict:
        title_text += f"<span style='color:{c.accent}'> · {verdict.upper()}</span>"
    fig.update_layout(**base_layout(c, title=dict(
        text=title_text, x=0.5, xanchor="center"),
        showlegend=False, margin=dict(l=60, r=32, t=120, b=88)))
    fig.update_yaxes(title_text="Equity (USDC)", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Return", row=2, col=2)
    fig.update_annotations(font=dict(size=13, color=c.accent))
    if tick_angle is not None:
        for r in (1, 2):
            fig.update_xaxes(tickangle=tick_angle, row=r, col=1)
        fig.update_xaxes(tickangle=tick_angle, row=1, col=2)

    _subplot_box(fig, c, 1, _card(c, [
        ("FINAL", num(pm.get("final_equity"), lambda v: f"&#36;{v:,.0f}", None)),
        ("TOTAL", num(pm.get("total_return"), "+.1%", None)),
        ("CAGR", num(cagr_val, "+.1%", "cagr")),
        ("SHARPE", num(pm.get("sharpe"), ".2f", "sharpe")),
        ("SORTINO", num(pm.get("sortino"), ".2f", "sortino")),
        ("MAX DD", num(-max_dd, ".1%", "max_drawdown", max_dd)),
        ("ULCER", num(pm.get("ulcer_index"), ".1%", "ulcer_index")),
    ]))
    _subplot_box(fig, c, 2, _card(c, [
        ("TRADES", num(ts["num"], ",d", None)),
        ("WIN", num(ts["win_rate"], lambda v: f"{v:.1f}%", "win_rate")),
        ("EXP", num(ts["expectancy"], lambda v: f"&#36;{v:+,.2f}", None)),
        ("PF", num(ts["profit_factor"], ".2f", "profit_factor")),
        ("AVG WIN", num(ts["avg_win"], lambda v: f"&#36;{v:+,.2f}", None)),
        ("AVG LOSS", num(ts["avg_loss"], lambda v: f"&#36;{v:+,.2f}", None)),
        ("LIQ", num(liq_count, ",d", None)),
    ]))
    _subplot_box(fig, c, 3, _card(c, [
        ("MAX DD", num(-max_dd, ".1%", "max_drawdown", max_dd)),
        ("AVG DD", num(avg_dd, ".1%", None)),
        ("ULCER", num(pm.get("ulcer_index"), ".1%", "ulcer_index")),
        ("LONG UW", num(longest_uw, lambda v: f"{v} bars", None)),
        ("CUR UW", num(cur_uw, lambda v: f"{v} bars", None)),
    ]))
    _subplot_box(fig, c, 4, _card(c, [
        ("N", num(n_close, ",d", None)),
        ("MEAN", num(mean_r, "+.2%", None)),
        ("MEDIAN", num(med_r, "+.2%", None)),
        ("P5/P95", f"{p5:.1%} / {p95:.1%}"),
        ("SKEW", num(skew_r, "+.2f", None)),
        ("% POS", num(pos_pct, ".1%", None)),
    ]))
    style_axes(fig, 2, 2, c, pct_rows={(1, 2)}, pct_cols={(2, 2)})
    return write_png(fig, path)


def figure2(ledger, path, theme="synthwave", verdict: str | None = None,
            bounds: dict | None = None):
    """Trade anatomy: leverage, collateral, direction and exit-type distributions."""
    c = Palette.of(theme)
    lev = np.array([float(t.get("leverage", 0) or 0) for t in ledger], dtype=float)
    coll = np.array([float(t.get("collateral", 0) or 0) for t in ledger], dtype=float)
    sides = [t.get("side", "?") for t in ledger]
    exits = [t.get("exit_type", "?") for t in ledger]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Leverage", "Collateral", "Direction", "Exit type"),
        horizontal_spacing=0.09, vertical_spacing=0.16,
    )

    fig.add_trace(go.Histogram(x=lev if lev.size else [0], nbinsx=24,
                               marker=dict(color=c.up, line=dict(color=c.bg, width=0.6),
                                           opacity=0.92),
                               showlegend=False), 1, 1)
    if lev.size:
        fig.add_vline(x=float(np.median(lev)), line=dict(color=c.accent, width=1.8,
                                                         dash="dash"), row=1, col=1)
    fig.add_trace(go.Histogram(x=coll if coll.size else [0], nbinsx=24,
                               marker=dict(color=c.accent, line=dict(color=c.bg, width=0.6),
                                           opacity=0.92),
                               showlegend=False), 1, 2)
    if coll.size:
        fig.add_vline(x=float(np.median(coll)), line=dict(color=c.accent, width=1.8,
                                                          dash="dash"), row=1, col=2)

    from collections import Counter

    sc = Counter(sides)
    fig.add_trace(go.Bar(x=["Long", "Short"],
                         y=[sc.get("long", 0), sc.get("short", 0)],
                         marker=dict(color=[c.up, c.down], line=dict(width=0), opacity=0.95),
                         text=[sc.get("long", 0), sc.get("short", 0)], textposition="outside",
                         textfont=dict(size=12, color=c.ink), showlegend=False), 2, 1)

    ec = Counter(exits)
    order = [k for k in ("take_profit", "stop_loss", "market_close", "liquidation")
             if ec.get(k)]
    if not order:
        order, vals = ["none"], [0]
    else:
        vals = [ec[k] for k in order]
    colors = {
        "take_profit": c.up, "stop_loss": c.accent,
        "market_close": c.muted, "liquidation": c.down,
    }
    fig.add_trace(go.Bar(x=order, y=vals,
                         marker=dict(color=[colors.get(k, c.muted) for k in order],
                                     line=dict(width=0), opacity=0.95),
                         text=vals, textposition="outside",
                         textfont=dict(size=12, color=c.ink), showlegend=False), 2, 2)

    n = len(ledger)
    med_lev = float(np.median(lev)) if lev.size else 0.0
    mean_lev = float(lev.mean()) if lev.size else 0.0
    max_lev = float(lev.max()) if lev.size else 0.0
    p95_lev = float(np.percentile(lev, 95)) if lev.size else 0.0
    med_coll = float(np.median(coll)) if coll.size else 0.0
    mean_coll = float(coll.mean()) if coll.size else 0.0
    max_coll = float(coll.max()) if coll.size else 0.0
    p95_coll = float(np.percentile(coll, 95)) if coll.size else 0.0
    num = partial(_num, c=c, bounds=bounds)

    title_text = "TRADE ANATOMY"
    if verdict:
        title_text += f"<span style='color:{c.accent}'> · {verdict.upper()}</span>"
    fig.update_layout(**base_layout(c, title=dict(
        text=title_text, x=0.5, xanchor="center"),
        showlegend=False, bargap=0.28,
        margin=dict(l=60, r=32, t=120, b=72)))
    for r in (1, 2):
        fig.update_yaxes(title_text="Count", row=r, col=1)
        fig.update_yaxes(title_text="Count", row=r, col=2)
    fig.update_xaxes(title_text="Leverage (x)", row=1, col=1)
    fig.update_xaxes(title_text="Initial margin (USDC)", row=1, col=2)
    fig.update_xaxes(title_text="Side", row=2, col=1)
    fig.update_xaxes(title_text="Exit type", row=2, col=2)
    fig.update_annotations(font=dict(size=13, color=c.accent))

    _subplot_box(fig, c, 1, _card(c, [
        ("MEDIAN", num(med_lev, lambda v: f"{v:.0f}x", None)),
        ("MEAN", num(mean_lev, lambda v: f"{v:.1f}x", None)),
        ("P95", num(p95_lev, lambda v: f"{v:.0f}x", None)),
        ("MAX", num(max_lev, lambda v: f"{v:.0f}x", "max_leverage")),
    ]))
    _subplot_box(fig, c, 2, _card(c, [
        ("MEDIAN", num(med_coll, ",.0f", None)),
        ("MEAN", num(mean_coll, ",.0f", None)),
        ("P95", num(p95_coll, ",.0f", None)),
        ("MAX", num(max_coll, ",.0f", None)),
    ]))

    side_map = {lab: st for lab, st in by_side_fn(ledger)}

    def _side(name, s):
        if s is None or not s["num"]:
            return f"{name}&nbsp;&nbsp;—"
        pct = s["num"] / n if n else 0.0
        return (f"{name}&nbsp;&nbsp;{s['num']:,}&nbsp;&nbsp;{pct:.1%}&nbsp;&nbsp;"
                f"w&nbsp;{s['win_rate']:.1f}%&nbsp;&nbsp;&#36;{s['net_pnl']:+,.0f}")

    _subplot_box(fig, c, 3, "<br>".join([
        _side("LONG", side_map.get("long")),
        _side("SHORT", side_map.get("short")),
    ]))

    exit_map = {lab: st for lab, st in by_exit_fn(ledger)}
    exit_labels = {"take_profit": "TP", "stop_loss": "SL",
                   "market_close": "MKT", "liquidation": "LIQ"}
    exit_lines = []
    for lab in ("take_profit", "stop_loss", "market_close", "liquidation"):
        st = exit_map.get(lab)
        if st is None or not st["num"]:
            continue
        pct = st["num"] / n if n else 0.0
        exit_lines.append(f"{exit_labels[lab]}&nbsp;&nbsp;{st['num']:,}&nbsp;&nbsp;"
                          f"{pct:.1%}&nbsp;&nbsp;w&nbsp;{st['win_rate']:.1f}%")
    _subplot_box(fig, c, 4, "<br>".join(exit_lines) if exit_lines else "—")
    style_axes(fig, 2, 2, c)
    return write_png(fig, path)


def training_figure(csv_path, out_path, title=None, theme="synthwave", ma=10):
    """4x3 diagnostic grid for one training CSV (PPO or SAC by filename)."""
    csv_path = Path(csv_path)
    df = load_training_csv(csv_path)
    if df.empty:
        return None
    algorithm = detect_algorithm(csv_path)
    c = Palette.of(theme)
    keys = PPO_KEYS if algorithm == "ppo" else SAC_KEYS

    xs = series(df, "time/total_timesteps")
    if not np.isfinite(xs).any():
        xs = None
        xlabel = "dump"
    else:
        xlabel = "timesteps"

    fig = make_subplots(rows=4, cols=3, subplot_titles=list(keys) + [""] * (12 - len(keys)),
                        horizontal_spacing=0.08, vertical_spacing=0.10)
    x_full = np.arange(len(df), dtype=float) if xs is None else xs
    for i, k in enumerate(keys):
        r, col = divmod(i, 3)
        r, col = r + 1, col + 1
        y = series(df, k)
        ok = np.isfinite(y)
        if not ok.any():
            fig.update_yaxes(title_text=k.split("/", 1)[-1], row=r, col=col)
            continue
        y_f = y[ok]
        mode = "lines+markers" if y_f.size < 80 else "lines"
        fig.add_trace(go.Scatter(x=x_full[ok], y=y_f, mode=mode, name="raw",
                                 legendgroup="raw", showlegend=(i == 0),
                                 line=dict(color=c.accent, width=1.5),
                                 marker=dict(size=5, color=c.accent, opacity=0.85),
                                 opacity=0.9), r, col)
        trend = moving_average(y, ma)
        if np.isfinite(trend).any():
            fig.add_trace(go.Scatter(x=x_full, y=trend, mode="lines", name=f"MA{ma}",
                                     legendgroup="ma", showlegend=(i == 0),
                                     line=dict(color=c.up, width=2.3),
                                     connectgaps=False), r, col)
        fig.update_yaxes(title_text=k.split("/", 1)[-1], row=r, col=col)
        fig.update_xaxes(title_text=xlabel, row=r, col=col)

    title = title or csv_path.stem
    fig.update_layout(**base_layout(c, title=dict(text=title, x=0.01, xanchor="left"),
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                x=1, xanchor="right"),
                                    margin=dict(l=72, r=36, t=88, b=64)))
    fig.update_annotations(font=dict(size=11, color=c.accent))
    style_axes(fig, 4, 3, c)
    return write_png(fig, out_path, width=1480, height=1180)