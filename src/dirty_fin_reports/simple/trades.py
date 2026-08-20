"""Per-trade descriptive statistics and cross-sectional groupings.

Sharpe / Sortino here are *per-trade* (mean/std of realized PnL), never
annualized — they describe trade-flow quality, not the portfolio time series
(which lives in ``metrics``).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np

from .ledger import EXIT_TYPES, SIDES


def _pnls(trades) -> np.ndarray:
    return np.array([float(t.get("realized_pnl") or 0.0) for t in trades], dtype=float)


def _episode_max_dd(pnls: np.ndarray, base: float) -> tuple[float, float]:
    """Peak-to-trough of a single-account PnL path → ``(dd_pnl, dd_pct)``.

    The path is anchored at 0 before the first trade so an episode that opens
    with losses still registers a drawdown from the starting account.
    """
    if pnls.size == 0:
        return 0.0, 0.0
    cum = np.concatenate([[0.0], np.cumsum(pnls)])
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    max_dd_pnl = float(-dd) if dd < -1e-9 else 0.0
    max_dd_pct = 100.0 * max_dd_pnl / max(abs(base), 1e-9)
    return max_dd_pnl, max_dd_pct


def trade_stats(trades, base: float = 1000.0, n_accounts: int = 1) -> dict:
    """Descriptive stats over a set of closed trades.

    ``max_dd_pct`` is the **worst per-episode** peak-to-trough of the trade-PnL
    cumsum relative to one account's ``base``. Parallel evaluation episodes are
    never serialized into one fake account (Ref #5). ``n_accounts`` is retained
    for API compatibility but no longer scales the drawdown denominator.
    ``profit_factor`` and ``expectancy_in_risks`` are ``None`` when there are no
    losing trades — the ratio is undefined, not "999".
    """
    del n_accounts  # drawdown is per-episode; do not invent a multi-account book
    trades = list(trades)
    pnls = _pnls(trades)
    n = int(pnls.size)
    empty = dict(
        num=0, net_pnl=0.0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
        profit_factor=None, expectancy=0.0, expectancy_in_risks=None,
        risk_reward=0.0, max_dd_pnl=0.0, max_dd_pct=0.0,
        sharpe_per_trade=None, sortino_per_trade=None,
    )
    if n == 0:
        return empty
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    net = float(pnls.sum())
    win_rate = 100.0 * float((pnls > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    gw = float(wins.sum()) if wins.size else 0.0
    gl = float(abs(losses.sum())) if losses.size else 0.0
    profit_factor = gw / gl if gl > 1e-12 else (None if gw > 0 else 0.0)
    risk_reward = avg_win / abs(avg_loss) if losses.size and abs(avg_loss) > 1e-12 else 0.0
    expectancy = net / n
    expectancy_in_risks = expectancy / abs(avg_loss) if losses.size and abs(avg_loss) > 1e-12 else None

    by_ep: dict[int, list] = {}
    for t in trades:
        by_ep.setdefault(int(t.get("episode") or 0) if isinstance(t, dict) else 0, []).append(t)
    if len(by_ep) <= 1:
        max_dd_pnl, max_dd_pct = _episode_max_dd(pnls, base)
    else:
        dds = [_episode_max_dd(_pnls(ep_trades), base) for ep_trades in by_ep.values()]
        max_dd_pnl = max(d for d, _ in dds)
        max_dd_pct = max(p for _, p in dds)

    std = float(pnls.std())
    sharpe = float(pnls.mean()) / std if n > 1 and std > 1e-12 else None
    dstd = float(losses.std()) if losses.size > 1 else 0.0
    sortino = float(pnls.mean()) / dstd if dstd > 1e-12 else None

    return dict(
        num=n, net_pnl=net, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=profit_factor, expectancy=expectancy,
        expectancy_in_risks=expectancy_in_risks, risk_reward=risk_reward,
        max_dd_pnl=max_dd_pnl, max_dd_pct=max_dd_pct,
        sharpe_per_trade=sharpe, sortino_per_trade=sortino,
    )


def by_group(trades, keyfn: Callable, order=None, base: float = 1000.0, n_accounts: int = 1) -> list:
    """Group trades by ``keyfn`` and return ``[(label, stats)]`` in ``order``."""
    groups: dict = OrderedDict()
    for t in trades:
        groups.setdefault(keyfn(t), []).append(t)
    labels = order if order is not None else list(groups)
    return [(lab, trade_stats(groups[lab], base=base, n_accounts=n_accounts))
            for lab in labels if lab in groups]


def by_symbol(trades, base: float = 1000.0, n_accounts: int = 1) -> list:
    syms = sorted({t["symbol"] for t in trades})
    return [(s, trade_stats([t for t in trades if t["symbol"] == s], base=base,
                            n_accounts=n_accounts)) for s in syms]


def by_exit(trades, base: float = 1000.0, n_accounts: int = 1) -> list:
    order = [e for e in EXIT_TYPES]
    return by_group(trades, lambda t: t.get("exit_type") or "unknown", order=order,
                    base=base, n_accounts=n_accounts)


def by_side(trades, base: float = 1000.0, n_accounts: int = 1) -> list:
    return by_group(trades, lambda t: t.get("side"), order=list(SIDES),
                    base=base, n_accounts=n_accounts)


def by_episode(trades, base: float = 1000.0, n_accounts: int = 1) -> list:
    eps = sorted({int(t.get("episode") or 0) for t in trades})
    return [(ep, trade_stats([t for t in trades if int(t.get("episode") or 0) == ep],
                             base=base, n_accounts=n_accounts)) for ep in eps]


def hold_stats(trades, bar_minutes: int = 5) -> dict:
    """Hold-duration stats in seconds (uses ``closed_at - opened_at`` bars)."""
    if not trades:
        return {"n": 0, "mean_seconds": 0.0, "median_seconds": 0.0, "max_seconds": 0.0}
    secs = np.array([
        (int(t.get("closed_at") or 0) - int(t.get("opened_at") or 0)) * int(bar_minutes) * 60
        for t in trades
    ], dtype=float)
    return {
        "n": int(secs.size),
        "mean_seconds": float(secs.mean()),
        "median_seconds": float(np.median(secs)),
        "max_seconds": float(secs.max()),
    }


def leverage_stats(trades) -> dict:
    if not trades:
        return {"n": 0, "mean": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0}
    lev = np.array([float(t.get("leverage") or 0.0) for t in trades], dtype=float)
    return {
        "n": int(lev.size),
        "mean": float(lev.mean()),
        "median": float(np.median(lev)),
        "max": float(lev.max()),
        "p95": float(np.percentile(lev, 95)),
    }