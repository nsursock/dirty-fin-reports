"""Portfolio risk metrics on a bar-indexed equity curve.

The headline risk ratios are computed at a *reporting cadence* (``freq``,
default ``"daily"``) rather than per 5-minute bar. Aggregating to daily returns
and annualizing by ``sqrt(252)`` keeps the Sharpe, Sortino, Calmar and UPI in
economically plausible territory; annualizing per-bar returns by
``sqrt(252*24*60/bar_min)`` is what produces the implausible "160 Sharpe".
"""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS = 252

_TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "daily": 1440,
    "1w": 10080,
    "weekly": 10080,
}

FREQ_PPY: dict[str, int] = {
    "1m": TRADING_DAYS * 24 * 60,
    "5m": TRADING_DAYS * 24 * 12,
    "15m": TRADING_DAYS * 24 * 4,
    "30m": TRADING_DAYS * 24 * 2,
    "1h": TRADING_DAYS * 24,
    "4h": TRADING_DAYS * 6,
    "1d": TRADING_DAYS,
    "daily": TRADING_DAYS,
    "1w": 52,
    "weekly": 52,
    "bar": None,
}

_EPS = 1e-12


def timeframe_minutes(timeframe: str) -> int:
    """Integer minutes for a kline-style timeframe like ``"5m"`` or ``"4h"``."""
    tf = str(timeframe).strip().lower()
    if tf in _TIMEFRAME_MINUTES:
        return _TIMEFRAME_MINUTES[tf]
    if tf.endswith("w"):
        return _TIMEFRAME_MINUTES["1w"]
    if tf.endswith("d"):
        return _TIMEFRAME_MINUTES["1d"]
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("m"):
        return int(tf[:-1])
    raise ValueError(f"unknown timeframe: {timeframe!r}")


def periods_per_year(timeframe: str = "5m") -> int:
    """Native bar cadence per year for ``timeframe`` (bar-minutes based)."""
    return TRADING_DAYS * 24 * 60 // timeframe_minutes(timeframe)


def returns(equity: np.ndarray) -> np.ndarray:
    """Simple per-bar returns ``(dE / E_prev)`` with a zero safeguard."""
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return np.array([], dtype=float)
    return np.diff(e) / (e[:-1] + _EPS)


def resample_returns(
    equity: np.ndarray, periods_per_year: int, freq: str
) -> tuple[np.ndarray | None, int | None, np.ndarray]:
    """Mark equity at every ``freq`` window and return the window simple returns.

    Returns ``(None, None, equity)`` when the cadence is coarser than the data
    or fewer than two full windows exist; ``(window_returns, ppy_freq,
    levels)`` otherwise, where ``levels`` is equity marked at each window close
    (drawdown metrics use these levels).
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return None, None, e
    fppy = FREQ_PPY.get(str(freq).lower())
    if fppy is None or fppy <= 0:
        raise ValueError(f"unknown reporting frequency: {freq!r}")
    bars_per_agg = max(round(int(periods_per_year) / fppy), 1)
    if bars_per_agg <= 1:
        return None, None, e
    n = e.size
    starts = e[np.arange(0, n - 1, bars_per_agg)]
    ends = e[np.minimum(np.arange(bars_per_agg, n, bars_per_agg), n - 1)]
    k = min(len(starts), len(ends))
    if k < 2:
        return None, None, e
    window_rets = ends[:k] / starts[:k] - 1.0
    levels = np.concatenate([starts[:1], ends[:k]])
    return window_rets, int(fppy), levels


def _excess(rets: np.ndarray, rf_period: float) -> np.ndarray:
    return rets - rf_period


def _sample_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=0))


def metrics(
    net: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    freq: str = "daily",
    rf_annual: float = 0.045,
) -> dict:
    """Risk metrics on a single bar-indexed equity curve.

    ``freq`` selects the Sharpe/Sortino/Calmar/UPI cadence: ``"bar"`` keeps the
    native per-bar series (legacy, annualized by ``sqrt(ppy)`` — suppressible
    via plausibility bounds), anything in ``FREQ_PPY`` aggregates the curve
    first. Dollar facts (``final_equity``, ``total_return``, ``cagr``) always
    come off the raw curve.

    ``sharpe``/``sortino``/``upi`` are ``None`` when their statistics are
    undefined (fewer than two periods, zero dispersion, no downside, zero
    Ulcer) — undefined rather than an inflated or infinite number.
    """
    e = np.asarray(net, dtype=float)
    out: dict = {
        "final_equity": float(e[-1]) if e.size else 0.0,
        "total_return": float(e[-1] / e[0] - 1.0) if e.size >= 2 and e[0] > 0 else 0.0,
        "max_drawdown": 0.0,
        "cagr": 0.0,
        "sharpe": None,
        "sortino": None,
        "ulcer_index": 0.0,
        "upi": None,
        "return_basis": "account",
        "freq": str(freq),
        "rf_annual": float(rf_annual),
        "periods_per_year": int(periods_per_year),
    }
    if e.size < 2:
        return out

    peak_net = np.maximum.accumulate(e)
    dd_net = (peak_net - e) / np.maximum(peak_net, _EPS)
    max_dd = float(np.max(dd_net))
    years = e.size / max(float(periods_per_year) or 1.0, 1e-12)
    cagr = (e[-1] / e[0]) ** (1.0 / years) - 1.0 if years > 0 and e[0] > 0 else 0.0

    ppy_freq = int(periods_per_year)
    levels = e
    window_rets = None
    if str(freq).lower() != "bar":
        window_rets, resampled_ppy, levels = resample_returns(e, periods_per_year, freq)
        if resampled_ppy is not None:
            ppy_freq = resampled_ppy
    if window_rets is not None:
        rets = window_rets
    else:
        rets = returns(e)

    sharpe = sortino = upi = None
    ulcer_index = 0.0
    if levels.size >= 2:
        peak = np.maximum.accumulate(levels)
        dd = (peak - levels) / np.maximum(peak, _EPS)
        ulcer_index = float(np.sqrt(np.mean(np.square(dd))))

    if rets.size >= 2:
        rf_period = float(rf_annual) / max(float(ppy_freq) or 1.0, 1e-12)
        excess = _excess(rets, rf_period)
        mean = float(np.mean(excess))
        std = _sample_std(excess)
        if std > _EPS:
            sharpe = mean / (std + _EPS) * math.sqrt(max(float(ppy_freq) or 1.0, 1e-12))
            down = excess[excess < 0]
            if down.size >= 2:
                dstd = _sample_std(down)
                if dstd > _EPS:
                    sortino = mean / (dstd + _EPS) * math.sqrt(max(float(ppy_freq) or 1.0, 1e-12))
            if ulcer_index > _EPS and mean > 0:
                upi = mean * max(float(ppy_freq) or 1.0, 1e-12) / ulcer_index

    out.update(
        {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "ulcer_index": ulcer_index,
            "upi": upi,
            "cagr": cagr,
            "periods_per_year": int(periods_per_year),
        }
    )
    return out