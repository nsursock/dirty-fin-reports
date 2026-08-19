"""Plausibility ("likelihood") validation for computed metrics.

The honest headline is not the raw number but the number *and* whether it is
economically plausible. This layer flags implausible values (a per-bar Sharpe of
160, a 100x+ win rate, a profit factor of 999) instead of printing them as fact.
Bounds are configurable; defaults live in ``DEFAULT_BOUNDS``.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "sharpe": (-5.0, 10.0),
    "sortino": (-10.0, 15.0),
    "calmar": (-5.0, 10.0),
    "cagr": (-0.99, 5.0),
    "max_drawdown": (0.0, 1.0),
    "ulcer_index": (0.0, 0.5),
    "upi": (None, 30.0),
    "win_rate": (0.0, 100.0),
    "profit_factor": (0.0, 20.0),
    "expectancy_in_risks": (None, 5.0),
    "max_leverage": (0.0, 100.0),
    "reward_delta_pct": (-50.0, 50.0),
}


@dataclass(frozen=True)
class Bound:
    metric: str
    low: float | None
    high: float | None
    actual: float | None
    ok: bool
    severity: str
    reason: str = ""


def _severity(value: float, low: float | None, high: float | None) -> tuple[bool, str]:
    ok = True
    severity = "ok"
    if high is not None and value > high:
        ok = False
        severity = "high" if value > high + max(abs(high), 1.0) else "low"
    elif low is not None and value < low:
        ok = False
        severity = "high" if value < low - max(abs(low), 1.0) else "low"
    return ok, severity


def check_value(metric: str, value, bounds: dict | None = None) -> Bound:
    """Validate one metric against a bounds dict; ``None`` values are undefined (ok)."""
    b = (bounds or DEFAULT_BOUNDS).get(metric, (None, None))
    low, high = b
    if value is None:
        return Bound(metric, low, high, None, True, "ok", "undefined")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return Bound(metric, low, high, None, True, "ok", "non-numeric")
    ok, sev = _severity(v, low, high)
    if not ok:
        reason = f"{metric}={v:g} outside [{_fmt(low)}, {_fmt(high)}]"
    else:
        reason = "within bounds"
    return Bound(metric, low, high, v, ok, sev, reason)


def _fmt(x: float | None) -> str:
    return "±inf" if x is None else f"{x:g}"


def aggregate(checks: list[Bound]) -> dict:
    """Roll individual checks into one verdict dict."""
    by_sev: dict[str, int] = {"ok": 0, "low": 0, "high": 0}
    failed = []
    for c in checks:
        if c.ok:
            by_sev["ok"] += 1
        else:
            by_sev[c.severity] += 1
            failed.append(c)
    if by_sev["high"] > 0:
        status = "implausible"
    elif by_sev["low"] > 0:
        status = "flagged"
    elif by_sev["ok"] == 0:
        status = "empty"
    else:
        status = "plausible"
    return {"status": status, "counts": by_sev, "failed": [c.reason for c in failed]}


def check_many(metrics: dict, bounds: dict | None = None) -> list[Bound]:
    """Validate a flat ``{metric_name: value}`` dict against bounds."""
    names = set((bounds or DEFAULT_BOUNDS))
    names |= set(metrics)
    return [check_value(n, metrics.get(n), bounds) for n in sorted(names)]