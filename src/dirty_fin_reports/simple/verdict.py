"""Interpretation layer: independent performance / plausibility / health axes.

The raw plausibility checks in :mod:`.plausibility` answer exactly one
question: *are the numbers statistically sane?*  Overloading that verdict to
mean *is the strategy good?* is what made a losing run read as "plausible"
and a strong run read as "implausible".  This module keeps the two ideas
apart and turns them into one human action:

* ``performance`` — did it make money?  (profitable / breakeven / losing)
* ``plausibility`` — are the numbers sane?  (computed in :mod:`.plausibility`)
* ``health``      — is the equity curve structurally intact?
                    (healthy / degraded / pathological)
* ``recommend``   — one action: accept / review / investigate.

A profitable-but-suspicious run (high alpha) is routed for *review*, never
silently accepted or hard-rejected.
"""

from __future__ import annotations

import numpy as np

BREAKEVEN_EPS = 0.005          # |total_return| within ±0.5% counts as breakeven
DEGRADED_MAX_DRAWDOWN = 0.50   # structural fragility: drew down more than half


def performance_axis(total_return) -> dict:
    """Classify profitability from the whole-window total return."""
    if total_return is None:
        return {"status": "breakeven", "return": None, "return_pct": None}
    ret = float(total_return)
    if not np.isfinite(ret):
        return {"status": "breakeven", "return": None, "return_pct": None}
    if ret > BREAKEVEN_EPS:
        status = "profitable"
    elif ret < -BREAKEVEN_EPS:
        status = "losing"
    else:
        status = "breakeven"
    return {"status": status, "return": ret, "return_pct": ret * 100.0}


def health_axis(net) -> dict:
    """Assess the structural integrity of an equity curve (magnitude-agnostic).

    ``pathological`` means the curve is not a valid account path at all
    (non-finite, non-positive or frozen); ``degraded`` means it is valid but
    structurally fragile (gave back more than half the account); ``healthy``
    otherwise.  High returns never make a curve "unhealthy" here.
    """
    arr = np.asarray(net, dtype=float)
    if arr.size == 0:
        return {"status": "pathological", "issues": ["empty equity curve"]}

    issues: list[str] = []
    if not np.all(np.isfinite(arr)):
        issues.append("non-finite equity values")
    if np.any(arr <= 0.0):
        issues.append("equity hit or fell below zero")
    if arr.size < 2 or np.allclose(arr, arr[0]):
        issues.append("frozen equity curve (zero variance)")
    if issues:
        return {"status": "pathological", "issues": issues}

    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.maximum(peak, 1e-12)
    max_dd = float(dd.min())
    if max_dd < -DEGRADED_MAX_DRAWDOWN:
        return {
            "status": "degraded",
            "issues": [f"drawdown exceeded {DEGRADED_MAX_DRAWDOWN:.0%} "
                       f"(max {max_dd:.1%})"],
        }
    return {"status": "healthy", "issues": []}


def recommend(performance: str, plausibility: str, health: str,
              violations: list[str] | None = None) -> dict:
    """Combine the three independent axes into one human action.

    Priority order:

    * pathological equity -> ``investigate`` (broken math, fix the pipeline)
    * implausible numbers -> ``investigate`` unless profitable, in which case
      ``review`` (a high-alpha outlier to verify, not discard)
    * degraded equity    -> ``review`` (structurally fragile)
    * flagged numbers    -> ``review`` (mild anomaly / high-alpha candidate)
    * losing             -> ``review`` (clean numbers but no edge)
    * otherwise          -> ``accept`` (trustworthy result)

    ``violations`` are the metric names that tripped the plausibility bounds;
    they are embedded in the reason so the action is self-explanatory.
    """
    violations = violations or []
    tag = f" ({', '.join(violations)})" if violations else ""

    if health == "pathological":
        return {"action": "investigate",
                "reason": "structurally broken equity curve; fix before trusting"}
    if plausibility == "implausible":
        if performance == "profitable":
            return {"action": "review",
                    "reason": f"profitable but statistically implausible{tag} — "
                              "high-alpha candidate, verify before trusting"}
        return {"action": "investigate",
                "reason": f"statistically implausible{tag} and not profitable"}
    if health == "degraded":
        return {"action": "review", "reason": "structurally fragile equity curve"}
    if plausibility == "flagged":
        if performance == "profitable":
            return {"action": "review",
                    "reason": f"profitable but statistically suspicious{tag} — "
                              "high-alpha candidate, verify before trusting"}
        return {"action": "review", "reason": f"statistically suspicious{tag}"}
    if performance == "losing":
        return {"action": "review",
                "reason": "statistically clean but unprofitable — review why"}
    return {"action": "accept",
            "reason": "statistically clean and structurally sound"}
