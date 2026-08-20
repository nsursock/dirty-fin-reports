"""Stage 0 validation gate scoring across folds and seeds.

Thresholds are **inputs** (from the bot's frozen ``stage0_gates.yaml``), never
hard-coded here. This module only aggregates fold/seed metrics and evaluates
the declared operators.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def _finite(xs: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def _metric(block: dict | None, name: str) -> float | None:
    if not isinstance(block, dict):
        return None
    v = block.get(name)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _scope_values(rows: list[dict], metric: str, scope: str) -> list[float]:
    return _finite(_metric(r.get(scope), metric) for r in rows)


def _median_scoped(
    rows: list[dict],
    metric: str,
    scope: str,
    *,
    min_n: int = 2,
    min_coverage: float = 0.5,
) -> tuple[float | None, dict]:
    """Median over a scope, with coverage guard for sparse/undefined metrics.

    Used for UPI (often ``None`` when Ulcer≈0). ``total_return`` is almost
    always defined, so coverage stays high and behavior is unchanged.
    """
    raw = [_metric(r.get(scope), metric) for r in rows]
    vals = _finite(raw)
    n_folds = len(rows)
    n_defined = len(vals)
    coverage = (n_defined / n_folds) if n_folds else 0.0
    diag = {
        "n_folds": n_folds,
        "n_defined": n_defined,
        "n_missing": n_folds - n_defined,
        "coverage": coverage,
        "min_n": int(min_n),
        "min_coverage": float(min_coverage),
    }
    # Only enforce coverage for metrics that are frequently undefined.
    if str(metric).lower() == "upi":
        if n_defined < int(min_n) or coverage < float(min_coverage):
            return None, diag
    return _median(vals), diag


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=float)))


def _fraction_positive(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(1 for v in vals if v > 0.0) / len(vals))


def _max_abs_share(vals: list[float]) -> float | None:
    if not vals:
        return None
    abs_vals = [abs(v) for v in vals]
    denom = float(sum(abs_vals))
    if denom <= 0.0:
        return None
    return float(max(abs_vals) / denom)


def _group_by_seed(rows: list[dict]) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = {}
    for r in rows:
        out.setdefault(r.get("seed"), []).append(r)
    return out


def _seeds_fraction_positive(rows: list[dict], metric: str, scope: str) -> float | None:
    groups = _group_by_seed(rows)
    if not groups:
        return None
    positives = 0
    n = 0
    for seed_rows in groups.values():
        vals = _scope_values(seed_rows, metric, scope)
        med = _median(vals)
        if med is None:
            continue
        n += 1
        if med > 0.0:
            positives += 1
    if n == 0:
        return None
    return float(positives / n)


def _median_retention(
    rows: list[dict],
    metric: str,
    *,
    min_pairs: int = 2,
    min_coverage: float = 0.5,
) -> tuple[float | None, dict]:
    """Median OOS/IS retention with coverage guard.

    UPI (and similar ratios) are often ``None`` when Ulcer is ~0 (monotone
    equity). Median over the *few* defined folds silently overstates or
    understates retention. Require enough defined pairs or return ``None``.
    """
    ratios: list[float] = []
    n_missing = 0
    for r in rows:
        is_v = _metric(r.get("is"), metric)
        oos_v = _metric(r.get("oos"), metric)
        if is_v is None or oos_v is None:
            n_missing += 1
            continue
        if is_v > 0.0:
            ratios.append(oos_v / is_v)
        elif oos_v > 0.0:
            # IS non-positive but OOS positive: treat as full retention for the gate.
            ratios.append(1.0)
        else:
            ratios.append(0.0)
    n_folds = len(rows)
    n_pairs = len(ratios)
    coverage = (n_pairs / n_folds) if n_folds else 0.0
    diag = {
        "n_folds": n_folds,
        "n_pairs": n_pairs,
        "n_missing": n_missing,
        "coverage": coverage,
        "min_pairs": int(min_pairs),
        "min_coverage": float(min_coverage),
    }
    if n_pairs < int(min_pairs) or coverage < float(min_coverage):
        return None, diag
    return _median(ratios), diag


def _eval_gate(name: str, spec: dict, rows: list[dict]) -> dict:
    metric = str(spec.get("metric", "total_return"))
    scope = str(spec.get("scope", "oos"))
    aggregate = str(spec.get("aggregate", "median"))
    op = str(spec.get("op", ">"))
    threshold = float(spec["threshold"])
    if op not in _OPS:
        raise ValueError(f"unsupported gate op {op!r} for {name}")

    if aggregate == "median":
        value, extra = _median_scoped(rows, metric, scope)
    elif aggregate == "fraction_positive":
        value = _fraction_positive(_scope_values(rows, metric, scope))
        extra = {}
    elif aggregate == "seeds_fraction_positive":
        value = _seeds_fraction_positive(rows, metric, scope)
        extra = {}
    elif aggregate == "median_retention":
        value, extra = _median_retention(rows, metric)
    elif aggregate == "max_abs_share":
        value = _max_abs_share(_scope_values(rows, metric, scope))
        extra = {}
    else:
        raise ValueError(f"unsupported aggregate {aggregate!r} for {name}")

    passed = False if value is None else bool(_OPS[op](value, threshold))
    out = {
        "name": name,
        "metric": metric,
        "scope": scope,
        "aggregate": aggregate,
        "op": op,
        "threshold": threshold,
        "value": value,
        "pass": passed,
    }
    out.update(extra)
    return out


def score_stage0_gates(fold_results: list[dict], gates: dict) -> dict:
    """Score locked Stage 0 gates against per-fold IS/OOS metric blocks.

    Parameters
    ----------
    fold_results:
        Rows like ``{"seed": 42, "fold": 0, "is": {...}, "oos": {...}}``.
        Metric blocks should include at least ``total_return`` and ``upi``.
    gates:
        Mapping of gate name → spec (``metric``, ``scope``, ``aggregate``,
        ``op``, ``threshold``), typically ``stage0_gates.yaml`` ``gates:``.
    """
    rows = [dict(r) for r in fold_results]
    gate_specs = dict(gates or {})
    results = [_eval_gate(name, spec, rows) for name, spec in gate_specs.items()]
    overall = all(g["pass"] for g in results) if results else False
    oos_returns = _scope_values(rows, "total_return", "oos")
    oos_upi = _scope_values(rows, "upi", "oos")
    return {
        "n_folds": len(rows),
        "n_seeds": len({r.get("seed") for r in rows}),
        "gates": {g["name"]: g for g in results},
        "gate_list": results,
        "overall_pass": overall,
        "summary": {
            "median_oos_return": _median(oos_returns),
            "median_oos_upi": _median(oos_upi),
            "folds_positive_fraction": _fraction_positive(oos_returns),
            "seeds_positive_fraction": _seeds_fraction_positive(rows, "total_return", "oos"),
            "max_fold_abs_share": _max_abs_share(oos_returns),
        },
    }


def format_gate_report(scored: dict) -> str:
    """Human-readable Stage 0 gate table."""
    lines = [
        "Stage 0 gate report",
        f"folds={scored.get('n_folds')}  seeds={scored.get('n_seeds')}  "
        f"overall={'PASS' if scored.get('overall_pass') else 'FAIL'}",
        "",
        f"{'gate':<32} {'value':>10} {'op':<3} {'thr':>8}  result",
        "-" * 64,
    ]
    for g in scored.get("gate_list") or []:
        val = g.get("value")
        val_s = "n/a" if val is None else f"{val:.4f}"
        thr_s = f"{float(g['threshold']):.4f}"
        flag = "PASS" if g.get("pass") else "FAIL"
        lines.append(
            f"{g['name']:<32} {val_s:>10} {g['op']:<3} {thr_s:>8}  {flag}"
        )
    return "\n".join(lines) + "\n"
