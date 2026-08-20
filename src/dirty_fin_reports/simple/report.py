"""Report orchestration: CSVs in, a validated metrics bundle out.

``run_reporter`` is the phase-1 CLI path: point it at a directory holding
``trades.csv`` (required) plus ``manager_ppo.csv`` / ``worker_sac.csv``
(optional), and it emits ``report.json`` and ``breakdown.txt`` with every
metric plausibility-checked — the daily-frequency Sharpe replaces the
per-bar "160 Sharpe".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Plausibility, ReportConfig
from .ledger import coerce_ledger, load, unique_keys, validate
from .equity import per_symbol_curves, portfolio_curve
from .metrics import metrics
from .plausibility import aggregate, check_many
from .verdict import health_axis, performance_axis, recommend
from .trades import by_episode, by_exit, by_side, by_symbol, hold_stats, leverage_stats, trade_stats
from .training import detect_algorithm, load_training_csv, series, training_health

_default_names = ("trades.csv", "manager_ppo.csv", "worker_sac.csv")


def _first_existing(cands) -> Optional[Path]:
    for c in cands:
        if c.exists():
            return c
    return None


def discover_sources(src: Path) -> dict:
    """Locate ledger + training CSVs in either the flat or trading-bot layout.

    Trading-bot runs keep ``testing/trades.csv`` and ``training/{manager_ppo,
    worker_sac}.csv``; the phase-1 fixture dir keeps them flat next to each
    other.  ``src`` may point at the run folder itself (both layouts work).
    """
    return {
        "trades": _first_existing((
            src / "trades.csv",
            src / "testing" / "trades.csv",
        )),
        "manager_ppo": _first_existing((
            src / "manager_ppo.csv",
            src / "training" / "manager_ppo.csv",
        )),
        "worker_sac": _first_existing((
            src / "worker_sac.csv",
            src / "training" / "worker_sac.csv",
        )),
    }


def _training_summary(df, algorithm: str, name: str, ma: int = 10) -> dict:
    h = training_health(df, algorithm=algorithm, ma=ma)
    h["source"] = name
    h["reward_delta_pct"] = h.get("reward_delta_pct")
    return h


def assemble(
    ledger_rows: list[dict],
    manager_df=None,
    worker_df=None,
    config: ReportConfig | None = None,
    ma: int = 10,
    plausibility: Plausibility | None = None,
) -> tuple[dict, dict, tuple, list]:
    """Compute the whole report bundle plus its curves in a single pass.

    This is the single source of truth for every derived value: the equity
    curves, the portfolio metrics (at ``cfg``'s cadence/rf), ``calmar``, the
    per-trade stats and the per-symbol curves are all computed exactly once
    here. Renderers (``figures``, ``breakdown``) consume the returned bundle
    instead of recomputing from raw inputs.

    Returns ``(report, eq, (symbols, per_symbol_curves), rows)``.
    """
    cfg = config or ReportConfig()
    rows = coerce_ledger(ledger_rows)
    warnings = validate(rows)
    if not rows:
        raise ValueError("ledger has no trades to report")

    eq = portfolio_curve(rows, start=cfg.initial_balance, n_steps=cfg.n_steps)
    per_sym = per_symbol_curves(rows, start=cfg.initial_balance, n_steps=cfg.n_steps)
    m = metrics(eq["net"], periods_per_year=cfg.periods_per_year,
                freq=cfg.reporting_freq, rf_annual=cfg.rf_annual)
    net_last = float(eq["net"][-1]) if eq["net"].size else cfg.initial_balance
    net_first = float(eq["net"][0]) if eq["net"].size else cfg.initial_balance
    total_return_eq = net_last / max(net_first, 1e-12) - 1.0
    m["total_return_equation"] = total_return_eq

    n_episodes = eq.get("n_episodes", 1)
    ts = trade_stats(rows, base=cfg.initial_balance, n_accounts=n_episodes)
    leverage = leverage_stats(rows)
    calmar = (
        m["cagr"] / m["max_drawdown"]
        if "cagr" in m and m["max_drawdown"] > 1e-12 else 0.0
    )
    m["calmar"] = calmar

    agents = {}
    plausibility_inputs: dict[str, float] = {
        "sharpe": m.get("sharpe"),
        "sortino": m.get("sortino"),
        "calmar": calmar,
        "cagr": m.get("cagr"),
        "max_drawdown": m.get("max_drawdown"),
        "ulcer_index": m.get("ulcer_index"),
        "upi": m.get("upi"),
        "win_rate": ts.get("win_rate"),
        "profit_factor": ts.get("profit_factor"),
        "expectancy_in_risks": ts.get("expectancy_in_risks"),
        "max_leverage": leverage.get("max"),
    }
    if manager_df is not None:
        h = _training_summary(manager_df, "ppo", "manager_ppo.csv", ma=ma)
        agents["manager_ppo"] = h
        plausibility_inputs["reward_delta_pct"] = h.get("reward_delta_pct")
    if worker_df is not None:
        h = _training_summary(worker_df, "sac", "worker_sac.csv", ma=ma)
        agents["worker_sac"] = h

    checks = check_many(plausibility_inputs, (plausibility or Plausibility()).as_dict())
    verdict = aggregate(checks)
    violations = [c.metric for c in checks if not c.ok]
    perf = performance_axis(m.get("total_return"))
    health = health_axis(eq["net"])
    recommendation = recommend(perf["status"], verdict["status"], health["status"],
                               violations)

    report = {
        "meta": {
            "validation": (
                "plausibility checks are heuristic sanity bounds on the computed "
                "metrics; they flag statistical outliers, not economic realism or "
                "model quality"
            ),
        },
        "config": {
            "timeframe": cfg.timeframe,
            "initial_balance": cfg.initial_balance,
            "reporting_freq": cfg.reporting_freq,
            "rf_annual": cfg.rf_annual,
            "periods_per_year": cfg.periods_per_year,
            "n_steps": cfg.n_steps,
            "start_date": cfg.start_date,
            "tick_tilt": cfg.tick_tilt,
            "tick_angle": cfg.x_tick_angle,
        },
        "ledger": {
            "n_trades": len(rows),
            "n_unique": len(unique_keys(rows)),
            "n_episodes": n_episodes,
            "symbols": sorted({t["symbol"] for t in rows}),
            "warnings": warnings,
        },
        "equity": {
            "length": eq["length"],
            "net_first": net_first,
            "net_last": net_last,
            "down_last_pct": float(_down_perc(net_first, net_last)),
        },
        "trades": {
            "stats": ts,
            "by_symbol": by_symbol(rows, base=cfg.initial_balance, n_accounts=n_episodes),
            "by_exit": by_exit(rows, base=cfg.initial_balance, n_accounts=n_episodes),
            "by_side": by_side(rows, base=cfg.initial_balance, n_accounts=n_episodes),
            "by_episode": by_episode(rows, base=cfg.initial_balance, n_accounts=n_episodes),
            "hold": hold_stats(rows, bar_minutes=_bar_minutes(cfg.timeframe)),
            "leverage": leverage,
        },
        "portfolio": m,
        "performance": perf,
        "health": health,
        "recommendation": recommendation,
        "agents": agents,
        "plausibility": verdict,
        "plausibility_checks": [
            {"metric": c.metric, "ok": c.ok, "severity": c.severity,
             "actual": c.actual, "reason": c.reason}
            for c in checks
        ],
    }
    return report, eq, per_sym, rows


def report_dict(
    ledger_rows: list[dict],
    manager_df=None,
    worker_df=None,
    config: ReportConfig | None = None,
    ma: int = 10,
    plausibility: Plausibility | None = None,
) -> dict:
    """Compute the whole report bundle from already-loaded inputs.

    Thin wrapper over :func:`assemble` that returns only the JSON-safe bundle.
    ``plausibility`` lets a caller tighten/relax the likelihood bounds; the
    ``Plausibility`` defaults flag history-style implausible numbers (a per-bar
    "160 Sharpe") as ``high`` severity instead of printing them as fact.
    """
    report, _, _, _ = assemble(ledger_rows, manager_df=manager_df, worker_df=worker_df,
                               config=config, ma=ma, plausibility=plausibility)
    return report


def _down_perc(first: float, last: float) -> float:
    if first <= 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def _bar_minutes(timeframe: str) -> int:
    from .metrics import timeframe_minutes

    return timeframe_minutes(timeframe)


def _assemble_sources(
    src_dir: str | Path,
    config: ReportConfig | None = None,
    n_steps: Optional[int] = None,
    plausibility: Plausibility | None = None,
) -> tuple[dict, dict, tuple, dict, list]:
    """Discover + load the CSVs under ``src_dir`` and assemble once.

    Returns ``(report, eq, per_sym, sources, rows)`` so the caller never has to
    re-load or re-compute any derived value.
    """
    src = Path(src_dir)
    cfg = config or ReportConfig()
    if n_steps is not None:
        cfg.n_steps = n_steps

    sources = discover_sources(src)
    if sources["trades"] is None:
        raise FileNotFoundError(f"no trades.csv found under {src} "
                                "(flat or testing/)")
    ledger_rows = load(sources["trades"])
    manager_df = worker_df = None
    if sources["manager_ppo"] is not None:
        manager_df = load_training_csv(sources["manager_ppo"])
    if sources["worker_sac"] is not None:
        worker_df = load_training_csv(sources["worker_sac"])

    report, eq, per_sym, rows = assemble(ledger_rows, manager_df=manager_df,
                                         worker_df=worker_df, config=cfg,
                                         plausibility=plausibility)
    report["sources"] = {k: (str(v) if v is not None else None)
                         for k, v in sources.items()}
    return report, eq, per_sym, sources, rows


def build_report(
    src_dir: str | Path,
    config: ReportConfig | None = None,
    n_steps: Optional[int] = None,
    plausibility: Plausibility | None = None,
) -> dict:
    """Auto-discover the CSV sources in ``src_dir`` and build the report."""
    report, _, _, _, _ = _assemble_sources(src_dir, config=config, n_steps=n_steps,
                                           plausibility=plausibility)
    return report


def format_breakdown(r: dict) -> str:
    """Render the report as a readable text breakdown (tabulate-based)."""
    from tabulate import tabulate

    lines: list[str] = ["TRADING REPORT", "=============", ""]
    cfg = r["config"]
    lines.append(f"timeframe={cfg['timeframe']}  ppy={cfg['periods_per_year']}  "
                 f"freq={cfg['reporting_freq']}  rf={cfg['rf_annual']}  start={cfg['initial_balance']}")
    for w in r["ledger"]["warnings"]:
        lines.append(f"WARN: {w}")
    lines.append("")

    t = r["trades"]["stats"]
    m = r["portfolio"]
    lines.append(f"trades: {t['num']}  net_pnl={t['net_pnl']:+.2f}  win_rate={t['win_rate']:.2f}%  "
                 f"pf={_fmt(t['profit_factor'])}  expectancy={t['expectancy']:+.4f}  "
                 f"expectancy_R={_fmt(t['expectancy_in_risks'])}")
    lines.append(f"portfolio: equity={r['equity']['net_last']:,.2f}  "
                 f"ret={m['total_return']:+.2%}  sharpe={_fmt(m['sharpe'])}  "
                 f"sortino={_fmt(m['sortino'])}  max_dd={m['max_drawdown']:.2%}  "
                 f"ulcer={m['ulcer_index']:.4f}  upi={_fmt(m['upi'])}  calmar={_fmt(m['calmar'])}  cagr={m['cagr']:+.2%}")
    lines.append("")

    for title, groups in (
        ("By symbol", r["trades"]["by_symbol"]),
        ("By exit type", r["trades"]["by_exit"]),
        ("By side", r["trades"]["by_side"]),
        ("By episode", r["trades"]["by_episode"]),
    ):
        rows = [["label", "num", "win%", "net pnl", "avg win", "avg loss", "PF"]]
        for lab, st in groups:
            rows.append([
                lab, st["num"], f"{st['win_rate']:.2f}", f"{st['net_pnl']:+.4f}",
                f"{st['avg_win']:.4f}", f"{st['avg_loss']:.4f}", _fmt(st["profit_factor"]),
            ])
        lines.append(title)
        lines.append(tabulate(rows, headers="firstrow", tablefmt="grid"))
        lines.append("")

    agents = r.get("agents") or {}
    if agents:
        lines.append("Agent training health")
        lines.append("-------------------")
        rows = [["agent", "algorithm", "rows", "nan_frac", "trailing_na",
                 "reward_delta_pct", "reward_trend"]]
        for name, h in agents.items():
            rows.append([
                name, h["algorithm"], h["rows"], f"{h['nan_fraction']:.3f}",
                h["trailing_na"],
                _fmt(h.get("reward_delta_pct")), h.get("reward_trend"),
            ])
        lines.append(tabulate(rows, headers="firstrow", tablefmt="grid"))
        lines.append("")

    p = r["plausibility"]
    perf = r["performance"]
    health = r["health"]
    rec = r["recommendation"]
    rpctxt = "n/a" if perf.get("return_pct") is None else f"{perf['return_pct']:+.1f}%"
    lines.append(f"Performance: {perf['status']} ({rpctxt})")
    lines.append(f"Health: {health['status']}"
                 + (f" — {'; '.join(health['issues'])}" if health["issues"] else ""))
    lines.append(f"Plausibility: {p['status']} ({p['counts']})")
    for reason in p["failed"]:
        lines.append(f"  FLAG: {reason}")
    lines.append(f"Recommendation: {rec['action']} — {rec['reason']}")
    return "\n".join(lines)


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_report(r: dict, out_path: str | Path) -> Path:
    """Serialize the report bundle to JSON (``None``/``numpy``-safe)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(r), fh, indent=2, sort_keys=True)
    return out


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def run_reporter(src_dir: str | Path, out_dir: str | Path | None = None,
                 config: ReportConfig | None = None, theme: str = "synthwave",
                 overlays: bool = False,
                 plausibility: Plausibility | None = None,
                 meta: dict | None = None) -> dict:
    """CLI entry: ``src_dir`` (trading-bot run layout) → JSON, PNGs, breakdown.

    Figures and text go where the bot keeps them: ``breakdown.txt`` +
    ``bot-performance-<verdict>.png`` + ``trade-anatomy-<verdict>.png`` land in
    the ``testing/`` folder (flat runs use ``out_dir``), the agent diagnostics
    in ``training/``, and ``report.json`` at the run root.

    ``meta`` is merged into ``report.json``'s top-level ``meta`` block (e.g.
    the synthetic generator's induced-edge profile), so downstream readers see
    the data's provenance next to the verdict.
    """
    src = Path(src_dir)
    out = Path(out_dir) if out_dir is not None else src
    out.mkdir(parents=True, exist_ok=True)
    r, eq, per_sym, sources, rows = _assemble_sources(src, config=config,
                                                      plausibility=plausibility)
    if meta:
        r.setdefault("meta", {}).update(meta)
    write_report(r, out / "report.json")
    r["artifacts"] = str(out)

    from .figures import figure1, figure2, training_figure

    cfg = config or ReportConfig()

    testing_dir = out
    if sources["trades"] is not None and Path(sources["trades"]).parent.name == "testing":
        testing_dir = Path(sources["trades"]).parent
    training_dir = out
    for cand in (sources["manager_ppo"], sources["worker_sac"]):
        if cand is not None and Path(cand).parent.name == "training":
            training_dir = Path(cand).parent
            break

    from .breakdown import breakdown, bot_cfg

    inferred_steps = max(
        (int(t.get("closed_at", 0) or 0) for t in rows), default=0
    ) if (cfg.n_steps is None or cfg.n_steps <= 0) else cfg.n_steps
    bd_cfg = bot_cfg(timeframe=cfg.timeframe, initial_balance=cfg.initial_balance,
                     n_steps=inferred_steps)
    breakdown(rows, eq["net"], testing_dir / "breakdown.txt", r["portfolio"],
              cfg=bd_cfg)

    per_sym_arg = per_sym if overlays else None
    status = r["plausibility"]["status"]
    bounds = (plausibility or Plausibility()).as_dict()
    figure1(eq["net"], eq["gross"], eq["steps"], rows,
            testing_dir / f"bot-performance-{status}.png",
            r["portfolio"], r["trades"]["stats"],
            per_symbol=per_sym_arg, theme=theme, verdict=status,
            periods_per_year=cfg.periods_per_year, freq=cfg.reporting_freq,
            bounds=bounds, start_date=cfg.start_dt,
            bar_minutes=_bar_minutes(cfg.timeframe),
            tick_angle=cfg.x_tick_angle)
    figure2(rows, testing_dir / f"trade-anatomy-{status}.png", theme=theme,
            verdict=status, bounds=bounds)

    ppo_path = sources["manager_ppo"]
    sac_path = sources["worker_sac"]
    if ppo_path is not None:
        training_figure(ppo_path, training_dir / "manager_diag.png",
                        title="PPO manager health", theme=theme)
        r["manager_diag_figure"] = str(training_dir / "manager_diag.png")
    if sac_path is not None:
        training_figure(sac_path, training_dir / "worker_diag.png",
                        title="SAC worker health", theme=theme)
        r["worker_diag_figure"] = str(training_dir / "worker_diag.png")

    r["breakdown"] = str(testing_dir / "breakdown.txt")
    r["figure1"] = str(testing_dir / f"bot-performance-{status}.png")
    r["figure2"] = str(testing_dir / f"trade-anatomy-{status}.png")
    r["out_dir"] = str(out)
    return r