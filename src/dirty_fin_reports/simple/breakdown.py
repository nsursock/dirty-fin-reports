"""Bot-format text breakdown: the exact metrics and tables the trading-bot
engine renders (16 subgroup tables over a 14-column schema, config header and
a baselines footer).

This is a reimplementation, not a copy: bucket rules, per-trade stats and the
portfolio risk row follow the same definitions as ``scripts/report.py`` in the
trading-bot, but every number is computed here from the ledger rows and the
bar-indexed net curve produced by this package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tabulate import tabulate

from .metrics import timeframe_minutes
from .env_params import TradingEnvParams

BD_COLS = ["label", "num trades", "win rate %", "avg win", "avg loss",
           "net profit", "sharpe", "max dd", "risk reward", "sortino", "calmar",
           "profit factor", "ulcer index", "upi"]


def bot_cfg(
    timeframe: str = "5m",
    initial_balance: float = 1000.0,
    n_steps: int = 5760,
    returns_basis: str = "collateral",
    env: TradingEnvParams | None = None,
    **overrides,
) -> dict:
    """Build a bot-compatible config dict for the breakdown header/tables.

    The ``env`` block comes from :class:`TradingEnvParams` (the single source
    shared with the synthetic generator); ``overrides`` may still replace it.
    Any top-level key in ``overrides`` is merged in (e.g. ``env``, ``data``).
    """
    env_params = (env or TradingEnvParams()).as_dict()
    env_params["initial_balance"] = initial_balance
    cfg = {
        "seed": 42,
        "data": {"n_symbols": 8, "n_steps": n_steps, "dt_days": None,
                 "timeframes": {"low": timeframe}},
        "env": env_params,
        "reward": {"mode": "normal", "drawdown_penalty": 1.0, "reward_clip": 10.0},
        "hrl": {"goal_every": 6, "goal_dim": 3},
        "manager": {"n_steps": 256},
        "worker": {"net_arch": [256, 256], "learning_rate": 0.001},
        "returns": {"basis": returns_basis, "freq": "daily", "rf_annual": 0.045},
        "eval": {"episodes": 8, "max_positions_per_symbol": 1, "deterministic": True},
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _config_lines(cfg) -> list[str]:
    d = cfg.get("data", {})
    e = cfg.get("env", {})
    r = cfg.get("reward", {})
    h = cfg.get("hrl", {})
    w = cfg.get("worker", {})
    ev = cfg.get("eval", {})
    n_sym = int(d.get("n_symbols", 0) or 0)
    per_sym = max(1, int(ev.get("max_positions_per_symbol", 1)))
    return [
        f"seed: {cfg.get('seed')}",
        f"symbols: {d.get('n_symbols')}  steps: {d.get('n_steps')}  dt_days: {d.get('dt_days')}",
        f"env: {n_sym} symbols x {per_sym} = {n_sym * per_sym} test envs  "
        f"lev {e.get('lev_min')}–{e.get('lev_max')}x  "
        f"risk {float(e.get('risk_min', 0)) * 100:.0f}–{float(e.get('risk_max', 0)) * 100:.0f}%  "
        f"eq={e.get('initial_balance')}  fees o/c={e.get('open_fee_rate')}/{e.get('close_fee_rate')}  "
        f"liq={e.get('liquidation_fee_rate')}  hold/day={e.get('holding_fee_daily')}",
        f"reward: {r.get('mode')}  dd_pen={r.get('drawdown_penalty')}  "
        f"clip={r.get('reward_clip')}  trade_knob={e.get('trade_knob')}",
        f"hrl: goal_every={h.get('goal_every')}  goal_dim={h.get('goal_dim')}  "
        f"manager n_steps={cfg.get('manager', {}).get('n_steps')}  "
        f"worker net={w.get('net_arch')} lr={w.get('learning_rate')}",
    ]


def breakdown_trade_stats(trades, base: float = 1000.0) -> dict:
    """Descriptive per-trade stats (mirrors the trading-bot engine).

    ``max_dd`` is the peak-to-trough drawdown of the trade-PnL cumsum scaled
    by the total account capital in play (``n_accounts * base``); Sharpe /
    Sortino (mean/std, mean/downside-std) and Calmar are per-trade, NOT
    annualized. ``pf`` is ``None`` when every trade wins.
    """
    pnls = np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in trades], dtype=float)
    n = int(pnls.size)
    if n == 0:
        return dict(num=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0, net=0.0,
                    sharpe=0.0, max_dd=0.0, rr=0.0, sortino=0.0, calmar=0.0, pf=0.0)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    net = float(pnls.sum())
    win_rate = 100.0 * float((pnls > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    accts = {(int(t.get("episode", 0) or 0), str(t.get("symbol", ""))) for t in trades}
    n_accts = max(len(accts), 1)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd_min = float((cum - peak).min())
    max_dd = 100.0 * abs(dd_min) / max(abs(base) * n_accts, 1.0) if abs(dd_min) > 1e-9 else 0.0
    rr = avg_win / abs(avg_loss) if losses.size and abs(avg_loss) > 1e-12 else 0.0
    gw = float(wins.sum()) if wins.size else 0.0
    gl = float(abs(losses.sum())) if losses.size else 0.0
    pf = (gw / gl) if gl > 1e-12 else (None if gw > 0 else 0.0)
    std = float(pnls.std())
    sharpe = float(pnls.mean()) / std if n > 1 and std > 1e-12 else 0.0
    dstd = float(losses.std()) if losses.size > 1 else 0.0
    sortino = float(pnls.mean()) / dstd if dstd > 1e-12 else 0.0
    total_ret = net / max(abs(base) * n_accts, 1.0)
    calmar = total_ret / (max_dd / 100.0) if max_dd > 1e-9 else 0.0
    return dict(num=n, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
                net=net, sharpe=sharpe, max_dd=max_dd, rr=rr, sortino=sortino,
                calmar=calmar, pf=pf)


def _bd_row(label, st) -> list:
    pf_fmt = "" if st["pf"] is None else f"{st['pf']:.4f}"
    sharpe = st.get("sharpe")
    sharpe_fmt = "" if sharpe is None else f"{sharpe:.4f}"
    sortino = st.get("sortino")
    sortino_fmt = "" if sortino is None else f"{sortino:.4f}"
    calmar = st.get("calmar")
    calmar_fmt = "" if calmar is None else f"{calmar:.4f}"
    ulcer = st.get("ulcer_index")
    ulcer_fmt = "" if ulcer is None else f"{ulcer:.4f}"
    upi = st.get("upi")
    upi_fmt = "" if upi is None else f"{upi:.4f}"
    return [label, st["num"], round(st["win_rate"], 2), round(st["avg_win"], 4),
            round(st["avg_loss"], 4), round(st["net"], 4), sharpe_fmt,
            round(st["max_dd"], 2), round(st["rr"], 4), sortino_fmt,
            calmar_fmt, pf_fmt, ulcer_fmt, upi_fmt]


def _bd_table(title, groups, portfolio) -> list[str]:
    rows = [BD_COLS]
    for label, st in groups:
        rows.append(_bd_row(label, st))
    rows.append(_bd_row("portfolio", portfolio))
    return [title, "", tabulate(rows, headers="firstrow", tablefmt="grid",
                                floatfmt=".4f",
                                colalign=("left",) + ("right",) * (len(BD_COLS) - 1)), ""]


def breakdown(ledger: list[dict], net, out_path: str | Path, pm: dict,
              cfg=None, rets=None) -> str:
    """Render the text breakdown (bot format) and write it to ``out_path``.

    ``ledger`` is the coerced trade rows; ``net`` the bar-indexed portfolio
    equity curve (mean over the per-episode books). ``pm`` is the canonical
    portfolio-metrics dict computed by ``report.assemble`` (Sharpe / Sortino /
    Calmar / ulcer / UPI at the reporting cadence) — this function consumes it
    instead of recomputing, so the breakdown can never drift from
    ``report.json``. Every subgroup row reports descriptive per-trade stats.
    """
    cfg = cfg or bot_cfg()
    net = np.asarray(net, dtype=float)
    base = float(net[0]) if net.size else 1000.0
    m = pm
    port = breakdown_trade_stats(ledger, base=base)
    if m:
        port["sharpe"] = m.get("sharpe", 0.0)
        port["sortino"] = m.get("sortino", 0.0)
        port["max_dd"] = 100.0 * float(m.get("max_drawdown", 0.0) or 0.0)
        port["calmar"] = m.get("calmar", 0.0)
        port["ulcer_index"] = m.get("ulcer_index")
        port["upi"] = m.get("upi")

    def _bucket(trades, keyfn, order):
        groups = {}
        for t in trades:
            groups.setdefault(keyfn(t), []).append(t)
        return [(lab, breakdown_trade_stats(groups[lab], base=base)) for lab in order if lab in groups]

    n = max(len(ledger), 1)
    by_symbol = sorted({t["symbol"] for t in ledger})
    sym_groups = [(s, breakdown_trade_stats([t for t in ledger if t["symbol"] == s], base=base))
                  for s in by_symbol]

    ep_order = sorted({int(t.get("episode", 0) or 0) for t in ledger})
    by_episode = []
    for e in ep_order:
        ep_trades = [t for t in ledger if int(t.get("episode", 0) or 0) == e]
        off = next((t.get("seed_offset") for t in ep_trades if t.get("seed_offset") is not None), "?")
        by_episode.append((f"episode {e} (seed+{off})", breakdown_trade_stats(ep_trades, base=base)))
    if len(ep_order) > 1:
        by_episode.append(("all", breakdown_trade_stats(ledger, base=base)))

    by_side = _bucket(ledger, lambda t: "bull (long)" if t["side"] == "long" else "bear (short)",
                      ["bull (long)", "bear (short)"])
    by_exit = _bucket(ledger, lambda t: t["exit_type"],
                      ["take_profit", "stop_loss", "market_close", "liquidation"])
    by_outcome = _bucket(ledger, lambda t: ("win" if (t.get("realized_pnl", 0) or 0) > 0 else
                                            "loss" if (t.get("realized_pnl", 0) or 0) < 0 else "breakeven"),
                         ["win", "loss", "breakeven"])

    def _lev(t):
        v = float(t.get("leverage", 0) or 0)
        if v < 5: return "cruiser (1-5x)"
        if v < 10: return "charger (5-10x)"
        if v < 20: return "turbo (10-20x)"
        if v < 50: return "warp (20-50x)"
        if v < 100: return "hyper (50-100x)"
        return "singularity (100x+)"
    by_lev = _bucket(ledger, _lev, ["cruiser (1-5x)", "charger (5-10x)", "turbo (10-20x)",
                                    "warp (20-50x)", "hyper (50-100x)", "singularity (100x+)"])

    low_tf = ((cfg.get("data") or {}).get("timeframes") or {}).get("low", "5m")
    bar_secs = float(timeframe_minutes(low_tf)) * 60.0
    base_eq = base

    def _hold(t):
        d = int(t.get("closed_at", 0) or 0) - int(t.get("opened_at", 0) or 0)
        secs = d * bar_secs
        if secs <= 0: return "unmatched (0s)"
        if secs < 30: return "flash (<30s)"
        if secs < 120: return "scalp (30s-2m)"
        if secs < 900: return "sprint (2-15m)"
        if secs < 3600: return "sit (15-60m)"
        return "camp (1h+)"
    by_hold = _bucket(ledger, _hold, ["unmatched (0s)", "flash (<30s)", "scalp (30s-2m)",
                                      "sprint (2-15m)", "sit (15-60m)", "camp (1h+)"])

    ret_basis = (dict(cfg.get("returns", {})) if cfg is not None else {}).get("basis", "account")

    def _roe(t):
        if ret_basis == "collateral":
            rbase = float(t.get("collateral", 0) or 0)
        else:
            rbase = float(t.get("equity_before", 0) or 0)
        r = 10_000.0 * (float(t.get("realized_pnl", 0) or 0)) / max(abs(rbase), 1e-9)
        if r < 0: return "dip (<0 bps)"
        if r < 10: return "scratch (<10 bps)"
        if r < 100: return "single-R (10-100 bps)"
        return "multi-R (>=100 bps)"
    by_roe = _bucket(ledger, _roe, ["dip (<0 bps)", "scratch (<10 bps)",
                                    "single-R (10-100 bps)", "multi-R (>=100 bps)"])

    def _collateral(t):
        v = float(t.get("collateral", 0) or 0)
        if v < 50: return "pocket (<$50)"
        if v < 200: return "small ($50-$200)"
        if v < 1000: return "standard ($200-$1k)"
        return "loaded (>=$1k)"
    by_collateral = _bucket(ledger, _collateral, ["pocket (<$50)", "small ($50-$200)",
                                                  "standard ($200-$1k)", "loaded (>=$1k)"])

    def _notional(t):
        v = float(t.get("notional", 0) or 0)
        if v < 1000: return "toy (<$1k)"
        if v < 10_000: return "standard ($1k-$10k)"
        if v < 50_000: return "size ($10k-$50k)"
        return "whale (>=$50k)"
    by_notional = _bucket(ledger, _notional, ["toy (<$1k)", "standard ($1k-$10k)",
                                              "size ($10k-$50k)", "whale (>=$50k)"])

    def _heat(t):
        eq = float(t.get("equity_before", base_eq) or base_eq)
        r = eq / max(base_eq, 1e-9)
        if r < 0.80: return "underwater (<80% eq)"
        if r < 0.95: return "bruised (80-95% eq)"
        if r < 1.05: return "par (95-105% eq)"
        if r < 1.30: return "green (105-130% eq)"
        return "moon (>=130% eq)"
    by_heat = _bucket(ledger, _heat, ["underwater (<80% eq)", "bruised (80-95% eq)",
                                      "par (95-105% eq)", "green (105-130% eq)",
                                      "moon (>=130% eq)"])

    def _bite(t):
        coll = max(abs(float(t.get("collateral", 0) or 0)), 1e-9)
        pct = 100.0 * abs(float(t.get("realized_pnl", 0) or 0)) / coll
        if pct < 0.5: return "dust (<0.5% margin)"
        if pct < 2: return "scratch (0.5-2%)"
        if pct < 8: return "nibble (2-8%)"
        if pct < 20: return "bite (8-20%)"
        return "feast (>=20%)"
    by_bite = _bucket(ledger, _bite, ["dust (<0.5% margin)", "scratch (0.5-2%)",
                                      "nibble (2-8%)", "bite (8-20%)", "feast (>=20%)"])

    def _liq(t):
        lev = max(abs(float(t.get("leverage", 0) or 0)), 1e-9)
        e = (cfg.get("env") or {}) if cfg is not None else {}
        defaults = TradingEnvParams()
        tbase = float(e.get("liq_threshold_base", defaults.liq_threshold_base))
        tfloor = float(e.get("liq_threshold_floor", defaults.liq_threshold_floor))
        ref = float(e.get("liq_threshold_ref_lev", defaults.liq_threshold_ref_lev))
        hi = float(e.get("liq_threshold_hi_lev", defaults.liq_threshold_hi_lev))
        lo = float(e.get("liq_threshold_lo_lev", defaults.liq_threshold_lo_lev))
        slope_lo = (tbase - 1.0) / max(ref - lo, 1e-6)
        slope_hi = (tfloor - tbase) / max(hi - ref, 1e-6)
        thr = (1.0 + slope_lo * (lev - lo)) if lev <= ref else (tbase + slope_hi * (lev - ref))
        thr = min(1.0, max(tfloor, thr))
        dist = 100.0 * thr / lev
        if dist < 2: return "knife-edge (<2%)"
        if dist < 5: return "tight (2-5%)"
        if dist < 15: return "cushion (5-15%)"
        return "fortress (>=15%)"
    by_liq = _bucket(ledger, _liq, ["knife-edge (<2%)", "tight (2-5%)",
                                    "cushion (5-15%)", "fortress (>=15%)"])

    def _fee_drag(t):
        notional = max(abs(float(t.get("notional", 0) or 0)), 1e-9)
        bps = 10_000.0 * float(t.get("fee", 0) or 0) / notional
        if bps <= 0: return "free (maker)"
        if bps < 5: return "light (<5 bps)"
        return "heavy (>=5 bps)"
    by_fee = _bucket(ledger, _fee_drag, ["free (maker)", "light (<5 bps)", "heavy (>=5 bps)"])

    margin_mode = ((cfg.get("env") or {}).get("margin_mode", "isolated")) or "isolated"
    by_margin = _bucket(ledger, lambda t: ("pool (cross)" if margin_mode == "cross"
                                           else "loner (isolated)"),
                        ["loner (isolated)", "pool (cross)"])

    def _vintage(i):
        frac = i / n
        return ("opening act (first 20%)" if frac < 0.2
                else "encore (last 20%)" if frac >= 0.8 else "mid-set (20-80%)")
    _vgroups = {}
    for _i, _t in enumerate(ledger):
        _vgroups.setdefault(_vintage(_i), []).append(_t)
    by_vintage = [(lab, breakdown_trade_stats(_vgroups[lab], base=base))
                  for lab in ("opening act (first 20%)", "mid-set (20-80%)", "encore (last 20%)")
                  if lab in _vgroups]

    lines: list[str] = ["BREAKDOWN", "=========", ""]
    if cfg is not None:
        lines += _config_lines(cfg)
        lines.append("")
    lines.append("Portfolio risk (Sharpe/Sortino/Calmar) is computed from the bar-indexed "
                 "net curve; Ulcer Index / UPI follow the reporting cadence. "
                 "Subgroup rows report descriptive trade stats only.")
    ui_str = f"{port['ulcer_index']:.4f}" if port.get("ulcer_index") is not None else "n/a"
    upi_str = f"{port['upi']:.3f}" if port.get("upi") is not None else "n/a"
    sharpe_str = f"{port['sharpe']:.3f}" if port.get("sharpe") is not None else "n/a"
    lines.append(f"portfolio: {port['num']} trades  final_equity={float(net[-1]):.2f}  "
                 f"ret={m.get('total_return', 0):+.2%}  sharpe={sharpe_str}  "
                 f"max_dd={port['max_dd']:.2f}%  ulcer_index={ui_str}  upi={upi_str}")
    lines.append("")
    for title, groups in [
        ("By symbol", sym_groups),
        ("By episode", by_episode),
        ("By position direction", by_side),
        ("By exit", by_exit),
        ("By outcome", by_outcome),
        ("By leverage", by_lev),
        ("By hold duration", by_hold),
        ("By return" if ret_basis == "collateral" else "By RoE", by_roe),
        ("By collateral", by_collateral),
        ("By notional", by_notional),
        ("By equity heat", by_heat),
        ("By bite size", by_bite),
        ("By liquidation distance", by_liq),
        ("By fee drag", by_fee),
        ("By margin type", by_margin),
        ("By trade vintage", by_vintage),
    ]:
        lines += _bd_table(title, groups, port)

    lines.append("Baselines (after fees/funding/slip)")
    lines.append(f"  policy: {float(net[0]):.2f} -> {float(net[-1]):.2f}  ({m.get('total_return', 0):+.3f})")
    lines.append("  flat: 1000.00 -> 1000.00  (+0.000)")
    text = "\n".join(lines)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(text)
    return text