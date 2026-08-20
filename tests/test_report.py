"""End-to-end report, breakdown text, JSON round-trip and reporter tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from dirty_fin_reports.simple.config import Plausibility, ReportConfig
from dirty_fin_reports.simple.report import (
    build_report,
    format_breakdown,
    report_dict,
    run_reporter,
    write_report,
)
from dirty_fin_reports.simple.ledger import coerce_ledger, load

RUN = Path(__file__).parent / "fixtures" / "runs" / "phase1"


def test_build_report_aggregates_fixture():
    r = build_report(RUN)
    assert r["ledger"]["n_trades"] == 8
    assert r["ledger"]["n_unique"] == 8
    assert r["ledger"]["n_episodes"] == 2
    assert r["ledger"]["symbols"] == ["BTC", "DOGE", "ETH", "SOL"]
    assert r["equity"]["net_last"] == 1050.0
    assert r["trades"]["stats"]["win_rate"] == pytest.approx(62.5)
    assert r["trades"]["stats"]["profit_factor"] == pytest.approx(2.0526315789, rel=1e-9)
    assert "manager_ppo" in r["agents"]
    assert "worker_sac" in r["agents"]
    assert r["config"]["periods_per_year"] == 72576
    assert r["config"]["reporting_freq"] == "daily"


def test_report_has_decoupled_verdict_axes():
    r = build_report(RUN)
    assert r["performance"]["status"] == "profitable"
    assert r["health"]["status"] == "healthy"
    # The fixture is profitable but statistically implausible: a high-alpha
    # candidate routed for review, not discarded.
    assert r["recommendation"]["action"] == "review"
    assert "high-alpha" in r["recommendation"]["reason"]


def test_report_has_validation_disclosure():
    r = build_report(RUN)
    assert "validation" in r["meta"]
    assert "heuristic sanity" in r["meta"]["validation"]


def test_run_reporter_merges_meta(tmp_path):
    out = tmp_path / "out"
    r = run_reporter(RUN, out_dir=out, meta={"data_origin": "synthetic"})
    data = json.loads((out / "report.json").read_text())
    assert data["meta"]["data_origin"] == "synthetic"
    assert "validation" in data["meta"]


def test_degenerate_fixture_is_flagged_implausible_not_trusted():
    r = build_report(RUN)
    p = r["plausibility"]
    assert p["status"] == "implausible"
    flagged = "\n".join(p["failed"])
    assert "calmar" in flagged
    assert "cagr" in flagged
    # Sharpe/Sortino/UPI are undefined (too few bars for a daily window), never
    # fabricated from per-bar returns into a fake "160 Sharpe" headline.
    assert r["portfolio"]["sharpe"] is None
    assert r["portfolio"]["upi"] is None


def test_long_realistic_ledger_passes_with_tuned_bounds():
    rows = _long_ledger()
    cfg = ReportConfig(timeframe="5m", initial_balance=1000.0, reporting_freq="daily")
    bounds = Plausibility(
        sharpe=(None, 12.0), sortino=(None, 30.0), upi=(None, 300.0),
        calmar=(None, 120.0),
    )
    r = report_dict(rows, config=cfg, plausibility=bounds)
    assert r["plausibility"]["status"] == "plausible"
    assert r["portfolio"]["sharpe"] is not None
    assert 0.0 < r["portfolio"]["sharpe"] < 12.0
    assert r["portfolio"]["max_drawdown"] <= 1.0


def test_report_headline_is_always_the_daily_cadence():
    rows = _long_ledger()
    from dirty_fin_reports.simple.metrics import metrics
    from dirty_fin_reports.simple.equity import portfolio_curve

    cfg = ReportConfig(timeframe="5m", initial_balance=1000.0, reporting_freq="daily")
    r = report_dict(rows, config=cfg,
                    plausibility=Plausibility(sharpe=(None, 12.0), sortino=(None, 30.0),
                                              upi=(None, 300.0), calmar=(None, 120.0)))
    eq = portfolio_curve(rows, start=1000.0)
    daily = metrics(eq["net"], periods_per_year=72576, freq="daily", rf_annual=0.045)
    assert r["portfolio"]["sharpe"] == daily["sharpe"]
    assert r["config"]["reporting_freq"] == "daily"


def test_report_dict_plus_agents_plausibility_inputs():
    from dirty_fin_reports.simple.training import load_training_csv

    rows = coerce_ledger(load(RUN / "trades.csv"))
    r = report_dict(rows, manager_df=load_training_csv(RUN / "manager_ppo.csv"),
                    worker_df=load_training_csv(RUN / "worker_sac.csv"))
    assert "manager_ppo" in r["agents"]
    assert r["agents"]["manager_ppo"]["reward_delta_pct"] == pytest.approx(800.0)
    # Both agents feed the plausibility surface and the worker's own reward
    # delta reaches it too.
    assert r["agents"]["worker_sac"]["reward_delta_pct"] == pytest.approx(180.0)
    metrics_checked = {c["metric"] for c in r["plausibility_checks"]}
    assert "sharpe" in metrics_checked
    assert "reward_delta_pct" in metrics_checked


def test_format_breakdown_renders_sections():
    r = build_report(RUN)
    text = format_breakdown(r)
    assert "TRADING REPORT" in text
    assert "By symbol" in text
    assert "By exit type" in text
    assert "By episode" in text
    assert "Agent training health" in text
    assert "Plausibility:" in text


def test_write_report_json_roundtrip(tmp_path):
    r = build_report(RUN)
    out = write_report(r, tmp_path / "nested" / "report.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["portfolio"]["sharpe"] == r["portfolio"]["sharpe"]
    assert data["trades"]["stats"]["num"] == 8
    assert isinstance(data["equity"]["net_last"], (int, float))


def test_run_reporter_writes_artifacts(tmp_path):
    out = tmp_path / "out"
    r = run_reporter(RUN, out_dir=out)
    assert (out / "report.json").exists()
    assert (out / "breakdown.txt").exists()
    assert "out_dir" in r


def test_build_report_missing_trades_raises(tmp_path):
    with pytest.raises(Exception):
        build_report(tmp_path)


def _long_ledger():
    rng = np.random.default_rng(21)
    days = 120
    bpd = 288
    pnl_daily = 1000.0 * (0.0015 + rng.normal(0, 0.0025, days))
    rows = []
    for d in range(days):
        rows.append({
            "trade_id": str(d + 1), "episode": 0, "symbol": "BTC",
            "side": "long" if pnl_daily[d] >= 0 else "short",
            "opened_at": max(d * bpd - 50, 1), "closed_at": (d + 1) * bpd,
            "entry_price": 100.0, "exit_price": 100.0, "notional": 1000.0,
            "leverage": 5.0, "collateral": 200.0, "entry_conviction": 0.8,
            "fee": 0.6, "funding": 0.0,
            "realized_pnl": float(pnl_daily[d]),
            "exit_type": "take_profit" if pnl_daily[d] >= 0 else "stop_loss",
        })
    return rows