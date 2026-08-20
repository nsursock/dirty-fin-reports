"""Bot-format breakdown + synthetic-run generator tests."""

from pathlib import Path

import numpy as np
import pytest

from dirty_fin_reports.simple.breakdown import BD_COLS, bot_cfg, breakdown, breakdown_trade_stats
from dirty_fin_reports.simple.synth import (
    PPO_COLUMNS,
    SAC_COLUMNS,
    generate_ppo,
    generate_run,
    generate_sac,
    generate_trades,
)
from dirty_fin_reports.simple.equity import portfolio_curve
from dirty_fin_reports.simple.ledger import coerce_ledger
from dirty_fin_reports.simple.metrics import metrics


def test_single_trade_stats_export():
    import importlib

    import dirty_fin_reports.simple as s
    import dirty_fin_reports.simple.trades as tr

    bd = importlib.import_module("dirty_fin_reports.simple.breakdown")
    assert s.trade_stats is tr.trade_stats
    assert not hasattr(bd, "trade_stats")
    assert s.breakdown_trade_stats is bd.breakdown_trade_stats


def test_synth_profile_shape():
    from dirty_fin_reports.simple.synth import synth_profile

    p = synth_profile()
    assert set(p["exit_type_mix"]) == {"take_profit", "stop_loss",
                                       "market_close", "liquidation"}
    assert set(p["gross_return_bps"]) == {"take_profit", "stop_loss", "market_close"}
    assert "note" in p


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    return generate_run(d, n_trades=360, n_episodes=8)


def test_generate_run_layout(synth_run):
    assert synth_run["trades"].exists()
    assert synth_run["manager_ppo"].exists()
    assert synth_run["worker_sac"].exists()
    assert synth_run["trades"].parent.name == "testing"
    assert synth_run["manager_ppo"].parent.name == "training"


def test_generate_trades_shape_and_schema():
    rows = generate_trades(n=320, n_episodes=8)
    assert 300 <= len(rows) <= 400
    eps = {int(t["episode"]) for t in rows}
    assert eps == {0, 1, 2, 3, 4, 5, 6, 7}
    for t in rows[:5]:
        assert t["closed_at"] > t["opened_at"]
        assert t["exit_type"] in ("market_close", "take_profit", "stop_loss", "liquidation")
        assert t["side"] in ("long", "short")
        assert t["notional"] == pytest.approx(t["collateral"] * t["leverage"], rel=1e-6)
        assert float(t["realized_pnl"]) >= -float(t["collateral"])
        assert "seed_offset" in t and "equity_before" in t


def test_liquidation_wipeout_and_pnl_floor():
    rows = generate_trades(n=400, n_episodes=8)
    liqs = [t for t in rows if t["exit_type"] == "liquidation"]
    assert liqs
    for t in liqs:
        assert float(t["realized_pnl"]) == pytest.approx(-float(t["collateral"]))
        assert float(t["fee"]) == 0.0 and float(t["funding"]) == 0.0
    for t in rows:
        assert float(t["realized_pnl"]) >= -float(t["collateral"])


def test_generated_ledger_is_plausible():
    rows = coerce_ledger(generate_trades(n=360, n_episodes=8))
    pnls = np.array([r["realized_pnl"] for r in rows])
    win_rate = 100.0 * (pnls > 0).mean()
    assert 48.0 < win_rate < 62.0
    net = float(pnls.sum())
    assert net > 0
    eq = portfolio_curve(rows, start=1000.0)
    ret = eq["net"][-1] / eq["net"][0] - 1.0
    assert 0.0 < ret < 0.5
    from dirty_fin_reports.simple.metrics import metrics

    m = metrics(eq["net"], periods_per_year=72576, freq="daily", rf_annual=0.045)
    assert m["sharpe"] is not None and 1.0 < m["sharpe"] < 12.0
    assert m["max_drawdown"] < 0.15


def test_generate_ppo_schema_and_rows(synth_run):
    import pandas as pd

    df = pd.read_csv(synth_run["manager_ppo"])
    assert sorted(df.columns) == sorted(PPO_COLUMNS)
    assert len(df) == 200
    assert df["time/total_timesteps"].is_monotonic_increasing
    rew = df["rollout/ep_rew_mean"]
    assert rew.iloc[-1] > rew.iloc[0]


def test_generate_sac_schema_and_rows(synth_run):
    import pandas as pd

    df = pd.read_csv(synth_run["worker_sac"])
    assert sorted(df.columns) == sorted(SAC_COLUMNS)
    assert len(df) == 200
    assert df["rollout/ep_rew_mean"].iloc[-1] > df["rollout/ep_rew_mean"].iloc[0]


def test_breakdown_header_and_tables(tmp_path, synth_run):
    from dirty_fin_reports.simple.ledger import load

    rows = coerce_ledger(load(synth_run["trades"]))
    eq = portfolio_curve(rows, start=1000.0)
    pm = metrics(eq["net"], periods_per_year=72576, freq="daily", rf_annual=0.045)
    out = tmp_path / "breakdown.txt"
    text = breakdown(rows, eq["net"], out, pm, cfg=bot_cfg())
    assert text.startswith("BREAKDOWN\n=========")
    for section in ("By symbol", "By episode", "By position direction", "By exit",
                    "By outcome", "By leverage", "By hold duration", "By return",
                    "By collateral", "By notional", "By equity heat", "By bite size",
                    "By liquidation distance", "By fee drag", "By margin type",
                    "By trade vintage", "Baselines (after fees/funding/slip)"):
        assert section in text, section
    assert "episode 0 (seed+1)" in text
    assert "| portfolio" in text
    assert out.read_text() == text


def test_breakdown_column_schema_matches_bot():
    assert BD_COLS == ["label", "num trades", "win rate %", "avg win", "avg loss",
                       "net profit", "sharpe", "max dd", "risk reward", "sortino",
                       "calmar", "profit factor", "ulcer index", "upi"]


def test_breakdown_writes_bot_like_lines(tmp_path, synth_run):
    from dirty_fin_reports.simple.ledger import load

    rows = coerce_ledger(load(synth_run["trades"]))
    eq = portfolio_curve(rows, start=1000.0)
    pm = metrics(eq["net"], periods_per_year=72576, freq="daily", rf_annual=0.045)
    text = breakdown(rows, eq["net"], tmp_path / "bd.txt", pm, cfg=bot_cfg())
    body = text.splitlines()
    assert body[0] == "BREAKDOWN"
    assert any(l.startswith("seed: 42") for l in body)
    assert any(l.startswith("env: 8 symbols x 1 = 8 test envs") for l in body)
    assert any(l.startswith("portfolio: ") for l in body)
    assert any(l.strip().startswith("flat: 1000.00 -> 1000.00") for l in body)
    assert "NOTES" in body
    assert any("empty cell means the metric is undefined" in l for l in body)
    assert any("ulcer index / upi" in l for l in body)
    # portfolio row is the last row of the symbol table and carries risk ratios
    sym_table = text.split("By symbol", 1)[1].split("By episode", 1)[0]
    assert "| portfolio" in sym_table
    assert "ulcer index" in sym_table and "upi" in sym_table


def test_trade_stats_basic():
    trades = [{"realized_pnl": 10.0, "episode": 0, "symbol": "BTC"},
              {"realized_pnl": 10.0, "episode": 0, "symbol": "BTC"},
              {"realized_pnl": -5.0, "episode": 0, "symbol": "BTC"}]
    st = breakdown_trade_stats(trades, base=1000.0)
    assert st["num"] == 3
    assert st["win_rate"] == pytest.approx(66.6666666667)
    assert st["net"] == pytest.approx(15.0)
    assert st["pf"] == pytest.approx(4.0)