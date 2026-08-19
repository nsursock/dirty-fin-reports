"""Trade-level descriptive statistics accuracy tests (hand-computed)."""

import numpy as np
import pytest

from dirty_fin_reports.simple.trades import (
    by_episode,
    by_exit,
    by_side,
    by_symbol,
    hold_stats,
    leverage_stats,
    trade_stats,
)


def _rows():
    from dirty_fin_reports.simple.ledger import coerce_ledger, load

    from conftest import TRADES_TWO

    return coerce_ledger(load(TRADES_TWO))


def test_trade_stats_episode_0_hand_computed():
    rows = [t for t in _rows() if int(t["episode"]) == 0]
    st = trade_stats(rows, base=1000.0, n_accounts=2)
    assert st["num"] == 5
    assert st["net_pnl"] == pytest.approx(83.0)
    assert st["win_rate"] == pytest.approx(60.0)
    assert st["avg_win"] == pytest.approx(51.0)
    assert st["avg_loss"] == pytest.approx(-35.0)
    assert st["profit_factor"] == pytest.approx(153.0 / 70.0)
    assert st["expectancy"] == pytest.approx(16.6)
    assert st["expectancy_in_risks"] == pytest.approx(16.6 / 35.0, rel=1e-9)
    assert st["risk_reward"] == pytest.approx(51.0 / 35.0, rel=1e-9)
    assert st["max_dd_pnl"] == pytest.approx(70.0)
    assert st["max_dd_pct"] == pytest.approx(100.0 * 70.0 / 2000.0)
    assert st["sharpe_per_trade"] == pytest.approx(0.33453602856642134, rel=1e-9)
    assert st["sortino_per_trade"] == pytest.approx(3.32, rel=1e-9)


def test_trade_stats_episode_1_hand_computed():
    rows = [t for t in _rows() if int(t["episode"]) == 1]
    st = trade_stats(rows, base=1000.0)
    assert st["num"] == 3
    assert st["net_pnl"] == pytest.approx(17.0)
    assert st["win_rate"] == pytest.approx(100 * 2 / 3, rel=1e-9)
    assert st["profit_factor"] == pytest.approx(42.0 / 25.0)
    assert st["expectancy_in_risks"] == pytest.approx(5.666666666666667 / 25.0, rel=1e-9)
    assert st["sortino_per_trade"] is None  # one loss => undefined


def test_trade_stats_empty():
    st = trade_stats([], base=1000.0)
    assert st["num"] == 0
    assert st["profit_factor"] is None
    assert st["sharpe_per_trade"] is None


def test_trade_stats_all_winners_profit_factor_undefined():
    rows = [
        {"realized_pnl": 5.0},
        {"realized_pnl": 7.0},
    ]
    st = trade_stats(rows, base=1000.0)
    assert st["profit_factor"] is None
    assert st["win_rate"] == 100.0
    assert st["expectancy_in_risks"] is None


def test_by_groupings(trades_rows):
    # fixture blend: 8 trades across 5 symbols, 4 exit types
    syms = by_symbol(trades_rows)
    total = sum(st["num"] for _, st in syms)
    assert total == 8
    assert [lab for lab, _ in syms] == ["BTC", "DOGE", "ETH", "SOL"]

    sides = by_side(trades_rows)
    assert [lab for lab, _ in sides] == ["long", "short"]
    long_n = dict(sides)["long"]["num"]
    assert long_n == 5

    exits = by_exit(trades_rows)
    assert dict(exits)["take_profit"]["num"] == 3
    assert dict(exits)["stop_loss"]["num"] == 3

    eps = by_episode(trades_rows)
    assert [lab for lab, _ in eps] == [0, 1]
    assert dict(eps)[0]["net_pnl"] == pytest.approx(83.0)


def test_hold_stats_hand_computed(trades_rows):
    h = hold_stats(trades_rows, bar_minutes=5)
    secs = np.array([9, 18, 27, 36, 45, 14, 23, 39], dtype=float) * 300.0
    assert h["n"] == 8
    assert h["mean_seconds"] == pytest.approx(secs.mean())
    assert h["median_seconds"] == pytest.approx(np.median(secs))
    assert h["max_seconds"] == pytest.approx(secs.max())


def test_leverage_stats_hand_computed(trades_rows):
    lev = np.array([5, 10, 3, 2, 5, 4, 3, 2], dtype=float)
    st = leverage_stats(trades_rows)
    assert st["n"] == 8
    assert st["mean"] == pytest.approx(lev.mean())
    assert st["median"] == pytest.approx(np.median(lev))
    assert st["max"] == pytest.approx(lev.max())
    assert st["p95"] == pytest.approx(np.percentile(lev, 95), rel=1e-9)