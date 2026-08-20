"""Equity-curve reconstruction tests against hand-computed values."""

import numpy as np
import pytest

from dirty_fin_reports.simple.equity import by_bar, equity_curves, infer_length, portfolio_curve
from dirty_fin_reports.simple.ledger import coerce_ledger, load
from conftest import TRADES_TWO


def _rows():
    return coerce_ledger(load(TRADES_TWO))


def test_infer_length_is_max_closed_at(_rows=_rows):
    assert infer_length(_rows()) == 50
    assert infer_length(_rows(), n_steps=60) == 60
    assert infer_length(_rows(), n_steps=40) == 50


def test_by_bar_aggregates_realized_fees_funding(_rows=_rows):
    realized, fees, funding = by_bar(_rows(), length=50)
    assert realized[10] == 90.0
    assert realized[30] == -30.0
    assert realized[50] == 8.0
    assert realized[15] == 30.0
    assert fees[50] == 0.1
    assert funding[40] == 0.0
    assert abs(realized.sum() - 100.0) < 1e-9


def test_portfolio_curve_hand_computed(_rows=_rows):
    eq = portfolio_curve(_rows(), start=1000.0, n_steps=None)
    assert eq["length"] == 50
    assert eq["n_episodes"] == 2
    assert eq["steps"].tolist() == list(range(1, 51))
    assert eq["net"][-1] == 1050.0
    assert eq["net"][0] == 1000.0
    assert eq["mtm"][-1] == pytest.approx(1050.0)
    assert "mtm" in eq["per_episode"][0]

    # Per-episode single-account books.
    assert eq["per_episode"][0]["net"][-1] == 1083.0
    assert eq["per_episode"][1]["net"][-1] == 1017.0

    # Portfolio book is the index-aligned mean of the two episodes.
    expected = {
        10: 1045.0, 15: 1060.0, 20: 1087.5, 25: 1075.0,
        30: 1060.0, 40: 1040.0, 45: 1046.0, 50: 1050.0,
    }
    net = eq["net"]
    for bar, value in expected.items():
        assert net[bar - 1] == value

    # Net only steps on trade closes: bar 9 still at start balance.
    assert eq["net"][8] == 1000.0
    assert eq["per_episode"][0]["net"][14] == 1090.0


def test_mtm_moves_while_position_open():
    """Open MTM must show underwater risk before the close bar (Ref #4)."""
    rows = coerce_ledger([{
        "trade_id": "1", "episode": 0, "symbol": "BTC", "side": "long",
        "opened_at": 1, "closed_at": 11, "entry_price": 100.0, "exit_price": 90.0,
        "notional": 1000.0, "leverage": 5.0, "collateral": 200.0,
        "entry_conviction": 1.0, "fee": 0.0, "funding": 0.0,
        "realized_pnl": -100.0, "exit_type": "stop_loss",
    }])
    eq = portfolio_curve(rows, start=1000.0, n_steps=11)
    # Realized stays flat until close.
    assert eq["net"][4] == 1000.0
    assert eq["net"][-1] == 900.0
    # Mid-hold MTM (bar 6 → index 5) is halfway to the -100 realized loss.
    assert eq["mtm"][5] < 1000.0
    assert eq["mtm"][5] == pytest.approx(950.0)
    assert eq["mtm"][-1] == pytest.approx(900.0)
    assert eq["mtm_gap_max_pct"] > 0.0


def test_gross_readds_fees_and_funding(_rows=_rows):
    eq = portfolio_curve(_rows(), start=1000.0)
    gross_ep0 = eq["per_episode"][0]["gross"]
    net_ep0 = eq["per_episode"][0]["net"]
    assert float(gross_ep0[-1]) == pytest.approx(1086.45, rel=1e-12)
    # cumulative open fees 2.6 + funding 0.85 for episode 0
    assert abs(float(gross_ep0[-1]) - float(net_ep0[-1]) - 2.6 - 0.85) < 1e-9


def test_empty_ledger_returns_empty_curves():
    eq = portfolio_curve([], start=1000.0)
    assert eq["length"] == 0
    assert eq["net"].size == 0


def test_equity_curves_from_path():
    eq = equity_curves(ledger_path=TRADES_TWO, start=1000.0)
    assert eq["net"][-1] == 1050.0