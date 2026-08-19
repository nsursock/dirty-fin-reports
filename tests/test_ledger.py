"""Ledger CSV loading, coercion and validation tests."""

import pytest

from dirty_fin_reports.simple.ledger import (
    EXIT_TYPES,
    SIDES,
    coerce_ledger,
    load,
    unique_keys,
    validate,
)
from conftest import TRADES_TWO, write_csv


def test_load_fixture_row_count():
    rows = load(TRADES_TWO)
    assert len(rows) == 8


def test_load_missing_required_column_raises(tmp_path):
    p = write_csv(
        tmp_path,
        "bad.csv",
        "trade_id,symbol,realized_pnl\n1,BTC,10.0\n",
    )
    with pytest.raises(ValueError, match="missing required column"):
        load(p)


def test_coerce_types():
    rows = coerce_ledger([
        {
            "trade_id": "1", "episode": "0", "symbol": "btc", "side": "Long",
            "opened_at": "1", "closed_at": "10", "entry_price": "100",
            "exit_price": "110", "notional": "1000", "leverage": "5",
            "collateral": "200", "entry_conviction": "0.6", "fee": "1.0",
            "funding": "0.5", "realized_pnl": "90.0", "exit_type": "take_profit",
        }
    ])
    r = rows[0]
    assert r["episode"] == 0
    assert r["trade_id"] == "1"
    assert r["symbol"] == "BTC"
    assert r["side"] == "long"
    assert r["opened_at"] == 1 and r["closed_at"] == 10
    assert r["realized_pnl"] == 90.0
    assert r["entry_price"] == 100.0


def test_coerce_tolerates_empty_and_non_numeric():
    rows = coerce_ledger([{
        "trade_id": "x", "episode": "1", "symbol": "ETH", "side": "long",
        "opened_at": "", "closed_at": "", "entry_price": "nan",
        "exit_price": "", "notional": "", "leverage": "", "collateral": "",
        "entry_conviction": "", "fee": "0", "funding": "", "realized_pnl": "",
        "exit_type": "",
    }])
    r = rows[0]
    assert r["opened_at"] is None
    assert r["entry_price"] is None
    assert r["fee"] == 0.0
    assert r["realized_pnl"] is None
    assert r["exit_type"] == ""


def test_validate_fixture_no_warnings(trades_rows):
    assert validate(trades_rows) == []


def test_validate_reports_data_problems():
    rows = coerce_ledger([
        {
            "trade_id": "1", "episode": "0", "symbol": "X", "side": "diagonal",
            "opened_at": "20", "closed_at": "10", "entry_price": "0",
            "exit_price": "5", "notional": "-1", "leverage": "-3",
            "collateral": "1", "entry_conviction": "0.5", "fee": "0",
            "funding": "0", "realized_pnl": "2", "exit_type": "warp",
        }
    ])
    warnings = validate(rows)
    text = "\n".join(warnings)
    assert "side" in text
    assert "exit_type" in text
    assert "closed_at < opened_at" in text
    assert "non-positive entry" in text
    assert "non-positive notional" in text
    assert "negative leverage" in text


def test_unique_keys_span_episodes(trades_rows):
    keys = unique_keys(trades_rows)
    assert len(keys) == 8
    assert (0, "1") in keys
    assert (1, "1") in keys


def test_validate_open_trade_warns():
    rows = coerce_ledger([{
        "trade_id": "1", "episode": "0", "symbol": "X", "side": "long",
        "opened_at": "1", "closed_at": "", "entry_price": "1", "exit_price": "",
        "notional": "10", "leverage": "1", "collateral": "1",
        "entry_conviction": "0.5", "fee": "0", "funding": "0",
        "realized_pnl": "", "exit_type": "open",
    }])
    warnings = validate(rows)
    assert any("still open" in w for w in warnings)


def test_exit_and_side_contracts():
    assert "liquidation" in EXIT_TYPES
    assert "short" in SIDES
    assert "long" in SIDES