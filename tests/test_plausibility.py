"""Plausibility / likelihood-bounds tests.

The core promise: an implausible number (a per-bar "160 Sharpe", a 999x
profit factor) is flagged and reported as such — never trusted as a headline.
"""

from dirty_fin_reports.simple.plausibility import (
    Bound,
    DEFAULT_BOUNDS,
    aggregate,
    check_many,
    check_value,
)


def test_check_value_ok_within_bounds():
    c = check_value("sharpe", 2.5)
    assert c.ok is True
    assert c.severity == "ok"


def test_check_value_flags_low_severity_near_overshoot():
    c = check_value("sharpe", 10.4)
    assert c.ok is False
    assert c.severity == "low"


def test_check_value_flags_high_severity_160_sharpe():
    c = check_value("sharpe", 160.0)
    assert c.ok is False
    assert c.severity == "high"
    assert "160" in c.reason


def test_check_value_high_severity_negative():
    c = check_value("sharpe", -50.0)
    assert c.ok is False
    assert c.severity == "high"
    assert c.low == -5.0


def test_check_value_undefined_is_ok():
    c = check_value("sharpe", None)
    assert c.ok is True
    assert c.actual is None


def test_check_value_inclusive_boundary():
    assert check_value("max_drawdown", 1.0).ok
    assert check_value("max_drawdown", 1.01).ok is False


def test_check_value_custom_bounds():
    c = check_value("sharpe", 200.0, {"sharpe": (None, 5.0)})
    assert c.ok is False
    assert c.severity == "high"


def test_aggregate_counts_and_status():
    checks = [
        check_value("sharpe", 2.0),
        check_value("profit_factor", 3.0),
        check_value("cagr", 400.0),
    ]
    agg = aggregate(checks)
    assert agg["status"] == "implausible"
    assert agg["counts"]["high"] == 1
    assert agg["counts"]["ok"] == 2
    assert len(agg["failed"]) == 1


def test_aggregate_all_ok():
    agg = aggregate([check_value("sharpe", 1.0), check_value("win_rate", 55.0)])
    assert agg["status"] == "plausible"
    assert agg["failed"] == []


def test_check_many_scans_default_bounds():
    data = {
        "sharpe": 3.0,
        "win_rate": 62.5,
        "profit_factor": 2.05,
        "max_leverage": 10.0,
    }
    checks = check_many(data)
    names = {c.metric for c in checks}
    assert "sharpe" in names and "win_rate" in names
    assert all(c.ok for c in checks)


def test_default_bounds_contract():
    assert DEFAULT_BOUNDS["sharpe"] == (-5.0, 10.0)
    assert DEFAULT_BOUNDS["max_drawdown"] == (0.0, 1.0)
    assert DEFAULT_BOUNDS["profit_factor"][1] == 20.0
    assert DEFAULT_BOUNDS["upi"][0] is None