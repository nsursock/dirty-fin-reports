"""Verdict-layer tests: performance / health axes and the recommendation.

The promise: profitability, statistical plausibility and structural health are
reported independently, and a profitable-but-suspicious run is routed for
review rather than silently accepted or hard-rejected.
"""

import numpy as np
import pytest

from dirty_fin_reports.simple.verdict import (
    health_axis,
    performance_axis,
    recommend,
)


def test_performance_axis_profitable():
    assert performance_axis(0.127)["status"] == "profitable"
    assert performance_axis(0.127)["return_pct"] == pytest.approx(12.7)


def test_performance_axis_losing():
    assert performance_axis(-0.019)["status"] == "losing"


def test_performance_axis_breakeven():
    assert performance_axis(0.001)["status"] == "breakeven"
    assert performance_axis(-0.001)["status"] == "breakeven"


def test_performance_axis_none_is_breakeven():
    assert performance_axis(None)["status"] == "breakeven"
    assert performance_axis(None)["return"] is None


def test_health_axis_healthy():
    h = health_axis([1000.0, 1010.0, 1020.0, 1015.0, 1030.0])
    assert h["status"] == "healthy"
    assert h["issues"] == []


def test_health_axis_degraded_deep_drawdown():
    h = health_axis([1000.0, 1000.0, 400.0, 450.0, 460.0])
    assert h["status"] == "degraded"
    assert any("drawdown" in i for i in h["issues"])


def test_health_axis_pathological_nonfinite():
    h = health_axis([1000.0, np.nan, 1010.0])
    assert h["status"] == "pathological"


def test_health_axis_pathological_nonpositive():
    h = health_axis([1000.0, 0.0, 500.0])
    assert h["status"] == "pathological"


def test_health_axis_pathological_flatline():
    h = health_axis([1000.0, 1000.0, 1000.0])
    assert h["status"] == "pathological"


def test_health_axis_empty():
    assert health_axis([])["status"] == "pathological"


def test_recommend_accept_clean_profitable():
    rec = recommend("profitable", "plausible", "healthy")
    assert rec["action"] == "accept"


def test_recommend_high_alpha_review():
    rec = recommend("profitable", "implausible", "healthy",
                    ["calmar", "sortino", "upi"])
    assert rec["action"] == "review"
    assert "high-alpha" in rec["reason"]
    assert "calmar" in rec["reason"]


def test_recommend_profitable_flagged_review():
    rec = recommend("profitable", "flagged", "healthy", ["calmar"])
    assert rec["action"] == "review"
    assert "high-alpha" in rec["reason"]


def test_recommend_losing_plausible_review():
    rec = recommend("losing", "plausible", "healthy")
    assert rec["action"] == "review"
    assert "unprofitable" in rec["reason"]


def test_recommend_pathological_investigate():
    rec = recommend("profitable", "plausible", "pathological")
    assert rec["action"] == "investigate"


def test_recommend_implausible_losing_investigate():
    rec = recommend("losing", "implausible", "healthy")
    assert rec["action"] == "investigate"


def test_recommend_degraded_review():
    rec = recommend("profitable", "plausible", "degraded")
    assert rec["action"] == "review"
