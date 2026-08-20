"""Stage 0 gate scoring tests."""

from __future__ import annotations

from dirty_fin_reports.simple.gates import format_gate_report, score_stage0_gates

GATES = {
    "median_oos_return_positive": {
        "metric": "total_return",
        "scope": "oos",
        "aggregate": "median",
        "op": ">",
        "threshold": 0.0,
    },
    "folds_positive_fraction": {
        "metric": "total_return",
        "scope": "oos",
        "aggregate": "fraction_positive",
        "op": ">=",
        "threshold": 0.60,
    },
    "seeds_positive_fraction": {
        "metric": "total_return",
        "scope": "oos",
        "aggregate": "seeds_fraction_positive",
        "op": ">=",
        "threshold": 0.60,
    },
    "oos_retention_vs_is": {
        "metric": "upi",
        "aggregate": "median_retention",
        "op": ">=",
        "threshold": 0.50,
    },
    "oos_upi_positive": {
        "metric": "upi",
        "scope": "oos",
        "aggregate": "median",
        "op": ">",
        "threshold": 0.0,
    },
    "no_single_fold_dominance": {
        "metric": "total_return",
        "scope": "oos",
        "aggregate": "max_abs_share",
        "op": "<=",
        "threshold": 0.50,
    },
}


def _row(seed, fold, is_ret, is_upi, oos_ret, oos_upi):
    return {
        "seed": seed,
        "fold": fold,
        "is": {"total_return": is_ret, "upi": is_upi},
        "oos": {"total_return": oos_ret, "upi": oos_upi},
    }


def test_stage0_gates_pass_on_healthy_planted_alpha():
    rows = [
        _row(1, 0, 0.20, 2.0, 0.12, 1.2),
        _row(1, 1, 0.18, 1.8, 0.10, 1.0),
        _row(1, 2, 0.22, 2.2, 0.11, 1.1),
        _row(2, 0, 0.15, 1.5, 0.09, 0.9),
        _row(2, 1, 0.16, 1.6, 0.08, 0.8),
        _row(2, 2, 0.14, 1.4, 0.07, 0.7),
    ]
    scored = score_stage0_gates(rows, GATES)
    assert scored["overall_pass"] is True
    assert scored["gates"]["median_oos_return_positive"]["pass"] is True
    assert scored["gates"]["folds_positive_fraction"]["value"] == 1.0
    assert scored["gates"]["seeds_positive_fraction"]["value"] == 1.0
    text = format_gate_report(scored)
    assert "PASS" in text
    assert "overall=PASS" in text


def test_stage0_gates_fail_when_oos_collapses():
    rows = [
        _row(1, 0, 0.20, 2.0, -0.05, -0.2),
        _row(1, 1, 0.18, 1.8, -0.04, -0.1),
        _row(1, 2, 0.22, 2.2, -0.03, -0.1),
        _row(2, 0, 0.15, 1.5, -0.02, -0.1),
        _row(2, 1, 0.16, 1.6, -0.01, -0.05),
        _row(2, 2, 0.14, 1.4, 0.01, 0.05),
    ]
    scored = score_stage0_gates(rows, GATES)
    assert scored["overall_pass"] is False
    assert scored["gates"]["median_oos_return_positive"]["pass"] is False
    assert scored["gates"]["oos_upi_positive"]["pass"] is False


def test_dominance_gate_catches_single_fold():
    rows = [
        _row(1, 0, 0.1, 1.0, 1.00, 1.0),
        _row(1, 1, 0.1, 1.0, 0.01, 0.5),
        _row(1, 2, 0.1, 1.0, 0.01, 0.5),
    ]
    scored = score_stage0_gates(rows, GATES)
    assert scored["gates"]["no_single_fold_dominance"]["pass"] is False
    assert scored["gates"]["no_single_fold_dominance"]["value"] > 0.50


def test_retention_undefined_when_upi_coverage_too_low():
    """Monotone equity → UPI=None; one defined pair must not drive the gate."""
    rows = [
        _row(1, 0, 0.20, None, 0.10, None),
        _row(1, 1, 0.18, None, 0.09, None),
        _row(1, 2, 0.22, 5942.0, 0.11, 1585.0),  # only defined pair
        _row(2, 0, 0.15, None, 0.08, None),
        _row(2, 1, 0.16, None, 0.07, None),
        _row(2, 2, 0.14, None, 0.06, None),
    ]
    scored = score_stage0_gates(rows, GATES)
    g = scored["gates"]["oos_retention_vs_is"]
    assert g["value"] is None
    assert g["pass"] is False
    assert g["n_pairs"] == 1
    assert g["coverage"] < 0.5


def test_retention_uses_median_when_coverage_ok():
    rows = [
        _row(1, 0, 0.20, 2.0, 0.12, 1.2),  # 0.60
        _row(1, 1, 0.18, 1.8, 0.10, 1.0),  # 0.556
        _row(1, 2, 0.22, 2.2, 0.11, 1.1),  # 0.50
        _row(2, 0, 0.15, 1.5, 0.09, 0.9),  # 0.60
        _row(2, 1, 0.16, 1.6, 0.08, 0.8),  # 0.50
        _row(2, 2, 0.14, 1.4, 0.07, 0.7),  # 0.50
    ]
    scored = score_stage0_gates(rows, GATES)
    g = scored["gates"]["oos_retention_vs_is"]
    assert g["n_pairs"] == 6
    assert g["pass"] is True
    assert g["value"] >= 0.50


def test_oos_upi_positive_undefined_when_coverage_too_low():
    rows = [
        _row(1, 0, 0.20, None, 0.10, None),
        _row(1, 1, 0.18, None, 0.09, None),
        _row(1, 2, 0.22, None, 0.11, 1585.0),  # only defined OOS UPI
        _row(2, 0, 0.15, None, 0.08, None),
        _row(2, 1, 0.16, None, 0.07, None),
        _row(2, 2, 0.14, None, 0.06, None),
    ]
    scored = score_stage0_gates(rows, GATES)
    g = scored["gates"]["oos_upi_positive"]
    assert g["value"] is None
    assert g["pass"] is False
    assert g["n_defined"] == 1
    assert g["coverage"] < 0.5
