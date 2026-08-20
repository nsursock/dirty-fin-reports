"""Risk-metric accuracy + plausibility ("no 160 Sharpe") tests.

Every formula is checked against a hand-computed value, and the annualization
tests pin down the exact behavior that used to produce inflated Sharpe ratios:
per-bar returns annualized by ``sqrt(72576)`` on a plateau ledger curve.
"""

import numpy as np
import pytest

from dirty_fin_reports.simple.metrics import (
    FREQ_PPY,
    metrics,
    periods_per_year,
    resample_returns,
    returns,
    timeframe_minutes,
)
from dirty_fin_reports.simple.plausibility import check_value, aggregate


def test_timeframe_minutes():
    assert timeframe_minutes("5m") == 5
    assert timeframe_minutes("1h") == 60
    assert timeframe_minutes("1d") == 1440
    assert timeframe_minutes("1w") == 10080
    assert timeframe_minutes("15m") == 15


def test_periods_per_year():
    assert periods_per_year("1d") == 252
    assert periods_per_year("5m") == 252 * 12 * 24
    assert periods_per_year("1h") == 252 * 24
    assert FREQ_PPY["daily"] == 252


def test_returns_hand_computed():
    r = returns(np.array([1000.0, 1100.0, 990.0, 1089.0]))
    assert np.allclose(r, [0.1, -0.1, 0.1], rtol=1e-12)


def test_resample_returns_hand_computed():
    e = np.array([1000, 1000, 1000, 1100, 1100, 1100, 990, 990, 990,
                  1089, 1089, 1089], dtype=float)
    rets, ppy, levels = resample_returns(e, periods_per_year=756, freq="daily")
    assert np.allclose(rets, [0.1, -0.1, 0.1])
    assert np.allclose(levels, [1000.0, 1100.0, 990.0, 1089.0])
    assert ppy == 252


def test_resample_too_short_returns_none():
    e = np.array([1000.0, 1100.0])
    rets, ppy, levels = resample_returns(e, periods_per_year=72576, freq="daily")
    assert rets is None
    assert ppy is None
    assert np.allclose(levels, e)


def test_metrics_sharpe_accuracy():
    e = np.array([1000, 1000, 1000, 1100, 1100, 1100, 990, 990, 990,
                  1089, 1089, 1089], dtype=float)
    m = metrics(e, periods_per_year=756, freq="daily", rf_annual=0.0)
    # mean/std(sample=0.0333/0.0942809) * sqrt(252)
    assert m["sharpe"] == pytest.approx(5.612486080160922, rel=1e-9)
    assert m["sortino"] is None  # a single downside window is insufficient
    assert m["freq"] == "daily"
    assert m["final_equity"] == pytest.approx(1089.0)
    assert m["total_return"] == pytest.approx(0.089)


def test_metrics_sharpe_with_rf():
    e = np.array([1000, 1000, 1000, 1100, 1100, 1100, 990, 990, 990,
                  1089, 1089, 1089], dtype=float)
    m = metrics(e, periods_per_year=756, freq="daily", rf_annual=0.06)
    assert m["sharpe"] == pytest.approx(5.57239689387406, rel=1e-9)
    assert m["rf_annual"] == pytest.approx(0.06)


def test_metrics_max_drawdown_hand_computed():
    e = np.array([1000.0, 1500.0, 800.0, 1200.0])
    m = metrics(e, periods_per_year=252, freq="daily")
    assert m["max_drawdown"] == pytest.approx(1.0 - 800.0 / 1500.0, rel=1e-9)


def test_metrics_ulcer_and_upi_hand_computed():
    net = np.array([1000.0, 1010.0, 1020.0, 1009.8, 1030.0, 1050.0])
    m = metrics(net, periods_per_year=252, freq="bar", rf_annual=0.0)
    bar_rets = net[1:] / net[:-1] - 1.0
    mean = bar_rets.mean()
    peak = np.maximum.accumulate(net)
    dd = (peak - net) / peak
    ui = float(np.sqrt(np.mean(np.square(dd))))
    assert m["ulcer_index"] == pytest.approx(ui, rel=1e-9)
    assert m["upi"] == pytest.approx(mean * 252.0 / ui, rel=1e-9)
    assert m["upi"] > 0.0


def test_metrics_cagr_hand_computed():
    e = np.array([1000.0, 1100.0])
    m = metrics(e, periods_per_year=252, freq="daily")
    years = 2.0 / 252.0
    expected = (1100.0 / 1000.0) ** (1.0 / years) - 1.0
    assert m["cagr"] == pytest.approx(expected, rel=1e-9)


def test_metrics_flat_curve_is_undefined_not_infinite():
    e = np.full(100, 1000.0)
    m = metrics(e, periods_per_year=72576, freq="bar")
    assert m["sharpe"] is None
    assert m["sortino"] is None
    assert m["upi"] is None
    assert m["max_drawdown"] == 0.0


def test_metrics_tiny_curve_defaults():
    m = metrics(np.array([1000.0]), periods_per_year=252, freq="daily")
    assert m["sharpe"] is None
    assert m["final_equity"] == 1000.0


def test_metrics_native_cadence_is_undefined_not_inflated():
    rng = np.random.default_rng(3)
    e = 1000.0 * np.cumprod(1 + rng.normal(0.0001, 0.001, 500))
    e = np.concatenate([[1000.0], e])
    m = metrics(e, periods_per_year=72576, freq="5m", rf_annual=0.045)
    # freq == native bar cadence: never per-bar annualized into a "160 Sharpe".
    assert m["sharpe"] is None
    assert m["sortino"] is None
    assert m["upi"] is None
    assert m["final_equity"] > 0.0


def test_metrics_too_short_resample_is_undefined_not_inflated():
    # Too few bars to form two daily windows: undefined, not per-bar annualized.
    e = np.array([1000.0, 1010.0, 1020.0])
    m = metrics(e, periods_per_year=72576, freq="daily", rf_annual=0.045)
    assert m["sharpe"] is None
    assert m["sortino"] is None
    assert m["upi"] is None


def _plateau_equity(days=60, start=1000.0, bars_per_day=288, daily_gain=0.01):
    return np.concatenate(
        [np.full(bars_per_day, start * (1 + daily_gain) ** d) for d in range(days)]
    )


def test_plateau_curve_inflates_per_bar_sharpe_but_not_the_headline():
    e = _plateau_equity(days=60)
    bar = metrics(e, periods_per_year=72576, freq="bar")
    daily = metrics(e, periods_per_year=72576, freq="daily")
    # ~15 is a grotesque amplification of a 1%-per-day ladder; the plausible
    # cap is out of reach for a real strategy and this number gets flagged.
    assert bar["sharpe"] > 10.0
    # A strictly monotone daily ladder is degenerate at the daily cadence —
    # undefined, never a monster headline.
    assert daily["sharpe"] is None
    check = check_value("sharpe", bar["sharpe"])
    assert not check.ok


def test_smooth_exponential_is_implausible_and_aggregate_flags_it():
    rng = np.random.default_rng(7)
    t = np.arange(3000, dtype=float)
    e = 1000.0 * np.exp(0.0004 * t) + 0.0005 * 1000.0 * np.exp(0.0004 * t) * np.sin(t / 40.0)
    for freq in ("bar", "daily"):
        m = metrics(e, periods_per_year=72576, freq=freq, rf_annual=0.0)
        assert m["sharpe"] is not None and m["sharpe"] > 100
        assert not check_value("sharpe", m["sharpe"]).ok
    agg = aggregate([check_value("sharpe", metrics(e, periods_per_year=72576, freq="daily")["sharpe"])])
    assert agg["status"] == "implausible"


def test_long_realistic_curve_daily_sharpe_stays_sane():
    rng = np.random.default_rng(21)
    days = 120
    daily = 1000.0 * np.cumprod(1 + rng.normal(0.0015, 0.0025, days))
    daily = np.concatenate([[1000.0], daily])
    e = np.concatenate([np.full(288, daily[d]) for d in range(days)] + [np.array([daily[-1]])])
    m = metrics(e, periods_per_year=72576, freq="daily")
    assert m["sharpe"] is not None and m["sharpe"] < 12.0
    assert check_value("sharpe", m["sharpe"]).ok