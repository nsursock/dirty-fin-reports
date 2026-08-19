"""config.yaml loading tests: every project parameter is YAML-driven."""

from pathlib import Path

from dirty_fin_reports.simple.config import (
    Plausibility,
    ProjectConfig,
    ReportConfig,
    SynthConfig,
    load_project,
)

ROOT = Path(__file__).parent.parent


def test_load_project_from_repo_config():
    p = load_project(ROOT / "configs" / "config.yaml")
    assert isinstance(p, ProjectConfig)
    assert isinstance(p.report, ReportConfig)
    assert isinstance(p.plausibility, Plausibility)
    assert isinstance(p.synth, SynthConfig)
    assert p.theme == "synthwave"
    assert p.overlays is False
    assert p.report.timeframe == "5m"
    assert p.report.initial_balance == 1000.0
    assert p.report.reporting_freq == "daily"
    assert p.report.rf_annual == 0.045
    assert p.report.n_steps == 17280
    assert p.synth.n_trades == 360
    assert p.synth.n_episodes == 8
    assert p.synth.ppo_rows == 200
    assert p.synth.seed == 42
    assert p.synth.seed_list == [3, 42, 7, 101, 999]
    assert p.plausibility.upi == (None, 30.0)
    assert p.plausibility.sharpe == (-5.0, 10.0)
    assert p.returns_basis == "collateral"


def test_load_project_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nope.yaml")


def test_load_project_minimal(tmp_path):
    cfg = tmp_path / "min.yaml"
    cfg.write_text("theme: ghibli\noverlays: true\n", encoding="utf-8")
    p = load_project(cfg)
    assert p.theme == "ghibli"
    assert p.overlays is True
    assert p.report.timeframe == "5m"
    assert p.synth.n_trades == 360


def test_load_project_custom_overrides(tmp_path):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text(
        "theme: valorant\n"
        "overlays: true\n"
        "report:\n  timeframe: 1h\n  initial_balance: 5000.0\n"
        "plausibility:\n  sharpe: [0.0, 5.0]\n  upi: [null, 20.0]\n"
        "synth:\n  n_trades: 400\n  ppo_rows: 300\n  seed: 7\n",
        encoding="utf-8",
    )
    p = load_project(cfg)
    assert p.theme == "valorant"
    assert p.overlays is True
    assert p.report.timeframe == "1h"
    assert p.report.initial_balance == 5000.0
    assert p.plausibility.sharpe == (0.0, 5.0)
    assert p.plausibility.upi == (None, 20.0)
    assert p.synth.n_trades == 400
    assert p.synth.ppo_rows == 300
    assert p.synth.seed == 7


def test_synth_seeds_fallback_to_seed(tmp_path):
    cfg = tmp_path / "s.yaml"
    cfg.write_text("synth:\n  seed: 5\n", encoding="utf-8")
    p = load_project(cfg)
    assert p.synth.seed == 5
    assert p.synth.seed_list == [5]