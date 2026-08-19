"""Figure-generation tests: every artifact must render to a non-empty PNG."""

from pathlib import Path

from dirty_fin_reports.simple.equity import per_symbol_curves, portfolio_curve
from dirty_fin_reports.simple.figures import figure1, figure2, training_figure
from dirty_fin_reports.simple.training import load_training_csv
from conftest import PPO, SAC, TRADES_TWO
from dirty_fin_reports.simple.ledger import coerce_ledger, load


def test_figure1_renders(tmp_path):
    rows = coerce_ledger(load(TRADES_TWO))
    eq = portfolio_curve(rows, start=1000.0)
    _, per_sym = per_symbol_curves(rows, start=1000.0)
    out = tmp_path / "figure1.png"
    figure1(eq["net"], eq["gross"], eq["steps"], rows, out, per_symbol=per_sym)
    assert out.exists()
    assert out.stat().st_size > 32_000  # kaleido wrote real pixels, not a stub


def test_figure1_handles_empty_ledger(tmp_path):
    rows = []
    out = tmp_path / "figure1_empty.png"
    figure1([1000.0, 1050.0], [1000.0, 1055.0], [1, 2], rows, out)
    assert out.exists() and out.stat().st_size > 32_000


def test_figure2_renders(tmp_path):
    rows = coerce_ledger(load(TRADES_TWO))
    out = tmp_path / "figure2.png"
    figure2(rows, out)
    assert out.exists() and out.stat().st_size > 32_000


def test_figure2_empty_ledger(tmp_path):
    out = tmp_path / "figure2_empty.png"
    figure2([], out)
    assert out.exists() and out.stat().st_size > 32_000


def test_training_figure_ppo(tmp_path):
    out = tmp_path / "manager_ppo.png"
    training_figure(PPO, out, title="PPO manager health")
    assert out.exists() and out.stat().st_size > 32_000


def test_training_figure_sac(tmp_path):
    out = tmp_path / "worker_sac.png"
    training_figure(SAC, out, title="SAC worker health")
    assert out.exists() and out.stat().st_size > 32_000


def test_run_reporter_emits_figures(tmp_path):
    from dirty_fin_reports.simple.report import run_reporter

    from conftest import FIXTURES

    src = FIXTURES / "runs" / "phase1"
    out = tmp_path / "out"
    r = run_reporter(src, out_dir=out)
    status = r["plausibility"]["status"]
    for name in (f"bot-performance-{status}.png", f"trade-anatomy-{status}.png",
                 "manager_diag.png", "worker_diag.png"):
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 32_000, name
    assert r["figure1"] == str(out / f"bot-performance-{status}.png")
    assert r["manager_diag_figure"] == str(out / "manager_diag.png")


def test_figure1_overlays_flag(tmp_path):
    """Overlays are opt-in: both branches must render without error."""
    rows = coerce_ledger(load(TRADES_TWO))
    eq = portfolio_curve(rows, start=1000.0)
    _, per_sym = per_symbol_curves(rows, start=1000.0)
    for overlays in (False, True):
        out = tmp_path / f"f1_{overlays}.png"
        figure1(eq["net"], eq["gross"], eq["steps"], rows, out,
                per_symbol=per_sym, overlays=overlays)
        assert out.exists() and out.stat().st_size > 32_000


def _make_bot_layout(tmp_path: Path):
    """Create a trading-bot-style run folder (testing/ + training/)."""
    testing = tmp_path / "testing"
    training = tmp_path / "training"
    testing.mkdir(parents=True)
    training.mkdir(parents=True)
    from conftest import PPO, SAC, TRADES_TWO

    (testing / "trades.csv").write_text(TRADES_TWO.read_text(), encoding="utf-8")
    (training / "manager_ppo.csv").write_text(PPO.read_text(), encoding="utf-8")
    (training / "worker_sac.csv").write_text(SAC.read_text(), encoding="utf-8")
    return tmp_path


def test_run_reporter_bot_layout(tmp_path):
    from dirty_fin_reports.simple.report import run_reporter

    run = _make_bot_layout(tmp_path / "run")
    out = tmp_path / "out"
    r = run_reporter(run, out_dir=out)
    # bot layout: figures/text into testing/, diagnostics into training/
    status = r["plausibility"]["status"]
    for name in (f"bot-performance-{status}.png", f"trade-anatomy-{status}.png",
                 "breakdown.txt"):
        assert (run / "testing" / name).exists(), name
    assert (run / "training" / "manager_diag.png").exists()
    assert (run / "training" / "worker_diag.png").exists()
    assert (out / "report.json").exists()
    assert r["sources"]["trades"].endswith("testing/trades.csv")
    assert r["sources"]["manager_ppo"].endswith("training/manager_ppo.csv")
    assert r["sources"]["worker_sac"].endswith("training/worker_sac.csv")
    assert r["breakdown"].endswith("testing/breakdown.txt")
    assert r["manager_diag_figure"].endswith("training/manager_diag.png")
    assert (run / "testing" / f"bot-performance-{status}.png").stat().st_size > 32_000


def test_build_report_flat_layout_unchanged(tmp_path):
    from dirty_fin_reports.simple.report import build_report
    from conftest import FIXTURES

    r = build_report(FIXTURES / "runs" / "phase1")
    assert r["sources"]["trades"].endswith("trades.csv")
    assert r["sources"]["manager_ppo"].endswith("manager_ppo.csv")


def test_build_report_missing_trades(tmp_path):
    from dirty_fin_reports.simple.report import build_report

    try:
        build_report(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")