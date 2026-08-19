"""Reporting configuration: cadence, annualization, plausibility bounds.

Everything is parameterized through ``config.yaml`` (see ``load_project``);
the dataclasses here are the typed surface it deserializes into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metrics import periods_per_year


@dataclass
class ReportConfig:
    timeframe: str = "5m"
    initial_balance: float = 1000.0
    reporting_freq: str = "daily"
    rf_annual: float = 0.045
    n_steps: Optional[int] = None
    ppo_csv: Optional[str] = None
    sac_csv: Optional[str] = None
    start_date: Optional[str] = None
    tick_tilt: bool = True
    tick_angle: float = 22.5
    tick_direction: str = "down"

    @property
    def periods_per_year(self) -> int:
        return periods_per_year(self.timeframe)

    @property
    def start_dt(self) -> datetime:
        from datetime import datetime

        raw = self.start_date or "2025-01-01"
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.fromisoformat("2025-01-01")

    @property
    def x_tick_angle(self) -> Optional[float]:
        """Signed date-axis tick rotation (``None`` when tilting is disabled)."""
        if not self.tick_tilt:
            return None
        sign = 1.0 if self.tick_direction == "down" else -1.0
        return sign * abs(float(self.tick_angle))


@dataclass
class Plausibility:
    sharpe: tuple[Optional[float], Optional[float]] = (-5.0, 10.0)
    sortino: tuple[Optional[float], Optional[float]] = (-10.0, 15.0)
    calmar: tuple[Optional[float], Optional[float]] = (-5.0, 10.0)
    cagr: tuple[Optional[float], Optional[float]] = (-0.99, 5.0)
    max_drawdown: tuple[Optional[float], Optional[float]] = (0.0, 1.0)
    ulcer_index: tuple[Optional[float], Optional[float]] = (0.0, 0.5)
    upi: tuple[Optional[float], Optional[float]] = (None, 30.0)
    win_rate: tuple[Optional[float], Optional[float]] = (0.0, 100.0)
    profit_factor: tuple[Optional[float], Optional[float]] = (0.0, 20.0)
    expectancy_in_risks: tuple[Optional[float], Optional[float]] = (None, 5.0)
    max_leverage: tuple[Optional[float], Optional[float]] = (0.0, 100.0)
    reward_delta_pct: tuple[Optional[float], Optional[float]] = (-50.0, 50.0)

    def as_dict(self) -> dict[str, tuple[Optional[float], Optional[float]]]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "ulcer_index": self.ulcer_index,
            "upi": self.upi,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy_in_risks": self.expectancy_in_risks,
            "max_leverage": self.max_leverage,
            "reward_delta_pct": self.reward_delta_pct,
        }


def _pair(v) -> tuple[Optional[float], Optional[float]]:
    if v is None:
        return (None, None)
    lo, hi = v
    return (None if lo is None else float(lo), None if hi is None else float(hi))


@dataclass
class SynthConfig:
    """Parameters for the synthetic-run generator (see ``synth.py``)."""
    n_trades: int = 360
    n_episodes: int = 8
    n_steps: int = 17280
    ppo_rows: int = 200
    sac_rows: int = 200
    seed: int = 42
    seeds: Optional[list[int]] = None   # one run per seed when generating

    @property
    def seed_list(self) -> list[int]:
        return list(self.seeds) if self.seeds else [self.seed]


@dataclass
class ProjectConfig:
    """The typed surface of ``config.yaml`` — every project parameter lives
    in the YAML and is loaded through :func:`load_project`."""
    report: ReportConfig = field(default_factory=ReportConfig)
    plausibility: Plausibility = field(default_factory=Plausibility)
    synth: SynthConfig = field(default_factory=SynthConfig)
    theme: str = "synthwave"
    overlays: bool = False
    env: dict = field(default_factory=dict)

    @property
    def returns_basis(self) -> str:
        return str(self.env.get("returns", {}).get("basis", "collateral")) if self.env else "collateral"


def load_project(path: str | Path) -> ProjectConfig:
    """Load the full project configuration from a YAML file.

    ``report`` keys map onto :class:`ReportConfig`, ``plausibility`` onto
    :class:`Plausibility`, ``synth`` onto :class:`SynthConfig`. Everything else
    (``theme``, ``overlays``, ``env``) is read by name.
    """
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    report = dict(raw.get("report") or {})
    cfg = ReportConfig(
        timeframe=str(report.get("timeframe", "5m")),
        initial_balance=float(report.get("initial_balance", 1000.0)),
        reporting_freq=str(report.get("reporting_freq", "daily")),
        rf_annual=float(report.get("rf_annual", 0.045)),
        n_steps=report.get("n_steps"),
        ppo_csv=report.get("ppo_csv"),
        sac_csv=report.get("sac_csv"),
        start_date=report.get("start_date"),
        tick_tilt=bool(report.get("tick_tilt", True)),
        tick_angle=float(report.get("tick_angle", 22.5)),
        tick_direction=str(report.get("tick_direction", "down")),
    )

    pl = raw.get("plausibility") or {}
    plausibility = Plausibility(**{k: _pair(v) for k, v in pl.items()}) if pl else Plausibility()

    syn = raw.get("synth") or {}
    synth = SynthConfig(
        n_trades=int(syn.get("n_trades", 360)),
        n_episodes=int(syn.get("n_episodes", 8)),
        n_steps=int(syn.get("n_steps", 17280)),
        ppo_rows=int(syn.get("ppo_rows", 200)),
        sac_rows=int(syn.get("sac_rows", 200)),
        seed=int(syn.get("seed", 42)),
        seeds=[int(s) for s in syn.get("seeds")] if syn.get("seeds") else None,
    )

    return ProjectConfig(
        report=cfg,
        plausibility=plausibility,
        synth=synth,
        theme=str(raw.get("theme", "synthwave")),
        overlays=bool(raw.get("overlays", False)),
        env=dict(raw.get("env") or {}),
    )