"""Phase 1: simple, deterministic reporting metrics.

The subpackage is deliberately decoupled from any trading environment. It takes
the two inputs the bot produces — a ledger/trade-history CSV and per-agent
training progress CSVs — and returns every metric on a strict, unit-tested
formula. Annualization happens at a sane reporting cadence (default daily), so
per-bar Sharpe inflation ("a 160 Sharpe ratio") cannot leak into the headline.
"""

from __future__ import annotations

from .ledger import (
    EXIT_TYPES,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    SIDES,
    coerce_ledger,
    load,
    to_frame,
    unique_keys,
    validate,
)
from .metrics import (
    FREQ_PPY,
    periods_per_year,
    resample_returns,
    returns,
    metrics,
)
from .equity import (
    by_bar,
    equity_curves,
    infer_length,
    per_symbol_curves,
    portfolio_curve,
)
from .trades import (
    by_group,
    by_symbol,
    by_exit,
    by_side,
    by_episode,
    hold_stats,
    leverage_stats,
    trade_stats,
)
from .training import (
    PPO_KEYS,
    SAC_KEYS,
    TRAIN_LOSS_KEYS,
    detect_algorithm,
    load_training_csv,
    series,
    moving_average,
    training_health,
)
from .plausibility import (
    DEFAULT_BOUNDS,
    Bound,
    check_value,
    aggregate,
)
from .verdict import health_axis, performance_axis, recommend
from .viz import Palette, resolve_theme, write_png
from .figures import figure1, figure2, training_figure
from .breakdown import BD_COLS, bot_cfg, breakdown, breakdown_trade_stats
from .synth import generate_ppo, generate_run, generate_sac, generate_trades
from .config import Plausibility, ProjectConfig, ReportConfig, SynthConfig, load_project
from .env_params import TradingEnvParams
from .report import (
    assemble,
    build_report,
    format_breakdown,
    report_dict,
    run_reporter,
    write_report,
)

__all__ = [
    "EXIT_TYPES",
    "NUMERIC_COLUMNS",
    "REQUIRED_COLUMNS",
    "SIDES",
    "coerce_ledger",
    "load",
    "to_frame",
    "unique_keys",
    "validate",
    "FREQ_PPY",
    "periods_per_year",
    "resample_returns",
    "returns",
    "metrics",
    "by_bar",
    "equity_curves",
    "infer_length",
    "per_symbol_curves",
    "portfolio_curve",
    "by_group",
    "by_symbol",
    "by_exit",
    "by_side",
    "by_episode",
    "hold_stats",
    "leverage_stats",
    "trade_stats",
    "PPO_KEYS",
    "SAC_KEYS",
    "TRAIN_LOSS_KEYS",
    "detect_algorithm",
    "load_training_csv",
    "series",
    "moving_average",
    "training_health",
    "DEFAULT_BOUNDS",
    "Bound",
    "check_value",
    "aggregate",
    "health_axis",
    "performance_axis",
    "recommend",
    "Palette",
    "resolve_theme",
    "write_png",
    "figure1",
    "figure2",
    "training_figure",
    "BD_COLS",
    "bot_cfg",
    "breakdown",
    "breakdown_trade_stats",
    "generate_ppo",
    "generate_run",
    "generate_sac",
    "generate_trades",
    "Plausibility",
    "ProjectConfig",
    "ReportConfig",
    "SynthConfig",
    "TradingEnvParams",
    "load_project",
    "assemble",
    "build_report",
    "format_breakdown",
    "report_dict",
    "run_reporter",
    "write_report",
]