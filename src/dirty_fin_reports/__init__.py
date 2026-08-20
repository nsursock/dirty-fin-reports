"""dirty_fin_reports: financial + ML-health reporting for trading bots.

Phase 1 ships the ``simple`` subpackage: a self-contained metrics engine that
consumes a trade ledger CSV and SB3-style training progress CSVs (PPO manager,
SAC worker) and produces accuracy-checked, plausibility-bounded risk metrics.
"""

from __future__ import annotations

__version__ = "0.0.3"