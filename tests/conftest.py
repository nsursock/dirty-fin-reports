"""Shared pytest fixtures: fixture CSV paths and pre-coerced ledger rows."""

from pathlib import Path

import pytest

from dirty_fin_reports.simple.ledger import coerce_ledger, load

FIXTURES = Path(__file__).parent / "fixtures"
TRADES_TWO = FIXTURES / "trades_two_episodes.csv"
PPO = FIXTURES / "manager_ppo.csv"
SAC = FIXTURES / "worker_sac.csv"


@pytest.fixture
def trades_rows():
    return coerce_ledger(load(TRADES_TWO))


def write_csv(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p