"""Training-CSV health metrics tests (PPO manager / SAC worker)."""

import numpy as np
import pandas as pd
import pytest

from dirty_fin_reports.simple.training import (
    PPO_KEYS,
    SAC_KEYS,
    detect_algorithm,
    load_training_csv,
    moving_average,
    series,
    training_health,
)
from conftest import PPO, SAC


def test_detect_algorithm_by_filename_stem():
    assert detect_algorithm(PPO) == "ppo"
    assert detect_algorithm(SAC) == "sac"
    assert detect_algorithm("some_path/manager_ppo.csv") == "ppo"
    assert detect_algorithm("some_path/worker_sac.csv") == "sac"


def test_series_resolves_sac_aliases():
    df = load_training_csv(SAC)
    alias = series(df, "train/loss/policy")
    raw = df["train/actor_loss"].to_numpy(dtype=float)
    assert np.allclose(alias, raw, rtol=0, atol=0)


def test_series_missing_column_is_nan():
    df = load_training_csv(PPO)
    out = series(df, "train/loss/policy")  # not present in PPO CSV
    assert out.size == len(df)
    assert np.isnan(out).all()


def test_moving_average_causal_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(moving_average(x, 2), [1.0, 1.5, 2.5, 3.5, 4.5])


def test_moving_average_skips_nan():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    assert np.allclose(moving_average(x, 2), [1.0, 1.0, 3.0, 3.5])


def test_ppo_health_accuracies():
    df = load_training_csv(PPO)
    h = training_health(df, "ppo", ma=3)
    assert h["algorithm"] == "ppo"
    assert h["rows"] == 5
    assert h["trailing_na"] == 0
    assert h["nan_fraction"] == 0.0
    assert h["reward_delta_pct"] == pytest.approx(800.0)  # 0.10 -> 0.90
    assert h["reward_trend"] == 1
    assert h["keys"]["train/loss"]["final"] == 0.1
    assert h["keys"]["rollout/ep_rew_mean"]["ma_final"] == pytest.approx(0.7)


def test_sac_health_accuracies():
    df = load_training_csv(SAC)
    h = training_health(df, "sac", ma=3)
    assert h["algorithm"] == "sac"
    assert h["reward_delta_pct"] == pytest.approx(180.0)  # -0.5 -> 0.4
    assert h["reward_trend"] == 1
    assert h["keys"]["train/loss/policy"]["final"] == pytest.approx(0.05)
    assert all(k in h["keys"] for k in ("train/loss/critic", "train/policy/alpha", "train/value/q_mean"))


def test_ppo_health_ignores_trailing_nan_row(tmp_path):
    df = load_training_csv(PPO)
    data = df.to_numpy().tolist() + [[np.nan] * len(df.columns)]
    df2 = pd.DataFrame(data, columns=df.columns)
    h = training_health(df2, "ppo", ma=3)
    assert h["rows"] == 6
    assert h["trailing_na"] == 1
    assert h["reward_delta_pct"] == pytest.approx(800.0)


def test_algorithm_key_intersection():
    assert "rollout/ep_rew_mean" in PPO_KEYS
    assert "rollout/ep_rew_mean" in SAC_KEYS