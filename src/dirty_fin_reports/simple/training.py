"""ML-training health metrics from SB3-style progress CSVs.

Supports the two agents the bot trains: the PPO manager and the SAC worker.
NaN tolerance follows the repo rule: a lone NaN in the *last* row is ignored
(the episode simply didn't finish); any other NaN is counted in ``nan_fraction``
and excluded from the moving average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PPO_KEYS: tuple[str, ...] = (
    "time/fps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/learning_rate",
    "train/n_updates",
    "train/loss",
)

SAC_KEYS: tuple[str, ...] = (
    "time/fps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss/policy",
    "train/loss/critic",
    "train/loss/alpha",
    "train/policy/alpha",
    "train/policy/log_pi_mean",
    "train/value/q_mean",
    "train/learning_rate",
    "train/n_updates",
    "train/ent_coef",
)

TRAIN_LOSS_KEYS: tuple[str, ...] = (
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/loss/policy",
    "train/loss/critic",
    "train/loss/alpha",
    "train/policy/alpha",
    "train/policy/log_pi_mean",
    "train/value/q_mean",
)

_SAC_ALIASES: dict[str, tuple[str, ...]] = {
    "train/loss/policy": ("train/actor_loss",),
    "train/loss/critic": ("train/critic_loss",),
    "train/loss/alpha": ("train/ent_coef_loss",),
    "train/policy/alpha": ("train/ent_coef",),
    "train/ent_coef": ("train/policy/alpha",),
}

_REWARD_KEYS = ("rollout/ep_rew_mean",)


def detect_algorithm(path: str | Path) -> str:
    """Identify the agent by filename stem (``*sac*`` → sac, else ppo)."""
    stem = Path(path).stem.lower()
    return "sac" if "sac" in stem else "ppo"


def load_training_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: empty training CSV")
    return df


def series(df: pd.DataFrame, key: str) -> np.ndarray:
    """Column as a float array (NaN preserved), resolving SAC alias columns."""
    col = key
    if col not in df.columns:
        for alt in _SAC_ALIASES.get(key, ()):
            if alt in df.columns:
                col = alt
                break
    if col not in df.columns:
        return np.full(len(df), np.nan)
    vals = df[col]
    try:
        return vals.astype("float64", copy=False).to_numpy(dtype=float)
    except (TypeError, ValueError):
        return pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float)


def moving_average(x: np.ndarray, w: int = 10) -> np.ndarray:
    """Causal trailing mean (expanding until ``w``), NaN-aware."""
    y = np.asarray(x, dtype=float)
    out = np.full(y.size, np.nan)
    if y.size == 0:
        return out
    w = max(1, int(w))
    ok = np.isfinite(y)
    y0 = np.where(ok, y, 0.0)
    cs_y = np.concatenate([[0.0], np.cumsum(y0)])
    cs_n = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    idx = np.arange(1, y.size + 1)
    j = np.maximum(0, idx - w)
    dn = cs_n[idx] - cs_n[j]
    good = dn > 0
    out[good] = (cs_y[idx][good] - cs_y[j][good]) / dn[good]
    return out


def _first_last_finite(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return np.nan, np.nan
    return float(f[0]), float(f[-1])


def training_health(df: pd.DataFrame, algorithm: str | None = None, ma: int = 10) -> dict:
    """Health summary for one training CSV.

    ``reward_delta_pct`` is the percentage change from the first to the last
    finite reward sample; ``reward_trend`` is ``+1/-1/0`` on that sign.
    Every monitored key gets its ``final``, ``min``, ``max`` and ``ma_final``.
    NaN handling: ``trailing_na`` counts the final run of NaN rows (tolerated);
    ``nan_fraction`` is the overall NaN share of monitored columns.
    """
    if algorithm is None:
        raise ValueError("algorithm must be 'ppo' or 'sac'")
    keys = PPO_KEYS if algorithm == "ppo" else SAC_KEYS

    def _present(k: str) -> bool:
        if k in df.columns:
            return True
        return any(a in df.columns for a in _SAC_ALIASES.get(k, ()))

    monitored = [k for k in keys if _present(k)]
    n_rows = len(df)

    total_nan = 0
    total_cells = 0
    key_stats: dict[str, dict] = {}
    for k in monitored:
        y = series(df, k)
        finite = np.isfinite(y)
        total_nan += int((~finite).sum())
        total_cells += int(y.size)
        f = y[finite]
        key_stats[k] = {
            "final": float(f[-1]) if f.size else None,
            "min": float(f.min()) if f.size else None,
            "max": float(f.max()) if f.size else None,
            "mean": float(f.mean()) if f.size else None,
            "ma_final": float(moving_average(y, ma)[-1]) if f.size else None,
            "nan_count": int((~finite).sum()),
        }

    reward = series(df, "rollout/ep_rew_mean")
    rf, rl = _first_last_finite(reward)
    reward_delta_pct = None
    if np.isfinite(rf) and np.isfinite(rl):
        reward_delta_pct = 100.0 * (rl - rf) / max(abs(rf), 1e-9)

    trailing_na = 0
    for row_idx in range(n_rows - 1, -1, -1):
        all_nan = all(np.isnan(series(df, k)[row_idx]) for k in monitored)
        if all_nan:
            trailing_na += 1
        else:
            break

    nan_fraction = (total_nan / total_cells) if total_cells else 0.0
    if np.isfinite(reward_delta_pct if reward_delta_pct is not None else np.nan):
        trend = 1 if reward_delta_pct > 0 else (-1 if reward_delta_pct < 0 else 0)
    else:
        trend = 0

    return {
        "algorithm": algorithm,
        "rows": n_rows,
        "keys": key_stats,
        "reward_delta_pct": reward_delta_pct,
        "reward_trend": trend,
        "nan_fraction": nan_fraction,
        "trailing_na": trailing_na,
    }