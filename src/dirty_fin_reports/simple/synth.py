"""Plausible synthetic run generators: a trade ledger, a PPO manager CSV and a
SAC worker CSV, written in the trading-bot run layout (``testing/trades.csv``
+ ``training/*.csv``). Used to exercise the phase-1 report without any real
bot inputs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .ledger import EXIT_TYPES, NUMERIC_COLUMNS

SYMBOLS = ("BTC", "ETH", "SOL", "AVAX", "BNB", "XRP", "DOGE", "LINK")
_BASE_PRICES = {
    "BTC": 58000.0, "ETH": 3100.0, "SOL": 145.0, "AVAX": 35.0,
    "BNB": 520.0, "XRP": 0.62, "DOGE": 0.12, "LINK": 16.5,
}
_BAR_VOL = {
    "BTC": 0.0012, "ETH": 0.0015, "SOL": 0.0022, "AVAX": 0.0025,
    "BNB": 0.0016, "XRP": 0.0020, "DOGE": 0.0030, "LINK": 0.0018,
}

PPO_COLUMNS = (
    "time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps",
    "rollout/ep_len_mean", "rollout/ep_rew_mean", "rollout/success_rate",
    "train/approx_kl", "train/clip_fraction", "train/clip_range",
    "train/entropy_loss", "train/explained_variance", "train/learning_rate",
    "train/loss", "train/n_updates", "train/policy_gradient_loss",
    "train/std", "train/value_loss",
)

SAC_COLUMNS = (
    "time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps",
    "rollout/ep_len_mean", "rollout/ep_rew_mean", "rollout/success_rate",
    "train/actor_loss", "train/critic_loss", "train/ent_coef",
    "train/ent_coef_loss", "train/learning_rate", "train/loss",
    "train/loss/alpha", "train/loss/policy", "train/loss/q1", "train/loss/q2",
    "train/n_updates", "train/policy/alpha", "train/policy/log_pi_mean",
    "train/value/q_mean",
)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_trades(
    n: int = 360,
    n_episodes: int = 8,
    n_steps: int = 17280,
    initial_balance: float = 1000.0,
    seed: int = 42,
) -> list[dict]:
    """A realistic closed-position ledger (bar-indexed times).

    Mirrors the bot's execution model: isolated margin, 20-150x leverage,
    risk-scaled collateral, per-trade fees + an occasional liquidation, and a
    mildly positive expectancy so the equity curve is a healthy staircase with
    visible drawdown episodes.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    per_ep = max(int(math.ceil(n / n_episodes)), 4)
    trade_id = 0
    for ep in range(n_episodes):
        equity = initial_balance
        # Regime drift per episode: most episodes grind gently up, one or two
        # are choppy or bearish so the mean book shows shallow recoverable
        # dips instead of a straight line (keeps Ulcer low, UPI healthy).
        regime = float(rng.normal(4.0, 44.0))
        ep_n = per_ep - 1 if ep == n_episodes - 1 else per_ep  # total stays near n
        for _ in range(ep_n):
            trade_id += 1
            symbol = SYMBOLS[int(rng.integers(0, len(SYMBOLS)))]
            side = "long" if rng.random() < 0.86 else "short"
            lev = float(rng.uniform(20.0, 100.0))
            risk = float(rng.uniform(0.01, 0.05))
            collateral = _clamp(equity * risk * rng.uniform(0.7, 1.4), 10.0, 5000.0)
            notional = collateral * lev
            entry_price = _BASE_PRICES[symbol] * math.exp(rng.normal(0, 0.004))

            opened = max(1, int(rng.integers(1, n_steps)))
            hold = int(rng.exponential(18.0)) + 1
            hold = _clamp(hold, 1, 96) if rng.random() < 0.96 else int(rng.integers(1, 200))
            closed = _clamp(opened + hold, opened + 1, n_steps)

            roll = rng.random()
            if roll < 0.30:
                exit_type = "take_profit"
            elif roll < 0.55:
                exit_type = "stop_loss"
            elif roll < 0.985:
                exit_type = "market_close"
            else:
                exit_type = "liquidation"

            if exit_type == "liquidation":
                realized = -collateral
                fee = 0.0
                funding = 0.0
            else:
                if exit_type == "take_profit":
                    ret_bps = float(rng.uniform(25.0, 70.0))
                elif exit_type == "stop_loss":
                    ret_bps = -float(rng.uniform(25.0, 65.0))
                else:
                    ret_bps = float(rng.normal(28.0, 60.0) + regime)
                gross = notional * ret_bps / 10_000.0
                fee = notional * (0.0003 + 0.0006)
                side_charge = 1.0 if side == "long" else -1.0
                funding = notional * side_charge * float(rng.normal(0.00001, 0.000004))
                realized = gross - fee - funding
                realized = _clamp(realized, -collateral, collateral * 8.0)

            if side == "long":
                exit_price = entry_price * (1.0 + realized / max(notional, 1e-9))
            else:
                exit_price = entry_price * (1.0 - realized / max(notional, 1e-9))
            exit_price = max(entry_price * 0.5, exit_price)

            row = {
                "trade_id": str(trade_id),
                "episode": ep,
                "seed_offset": int(ep + 1),
                "symbol": symbol,
                "side": side,
                "opened_at": opened,
                "closed_at": closed,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": notional,
                "leverage": lev,
                "collateral": collateral,
                "equity_before": equity,
                "entry_conviction": float(rng.uniform(0.05, 1.0)),
                "fee": fee,
                "funding": funding,
                "realized_pnl": realized,
                "exit_type": exit_type,
            }
            rows.append(row)
            equity = max(equity + realized, 10.0)
    return rows


def _write_csv(rows, columns: tuple[str, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def generate_ppo(n: int = 200, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    timestep = 0
    for i in range(n):
        timestep = i * 2560
        rows.append({
            "time/fps": float(rng.normal(2400, 260)),
            "time/iterations": i,
            "time/time_elapsed": float(i * 1.06 + rng.normal(0, 0.4)),
            "time/total_timesteps": timestep,
            "rollout/ep_len_mean": float(140 - 25 * min(i / n, 1.0) + rng.normal(0, 3)),
            "rollout/ep_rew_mean": float(2.6 + 1.0 * min(i / n, 1.0) ** 1.2 + rng.normal(0, 0.10)),
            "rollout/success_rate": float(0.20 + 0.35 * min(i / n, 1.0) + rng.normal(0, 0.02)),
            "train/approx_kl": float(0.01 * math.exp(-2.5 * i / n) + rng.normal(0, 0.0003)),
            "train/clip_fraction": float(0.12 * math.exp(-2.0 * i / n) + rng.normal(0, 0.004)),
            "train/clip_range": 0.2,
            "train/entropy_loss": float(1.4 * math.exp(-1.2 * i / n) + rng.normal(0, 0.02)),
            "train/explained_variance": float(0.15 + 0.45 * min(i / n, 1.0) + rng.normal(0, 0.02)),
            "train/learning_rate": float(3e-4 * math.exp(-2.0 * i / n)),
            "train/loss": float(2.2 * math.exp(-1.6 * i / n) + rng.normal(0, 0.03)),
            "train/n_updates": i,
            "train/policy_gradient_loss": float(1.1 * math.exp(-1.5 * i / n) + rng.normal(0, 0.02)),
            "train/std": float(1.5 * math.exp(-1.0 * i / n) + rng.normal(0, 0.02)),
            "train/value_loss": float(0.9 * math.exp(-1.2 * i / n) + rng.normal(0, 0.015)),
        })
    return rows


def generate_sac(n: int = 200, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = i / n
        rows.append({
            "time/fps": float(rng.normal(3800, 320)),
            "time/iterations": i,
            "time/time_elapsed": float(i * 0.72 + rng.normal(0, 0.3)),
            "time/total_timesteps": i * 4096,
            "rollout/ep_len_mean": float(160 - 35 * t + rng.normal(0, 4)),
            "rollout/ep_rew_mean": float(2.8 + 0.9 * min(t, 1.0) ** 1.2 + rng.normal(0, 0.12)),
            "rollout/success_rate": float(0.18 + 0.4 * t + rng.normal(0, 0.02)),
            "train/actor_loss": float(0.8 * math.exp(-1.4 * t) + rng.normal(0, 0.02)),
            "train/critic_loss": float(1.6 * math.exp(-1.3 * t) + rng.normal(0, 0.03)),
            "train/ent_coef": float(0.2 * math.exp(-1.0 * t) + rng.normal(0, 0.003)),
            "train/ent_coef_loss": float(0.05 * math.exp(-1.0 * t) + rng.normal(0, 0.002)),
            "train/learning_rate": float(1e-3 * math.exp(-2.0 * t)),
            "train/loss": float(1.4 * math.exp(-1.2 * t) + rng.normal(0, 0.03)),
            "train/loss/alpha": float(0.18 * (1 - t) ** 2 + rng.normal(0, 0.004)),
            "train/loss/policy": float(0.5 * math.exp(-1.4 * t) + rng.normal(0, 0.015)),
            "train/loss/q1": float(1.1 * math.exp(-1.3 * t) + rng.normal(0, 0.03)),
            "train/loss/q2": float(1.1 * math.exp(-1.3 * t) + rng.normal(0, 0.03)),
            "train/n_updates": i,
            "train/policy/alpha": float(0.2 * math.exp(-0.9 * t) + rng.normal(0, 0.003)),
            "train/policy/log_pi_mean": float(0.9 * math.exp(-1.0 * t) + rng.normal(0, 0.02)),
            "train/value/q_mean": float(1.2 * math.exp(-1.2 * t) + rng.normal(0, 0.03)),
        })
    return rows


def generate_run(
    out_dir: str | Path,
    n_trades: int = 360,
    n_episodes: int = 8,
    n_steps: int = 17280,
    ppo_rows: int = 200,
    sac_rows: int = 200,
    seed: int = 42,
) -> dict:
    """Write a full synthetic run folder in the trading-bot layout."""
    out = Path(out_dir)
    testing = out / "testing"
    training = out / "training"
    testing.mkdir(parents=True, exist_ok=True)
    training.mkdir(parents=True, exist_ok=True)

    rows = generate_trades(n=n_trades, n_episodes=n_episodes, n_steps=n_steps, seed=seed)

    cols = list(NUMERIC_COLUMNS)
    header = ("trade_id", "episode", "symbol", "side", *cols, "seed_offset",
              "equity_before", "exit_type")
    _write_csv(rows, header, testing / "trades.csv")
    _write_csv(generate_ppo(ppo_rows), PPO_COLUMNS, training / "manager_ppo.csv")
    _write_csv(generate_sac(sac_rows), SAC_COLUMNS, training / "worker_sac.csv")
    return {"trades": testing / "trades.csv",
            "manager_ppo": training / "manager_ppo.csv",
            "worker_sac": training / "worker_sac.csv"}