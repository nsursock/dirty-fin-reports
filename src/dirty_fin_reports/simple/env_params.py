"""Single source of truth for trading-environment parameters.

Fees, slippage, leverage and liquidation settings were previously hardcoded in
three places (``synth.generate_trades``, ``breakdown.bot_cfg`` and
``configs/config.yaml``'s ``env`` block) and had already drifted (e.g. the
synthetic leverage draw used a different ceiling than ``bot_cfg``). ``bot_cfg``
and the synthetic generator both consume :class:`TradingEnvParams` so there is
exactly one definition of each constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TradingEnvParams:
    lev_min: float = 20.0
    lev_max: float = 100.0
    risk_min: float = 0.01
    risk_max: float = 0.05
    open_fee_rate: float = 0.0003
    close_fee_rate: float = 0.0006
    slippage_bps: float = 1.0
    holding_fee_daily: float = 0.00015
    bars_per_day: int = 288
    liquidation_fee_rate: float = 0.003
    liq_threshold_base: float = 0.90
    liq_threshold_floor: float = 0.67
    liq_threshold_ref_lev: float = 2.0
    liq_threshold_hi_lev: float = 150.0
    liq_threshold_lo_lev: float = 1.0
    min_collateral: float = 10.0
    max_collateral: float = 10000.0
    initial_balance: float = 1000.0
    margin_mode: str = "isolated"
    trade_knob: float = 2.5
    n_envs_per_symbol: int = 256

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TradingEnvParams":
        """Build from a (possibly partial) dict, e.g. the ``env`` YAML block."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in known})
