"""Ledger / trade-history CSV loading, coercion and validation.

The phase-1 schema matches the bot's ``trades.csv`` (written by
``write_ledger``): one row per *closed* position, bar-indexed open/close times.
Extra columns are tolerated; every required column must be present.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "episode",
    "symbol",
    "side",
    "opened_at",
    "closed_at",
    "entry_price",
    "exit_price",
    "notional",
    "leverage",
    "collateral",
    "entry_conviction",
    "fee",
    "funding",
    "realized_pnl",
    "exit_type",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "opened_at",
    "closed_at",
    "entry_price",
    "exit_price",
    "notional",
    "leverage",
    "collateral",
    "entry_conviction",
    "fee",
    "funding",
    "realized_pnl",
)

EXIT_TYPES: tuple[str, ...] = ("market_close", "take_profit", "stop_loss", "liquidation")
SIDES: tuple[str, ...] = ("long", "short")


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if v in ("", "nan", "null", "None"):
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _maybe_int(v) -> int | None:
    f = _maybe_float(v)
    if f is None:
        return None
    return int(round(f))


def load(path: str | Path) -> list[dict]:
    """Read a ledger CSV into raw rows, raising on a missing required column."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: empty CSV (no header row)")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{p}: missing required column(s): {', '.join(missing)}")
        rows = [dict(r) for r in reader]
    return rows


def coerce_ledger(rows: Iterable[dict]) -> list[dict]:
    """Normalize raw rows: typed numbers, trimmed strings, keyed identifiers."""
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        for c in NUMERIC_COLUMNS:
            val = _maybe_float(d.get(c))
            if c in ("opened_at", "closed_at"):
                d[c] = _maybe_int(d.get(c))
            else:
                d[c] = val
        d["episode"] = _maybe_int(d.get("episode")) or 0
        d["trade_id"] = str(d.get("trade_id") or "").strip()
        d["symbol"] = str(d.get("symbol") or "").strip().upper()
        d["side"] = str(d.get("side") or "").strip().lower()
        d["exit_type"] = str(d.get("exit_type") or "").strip().lower()
        out.append(d)
    return out


def validate(rows: Iterable[dict]) -> list[str]:
    """Return a list of descriptive warnings about data-quality problems."""
    warnings: list[str] = []
    for i, r in enumerate(rows):
        tag = f"row {i} (ep{int(r.get('episode', 0) or 0)}/{r.get('trade_id', '?')})"
        if r.get("side") not in SIDES:
            warnings.append(f"{tag}: unexpected side {r.get('side')!r}")
        if r.get("exit_type") not in EXIT_TYPES:
            warnings.append(f"{tag}: unexpected exit_type {r.get('exit_type')!r}")
        clos, open_ = r.get("closed_at"), r.get("opened_at")
        if clos is not None and open_ is not None and clos < open_:
            warnings.append(f"{tag}: closed_at < opened_at ({clos} < {open_})")
        if r.get("closed_at") is None:
            warnings.append(f"{tag}: trade still open (no closed_at)")
        if r.get("leverage") is not None and r.get("leverage", 0) < 0:
            warnings.append(f"{tag}: negative leverage")
        if r.get("notional") is not None and r.get("notional", 0) <= 0:
            warnings.append(f"{tag}: non-positive notional")
        if r.get("entry_price") is not None and r.get("entry_price", 0) <= 0:
            warnings.append(f"{tag}: non-positive entry price")
        pnl, coll = r.get("realized_pnl"), r.get("collateral")
        if pnl is not None and coll is not None and float(pnl) < -float(coll):
            warnings.append(f"{tag}: pnl below -100% of collateral ({pnl} < {-coll})")
    return warnings


def unique_keys(rows: Iterable[dict]) -> list[tuple[int, str]]:
    """Distinct ``(episode, trade_id)`` keys — trade_id may reuse across episodes."""
    return sorted({(int(r.get("episode", 0) or 0), str(r.get("trade_id") or "")) for r in rows})


def to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Typed DataFrame view of the ledger (usual if pandas is available)."""
    return pd.DataFrame(coerce_ledger(rows))