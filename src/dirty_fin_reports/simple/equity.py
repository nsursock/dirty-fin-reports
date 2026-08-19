"""Equity-curve reconstruction from a closed-trade ledger.

The ledger only records closed positions (bar ``opened_at``/``closed_at``), so
the per-account book is rebuilt the same way the bot's ``run_test`` does it:
start each episode at ``initial_balance`` and step the book by a trade's
``realized_pnl`` at its close bar. ``gross`` re-adds cumulative open fees and
funding. The portfolio curve is the index-aligned mean of the per-episode
curves (one account per episode is assumed — the ledger carries no
per-env/slot breakdown).

The timeline length is ``max(closed_at)`` unless ``n_steps`` is given.
"""

from __future__ import annotations

import numpy as np

from .ledger import coerce_ledger


def infer_length(rows: list[dict], n_steps: int | None = None) -> int:
    """Bar count of the reconstructed timeline."""
    closes = [
        int(t["closed_at"])
        for t in rows
        if t.get("closed_at") is not None
    ]
    inferred = max(closes) if closes else 0
    if n_steps is not None:
        return max(int(n_steps), inferred)
    return max(inferred, 1)


def by_bar(rows: list[dict], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per close-bar ``(realized, fees, funding)`` for one episode."""
    realized = np.zeros(length + 1, dtype=float)
    fees = np.zeros(length + 1, dtype=float)
    funding = np.zeros(length + 1, dtype=float)
    for t in rows:
        b = int(t.get("closed_at") or 0)
        if not (1 <= b <= length):
            continue
        realized[b] += float(t.get("realized_pnl") or 0.0)
        fees[b] += float(t.get("fee") or 0.0)
        funding[b] += float(t.get("funding") or 0.0)
    return realized, fees, funding


def portfolio_curve(
    rows: list[dict], start: float = 1000.0, n_steps: int | None = None
) -> dict:
    """Reconstruct the portfolio book curve from a ledger.

    Returns 1-indexed ``steps`` (bars ``1..T``) with the corresponding ``net``
    and ``gross`` equity, plus ``per_episode`` curves and ``length``. Rows must
    already be coerced (see ``coerce_ledger``).
    """
    rows = list(rows)
    if not rows:
        return {"steps": np.array([], dtype=int), "net": np.array([]), "gross": np.array([]),
                "per_episode": {}, "length": 0}
    length = infer_length(rows, n_steps)
    episodes: dict[int, list[dict]] = {}
    for t in rows:
        episodes.setdefault(int(t.get("episode") or 0), []).append(t)

    net = np.zeros(length + 1, dtype=float)
    gross = np.zeros(length + 1, dtype=float)
    per_episode: dict[int, dict[str, np.ndarray]] = {}
    for ep, ep_rows in episodes.items():
        realized, fees, funding = by_bar(ep_rows, length)
        book = np.full(length + 1, start, dtype=float)
        base = np.full(length + 1, start, dtype=float)
        for b in range(1, length + 1):
            book[b] = book[b - 1] + realized[b]
            base[b] = base[b - 1] + realized[b] + fees[b] + funding[b]
        per_episode[int(ep)] = {"net": book, "gross": base}
        net += book
        gross += base
    n_ep = len(episodes)
    net /= n_ep
    gross /= n_ep
    steps = np.arange(1, length + 1, dtype=int)
    return {
        "steps": steps,
        "net": net[1:],
        "gross": gross[1:],
        "per_episode": per_episode,
        "length": length,
        "n_episodes": n_ep,
    }


def per_symbol_curves(
    rows: list[dict], start: float = 1000.0, n_steps: int | None = None
) -> tuple[list[str], np.ndarray]:
    """Per-symbol equity curves for overlay traces.

    Each symbol's curve is ``start`` plus its cumulative realized PnL (by close
    bar) averaged across episodes, so the overlay sits on the same scale as the
    portfolio book.
    """
    rows = list(rows)
    if not rows:
        return [], np.zeros((0, 0))
    length = infer_length(rows, n_steps)
    syms = sorted({t["symbol"] for t in rows})
    n_ep = max(len({int(t.get("episode") or 0) for t in rows}), 1)
    idx = {s: k for k, s in enumerate(syms)}
    realized = np.zeros((len(syms), length + 1), dtype=float)
    for t in rows:
        b = int(t.get("closed_at") or 0)
        if 1 <= b <= length:
            realized[idx[t["symbol"]], b] += float(t.get("realized_pnl") or 0.0)
    curves = start + np.cumsum(realized, axis=1)[:, 1:] / n_ep
    return syms, curves


def equity_curves(
    ledger_path=None, rows: list[dict] | None = None, start: float = 1000.0,
    n_steps: int | None = None,
) -> dict:
    """Load (path or rows) and rebuild the equity curves for a ledger."""
    if rows is None:
        if ledger_path is None:
            raise ValueError("either ledger_path or rows is required")
        from .ledger import load

        rows = coerce_ledger(load(ledger_path))
    return portfolio_curve(coerce_ledger(rows), start=start, n_steps=n_steps)