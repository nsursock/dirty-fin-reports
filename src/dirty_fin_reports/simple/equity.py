"""Equity-curve reconstruction from a closed-trade ledger.

The ledger only records closed positions (bar ``opened_at``/``closed_at``).
Two books are rebuilt per episode:

* **realized** — steps by ``realized_pnl`` at each close bar only (flat while
  a position is open).
* **mtm** — mark-to-market while open, using linear price interpolation from
  ``entry_price`` → ``exit_price`` over ``[opened_at, closed_at)``, with fees
  and funding accrued proportionally. At the close bar the MTM book matches
  the realized step (cash absorbs ``realized_pnl``).

The portfolio ``net`` / ``mtm`` curves are the index-aligned mean of the
per-episode books (display aid only — risk ratios should aggregate
per-episode metrics, not treat the mean as one traded account).
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


def _signed_qty(t: dict) -> float:
    entry = float(t.get("entry_price") or 0.0)
    notional = float(t.get("notional") or 0.0)
    if entry <= 0.0 or notional <= 0.0:
        return 0.0
    q = notional / entry
    return q if str(t.get("side") or "").lower() == "long" else -q


def _floating_pnl(t: dict, bar: int) -> float:
    """Linear-interpolated open-position MTM at ``bar`` (before close)."""
    opened = int(t.get("opened_at") or 0)
    closed = int(t.get("closed_at") or 0)
    if closed <= opened:
        return float(t.get("realized_pnl") or 0.0)
    alpha = (bar - opened) / (closed - opened)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    entry = float(t.get("entry_price") or 0.0)
    exit_px = float(t.get("exit_price") or entry)
    mark = entry + alpha * (exit_px - entry)
    fee = float(t.get("fee") or 0.0)
    funding = float(t.get("funding") or 0.0)
    return _signed_qty(t) * (mark - entry) - alpha * (fee + funding)


def _episode_books(
    ep_rows: list[dict], length: int, start: float
) -> dict[str, np.ndarray]:
    realized_bar, fees, funding = by_bar(ep_rows, length)
    realized = np.full(length + 1, start, dtype=float)
    gross = np.full(length + 1, start, dtype=float)
    mtm = np.full(length + 1, start, dtype=float)
    cash = float(start)
    for b in range(1, length + 1):
        realized[b] = realized[b - 1] + realized_bar[b]
        gross[b] = gross[b - 1] + realized_bar[b] + fees[b] + funding[b]
        cash += float(realized_bar[b])
        floating = 0.0
        for t in ep_rows:
            opened = int(t.get("opened_at") or 0)
            closed = int(t.get("closed_at") or 0)
            if opened <= b < closed:
                floating += _floating_pnl(t, b)
        mtm[b] = cash + floating
    return {"net": realized, "gross": gross, "mtm": mtm}


def portfolio_curve(
    rows: list[dict], start: float = 1000.0, n_steps: int | None = None
) -> dict:
    """Reconstruct portfolio realized and MTM books from a ledger.

    Returns 1-indexed ``steps`` (bars ``1..T``) with mean ``net`` (realized),
    ``gross``, and ``mtm`` equity, plus ``per_episode`` curves and ``length``.
    Rows must already be coerced (see ``coerce_ledger``).
    """
    rows = list(rows)
    if not rows:
        return {
            "steps": np.array([], dtype=int),
            "net": np.array([]),
            "gross": np.array([]),
            "mtm": np.array([]),
            "per_episode": {},
            "length": 0,
            "n_episodes": 0,
            "mtm_gap_max_pct": 0.0,
        }
    length = infer_length(rows, n_steps)
    episodes: dict[int, list[dict]] = {}
    for t in rows:
        episodes.setdefault(int(t.get("episode") or 0), []).append(t)

    net = np.zeros(length + 1, dtype=float)
    gross = np.zeros(length + 1, dtype=float)
    mtm = np.zeros(length + 1, dtype=float)
    per_episode: dict[int, dict[str, np.ndarray]] = {}
    gap_max = 0.0
    for ep, ep_rows in episodes.items():
        books = _episode_books(ep_rows, length, start)
        per_episode[int(ep)] = books
        net += books["net"]
        gross += books["gross"]
        mtm += books["mtm"]
        denom = np.maximum(np.abs(books["net"]), 1e-12)
        gap = np.max(np.abs(books["mtm"] - books["net"]) / denom)
        gap_max = max(gap_max, float(gap))
    n_ep = len(episodes)
    net /= n_ep
    gross /= n_ep
    mtm /= n_ep
    steps = np.arange(1, length + 1, dtype=int)
    return {
        "steps": steps,
        "net": net[1:],
        "gross": gross[1:],
        "mtm": mtm[1:],
        "per_episode": per_episode,
        "length": length,
        "n_episodes": n_ep,
        "mtm_gap_max_pct": 100.0 * gap_max,
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
