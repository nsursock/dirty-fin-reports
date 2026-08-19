# dirty-fin-reports

Financial + ML-health reporting engine for a synthetic trading bot. Phase 1 is a
simple report pipeline: it synthesizes a plausible ledger of leveraged trades,
computes trading + risk metrics, validates them against plausibility bounds, and
renders PNG dashboards (`bot-performance-*`, `trade-anatomy-*`) plus a
machine-readable `report.json`.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
```

kaleido (PNG export) needs its Chromium binary; if `write_image` fails, install it via
your package manager or let `venv/bin/python -m dirty_fin_reports.simple.main` retry
the kaleido graphviz fallback.

## Configuration

All knobs live in `configs/config.yaml`:

- `synth.seeds` — one run folder per seed. The default is `[3, 42, 7, 101, 999]`.
- `plausibility.*` — bounds that drive the per-metric `plausible / flagged /
  implausible` verdict and color-flag out-of-range stats on the figures.
- `report.timeframe`, `report.initial_balance`, `report.reporting_freq`,
  `report.rf_annual`, `report.n_steps`.
- `report.start_date` — date anchor for the equity / trade-returns / drawdown x-axes.
- `report.tick_tilt` (`on/off`), `report.tick_angle` (degrees),
  `report.tick_direction` (`down` | `up`) — rotation of the date-axis tick labels.

## Generate the five-seed figures

Run the pipeline from the project root with the venv interpreter:

```bash
PYTHONPATH=src venv/bin/python -m dirty_fin_reports.simple.main configs/config.yaml
```

Each seed gets a timestamped folder:

```
runs/20260819-223039-synth-s42/
├── testing/
│   ├── trades.csv                 # ledger (coerced)
│   ├── breakdown.txt              # text report
│   ├── bot-performance-<verdict>.png
│   └── trade-anatomy-<verdict>.png
├── training/
│   ├── manager_diag.png           # PPO health grid
│   └── worker_diag.png            # SAC health grid
└── report.json                    # metrics, plausibility checks, verdict
```

The `<verdict>` disk suffix is one of `plausible`, `flagged` or `implausible`,
mirroring the run-level aggregate of all plausibility checks.

## Tests

```bash
venv/bin/python -m pytest -q
```

## Project layout

- `src/dirty_fin_reports/simple/` — config, synth data generation, ledger handling,
  metrics, plausibility checks, figures, report assembly, CLI entry (`main.py`).
- `tests/` — pytest suite covering config parsing, synthesis, ledger validation,
  metrics, report assembly and figure rendering.
- `configs/config.yaml` — pipeline configuration.
- `sota/` — working design notes (architecture, synthetic data, reporting, etc.).