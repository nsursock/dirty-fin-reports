"""CLI entrypoint: drive the phase-1 pipeline from ``configs/config.yaml``.

Usage:
  python -m dirty_fin_reports.simple.main [config.yaml] [run_dir]

With no ``run_dir`` a fresh synthetic run is generated per ``synth.seeds`` (or
per ``synth.seed``) into ``runs/<ts>-synth-s<seed>/`` and reported. With a
``run_dir`` the existing run (``testing/trades.csv`` + ``training/*.csv``) is
reported instead. All report parameters come from the YAML: theme, overlays,
cadence, plausibility bounds and synthetic-generation knobs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .config import load_project

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _DEFAULT_CONFIG
    rest = list(argv)
    if argv and Path(argv[0]).suffix in (".yaml", ".yml"):
        config_path = Path(argv[0])
        rest = argv[1:]
    run_dir = Path(rest[0]) if rest else None

    if not config_path.exists():
        config_path = Path("configs/config.yaml")
    project = load_project(config_path)
    cfg = project.report
    if cfg.n_steps is None:
        cfg.n_steps = project.synth.n_steps

    run_dirs: list[Path] = []
    if run_dir is None:
        from .synth import generate_run

        ts = f"{datetime.now():%Y%m%d-%H%M%S}"
        for seed in project.synth.seed_list:
            rd = Path(f"runs/{ts}-synth-s{seed}")
            generate_run(rd, n_trades=project.synth.n_trades,
                         n_episodes=project.synth.n_episodes,
                         n_steps=project.synth.n_steps,
                         ppo_rows=project.synth.ppo_rows,
                         sac_rows=project.synth.sac_rows,
                         seed=seed, env=project.env_params)
            run_dirs.append(rd)
    else:
        run_dirs = [run_dir]

    from .report import run_reporter

    for rd in run_dirs:
        r = run_reporter(rd, out_dir=rd, config=cfg, theme=project.theme,
                         overlays=project.overlays, plausibility=project.plausibility)
        print(f"run: {rd}")
        print(f"status: {r['plausibility']['status']} ({r['plausibility']['counts']})")
        print(f"breakdown: {r['breakdown']}")
        print(f"figures: {r['figure1']} {r['figure2']}")
        if "manager_diag_figure" in r:
            print(f"diags: {r['manager_diag_figure']} {r['worker_diag_figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())