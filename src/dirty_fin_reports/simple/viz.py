"""Minimal plotly theme scaffolding for figure rendering.

Palettes are ported from the dirty-trading-bot report engine so the phase-1
figures share the synthwave / cyberpunk look the user visually recognizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import plotly.graph_objects as go

THEMES: dict[str, dict[str, str]] = {
    # Ported verbatim from dirty-mkt-data (the trading bot's theme source).
    "synthwave": dict(
        background="#241B2F", plot_background="#1B1424", grid="#3A2E52",
        text="#F2EFFF", up="#FF2E88", down="#22E4FF", accent="#B967FF",
    ),
    "ghibli": dict(
        background="#F3EEDC", plot_background="#FBF8EE", grid="#E4DCC3",
        text="#3D3A2A", up="#5F9266", down="#D97757", accent="#8FB0C9",
    ),
    "valorant": dict(
        background="#0F1923", plot_background="#141E2A", grid="#1F2F3B",
        text="#ECE8E1", up="#FF4655", down="#40C4A6", accent="#FFD166",
    ),
    "cyberpunk": dict(
        background="#0b0b16", plot_background="#121222", grid="#1f1f38",
        text="#e0e0e8", up="#00f3ff", down="#ff0055", accent="#00ff66",
    ),
    "tokyo_midnight": dict(
        background="#1a1b26", plot_background="#24283b", grid="#414868",
        text="#c0caf5", up="#7aa2f7", down="#f7768e", accent="#bb9af7",
    ),
    "nordic_frost": dict(
        background="#f4f7f6", plot_background="#ffffff", grid="#e2e8e6",
        text="#2e4057", up="#048a81", down="#858ae3", accent="#ff8a5b",
    ),
    "brutalist_terminal": dict(
        background="#000000", plot_background="#000000", grid="#1a1a1a",
        text="#00ff00", up="#00ff00", down="#ff3333", accent="#00ffff",
    ),
}


def resolve_theme(theme: str) -> str:
    if theme == "random":
        return random.choice(sorted(THEMES))
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r} (have {sorted(THEMES)})")
    return theme


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


@dataclass
class Palette:
    theme: str
    bg: str
    panel: str
    ink: str
    grid: str
    up: str
    down: str
    accent: str
    muted: str
    up_soft: str
    down_soft: str
    accent_soft: str

    @classmethod
    def of(cls, theme: str) -> "Palette":
        name = resolve_theme(theme)
        t = THEMES[name]
        r, g, b = (int(t["background"][i : i + 2], 16) for i in (1, 3, 5))
        dark = (r + g + b) / 3.0 < 128.0
        muted = "#9AA4B2" if dark else "#5C6570"
        return cls(
            theme=name,
            bg=t["background"], panel=t["plot_background"], ink=t["text"],
            grid=t["grid"], up=t["up"], down=t["down"], accent=t["accent"],
            muted=muted,
            up_soft=_rgba(t["up"], 0.22), down_soft=_rgba(t["down"], 0.28),
            accent_soft=_rgba(t["accent"], 0.20),
        )


_FONT = dict(family="JetBrains Mono, Menlo, Monaco, Consolas, monospace")


def base_layout(c: Palette, **extra) -> dict:
    layout = dict(
        paper_bgcolor=c.bg,
        plot_bgcolor=c.panel,
        font=dict(family=_FONT["family"], color=c.ink),
        title_font=dict(size=18, color=c.up, family=_FONT["family"]),
        margin=dict(l=60, r=32, t=80, b=56),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=11, color=c.ink)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=c.panel, bordercolor=c.up,
                        font=dict(color=c.ink, size=11, family=_FONT["family"])),
    )
    layout.update(extra)
    return layout


def style_axes(fig, rows: int, cols: int, c: Palette, pct_rows=None, pct_cols=None):
    pct_rows = pct_rows or set()
    pct_cols = pct_cols or set()
    for r in range(1, rows + 1):
        for cc in range(1, cols + 1):
            fig.update_xaxes(
                row=r, col=cc, showgrid=True, gridcolor=c.grid, gridwidth=1,
                zeroline=False, showline=True, linecolor=c.grid, linewidth=1,
                tickfont=dict(size=10, color=c.muted),
                title_font=dict(size=11, color=c.muted), automargin=True,
                tickformat=".1%" if (r, cc) in pct_cols else None,
            )
            fig.update_yaxes(
                row=r, col=cc, showgrid=True, gridcolor=c.grid, gridwidth=1,
                zeroline=False, showline=True, linecolor=c.grid, linewidth=1,
                tickfont=dict(size=10, color=c.muted),
                title_font=dict(size=11, color=c.muted), automargin=True,
                tickformat=".1%" if (r, cc) in pct_rows else None,
            )


def write_png(fig: go.Figure, path, width=1280, height=920, scale=2):
    fig.write_image(str(path), format="png", width=width, height=height, scale=scale)
    return path