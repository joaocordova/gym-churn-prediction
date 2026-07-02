"""Shared Plotly theming and figure persistence.

One visual system for every chart in the project: a validated categorical
palette (colorblind-safe adjacent-pair separation), a single-hue sequential
ramp for magnitude, a blue↔red diverging pair for polarity (SHAP), recessive
gridlines and system-native typography. Figures are written as interactive
HTML under ``assets/`` and, when the ``kaleido`` engine is available, as PNG
snapshots under ``assets/img/`` for README embedding.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from gym_churn.logging_utils import get_logger

log = get_logger(module="plotting")

# Validated categorical palette (fixed order — the ordering is the CVD-safety
# mechanism; never cycle or re-sort it).
SERIES = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

SEQUENTIAL_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

DIVERGING_MID = "#f0efec"
POSITIVE = "#e34948"   # pushes churn risk up
NEGATIVE = "#2a78d6"   # pushes churn risk down

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=SERIES,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        title=dict(font=dict(color=INK, size=17), x=0.02, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(
            gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
            tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
            tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY),
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
        ),
        margin=dict(l=70, r=30, t=70, b=60),
        hoverlabel=dict(font=dict(family=FONT)),
    )
)

pio.templates["gym"] = _TEMPLATE


def themed_figure(**layout_kwargs) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="gym", **layout_kwargs)
    return fig


def apply_theme(fig: go.Figure, **layout_kwargs) -> go.Figure:
    fig.update_layout(template="gym", **layout_kwargs)
    return fig


def save_figure(
    fig: go.Figure, name: str, assets_dir: str | Path, img_dir: str | Path | None = None
) -> Path:
    """Persist a figure as interactive HTML (+ PNG snapshot when possible)."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    html_path = assets_dir / f"{name}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    if img_dir is not None:
        img_dir = Path(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(img_dir / f"{name}.png", width=980, height=560, scale=2)
        except Exception as exc:  # kaleido missing or headless failure
            log.warning("PNG export skipped for {name}: {exc}", name=name, exc=exc)

    log.info("Figure saved: {path}", path=str(html_path))
    return html_path
