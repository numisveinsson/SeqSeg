"""Violin plot utilities in Nature journal style.

Nature journal figure guidelines (https://research-figure-guide.nature.com/):
- Font: sans-serif (Arial/Helvetica); DPI: 600+ for combination; vector (PDF) preferred
- Colorblind-safe palettes
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import mannwhitneyu

if TYPE_CHECKING:
	from matplotlib.axes import Axes

# Nature journal figure guidelines
NATURE_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
}

# Colorblind-safe palette (blue-orange; avoid red-green)
NATURE_COLORS_DEFAULT = {'Reference vs Others': '#0173b2', 'Others vs Others': '#de8f05'}

# Extended palette for 3+ groups (colorblind-safe: blue, orange, green, teal, purple, pink)
NATURE_COLORS_EXTENDED = [
    '#0173b2',  # blue
    '#de8f05',  # orange
    '#029e73',  # green
    '#56b4e9',  # light blue
    '#cc78bc',  # pink
    '#ca9161',  # tan
    '#949494',  # gray
    '#ece133',  # yellow
]


def get_nature_colors(n: int) -> list[str]:
    """Return n colorblind-safe colors from the extended palette. Cycles if n > len(palette)."""
    palette = NATURE_COLORS_EXTENDED
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def apply_nature_style() -> None:
    """Apply Nature journal figure defaults to matplotlib rcParams."""
    plt.rcParams.update(NATURE_RCPARAMS)


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return 'p < 0.001'
    if p < 0.01:
        return f'p = {p:.3f}'
    return f'p = {p:.2f}'


def add_wilcoxon_bracket(
    ax,
    vals1: np.ndarray,
    vals2: np.ndarray,
    pos1: float,
    pos2: float,
) -> None:
    """Run Wilcoxon rank-sum (Mann-Whitney U) and draw significance bracket on axes."""
    if len(vals1) < 2 or len(vals2) < 2:
        return
    try:
        _, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
    except Exception:
        return
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    y_bracket = ymax + 0.04 * y_span
    ax.plot(
        [pos1, pos1, pos2, pos2],
        [y_bracket, y_bracket + 0.02 * y_span, y_bracket + 0.02 * y_span, y_bracket],
        color='black',
        linewidth=0.5,
    )
    ax.text(
        (pos1 + pos2) / 2,
        y_bracket + 0.03 * y_span,
        format_pvalue(p),
        ha='center',
        va='bottom',
        fontsize=5,
    )
    ax.set_ylim(ymin, y_bracket + 0.08 * y_span)


def draw_violin_ax(
    ax,
    vals_by_group: list[np.ndarray],
    positions: list[int],
    labels: list[str],
    colors: dict[str, str] | list[str],
    ylabel: str,
    *,
    group_order: list[str] | None = None,
    set_ylim: Callable[["Axes"], None] | None = None,
    add_wilcoxon: bool = True,
    subplot_label: str | None = None,
    xtick_rotation: float = 15,
) -> None:
    """Draw a single violin plot on the given axes in Nature journal style.

    Args:
        ax: Matplotlib axes to draw on.
        vals_by_group: List of 1D arrays, one per group (same order as positions).
        positions: X positions for each violin.
        labels: X-axis tick labels for each position.
        colors: Either a dict mapping group name -> color, or a list of colors
            (one per position). Fallback color '#666666' if group not in dict.
        ylabel: Y-axis label.
        group_order: If colors is a dict, order of group names for lookup.
            Default: use positions index to index into group_order.
        set_ylim: Optional callable(ax) to set y-axis limits.
        add_wilcoxon: If True and exactly 2 groups, add Mann-Whitney U bracket.
        subplot_label: Optional label (e.g. 'a', 'b') in top-left.
        xtick_rotation: Rotation of x-tick labels in degrees.
    """
    if subplot_label:
        ax.text(
            -0.28, 1.05, subplot_label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight='bold',
            va='top',
            ha='right',
        )

    if not vals_by_group:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    parts = ax.violinplot(
        vals_by_group,
        positions=positions,
        showmeans=False,
        showmedians=True,
    )
    for i, pc in enumerate(parts['bodies']):
        if isinstance(colors, dict) and group_order is not None:
            pos = positions[i] if i < len(positions) else i
            g = group_order[pos] if pos < len(group_order) else None
            c = colors.get(g, '#666666') if g else '#666666'
        elif isinstance(colors, (list, tuple)):
            c = colors[i] if i < len(colors) else '#666666'
        else:
            c = '#666666'
        pc.set_facecolor(c)
        pc.set_alpha(0.75)
        pc.set_edgecolor(c)
        pc.set_linewidth(0.5)
    if parts.get('cmedians'):
        parts['cmedians'].set_linewidth(0.75)
        parts['cmedians'].set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha='right')
    ax.set_ylabel(ylabel)
    if set_ylim:
        set_ylim(ax)
    if add_wilcoxon and len(vals_by_group) == 2:
        add_wilcoxon_bracket(ax, vals_by_group[0], vals_by_group[1], positions[0], positions[1])
    ax.grid(axis='y', alpha=0.25, linewidth=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_violin_grid(
    data_per_panel: list[tuple[list[np.ndarray], list[int], list[str]]],
    ylabels: list[str],
    colors: dict[str, str] | list[str],
    group_order: list[str] | None,
    out_path: str,
    *,
    figsize_per_panel: tuple[float, float] = (2.4, 2.0),
    ncols: int = 3,
    format: str = 'png',
    set_ylim: Callable[["Axes", str], None] | None = None,
    metric_keys: list[str] | None = None,
) -> None:
    """Create a grid of violin subplots and save to file.

    Args:
        data_per_panel: List of (vals_by_group, positions, labels) per subplot.
        ylabels: Y-axis label per panel.
        colors: Color mapping or list (see draw_violin_ax).
        group_order: Group order for color lookup.
        out_path: Output file path.
        figsize_per_panel: Size per subplot.
        ncols: Number of columns in grid.
        format: 'png' or 'pdf'.
        set_ylim: Optional callable(ax, metric_key) to set y-limits.
        metric_keys: If set_ylim is used, metric key per panel for the callback.
    """
    apply_nature_style()
    n = len(data_per_panel)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ((vals_by_group, positions, labels), ylabel) in enumerate(zip(data_per_panel, ylabels)):
        ax = axes[idx]
        ylim_fn = None
        if set_ylim and metric_keys and idx < len(metric_keys):
            mk = metric_keys[idx]
            ylim_fn = lambda a, m=mk: set_ylim(a, m)
        draw_violin_ax(
            ax,
            vals_by_group,
            positions,
            labels,
            colors,
            ylabel,
            group_order=group_order,
            set_ylim=ylim_fn,
            subplot_label=chr(97 + idx),
        )

    for j in range(len(data_per_panel), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    dpi = 600 if format == 'png' else None
    base, _ = os.path.splitext(out_path)
    save_path = f'{base}.{format}' if format else out_path
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=format or 'png')
    plt.close(fig)


def save_violin_figure(
    fig,
    out_path: str,
    format: str = 'png',
    dpi: int = 600,
) -> None:
    """Save figure with Nature-style defaults (600 DPI for PNG)."""
    base, _ = os.path.splitext(out_path)
    save_path = f'{base}.{format}' if format else out_path
    fig.savefig(
        save_path,
        dpi=dpi if format == 'png' else None,
        bbox_inches='tight',
        format=format or 'png',
    )
    plt.close(fig)
