"""Shared heatmap drawing for the validation figures.

Extracted from `naive_prevalence.py` so `symptom_embed.py` can produce figures
in the same format without importing a sibling CLI script. Both analyses answer
"condition x symptom", so both read far better as a matrix than as bars: the
matrix shows the off-diagonal, which is where most of what we have learned
actually lives (a case's base rate, leakage into a neighbouring domain).

Three views per analysis, always in this order, because the third is only
interpretable after the second:

  rate       what each condition produces
  base rate  what each CASE produces with nothing injected
  contrast   condition minus that case's own control

The value formatter differs between analyses — the lexicon reports fractions of
sessions, the embedding reports z — so callers pass `fmt`.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two-sided views (a contrast) need a diverging map centred on zero; one-sided
# views (a rate) need a sequential one. Using the same map for both makes a
# contrast of -20% look like a rate of 20%.
SEQUENTIAL = "Reds"
DIVERGING = "RdBu_r"


def heat(ax, grid, *, title, xlabel, targets=None, mark_control=True,
         cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, fmt="{:.0%}"):
    """One heatmap panel. Returns the image, for the caller's colorbar.

    `targets` maps a row label to the column that row's condition injected, so
    the cell carrying the actual claim can be outlined. Without it a reader has
    no way to tell which of nine numbers in a row is the one being asserted.
    """
    image = ax.imshow(grid.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=8)

    # White text only where the cell is dark enough to need it. On a diverging
    # map that is either end, not just the top.
    dark = 0.55 * max(abs(vmin), abs(vmax))
    # Only prepend a sign if the caller's format does not already carry one,
    # otherwise "{:+.2f}" renders as "++0.01".
    explicit_sign = "+" in fmt.split(":")[-1]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            if np.isnan(value):
                continue
            sign = "+" if (vmin < 0 and value > 0 and not explicit_sign) else ""
            ax.text(j, i, f"{sign}{fmt.format(value)}", ha="center", va="center",
                    fontsize=7, color="white" if abs(value) > dark else "0.25")

    if targets:
        columns = list(grid.columns)
        for i, row in enumerate(grid.index):
            target = targets.get(row)
            if row == "no_symptoms" or target not in columns:
                continue
            ax.add_patch(plt.Rectangle(
                (columns.index(target) - 0.5, i - 0.5), 1, 1, fill=False,
                edgecolor="tab:blue", lw=2.2, zorder=3))

    if mark_control and len(grid.index) and grid.index[0] == "no_symptoms":
        ax.axhline(0.5, color="0.2", lw=1.6, ls="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    return image


def draw_panels(panels, grid_of, out_path, *, title, ylabel, bar_label,
                xlabel="symptom", targets_of=None, cmap=SEQUENTIAL,
                vmin=0.0, vmax=1.0, height=7.0, mark_control=True,
                fmt="{:.0%}"):
    """One figure, one panel per entry in `panels`.

    `panels` is whatever splits this analysis: the two instruments for the
    lexicon, the two models for the embedding. `grid_of(panel)` returns the
    matrix; `targets_of(panel)` the row->column map for outlining, or None.
    """
    panels = list(panels)
    fig, axes = plt.subplots(1, max(len(panels), 1), figsize=(9.5 * len(panels), height),
                             squeeze=False)
    drew = False
    for ax, panel in zip(axes[0], panels, strict=True):
        grid = grid_of(panel)
        if grid is None or grid.empty:
            ax.set_visible(False)
            continue
        drew = True
        image = heat(ax, grid, title=str(panel), xlabel=xlabel,
                     targets=targets_of(panel) if targets_of else None,
                     mark_control=mark_control, cmap=cmap, vmin=vmin, vmax=vmax,
                     fmt=fmt)
        if panel == panels[0]:
            ax.set_ylabel(ylabel)
        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label(bar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
