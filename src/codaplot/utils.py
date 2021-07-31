from typing import Dict, Tuple, Literal
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def warn(s):
    print(f"\033[1;101m\n\nWARNING: {s}\n\n\033[0m")


class FacetGridAxes:
    def __init__(
        self,
        n_plots: int,
        n_cols: int,
        figsize: Tuple[float, float],
        figure_kwargs: Dict = None,
        constrained_layout_pads: Dict = None,
    ) -> None:

        if figure_kwargs is None:
            figure_kwargs = dict(constrained_layout=True, dpi=180)
        else:
            assert "figsize" not in figure_kwargs

        if constrained_layout_pads is None:
            constrained_layout_pads = dict(h_pad=0, w_pad=0, hspace=0, wspace=0)

        n_rows = int(np.ceil(n_plots / n_cols))
        fig = plt.figure(**figure_kwargs, figsize=figsize)
        if figure_kwargs.get("constrained_layout", False):
            fig.set_constrained_layout_pads(**constrained_layout_pads)

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        axes = np.empty((n_rows, n_cols), dtype="object")
        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j + 1 <= n_plots:
                    axes[i, j] = fig.add_subplot(gs[i, j])

        big_ax = fig.add_subplot(gs[:, :], frameon=False)
        big_ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
            labelsize=0,
            length=0,
        )

        self.big_ax = big_ax
        self.axes = axes
        self.axes_flat = axes.flatten()
        self.fig = fig

    def add_y_marginlabel(self, label):
        self._add_axeslabels_to_facet_grid_with_hidden_full_figure_axes(
            axis="y", label=label
        )

    def add_x_marginlabel(self, label):
        self._add_axeslabels_to_facet_grid_with_hidden_full_figure_axes(
            axis="x", label=label
        )

    def _add_axeslabels_to_facet_grid_with_hidden_full_figure_axes(self, axis, label):
        add_margin_label_via_encompassing_big_ax(
            fig=self.fig, axes=self.axes, big_ax=self.big_ax, axis=axis, label=label
        )


def add_margin_label_via_encompassing_big_ax(
    fig: Figure, axes: np.ndarray, big_ax: Axes, axis: Literal["x", "y"], label: str
):
    axes_flat = axes.flatten()
    fig.draw(fig.canvas.get_renderer())

    method_name = f"get_{axis}ticklabels"

    # quickfix:
    # would be more correct to check text width instead of label length
    max_ticklabel_length = 0
    max_ticklabel = ""
    for ax in axes_flat:
        for ticklabel in getattr(ax, method_name)():
            s = ticklabel.get_text()
            if len(s) > max_ticklabel_length:
                max_ticklabel_length = len(s)
                max_ticklabel = s
    if not max_ticklabel:
        raise ValueError("No ticklabels found")

    max_ticklabel_width_pt = (
        get_text_width_inch(
            s=max_ticklabel,
            size=mpl.rcParams[f"{axis}tick.labelsize"],
            ax=axes_flat[-1],
            draw_figure=False,
        )
        * 72
    )

    labelpad = (
        max_ticklabel_width_pt
        + mpl.rcParams[f"{axis}tick.major.size"]
        + mpl.rcParams[f"{axis}tick.major.pad"]
        + mpl.rcParams["axes.labelpad"]
    )

    big_ax.set_ylabel(label, labelpad=labelpad)


def get_text_width_inch(s, size, ax, fontfamily=None, draw_figure=True):
    r = ax.figure.canvas.get_renderer()
    if draw_figure:
        ax.figure.draw(r)
    if not s:
        raise ValueError(
            "s is an emtpy string - did you draw the figure prior to calling this function?"
        )
    # get window extent in display coordinates
    text_kwargs = dict(fontsize=size)
    if fontfamily:
        text_kwargs |= dict(fontfamily=fontfamily)
    artist = ax.text(0, 0, s, **text_kwargs)
    bbox = artist.get_window_extent(renderer=r)
    data_coord_bbox = bbox.transformed(ax.figure.dpi_scale_trans.inverted())
    artist.remove()
    # data_coord_bbox.height
    return data_coord_bbox.width


# either there is a bug in ScalarFormatter, or I have an issue locally, perhaps with the locale?
# anyhoo, I get trailing zeros in the offset label, and here is a quick fix for that:
class ScalarFormatterQuickfixed(mticker.ScalarFormatter):
    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        s = super().get_offset()
        res = re.sub(r"0+e", "e", s.rstrip("0"))
        return res


def find_offset(ax=None, xlim=None):
    """Find offset for usage with ScalarFormatter

    Works by comparing xlim[0] and xlim[1] until a differing position is found.
    Everything starting from this position is set to 0 to give the offset


    Parameters
    ----------
    specify either ax OR xlim. if ax is specified, ax.get_xlim() is used

    """
    if xlim != None:
        xmin, xmax = xlim
    else:
        xmin, xmax = ax.get_xlim()
    for i, (s1, s2) in enumerate(zip(str(xmin), str(xmax))):
        print(s1, s2)
        if s1 != s2:
            res = str(xmin)[:i]
            a1, *a2 = str(xmin)[i:].split(".")
            res += "0" * len(a1)
            if a2:
                res += "." + "0" * len(a2)
            break
    else:
        raise ValueError("No offset found")
    return float(res)
