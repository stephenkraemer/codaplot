# # Imports

import os
from typing import Any, Dict, Literal, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import codaplot as co
import codaplot.utils as coutils

# # jointplot


def jointplot(
    xscore,
    yscore,
    xlabel,
    ylabel,
    freq_label: str,
    hist2d_kwargs: Optional[Dict[str, Any]] = None,
    x_hist1d_kwargs: Optional[Dict[str, Any]] = None,
    y_hist1d_kwargs: Optional[Dict[str, Any]] = None,
    xlim=None,
    ylim=None,
    pcc_pos: Optional[
        Union[Literal["title", "upper center"], Tuple[float, float, str, str]]
    ] = "title",
    show_pvalue=False,
    dist_axes_height_and_width_in=6 / 2.54,
    hist_height=1 / 2.54,
    cbar_height=0.3 / 2.54,
    cbar_spacer_height=0.2 / 2.54,
) -> Tuple[Figure, Dict[str, Axes]]:
    """jointplot

    the plot uses equal aspect ratio for the scatterplot axes

    Parameters
    ----------
    dist_axes_height_and_width_in
        2d dist axes is square
    hist_height
        size of the hist axes not aligned with the 2d axes
    cbar_height
    cbar_spacer_height
    pcc_pos
       if not None, place as title of 2D axes ('title'), aligned to the top of the 2D axes ('upper center'), or at (x, y, ha, va) position
    hist2d_kwargs
       not: rasterized
    x/y_hist1d_kwargs
       not: rasterized, orientiation

    Returns
    -------
    Figure, Dict: 'main'|'x_hist'|'y_hist'|'cbar -> Axes
    """

    # %%
    pd.testing.assert_index_equal(xscore.index, yscore.index)
    if x_hist1d_kwargs is None:
        x_hist1d_kwargs = {}
    if y_hist1d_kwargs is None:
        y_hist1d_kwargs = {}
    if hist2d_kwargs is None:
        hist2d_kwargs = {}
    assert not {"rasterized"}.intersection(set(hist2d_kwargs.values()))
    assert not {"rasterized", "orientation"}.intersection(set(x_hist1d_kwargs.values()))
    assert not {"rasterized", "orientation"}.intersection(set(y_hist1d_kwargs.values()))

    """
    Note that if the fig height /  fig width ratio is not chosen appropriately (such that the space for the 2d axes is square), there will be a lot of whitespace introduced by CL. It could be that the figure can also be arranged with little whitespace if these ratios are off, but that this is just not handled optimally by CL atm, see https://github.com/matplotlib/matplotlib/pull/17246 and https://github.com/mwaskom/seaborn/issues/2051. I am not sure if thats the case
    """

    height_ratios = np.array(
        (hist_height, dist_axes_height_and_width_in, cbar_spacer_height, cbar_height)
    )
    width_ratios = np.array((dist_axes_height_and_width_in, hist_height))
    fig_height = height_ratios.sum()
    fig_width = width_ratios.sum()

    fig = plt.figure(
        constrained_layout=True,
        dpi=180,
        figsize=(fig_width, fig_height),  # type: ignore
    )
    fig.set_constrained_layout_pads(hspace=0.05, wspace=0.05, h_pad=0, w_pad=0)
    axd = fig.subplot_mosaic(
        [
            ["x_hist", "."],
            ["main", "y_hist"],
            [".", "."],
            ["cbar", "."],
        ],
        gridspec_kw=dict(
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        ),
    )
    hist2d_ax = axd["main"]
    xhist_ax = axd["x_hist"]
    yhist_ax = axd["y_hist"]

    cbar_ax = axd["cbar"]
    _, _, _, qmesh = hist2d_ax.hist2d(
        x=xscore.to_numpy(),
        y=yscore.loc[xscore.index].to_numpy(),
        **hist2d_kwargs,
        rasterized=True,
    )
    hist2d_ax.set_aspect("equal")
    hist2d_ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
    )

    cb = fig.colorbar(
        qmesh,
        cax=cbar_ax,
        orientation="horizontal",
    )  # location="bottom",
    axd["cbar"].set_xlabel(freq_label)
    co.cbar_change_style_to_inward_white_ticks(cb, remove_ticks_at_ends="always")

    xhist_ax.hist(xscore, rasterized=True, **x_hist1d_kwargs)
    xhist_ax.xaxis.set_visible(False)
    xhist_ax.set(ylabel=freq_label)

    yhist_ax.hist(yscore, orientation="horizontal", rasterized=True, **y_hist1d_kwargs)
    yhist_ax.yaxis.set_visible(False)
    yhist_ax.set(xlabel=freq_label)

    pd.testing.assert_index_equal(xscore.index, yscore.index)
    pcc, two_tailed_pvalue = scipy.stats.pearsonr(
        x=xscore.to_numpy(), y=yscore.to_numpy()
    )


    pcc_str = f"PCC = {pcc:.2f}"
    if show_pvalue and pcc_pos == "title":

        pcc_str = TextArea(
            f"PCC = {pcc:.2f}", textprops=dict(size=mpl.rcParams["axes.titlesize"])
        )
        log10_pvalue = np.log10(two_tailed_pvalue)
        if log10_pvalue == -np.inf:
            pval_text_size = max(5, mpl.rcParams["axes.titlesize"] - 2)
            pval_str = TextArea("log10(p) < -100", textprops=dict(size=pval_text_size))
        else:
            pval_str = TextArea(
                f"\nlog10(p) = {log10_pvalue.round(0):.0f}",
                textprops=dict(size=mpl.rcParams["axes.titlesize"] - 1),
            )
        hpack = VPacker(children=[pcc_str, pval_str], align="center", pad=0, sep=2)
        anchored_box = AnchoredOffsetbox(
            loc="lower center",
            child=hpack,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
            bbox_transform=hist2d_ax.transAxes,
            borderpad=0.0,
        )
        hist2d_ax.add_artist(anchored_box)
    elif pcc_pos == "title":
        hist2d_ax.set_title(pcc_str)
    elif pcc_pos == "upper center":
        xstart, xend = hist2d_ax.get_xlim()
        xcenter = xstart + (xend - xstart) / 2
        _, yend = hist2d_ax.get_ylim()
        hist2d_ax.text(
            x=xcenter,
            y=yend,
            s=pcc_str,
            va="top",
            ha="center",
            fontsize=mpl.rcParams["axes.labelsize"],
        )
    elif pcc_pos is not None:
        hist2d_ax.text(
            x=pcc_pos[0],
            y=pcc_pos[1],
            s=pcc_str,
            fontsize=mpl.rcParams["axes.labelsize"],
        )

    fig.canvas.draw()
    fig.set_constrained_layout(False)

    # bbox: [[xmin, ymin], [xmax, ymax]]
    bbox_main = axd["main"].get_position()

    bbox_x_hist = xhist_ax.get_position()
    xhist_ax.set_position(
        mtransforms.Bbox(
            [[bbox_main.xmin, bbox_x_hist.ymin], [bbox_main.xmax, bbox_x_hist.ymax]]
        )
    )
    bbox_y_hist = yhist_ax.get_position()
    yhist_ax.set_position(
        mtransforms.Bbox(
            [[bbox_y_hist.xmin, bbox_main.ymin], [bbox_y_hist.xmax, bbox_main.ymax]]
        )
    )

    bbox_cbar = axd["cbar"].get_position()
    axd["cbar"].set_position(
        mtransforms.Bbox(
            [[bbox_main.xmin, bbox_cbar.ymin], [bbox_main.xmax, bbox_cbar.ymax]]
        )
    )

    # %%

    return fig, axd
