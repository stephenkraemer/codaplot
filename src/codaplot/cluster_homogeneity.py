# # Imports

from typing import Any, Dict, Optional, Tuple
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import codaplot as co
from typing import List

# # Runner

# moved to pseudotime lib

# # Helpers

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False) -> None:
        self.vcenter = vcenter
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None) -> np.ndarray:  # type: ignore
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x: List[float] = [self.vmin, self.vcenter, self.vmax]  # type: ignore
        y: List[float] = [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def cm(x):
    return x/2.54


# # Multiline plot


def cluster_homogeneity_multiline_plot(
    data: pd.DataFrame,
    figsize: Tuple[float, float],
    data_name: Optional[str] = None,
    color_ser: Optional[pd.Series] = None,
    color_name: Optional[str] = None,
    color_norm: Optional[mcolors.Normalize] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    line_collection_kwds: Optional[Dict[str, Any]] = None,
    cbar_kwds: Optional[Dict[str, Any]] = None,
):
    """

    Parameters
    ----------
    data
        (vars x samples)
    color_ser
        must have same index as data
    stat_name
        name of the
    line_collection_kwargs
        defaults to alpha=0.01, linewidth=1
    color_norm
        defaults to mid point normalization with vmin = 10th percentile, vcenter = median, vmax = 90th percentile
    ylim
        defaults to (floor(min(data)), ceil(max(data)))



    """

    # Args assertions
    if color_ser is not None:
        pd.testing.assert_index_equal(data.index, color_ser.index)

    # Handle default args
    line_collection_kwargs_complete = dict(alpha=0.01, linewidth=1)
    if line_collection_kwds:
        line_collection_kwargs_complete.update(line_collection_kwds)
    alpha = line_collection_kwargs_complete.pop("alpha")

    # - it seems that line collection does not adapt is cmap normalization to the range
    #   of the scores passed with lc.set_array in the next step
    # - therefore in all cases use norm to set vmin and vmax, and if necessary vcenter
    if color_norm is None and color_ser is not None:
        vmin, mid, vmax = color_ser.quantile([0.1, 0.5, 0.9])
        color_norm = MidpointNormalize(vmin=vmin, vmax=vmax, vcenter=mid)

    if ylim is None:
        ylim = (np.floor(data.min().min()), np.ceil(data.max().max()))  # type: ignore

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True, dpi=180)

    # create line collections and add to figure
    # data for line collections
    segments_arr = np.empty(data.shape + (2,))
    # Set y values (data matrix)
    segments_arr[:, :, 1] = data.values
    # Set x values (arange, starting at 0.5 to plot in center of unit size bins)
    xticks = np.arange(0.5, data.shape[1])
    segments_arr[:, :, 0] = xticks

    # if line plots are colored:
    # - one dummy line collection with alpha=1 necessary for correctly displayed colorbar
    if color_ser is not None:
        lc_dummy, lc = [
            LineCollection(
                segments_arr,  # type: ignore
                cmap="viridis",
                alpha=curr_alpha,
                norm=color_norm,
                **line_collection_kwargs_complete,
            )
            for curr_alpha in [1, alpha]
        ]
        lc.set_array(color_ser.to_numpy())
        ax.add_collection(lc)  # type: ignore
        cbar = fig.colorbar(lc_dummy, **cbar_kwds)  # type: ignore
        cbar.set_label(color_name)
        co.cbar_change_style_to_inward_white_ticks(
            cbar=cbar,
            remove_ticks_at_ends="always",
            outline_is_visible=False,
            # tick_width: int = 1,
            # tick_length: int = 2,
            # tick_params_kwargs: Optional[Dict] = None,
            # remove_ticks_at_ends: Literal["always", "if_on_edge", "never"] = "if_on_edge",
        )
    else:
        lc = LineCollection(segments_arr, alpha=alpha, **line_collection_kwds)  # type: ignore
        ax.add_collection(lc)  # type: ignore

    # adjust limits,
    # X and Y ticks are not automatically updated after the Artist was added
    ax.set(
        xticks=xticks,
        xlim=(0, data.shape[1]),
        ylim=ylim,
    )

    # set labels
    if title is not None:
        ax.set_title(title)
    if data_name is not None:
        ax.set_ylabel(data_name)
    ax.set_xticklabels(data.columns, rotation=90)

    return fig


# # Unsorted

'''




def cluster_homogeneity_jitter_plot(
    group_df, stat_name, color, color_name, geom_point_kwargs, figsize
) -> Figure:
    """

    Notes
    -----
    - This returns a figure created from a plotnine object,
      the figure must be saved with bbox_inches='tight'

    Parameters
    ----------
    group_df
    stat_name
    color
    color_name
    geom_point_kwargs
    figsize

    Returns
    -------

    """

    if stat_name is None:
        stat_name = "score"

    vmin, mid, vmax = color.quantile([0.1, 0.5, 0.9])

    geom_point_kwargs_complete = dict(
        size=1, alpha=0.5, stroke=0, position=pn.position_jitter(width=0.2, height=0)
    )
    if geom_point_kwargs:
        geom_point_kwargs_complete.update(geom_point_kwargs)

    plot_df = (
        group_df.stack()
        .reset_index(-1)
        .set_axis(["pop", stat_name], axis=1, inplace=False)
    )
    plot_df["color"] = color
    plot_df["pop"] = pd.Categorical(plot_df["pop"], categories=group_df.columns)

    def pn_midpoint_norm(x, vmin, vmax, mid=0, to=(0, 1), **kwargs):
        # _from is overwritten by plotnine
        return np.interp(x, [vmin, mid, vmax], [0, 0.5, 1])

    sampled_plot_df = plot_df.sample(30000) if plot_df.shape[0] > 30000 else plot_df
    g = (
        pn.ggplot(data=sampled_plot_df)
        + pn.aes(
            x="pop",
            y=stat_name,
            color="color",
            # fill="membership_score",
        )
        + pn.geom_point(**geom_point_kwargs_complete)
        # + pn.theme_classic()
        + mhstyle.pn_paper_theme()
        + pn.theme(figure_size=figsize, dpi=180)
        + pn.theme(axis_text_x=pn.element_text(rotation=90))
        + pn.scale_color_cmap(
            name="viridis",
            rescaler=partial(pn_midpoint_norm, vmin=vmin, vmax=vmax, mid=mid),
        )
        + pn.labs(x="", color=color_name)
    )
    # https://github.com/has2k1/plotnine/issues/347
    fig = g.draw()
    points = fig.axes[0].collections[0]
    points.set_rasterized(True)
    return fig
'''
