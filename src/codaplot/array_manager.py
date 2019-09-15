"""Use array of plotting functions to facilitate creation of complex gridspecs"""

# * Setup
import itertools
from copy import copy, deepcopy
from functools import wraps
from inspect import getfullargspec, signature
from typing import Callable, Tuple, Dict, Optional, List, Iterable, Union

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.legend as mlegend
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
from dataclasses import dataclass, field

from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

import codaplot as co
from codaplot.plotting import adjust_coords

p = mpl.rcParams
p["legend.fontsize"] = 7
p["legend.title_fontsize"] = 8
# in font size units
# details: https://matplotlib.org/3.1.1/api/legend_api.html
p["legend.handlelength"] = 0.5
p["legend.handleheight"] = 0.5
# vertical space between legend entries
p["legend.labelspacing"] = 0.5
# pad between axes and legend border
p["legend.borderaxespad"] = 0
# fractional whitspace inside legend border, still measured in font size units
p["legend.borderpad"] = 0


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# * Arrays of plots
# *** ArrayElement


@dataclass
class ArrayElement:
    name: Optional[str] = None
    # without func, a placeholder Axes is created
    func: Optional[Callable] = None
    args: Tuple = tuple()
    kwargs: Dict = field(default_factory=dict)

    def __repr__(self):
        if self.name:
            return f"AE: {self.name}"
        else:
            return f"AE {self.func}"


# *** array to figure
def array_to_figure(
    plot_array: np.ndarray,
    figsize: Tuple[float],
    merge_by_name: bool = True,
    constrained_layout_pads: Optional[Dict] = None,
    gridspec_pads: Optional[Dict] = None,
    constrained_layout=True,
) -> Dict:
    """Convert array of plotting instructions into figure, provide access to Axes

    plot array format
    - Array of ArrayElements
    - all ArrayElements must have names
    - ArrayElement names must be unique, they will e.g. be used as identifiers
      in the returned Axes dict, but the unique names are also relied upon by
      internal code.


    [[ArrayElement, ArrayElement, (1, 'rel')],
     [ArrayElement, ArrayElement, (1, 'abs')],
     [  (1, 'abs'),   (1, 'rel'),       None][,


    Args:
        plot_array: array of ArrayElements detailing plotting instructions

    Returns:
        dict with keys
          - axes_d (dict name -> Axes)
          - axes_arr (array of Axes matching figure layout)
          - fig
    """

    if constrained_layout_pads is None:
        constrained_layout_pads = dict()
    if gridspec_pads is None:
        gridspec_pads = dict()

    # The margins of the array contain the height and width specs as (size, 'abs'|'rel')
    # retrieve the actual plot array (inner_array) and convert the size specs
    # to relative-only ratios, given the figsize

    inner_array = plot_array[:-1, :-1]
    widths = plot_array[-1, :-1]
    width_ratios = compute_gridspec_ratios(widths, figsize[0])
    heights = plot_array[:-1, -1]
    height_ratios = compute_gridspec_ratios(heights, figsize[1])

    # assert (
    #     np.unique(
    #         [elem.name for elem in inner_array.ravel() if elem is not None],
    #         return_counts=True,
    #     )[-1]
    #     == 1
    # ).all()
    #
    fig = plt.figure(constrained_layout=constrained_layout, dpi=180, figsize=figsize)
    # constrained layout pads need to be set directly,
    # setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(**constrained_layout_pads)
    gs = gridspec.GridSpec(
        inner_array.shape[0],
        inner_array.shape[1],
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        figure=fig,
        **gridspec_pads,
    )

    axes_d = {}
    axes_arr = np.empty_like(inner_array)
    plot_returns = {}
    seen = []
    for t in itertools.product(
        range(inner_array.shape[0]), range(inner_array.shape[1])
    ):
        ge = inner_array[t]
        # Do not create Axes for Spacers (np.nan)
        # However, do create Axes for GridElements without plotting function (below)
        if ge is None:
            continue

        if merge_by_name:
            if ge.name in seen:
                continue
            else:
                next_col = t[1]
                for next_col in range(t[1] + 1, inner_array.shape[1]):
                    next_ge = inner_array[t[0], next_col]
                    if next_ge is None or not next_ge.name == ge.name:
                        break
                else:
                    next_col += 1
                next_row = t[0]
                for next_row in range(t[0] + 1, inner_array.shape[0]):
                    next_ge = inner_array[next_row, t[1]]
                    if next_ge is None or not next_ge.name == ge.name:
                        break
                else:
                    next_row += 1
                if next_row > t[0] + 1 or next_col > t[1] + 1:
                    assert (
                        len(
                            {
                                ge.name
                                for ge in np.ravel(
                                    inner_array[t[0] : next_row, t[1] : next_col]
                                )
                            }
                        )
                        == 1
                    )
                    t = (slice(t[0], next_row), slice(t[1], next_col))
                seen.append(ge.name)

        ax = fig.add_subplot(gs[t])
        axes_arr[t] = ax
        if ge.name is not None:
            if ge.name in axes_d:
                if isinstance(axes_d[ge.name], list):
                    axes_d[ge.name].append(ax)
                else:
                    axes_d[ge.name] = [axes_d[ge.name], ax]
            else:
                axes_d[ge.name] = ax

        # Without plotting function, we stop after having created the Axes
        if ge.func is None:
            continue
        # Plotting func is given, create plot
        # Add Axes to func args if necessary
        # If ax is not in plotter args, we assume that this is a pyplot function relying on state

        # Do no use getfullargspec, it will fail on decorated funtions, even if they correctly use functools.wrap
        # see also: https://hynek.me/articles/decorators/
        plotter_args = signature(ge.func).parameters.keys()
        if "ax" in plotter_args:
            ge.kwargs["ax"] = ax
        plot_returns[ge.name] = ge.func(*ge.args, **ge.kwargs)

    return {
        "axes_d": axes_d,
        "axes_arr": axes_arr,
        "plot_returns": plot_returns,
        "fig": fig,
    }


# ***** Compute gridspec ratios
def compute_gridspec_ratios(sizes: np.ndarray, total_size_in: float) -> np.ndarray:
    """Convert mixture of absolute and relative column/row sizes to relative-only

    The result depends on the figsize, the function will first allocate space for
    all absolute sizes, and then divide the rest proportionally between the relative
    sizes.

    Args:
        sizes: array of size specs, eg. [(1, 'rel'), (2, 'abs'), (3, 'rel')]
            all sizes in inch
        total_size_in: size of the corresponding dimension of the figure (inch)

    Returns:
        array of ratios which can be passed to width_ratios or height_ratios of
        the Gridspec constructor. Note that this result is only valid for the
        given figsize.

    """
    heights = np.array([t[0] for t in sizes]).astype(float)
    anno = np.array([t[1] for t in sizes])
    total_abs_heights = heights[anno == "abs"].sum()
    remaining_height = total_size_in - total_abs_heights
    rel_heights = heights[anno == "rel"]
    heights[anno == "rel"] = rel_heights / rel_heights.sum() * remaining_height
    return heights


# * Add guides


def add_guides(guide_spec_l, ax, xpad_in, ypad_in):
    ax.axis("off")
    guide_placement_params_df = get_guide_placement_params(guide_spec_l, ax)
    place_guides(
        ax=ax, placement_df=guide_placement_params_df, xpad_in=xpad_in, ypad_in=ypad_in
    )


# *** Get guide placement params


def get_guide_placement_params(guide_spec_l: List[Optional[Dict]], ax):
    """Get dataframe with required height and width for all guides

    Args:
        guide_spec_l: list of optional dicts, each dict provides key-value pairs
            required to either draw a legend or a colorbar.
            For legend
                - handles
                - labels
                - optional: other Legend args, eg. title
            For colorbars:
                - mappable (ScalarMappable, eg. returned by pcolormesh, scatter)
                - optional: other colorbar args
                - optional: title
        ax: Axes onto which to draw the guides. The Axes is assumed to be completely empty.

    Returns:
         pd.DataFrame with columns: title, height, width, contents
         contents is a reference to the guide specification dictionary from guide_spec_l
         (because its convenient to have these things together in the same frame)

    """
    # Note that 'title' is not a colorbar arg. It is removed before passing on the cbar_args
    placement_df = pd.DataFrame(columns=["title", "height", "width", "contents"])
    for guide_d in guide_spec_l:
        # Empty slots are allowed and should be skipped
        if guide_d is None:
            continue
        # A guide spec without handles is expected to define a colorbar
        if "handles" not in guide_d:
            assert "mappable" in guide_d
            # compute width and height of the inset axes in parent axes coordinates
            # shrink gives the width or height in parent axes coordinates, depending on the orientation
            shrink = guide_d.get("shrink", 1)
            aspect = guide_d.get("aspect", 20)
            orientation = guide_d.get("orientation", "vertical")
            if orientation == "vertical":
                height = shrink
                width = shrink / aspect
            else:
                width = shrink
                height = shrink / aspect
            placement_df = placement_df.append(
                dict(
                    # remove title, it is not a cbar arg, so that the guide specification
                    # from now on only contains cbar args
                    title=guide_d.pop("title", None),
                    height=height,
                    width=width,
                    contents=guide_d,
                ),
                ignore_index=True,
            )
        else:  # We have a standard legend
            elem = guide_d["handles"], guide_d["labels"]
            # Create a throw-away legend object without appropriate placing parameters,
            # to get the legend size
            l = mlegend.Legend(ax, *elem, title=guide_d.get("title", None))
            r = ax.figure.canvas.get_renderer()
            bbox = l.get_window_extent(r).transformed(ax.transAxes.inverted())
            placement_df = placement_df.append(
                dict(
                    title=guide_d.get("title", None),
                    height=bbox.height,
                    width=bbox.width,
                    contents=guide_d,
                ),
                ignore_index=True,
            )
    return placement_df


# *** Place guides


def place_guides(ax, placement_df, xpad_in=0.2, ypad_in=0.2 / 2.54):
    """Pack guides into Axes

    If constrained_layout is enabled, the Axes may be shrunk. This happens after
    this function was applied, and may bring the guides to close together, or even
    make them overlap. This can be adjusted for by increasing xpad_in and ypad_in, which
    also become shrinked by constrained_layout.

    Args:
        ax: guide Axes, assumed to be empty
        placement_df: columns title, height, width, contents
            contents is a guide spec dictionary, see get_guide_placement_params
            for details
        xpad_in: padding between guides along x axis, in inch
        ypad_in: padding between guides along y axis, in inch
    """
    ax_width_in, ax_height_in = get_axes_dim_in(ax)
    min_xpad_ax_coord = xpad_in / ax_width_in
    min_ypad_ax_coord = ypad_in / ax_width_in

    # Iterate over guides, from largest to smallest
    # Fill guide Axes column by column
    placement_df = placement_df.sort_values("height", ascending=False)
    curr_x = 0
    next_x = 0
    curr_y = 1
    for _, row_ser in placement_df.iterrows():

        # TODO: row_ser.height may be larger 1
        if curr_y - row_ser.height < 0:
            # The current guide cannot be placed in the current column
            # => start a new column
            curr_y = 1
            curr_x = next_x

        # Place the guide, even if its height is larger than the Axes height
        if "handles" in row_ser.contents:
            _add_legend(ax, curr_x, curr_y, row_ser)
            curr_xpad = min_xpad_ax_coord
            curr_ypad = min_ypad_ax_coord
        else:
            # The colorbar is added as inset axes
            # we need to add additional padding to the predefined padding to
            # account for title and tick(labels)
            add_xpad_ax_coord, add_ypad_ax_coord = _add_cbar_inset_axes(
                row_ser, ax, curr_x, curr_y
            )
            curr_xpad = min_xpad_ax_coord + add_xpad_ax_coord
            curr_ypad = min_ypad_ax_coord + add_ypad_ax_coord

        # TODO: padding around colorbars hardcoded in axes coordinates
        curr_y = curr_y - (row_ser.height + curr_ypad)
        next_x = max(next_x, curr_x + row_ser.width + curr_xpad)


# ***** Add legend


def _add_legend(ax, curr_x, curr_y, row_ser):
    # This is a Legend
    l = mlegend.Legend(
        ax,
        handles=row_ser.contents["handles"],
        labels=row_ser.contents["labels"],
        title=row_ser.contents.get("title", None),
        # avoid padding between legend corner and anchor point
        borderaxespad=0,
        loc="upper left",
        bbox_to_anchor=(curr_x, curr_y),
    )
    l._legend_box.align = "left"
    ax.add_artist(l)


# ***** Add cbar inset axes


def _add_cbar_inset_axes(row_ser, ax, curr_x, curr_y):
    fig: Figure = ax.figure
    cbar_ax: Axes
    ax_width_in, ax_height_in = get_axes_dim_in(ax)
    fontdict = {"fontsize": p["legend.title_fontsize"], "verticalalignment": "top"}

    if row_ser["title"]:
        # What would be more idiomatic code? This seems suboptimal
        # Just using fontsize and padding sizes is not ideal for estimating width...
        # Could not get size of Text without adding to Axes
        # so first add text, then remove it, then add at correct coord again
        title_text = ax.text(0, 0, row_ser["title"], fontdict=fontdict)
        r = fig.canvas.get_renderer()
        title_axes_bbox = title_text.get_window_extent(r).transformed(
            ax.transAxes.inverted()
        )
        title_width_axes_coord = title_axes_bbox.width
        title_height_axes_coord = (
            title_axes_bbox.height
            + mpl.rcParams["legend.labelspacing"]
            * mpl.rcParams["legend.fontsize"]
            * 1
            / 72
        )
        title_text.remove()
    else:
        title_width_axes_coord = 0
        title_height_axes_coord = 0

    # make room for title and same padding as with legends
    cbar_ax = ax.inset_axes(
        [
            curr_x,
            curr_y - row_ser.height - title_height_axes_coord,
            row_ser.width,
            row_ser.height,
        ]
    )
    fig.colorbar(**row_ser.contents, cax=cbar_ax)
    if row_ser["title"]:
        cbar_ax.text(
            curr_x, curr_y, row_ser["title"], fontdict=fontdict, transform=ax.transAxes
        )

    # To get the dimensions of this guide, we must take into account
    # ticklabels and the colorbar title (which is placed as axes title)
    # Depending on the cbar orientation, the axis ticks contribute to height or width
    # We calculate the size of the padding needed to account for this
    # (this will be added to the whitespace padding defined in the guide placement function)
    # ## Get title dim

    # ## Get tick label width
    # draw is necessary to get yticklabels
    fig.canvas.draw()
    # assumption: all labels have same length
    # ticks are on x or y axis depending on orientation of cbar
    orientation = row_ser.contents.get("orientation", "vertical")
    if orientation == "vertical":
        first_ticklabel = cbar_ax.get_yticklabels()[0]
    else:
        first_ticklabel = cbar_ax.get_xticklabels()[0]
    first_ticklabel_axes_bbox = first_ticklabel.get_window_extent().transformed(
        ax.transAxes.inverted()
    )
    ytick_size_axes_coord = p["ytick.major.size"] * 1 / 72 / ax_width_in

    if orientation == "vertical":
        # no additional y padding required
        ypad_ax_coord = title_height_axes_coord
        # we have to add additional x padding
        # either the title or the yticklabels will protrude more, select
        # the outmost boundary
        xpad_ax_coord = max(
            title_width_axes_coord - row_ser["width"],
            (first_ticklabel_axes_bbox.width + ytick_size_axes_coord),
        )
    else:
        # no additional x padding required
        if title_width_axes_coord > row_ser.contents["width"]:
            xpad_ax_coord = title_width_axes_coord - row_ser.contents["width"]
        else:
            xpad_ax_coord = 0
        # we have to add additional y padding for horizontal bar due to title *and*
        # tick(labels) together
        ypad_ax_coord = (
            title_height_axes_coord
            + ytick_size_axes_coord
            + first_ticklabel_axes_bbox.height
        )
    return xpad_ax_coord, ypad_ax_coord


# * helpers

# *** get axes dim
def get_axes_dim_in(ax):
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    ax_width_in, ax_height_in = bbox.width, bbox.height
    return ax_height_in, ax_width_in


# * Tests
# tests
def object_plotter(ax, a=3, b=10, title="title"):
    """Simple test function which accepts ax, and fails if b is not overwritten"""
    ax.plot(range(a), range(b), label="object")
    ax.plot(range(a), range(b), c="green", label="o_g")
    handles, labels, *_ = mlegend._parse_legend_args([ax])
    return dict(handles=handles, labels=labels, title=title)


def test_plot():
    """Create simple array[ArrayElement] and create figure and axes containers

    displays figure if successful

    Tested features
    - array handling
        - specification of a combination of relative and absolute column and row sizes in margins
        - Creating of empty Axes
        - returned objects which are queried: axes_d, results
        - None acts as placeholder without Axes creation
        - concatenation of arrays prior to plotting
    - manual legend addition
    - ArrayElement
        - plotting works with
            - pyplot functions (state machinery relying functions)
            - function taking Axes object

    """

    plot_array1 = np.array(
        [
            [
                ArrayElement(
                    "scatter1",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(marker=".", s=10, c="green", label="green"),
                ),
                ArrayElement(
                    "scatter2",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="black", label="black"),
                ),
                None,
                (3, "abs"),
            ],
            [
                ArrayElement("plot", object_plotter, args=(), kwargs=dict(b=3)),
                ArrayElement(
                    "scatter4",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="blue", marker=".", s=10, label="blue"),
                ),
                ArrayElement("legend"),
                (5, "rel"),
            ],
            [
                None,
                ArrayElement(
                    "heatmap",
                    co.plotting.heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame(np.random.randn(3, 3)),
                        pcolormesh_args=dict(cmap="RdBu_r"),
                        xticklabels=True,
                        xticklabel_rotation=90,
                        xlabel="Xlabel",
                        yticklabels=True,
                        ylabel="Ylabel",
                        cbar_args=dict(
                            shrink=0.7,
                            aspect=5,
                            ticks=[-2, 0, 2],
                            orientation="vertical",
                            title="% Methylation",
                        ),
                        add_colorbar=False,
                        title=None,
                    ),
                ),
                None,
                (2, "rel"),
            ],
            [(2, "rel"), (3, "rel"), (2, "abs"), None],
        ]
    )

    plot_array2 = [
        [
            ArrayElement("plot3", object_plotter, args=(), kwargs=dict(b=3)),
            ArrayElement(
                "scatter34",
                plt.scatter,
                args=([1, 2, 3], [1, 2, 3]),
                kwargs=dict(c="blue"),
            ),
            None,
            (1, "rel"),
        ],
        [(2, "rel"), (3, "rel"), (2, "abs"), None],
    ]

    plot_array_combined = np.vstack([plot_array1[:-1, :], plot_array2])
    res = array_to_figure(plot_array_combined, figsize=(15 / 2.54, 15 / 2.54))
    axes_d = res["axes_d"]

    for k, v in res["plot_returns"].items():
        # check whether we already have a complete guide spec
        if not isinstance(v, dict) or ("mappable" not in v and "handles" not in v):
            handles, labels, *_ = mlegend._parse_legend_args([axes_d[k]])
            if handles:
                res["plot_returns"][k] = dict(
                    handles=handles, labels=labels, title="TODO"
                )
            else:
                res["plot_returns"][k] = None

    add_guides(
        guide_spec_l=res["plot_returns"].values(),
        ax=res["axes_d"]["legend"],
        xpad_in=0.5 / 2.54,
        ypad_in=0.5 / 2.54,
    )

    # display(res["fig"])


def heatmap_test_plot():

    spaced_hmap_args_ser = pd.Series(
        dict(
            row_clusters=np.repeat([1, 2, 3], 4),
            col_clusters=np.repeat([1, 2], 6),
            row_spacer_size=0.05,
            col_spacer_size=0.02,
        )
    )

    plot_array = np.array(
        [
            [
                None,
                ArrayElement(
                    "col_anno",
                    co.plotting.categorical_heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame({"a": np.repeat(["A", "B", "C", "D"], 3)}).T,
                        cmap="Set1",
                        # does not work with spacer
                        label_stretches=False,
                        legend=False,
                        spaced_heatmap_args=dict(
                            col_clusters=np.repeat([1, 2], 6),
                            col_spacer_size=0.02,
                            xticklabels=False,
                            xticklabel_rotation=0,
                            xlabel=None,
                            yticklabels=False,
                            ylabel=None,
                        ),
                    ),
                ),
                None,
                (1 / 2.54, "abs"),
            ],
            [
                ArrayElement(
                    "anno1",
                    co.plotting.categorical_heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame({"a": np.repeat(["A", "B", "C"], 4)}),
                        cmap="Set2",
                        # does not work with spacer
                        label_stretches=False,
                        legend=False,
                        spaced_heatmap_args=dict(
                            row_clusters=np.repeat([1, 2, 3], 4),
                            row_spacer_size=0.05,
                            xticklabels=False,
                            xticklabel_rotation=0,
                            xlabel=None,
                            yticklabels=False,
                            ylabel=None,
                        ),
                    ),
                ),
                ArrayElement(
                    "beta_hmap",
                    func=co.plotting.spaced_heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame(np.random.randn(12, 12)),
                        pcolormesh_args=dict(cmap="YlOrBr"),
                        xticklabels=True,
                        xticklabel_rotation=0,
                        xlabel=None,
                        yticklabels=False,
                        ylabel=None,
                        cbar_args=dict(shrink=0.5, aspect=10),
                        add_colorbar=True,
                        title="Beta values",
                        **spaced_hmap_args_ser,
                    ),
                ),
                None,
                (1, "rel"),
            ],
            # Second row
            [
                ArrayElement(
                    "anno1_2",
                    co.plotting.categorical_heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame({"a": np.repeat(["A", "B", "C"], 4)}),
                        cmap="Set2",
                        # does not work with spacer
                        label_stretches=False,
                        legend=False,
                        spaced_heatmap_args=dict(
                            row_clusters=np.repeat([1, 2, 3], 4),
                            row_spacer_size=0.05,
                            col_spacer_size=0.1,
                            xticklabels=False,
                            xticklabel_rotation=0,
                            xlabel=None,
                            yticklabels=False,
                            ylabel=None,
                        ),
                    ),
                ),
                ArrayElement(
                    "beta_hmap_2",
                    func=co.plotting.spaced_heatmap2,
                    kwargs=dict(
                        df=pd.DataFrame(np.random.randn(12, 12)),
                        pcolormesh_args=dict(
                            cmap="RdBu_r", norm=MidpointNormalize(-3, 3, 0)
                        ),
                        xticklabels=True,
                        xticklabel_rotation=0,
                        xlabel=None,
                        yticklabels=False,
                        ylabel=None,
                        cbar_args=dict(shrink=0.5, aspect=10),
                        add_colorbar=True,
                        title="Beta values",
                        **spaced_hmap_args_ser,
                    ),
                ),
                ArrayElement("legend"),
                (1, "rel"),
            ],
            [(1, "rel"), (8, "rel"), (3 / 2.54, "abs"), None],
        ]
    )

    res = array_to_figure(
        plot_array,
        figsize=(10 / 2.54, 10 / 2.54),
        constrained_layout_pads=dict(h_pad=0.1, w_pad=0.1, hspace=0, wspace=0),
    )

    add_guides(
        res["plot_returns"].values(),
        res["axes_d"]["legend"],
        xpad_in=0.4 / 2.54,
        ypad_in=0.4 / 2.54,
    )


# * Spaced dendrogram


def test_spaced_dendrogram():

    # default dendrogram goes from top to bottom
    # - y coords are linkage heights
    # - x coords connect to middle of cluster rows
    # - does not change with orientation passed to dendro func

    # %%
    spacer_size = 0.05
    rng = np.random.RandomState(1234)
    row_clusters = np.array([2, 1, 1, 2, 3, 3, 2, 2, 3, 1, 2])
    col_clusters = np.array([2, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1])
    df = (
        pd.DataFrame(rng.randn(11, 11))
        .add(row_clusters * 2, axis=0)
        .add(col_clusters * 4, axis=1)
    )
    row_linkage = linkage(df)
    col_linkage = linkage(df.T)
    fig, axes = plt.subplots(
        2, 2, gridspec_kw=dict(width_ratios=[1, 3]), constrained_layout=True
    )
    fig.set_size_inches(10, 10)
    fig.set_dpi(180)
    co.plotting.spaced_heatmap2(
        df.iloc[leaves_list(row_linkage), leaves_list(col_linkage)],
        axes[1, 1],
        pcolormesh_args=dict(cmap="RdBu_r"),
        xticklabels=True,
        yticklabels=True,
        row_clusters=row_clusters[leaves_list(row_linkage)],
        col_clusters=col_clusters[leaves_list(col_linkage)],
        col_spacer_size=spacer_size,
        row_spacer_size=spacer_size,
    )

    co.plotting.cut_dendrogram(
        linkage_mat=row_linkage,
        cluster_ids_data_order=pd.Series(row_clusters),
        ax=axes[1, 0],
        spacing_groups=row_clusters,
        spacer_size=spacer_size,
        pretty=True,
        stop_at_cluster_level=True,
        orientation="horizontal",
        show_cluster_points=True,
        point_params=None,
        min_cluster_size=0,
        min_height=0,
    )

    co.plotting.cut_dendrogram(
        linkage_mat=col_linkage,
        cluster_ids_data_order=pd.Series(col_clusters),
        ax=axes[0, 1],
        spacing_groups=col_clusters,
        spacer_size=spacer_size,
        pretty=True,
        stop_at_cluster_level=False,
        orientation="vertical",
        show_cluster_points=True,
        point_params=None,
        min_cluster_size=0,
        min_height=0,
    )

    # %%

    """
    dendrogram_dict = dendrogram(row_linkage, orientation='right', no_plot=True)
    dcoords = pd.DataFrame.from_records(dendrogram_dict["dcoord"])
    dcoords.columns = ["ylow_left", "yhigh1", "yhigh2", "ylow_right"]
    dcoords = dcoords.sort_values("yhigh1")
    icoords = pd.DataFrame.from_records(dendrogram_dict["icoord"])
    icoords.columns = ["xleft1", "xleft2", "xright1", "xright2"]
    icoords = icoords.loc[dcoords.index, :]
    # x coordinates point to middle of observations
    obs_coord = icoords.reset_index(drop=True)
    # y coordinates give height of links (distance between linked clusters)
    linkage_coords = dcoords.reset_index(drop=True)

    clusters_ord = row_clusters[leaves_list(row_linkage)]

    xcoords_adjusted = obs_coord.divide(10).apply(adjust_coords,
                                                  clusters=clusters_ord,
                                                  spacer_size = spacer_size)
    obs_coord = xcoords_adjusted

    orientation = 'bottom'
    if orientation in ['left', 'right']:
        x = linkage_coords
        y = obs_coord
    else:  # top, bottom
        x = obs_coord
        y = linkage_coords
    if orientation == 'left':
        axes[0].invert_xaxis()
    if orientation == 'bottom':
        axes[0].invert_yaxis()
        # axes[0].invert_xaxis()
    if orientation == 'top':
        # axes[0].invert_xaxis()


    for i in range(len(obs_coord)):
        axes[0].plot(x.iloc[i], y.iloc[i], color='black')
    """
    # %%


# * Adjust coords
def test_adjust_coords():

    co.plotting.find_stretches2(np.repeat([1, 2, 3], 4))

    clusters = np.array([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3])
    adjusted_y = adjust_coords(
        coords=np.arange(0, 11),
        # y_coords=np.array(list('abcdefghij')),
        spacing_groups=clusters,
        spacer_size=0.2,
    )

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x=np.arange(0, 11), y=adjusted_y, hue=clusters - 1, ax=ax)

    print(adjusted_y)


def test_array_merging():

    plot_array = np.array(
        [
            [ArrayElement("name2"), ArrayElement("name1"), (1, "rel")],
            [ArrayElement("name1"), ArrayElement("name1"), (1, "rel")],
            [(1, "rel"), (1, "rel"), None],
        ]
    )
    array_to_figure(plot_array=plot_array, figsize=(6, 6), merge_by_name=True)


ArrayLike = Union[pd.Series, np.ndarray]


# %%
def cross_plot(
    center: Union[List, np.ndarray],
    center_row_sizes=None,
    center_col_sizes=None,
    center_row_spacer=(0.05, "rel"),
    center_col_spacer=(0.05, "rel"),
    figsize=(5, 5),
    constrained_layout=True,
    constrained_layout_pads: Optional[Dict] = None,
    gridspec_pads: Optional[Dict] = None,
    top: Optional[Iterable[Dict]] = None,
    top_sizes=None,
    left: Optional[Iterable[Dict]] = None,
    left_sizes=None,
    bottom: Optional[Iterable[Dict]] = None,
    bottom_sizes=None,
    right: Optional[Iterable[Dict]] = None,
    right_sizes=None,
    row_dendrogram: Optional[Dict] = None,
    row_dendrogram_size=(1 / 2.54, "abs"),
    col_dendrogram: Optional[Dict] = None,
    col_dendrogram_size=(1 / 2.54, "abs"),
    row_order: Optional[Union[ArrayLike, List[ArrayLike]]] = None,
    col_order: Optional[Union[ArrayLike, List[ArrayLike]]] = None,
    default_func=co.plotting.heatmap3,
    row_clusters=None,
    row_spacer_size=0.02,
    col_clusters=None,
    col_spacer_size=0.02,
    # aligned_arg_names: Optional[Tuple[str]] = ('df', 'data'),
    # cluster_data = None,
    # col_cluster: Union[bool, int] = False,
    # col_cluster_metric = 'euclidean',
    # col_cluster_method = 'average',
    # row_cluster: Union[bool, int] = False,
    # row_cluster_metric = 'euclidean',
    # row_cluster_method = 'average',
):
    """

    Args:
        center: plots at the center of the cross, either 1D List / array (interpreted as a row), or a numpy 2D array. Aligned by row and col if align=True
        top: annotation plots above each center plot
        left: annotation plots at left of each center plot
        bottom: annotation plots below each center plot
        right: annotation plots at right of each center plot
        row_dendrogram: kwargs passed to co.plotting.cutdendrogram
        col_dendrogram: kwargs passed to co.plotting.cutdendrogram
        default_func: called if plot spec dict does not contain _func key
        aligned_arg_names: if not None and plot spec does not override with a _align key, align arguments of that name based on row order (left, right), col order (top, bottom) or both (center)


    Returns:
        dict with keys
        - fig: figure
        - array: array of plot specs with aligned
        - axes_d
        - axes_arr

    """

    # add default func
    for elem in itertools.chain.from_iterable(
        it for it in [left, top, bottom, right, np.ravel(center)] if it is not None
    ):
        if elem.func is None:
            elem.func = default_func

    # add dendrograms to left / top
    if isinstance(col_dendrogram, dict) or col_dendrogram:
        if not isinstance(col_dendrogram, dict):
            col_dendrogram = {}
        else:
            col_dendrogram = deepcopy(col_dendrogram)
        col_dendrogram["orientation"] = "vertical"
        assert (
            col_order.ndim == 2
        ), "col order not linkage mat, but dendrogram requested"
        top = [
            ArrayElement(
                func=co.plotting.cut_dendrogram,
                kwargs=dict(
                    linkage_mat=col_order,
                    cluster_ids_data_order=col_clusters,
                    spacing_groups=col_clusters,
                    stop_at_cluster_level=False,
                    min_height=0,
                    min_cluster_size=0,
                    spacer_size=col_spacer_size,
                    **col_dendrogram,
                ),
            )
        ] + (top if top else [])
        top_sizes = [col_dendrogram_size] + (top_sizes if top_sizes else [])

    if isinstance(row_dendrogram, dict) or row_dendrogram:
        if not isinstance(row_dendrogram, dict):
            row_dendrogram = {}
        else:
            row_dendrogram = deepcopy(row_dendrogram)
        row_dendrogram["orientation"] = "horizontal"
        assert (
            row_order.ndim == 2
        ), "row order not linkage mat, but dendrogram requested"
        left = [
            ArrayElement(
                func=co.plotting.cut_dendrogram,
                kwargs=dict(
                    linkage_mat=row_order,
                    cluster_ids_data_order=row_clusters,
                    stop_at_cluster_level=False,
                    spacing_groups=row_clusters,
                    min_height=0,
                    spacer_size=row_spacer_size,
                    min_cluster_size=0,
                    **row_dendrogram,
                ),
            )
        ] + (left if left else [])
        left_sizes = [row_dendrogram_size] + (left_sizes if left_sizes else [])

    # get array size, init array
    def len_or_none(x):
        if x is None:
            return 0
        else:
            return len(x)

    # center is list, list of list, 1d array or 2d array
    # convert to 2d array
    center_arr = np.array(center)
    if center_arr.ndim == 1:
        center_arr = center_arr[np.newaxis, :]
    if center_row_sizes is None:
        center_row_sizes = [(1, "rel")] * center_arr.shape[0]
    if center_col_sizes is None:
        center_col_sizes = [(1, "rel")] * center_arr.shape[1]

    n_cols = len_or_none(left) + len_or_none(right) + center_arr.shape[1]
    n_rows = len_or_none(top) + len_or_none(bottom) + center_arr.shape[0]
    # Leave space for sizes
    plot_arr = np.empty((n_rows + 1, n_cols + 1), dtype=object)

    # add ArrayElements at each side and in center, add sizes into array
    first_center_row = len(top)
    first_center_col = len(left)
    # last index plus 1
    last_center_row_p1 = len(top) + center_arr.shape[0]
    last_center_col_p1 = len(left) + center_arr.shape[1]

    # Add plots to array
    if left is not None:
        plot_arr[first_center_row:last_center_row_p1, :first_center_col] = left
    if right is not None:
        plot_arr[first_center_row:last_center_row_p1, last_center_col_p1:-1] = right
    if top is not None:
        plot_arr[:first_center_row, first_center_col:last_center_col_p1] = np.array(
            top
        )[:, np.newaxis]
    if bottom is not None:
        plot_arr[last_center_row_p1:-1, first_center_col:last_center_col_p1] = np.array(
            bottom
        )[:, np.newaxis]
    # add center plots
    plot_arr[
        first_center_row:last_center_row_p1, first_center_col:last_center_col_p1
    ] = center_arr

    # add size specs
    height_ratios = list(
        itertools.chain.from_iterable(
            (x for x in [top_sizes, center_row_sizes, bottom_sizes] if x is not None)
        )
    ) + [None]
    width_ratios = list(
        itertools.chain.from_iterable(
            (x for x in [left_sizes, center_col_sizes, right_sizes] if x is not None)
        )
    ) + [None]
    plot_arr[:, -1] = height_ratios
    plot_arr[-1, :] = width_ratios

    # add row spacers
    row_spacer = np.empty((1, plot_arr.shape[1]), dtype=object)
    row_spacer[0, -1] = center_row_spacer
    plot_arr = np.insert(
        plot_arr, slice(first_center_row + 1, last_center_row_p1), row_spacer, axis=0
    )
    # taking into account the new rows, add enlarged col spacers
    col_spacer = np.empty((plot_arr.shape[0], 1), dtype=object)
    col_spacer[-1, 0] = center_col_spacer
    plot_arr = np.insert(
        plot_arr, slice(first_center_col + 1, last_center_col_p1), col_spacer, axis=1
    )

    # plot
    res = array_to_figure(
        plot_array=plot_arr,
        figsize=figsize,
        merge_by_name=False,
        constrained_layout_pads=constrained_layout_pads,
        gridspec_pads=gridspec_pads,
        constrained_layout=constrained_layout,
    )

    return res


# %%


def test_cross_plot():

    global_spacer_size = 0.06
    rng = np.random.RandomState(1234)
    row_clusters = np.array([2, 1, 1, 2, 3, 3, 2, 2, 3, 1, 2])
    col_clusters = np.array([2, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1])
    df = (
        pd.DataFrame(rng.randn(11, 11))
        .add(row_clusters * 2, axis=0)
        .add(col_clusters * 4, axis=1)
    )
    row_linkage = linkage(df)
    col_linkage = linkage(df.T)
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    spacers = dict(
        row_clusters=row_clusters[row_order],
        col_clusters=col_clusters[col_order],
        row_spacer_size=global_spacer_size,
        col_spacer_size=global_spacer_size,
    )
    row_spacers = dict(
        row_clusters=row_clusters[row_order], row_spacer_size=global_spacer_size
    )
    col_spacers = dict(
        col_clusters=col_clusters[col_order], col_spacer_size=global_spacer_size
    )
    figsize = (20 / 2.54, 20 / 2.54)
    figsize_ratio = figsize[0] / figsize[1]
    res = cross_plot(
        figsize=figsize,
        constrained_layout=False,
        constrained_layout_pads=dict(h_pad=0, w_pad=0, hspace=0.03, wspace=0.03),
        gridspec_pads=dict(hspace=0.03, wspace=0.03),
        center_col_spacer=(0.2 / figsize_ratio, "rel"),
        center_row_spacer=(0.2, "rel"),
        center=np.array(
            [
                [
                    ArrayElement(
                        kwargs=dict(
                            df=df.iloc[row_order, col_order],
                            pcolormesh_args=dict(cmap="RdBu_r"),
                            xticklabels=True,
                            yticklabels=True,
                            **spacers,
                        )
                    ),
                    ArrayElement(
                        kwargs=dict(
                            df=df.iloc[row_order, col_order] * 10,
                            pcolormesh_args=dict(cmap="YlOrBr"),
                            xticklabels=True,
                            yticklabels=True,
                            **spacers,
                        )
                    ),
                ],
                [
                    ArrayElement(
                        kwargs=dict(
                            df=df.iloc[row_order, col_order] * 5,
                            pcolormesh_args=dict(cmap="RdBu_r"),
                            xticklabels=True,
                            yticklabels=True,
                            **spacers,
                        )
                    ),
                    ArrayElement(
                        kwargs=dict(
                            df=df.iloc[row_order, col_order] * 2,
                            pcolormesh_args=dict(cmap="viridis"),
                            xticklabels=True,
                            yticklabels=True,
                            **spacers,
                        )
                    ),
                ],
            ]
        ),
        top=[
            ArrayElement(
                kwargs=dict(
                    df=pd.DataFrame({"anno1": col_clusters}).T.iloc[:, col_order],
                    is_categorical=True,
                    pcolormesh_args=dict(cmap="Set1"),
                    **col_spacers,
                )
            ),
            ArrayElement(
                kwargs=dict(df=pd.DataFrame({"values": np.arange(11)}).T, **col_spacers)
            ),
            ArrayElement(func=spacer),
        ],
        top_sizes=[(1 / 2.54, "abs"), (1 / 2.54, "abs"), (0.5 / 2.54, "abs")],
        left=[
            ArrayElement(
                kwargs=dict(
                    df=pd.DataFrame(
                        {"anno2": pd.Series(row_clusters)[row_order].astype(str)}
                    ),
                    **row_spacers,
                )
            ),
            ArrayElement(func=spacer),
        ],
        left_sizes=[(1 / 2.54, "abs"), (0.5 / 2.54, "abs")],
        right=[
            ArrayElement(func=spacer),
            ArrayElement(
                func=spaced_barplot,
                kwargs=dict(
                    y=adjust_coords(
                        np.arange(0, df.shape[0]) + 0.5,
                        row_clusters[row_order],
                        global_spacer_size,
                    ),
                    width=df.sum(axis=1),
                    # TODO: height not calculated automatically
                    height=0.05,
                    color="gray",
                ),
            ),
        ],
        right_sizes=[(0.5, "abs"), (2, "abs")],
        # left=None,
        # left_sizes=None,
        bottom=[
            ArrayElement(func=spacer),
            ArrayElement(
                func=anno_axes("bottom")(plt.bar),
                kwargs=dict(
                    x=adjust_coords(
                        co.plotting.find_stretches2(col_clusters[col_order])[1],
                        col_clusters[col_order],
                        global_spacer_size,
                        max_coord=df.shape[1],
                    ),
                    height=df.groupby(col_clusters, axis=1).sum().sum(axis=0),
                    width=0.1,
                    color="gray",
                ),
            ),
        ],
        bottom_sizes=[(0.5, "abs"), (2, "abs")],
        row_dendrogram=True,
        col_dendrogram=True,
        row_order=row_linkage,
        col_order=col_linkage,
        row_clusters=pd.Series(row_clusters),
        col_clusters=pd.Series(col_clusters),
        row_spacer_size=global_spacer_size,
        col_spacer_size=global_spacer_size,
        default_func=co.plotting.heatmap3,
    )


def spacer(ax):
    ax.axis("off")
    ax.tick_params(
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        axis="both",
        which="both",
        labelsize=0,
        length=0,
    )


def anno_axes(loc):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "ax" in kwargs:
                ax = kwargs["ax"]
            else:
                ax = plt.gca()
            ax.axis("off")
            if loc in ["top", "bottom"]:
                ax.set_xlim((0, 1))
                if loc == "bottom":
                    ax.invert_yaxis()
            else:
                ax.set_ylim(0, 1)
                if loc == "left":
                    ax.invert_xaxis()
            func(*args, **kwargs)

        return wrapper

    return decorator


@anno_axes("right")
def spaced_barplot(ax, **kwargs):
    ax.barh(**kwargs)


pcm_display_kwargs = dict(edgecolor="face", linewidth=0.2)


def test_heatmap3():

    df = pd.DataFrame(np.random.random_integers(0, 10, (9, 9))).set_axis(
        ["Aga", "bg", "Ag", "CD", "pP", "1", "8", "3", "0"], axis=1, inplace=False
    )
    row_clusters = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    col_clusters = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1
    col_label_colors = np.array(
        sns.color_palette("Set1", np.unique(col_clusters).shape[0])
    )[col_clusters]
    cluster_to_color = pd.Series({1: "blue", 2: "green", 3: "red"})
    # labels are always str
    label_to_color = dict(
        zip(list(df.index.astype(str)), np.repeat(["blue", "red", "black"], 3))
    )

    co.plotting.find_stretches2()



    fig, ax = plt.subplots(1, 1, dpi=180, figsize=(2.5, 2.5))
    # ax.imshow(pd.DataFrame(np.random.randn(9, 9)), rasterized=True)
    res = co.plotting.heatmap3(
        df=df,
        ax=ax,
        xticklabels=True,
        xticklabel_colors=col_label_colors,
        yticklabels=True,
        yticklabel_colors=label_to_color,
        pcolormesh_args=dict(**pcm_display_kwargs),
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        row_spacer_size=0.1,
        col_spacer_size=0.1,
        show_guide=True,
        # is_categorical=True,
        # heatmap_args={},
        # cmap='Set3'
    )
    ax.tick_params(bottom=True, left=True, labelleft=True, size=5, width=0.5)

    # plt.setp(
    #     ax.get_xticklabels(),
    #     rotation=90,
    #         # fontstretch='extra-condensed',
    #         # rotation_mode='anchor',
    #     ha="center",
    #     va="center",
    #         # y=-0.2,
    #         # x = 0.3,
    #         # clip_on=True,
    #     # bbox=dict(color="blue", alpha=0.8),
    #     # multialignment="center",
    # )
    # fig.savefig('/home/stephen/temp/test.pdf')
    #
    # dir(ax.get_xticklabels()[0])


def npinsert():
    arr = np.ones((4, 4))
    np.insert(arr, [1, 2], [0] * 4, axis=0)


def end():
    pass
