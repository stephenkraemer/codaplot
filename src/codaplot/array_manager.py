"""Use array of plotting functions to facilitate creation of complex gridspecs"""


# * Setup
import itertools
from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Tuple, Dict, Optional, List, Iterable, Union
import toolz as tz

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.legend as mlegend
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np

from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list

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


def array_to_figure(
    plot_array: np.ndarray,
    figsize: Tuple[float],
    merge_by_name: Union[bool, List[str]] = True,
    layout_pads: Optional[Dict] = None,
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


    Parameters
    ----------
    plot_array: array of dicts detailing plotting instructions
    merge_by_name: bool or List of names to be considered for merging

    Returns
    --------
    dict with keys
      - axes_d (dict name -> Axes)
      - axes_arr (array of Axes matching figure layout)
      - fig
    """

    if layout_pads is None:
        layout_pads = dict()
    if not constrained_layout:
        layout_pads.pop("h_pad", None)
        layout_pads.pop("w_pad", None)

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
    fig.set_constrained_layout_pads(**layout_pads)
    gs = gridspec.GridSpec(
        inner_array.shape[0],
        inner_array.shape[1],
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        figure=fig,
        **(layout_pads if not constrained_layout else {}),
    )

    axes_d = {}
    axes_arr = np.empty_like(inner_array)
    plot_returns_d = {}
    plot_returns_arr = np.empty_like(inner_array)
    seen = []
    for t in itertools.product(
        range(inner_array.shape[0]), range(inner_array.shape[1])
    ):
        ge = inner_array[t]
        # Do not create Axes for Spacers (np.nan)
        # However, do create Axes for GridElements without plotting function (below)
        if ge is None:
            continue

        # do not pop elements, they are still needed for inspection after this function
        ge_name = ge.get("_name", None)
        ge_func = ge.get("_func", None)
        ge_args = ge.get("_args", [])
        ge_kwargs = tz.dissoc(ge, "_name", "_func", "_args", "_supply")

        if (
            ge_name is not None
            and merge_by_name
            and (ge_name in merge_by_name if isinstance(merge_by_name, list) else True)
        ):
            if ge_name in seen:
                continue
            else:
                # allow skipping None elements, assume these are spacers
                col_slice_end = t[1] + 1
                for next_col in range(t[1] + 1, inner_array.shape[1]):
                    next_ge = inner_array[t[0], next_col]
                    if next_ge is None:
                        continue
                    next_ge_name = next_ge.get("_name")
                    if next_ge_name == ge_name:
                        col_slice_end = next_col + 1
                    if next_ge_name != ge_name:
                        break

                # allow skipping None elements, assume these are spacers
                row_slice_end = t[0] + 1
                for next_row in range(t[0] + 1, inner_array.shape[0]):
                    next_ge = inner_array[next_row, t[1]]
                    if next_ge is None:
                        continue
                    next_ge_name = next_ge.get("_name")
                    if next_ge_name == ge_name:
                        row_slice_end = next_row + 1
                    if next_ge_name != ge_name:
                        break

                # TODO: this assertion has a naive handling of None elements
                if row_slice_end > t[0] + 1 or col_slice_end > t[1] + 1:
                    assert (
                        len(
                            {
                                ge_name
                                for ge in np.ravel(
                                    inner_array[
                                        t[0] : row_slice_end, t[1] : col_slice_end
                                    ]
                                )
                                if ge is not None
                            }
                        )
                        == 1
                    )
                    t = (slice(t[0], row_slice_end), slice(t[1], col_slice_end))
                seen.append(ge_name)

        ax = fig.add_subplot(gs[t])
        axes_arr[t] = ax
        if ge_name is not None:
            if ge_name in axes_d:
                if isinstance(axes_d[ge_name], list):
                    axes_d[ge_name].append(ax)
                else:
                    axes_d[ge_name] = [axes_d[ge_name], ax]
            else:
                axes_d[ge_name] = ax

        # Without plotting function, we stop after having created the Axes
        if ge_func is None:
            continue
        # Plotting func is given, create plot
        # Add Axes to func args if necessary
        # If ax is not in plotter args, we assume that this is a pyplot function relying on state

        # Do no use getfullargspec, it will fail on decorated funtions, even if they correctly use functools.wrap
        # see also: https://hynek.me/articles/decorators/
        plotter_args = signature(ge_func).parameters.keys()
        if "ax" in plotter_args:
            ge_kwargs["ax"] = ax
        res = ge_func(*ge_args, **ge_kwargs)
        plot_returns_arr[t] = res
        if ge_name is not None:
            plot_returns_d[ge_name] = res

    return {
        "axes_d": axes_d,
        "axes_arr": axes_arr,
        "plot_returns_d": plot_returns_d,
        "plot_returns_arr": plot_returns_arr,
        "fig": fig,
    }


def get_plot_id_array(plot_arr, none_repr="") -> np.ndarray:
    """Create readable array with str representations of plot array elements

    Parameters
    ----------
    plot_arr: array specifying plot layout
    none_repr: str, representing empty array fields

    Returns
    -------
    array of strings
        Readable description of all elements in a plot array
    """
    max_chars = 20

    def format_object_strings(o):
        # For dict, print name or func if available, otherwise it is a placeholder to create an emtpy Axes
        if isinstance(o, dict):
            if o.get("_name"):
                s = o["_name"]
            elif o.get("_func"):
                s = o["_func"].__name__
            else:
                s = "spacer"
        # For tuple (size spec), print abbreviation of the size
        elif isinstance(o, tuple):
            size = f"{o[0]:.2f}"
            kind = "a" if o[1] == "abs" else "r"
            s = size + kind
        else:  # None
            s = none_repr
        # return at most 10 chars, right aligned
        if len(s) > max_chars:
            s = s[0 : max_chars // 2] + ".." + s[-max_chars // 2 :]
        # noinspection PyStringFormat
        return f"{{:>{max_chars}}}".format(s)

    return np.vectorize(format_object_strings)(plot_arr)


def print_plot_id_arr(plot_arr):
    """Print readable representation of contents of a plot array"""
    id_arr = get_plot_id_array(plot_arr, none_repr="")
    print(np.array2string(id_arr, max_line_width=10000, separator=""))


def compute_gridspec_ratios(sizes: np.ndarray, total_size_in: float) -> np.ndarray:
    """Convert mixture of absolute and relative column/row sizes to relative-only

    The result depends on the figsize, the function will first allocate space for
    all absolute sizes, and then divide the rest proportionally between the relative
    sizes.

    Parameters
    ----------
    sizes: array of size specs
        eg. [(1, 'rel'), (2, 'abs'), (3, 'rel')] all sizes in inch
    total_size_in:
        size of the corresponding dimension of the figure (inch)


    Returns
    -------
    np.ndarray
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



def add_guides(guide_spec_l, ax, guide_titles=None, xpad_in=0.1, ypad_in=0.1) -> None:
    """Pack Legends and colorbars into ax

    Parameters
    ----------
    guide_spec_l: list of dicts (None elements are allowed and ignored),
        each dict provides key-value pairs required to either draw a legend or a colorbar.
           For legend
               - handles
               - labels
               - optional: other Legend args, eg. title
           For colorbars:
               - mappable (ScalarMappable, eg. returned by pcolormesh, scatter)
               - optional: other colorbar args
               - optional: title
    guide_titles:
        if given, only guides with these titles are considered, in the given order. Multiple guides with the same title will all be printed, in indefinite order.
    ax: Axes onto which to draw the guides.
        The Axes is assumed to be completely empty.
    xpad_in: padding between guides in inch
        (final size may differ if a layout optimization such as constrained_layout is applied)
    ypad_in: padding between guides in inch (size may vary, see xpad_in)
    """
    ax.axis("off")
    if guide_titles:
        selected_guide_specs_l = []
        for title in guide_titles:
            for guide_spec in guide_spec_l:
                if guide_spec is not None and guide_spec.get("title") == title:
                    selected_guide_specs_l.append(guide_spec)
    else:
        selected_guide_specs_l = guide_spec_l

    guide_placement_params_df = get_guide_placement_params(selected_guide_specs_l, ax)
    place_guides(
        ax=ax, placement_df=guide_placement_params_df, xpad_in=xpad_in, ypad_in=ypad_in
    )



def get_guide_placement_params(guide_spec_l: List[Optional[Dict]], ax):
    """Get dataframe with required height and width for all guides

    Parameters
    ----------
    guide_spec_l: list of guidespec dicts
    ax: Axes to draw legend on
        legend will not be drawn by this function, Axes required to estimate guide size requirements

    Returns
    -------
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


def place_guides(ax: Axes, placement_df: pd.DataFrame, xpad_in=0.2, ypad_in=0.2 / 2.54) -> None:
    """Pack guides into Axes

    If constrained_layout is enabled, the Axes may be shrunk. This happens after
    this function was applied, and may bring the guides to close together, or even
    make them overlap. This can be adjusted for by increasing xpad_in and ypad_in, which
    also become shrinked by constrained_layout.

    Parameters
    ----------
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


def _add_legend(ax, curr_x, curr_y, row_ser):
    """Add legend to axis
    Helper method for place_guides
    """
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



def _add_cbar_inset_axes(row_ser, ax, curr_x, curr_y):
    """Add cbar to ax (as inset axes)

    Helper method for place_guides
    """
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
        title_height_axes_coord = title_axes_bbox.height + (
            mpl.rcParams["legend.labelspacing"]
            * mpl.rcParams["legend.fontsize"]
            / 72
            / ax_height_in
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
        ax.text(curr_x, curr_y, row_ser["title"], fontdict=fontdict)

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


# helpers

def get_axes_dim_in(ax):
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    ax_width_in, ax_height_in = bbox.width, bbox.height
    return ax_width_in, ax_height_in


cross_plot_supply_tasks = dict(
        center=dict(
                row_spacing_group_ids=["row_spacing_group_ids"],
                row_spacer_sizes=["row_spacer_sizes"],
                col_spacing_group_ids=["col_spacing_group_ids"],
                col_spacer_sizes=["col_spacer_sizes"],
        ),
        leftright=dict(
                row_spacing_group_ids=[
                    "row_spacing_group_ids",
                    "spacing_group_ids",
                    "group_ids",
                ],
                row_spacer_sizes=["row_spacer_sizes", "spacer_sizes"],
        ),
        topbottom=dict(
                col_spacing_group_ids=[
                    "col_spacing_group_ids",
                    "spacing_group_ids",
                    "group_ids",
                ],
                col_spacer_sizes=["col_spacer_sizes", "spacer_sizes"],
        ),
)
cross_plot_adjust_coord_tasks = dict(topbottom=["x"], leftright=["y"])
cross_plot_align_tasks = ("data", "df", "arr")

Array1DLike = Union[List, pd.Series, np.ndarray]

# noinspection PyDefaultArgument
def cross_plot(
    center: Union[List, np.ndarray],
    center_row_sizes=None,
    center_col_sizes=None,
    center_row_pad=(0.05, "rel"),
    center_col_pad=(0.05, "rel"),
    center_margin_ticklabels=False,
    xticklabels=None,
    yticklabels=None,
    legend_side="right",
    legend_extent=("center",),  # select from 'top, 'bottom', 'center'
    legend_args: Optional[Dict] = None,
    legend_axes_selectors: Optional[List[Union[str, Tuple[int, int]]]] = None,
    legend_size=(1, "rel"),
    legend_pad=(0.2, "abs"),
    pads_around_center: Optional[Union[float, Tuple[float]]] = None,
    figsize=(5, 5),
    constrained_layout=True,
    layout_pads: Optional[Dict] = None,
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
    row_order: Optional[Union[Array1DLike, List[Array1DLike]]] = None,
    col_order: Optional[Union[Array1DLike, List[Array1DLike]]] = None,
    default_func=co.plotting.heatmap3,
    default_func_kwargs: Optional[Dict] = None,
    row_spacing_group_ids=None,
    row_spacer_sizes=0.02,
    col_spacing_group_ids=None,
    col_spacer_sizes=0.02,
    supply_tasks=cross_plot_supply_tasks,
    adjust_coords_tasks=cross_plot_adjust_coord_tasks,
    align_args=cross_plot_align_tasks,
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
        pads_around_center: (top, right, bottom, left) - clockwise sides. pads around center plots as (number, type), e.g. (1, 'rel') or (0.5, 'abs')
        legend_axes_selectors: list of Axes names and axes coordinates (x, y), specifies selection and order of displayed guides

    Returns:
        dict with keys
        - fig: figure
        - array: array of plot specs with aligned
        - axes_d
        - axes_arr

    """

    # TODO: copy dicts before modifying in place

    if legend_args is None:
        legend_args = {}
    if default_func_kwargs is None:
        default_func_kwargs = {}
    if isinstance(pads_around_center, tuple):
        pads_around_center = [pads_around_center] * 4
    elif pads_around_center is None:
        pads_around_center = [None] * 4
    pads_around_center_d = dict(
        zip("top right bottom left".split(), pads_around_center)
    )
    # replace None sides with empty lists, to allow for iteration
    top, right, bottom, left = [
        x if x is not None else [] for x in [top, right, bottom, left]
    ]
    # center is list, list of list, 1d array or 2d array
    # convert to 2d array
    center_arr = np.array(center)
    if center_arr.ndim == 1:
        center_arr = center_arr[np.newaxis, :]

    if row_order.ndim == 2:
        row_linkage = row_order
        row_idx = leaves_list(row_linkage)
    else:
        assert row_order.ndim == 1
        row_linkage = None
        row_idx = row_order

    if col_order.ndim == 2:
        col_linkage = col_order
        col_idx = leaves_list(col_linkage)
    else:
        assert col_order.ndim == 1
        col_linkage = None
        col_idx = col_order

    if align_args:
        if row_spacing_group_ids is not None:
            row_spacing_group_ids = index_list_np_or_pd(row_spacing_group_ids, row_idx)
        if col_spacing_group_ids is not None:
            col_spacing_group_ids = index_list_np_or_pd(col_spacing_group_ids, col_idx)
        # spacer sizes are not aligned

    # add default func and default func kwargs
    for elem in itertools.chain.from_iterable(
        it for it in [left, top, bottom, right, np.ravel(center_arr)] if it is not None
    ):
        if elem.get("_func") is None:
            elem["_func"] = default_func
        if elem["_func"] == default_func:
            # edit elem in place
            for default_k, default_v in default_func_kwargs.items():
                if default_k in elem:
                    continue
                else:
                    elem[default_k] = default_v

    # fill supplied kwargs
    for name, elems in zip(
        "center topbottom leftright topbottom leftright".split(),
        [center_arr.flatten(), top, right, bottom, left],
    ):
        for elem in elems:
            for var_name, supply_targets in cross_plot_supply_tasks[name].items():
                if locals().get("var_name") is None:
                    continue
                if any(supply_target in elem for supply_target in supply_targets):
                    continue
                elem_func = elem.get("_func")
                if elem_func:
                    keys = signature(elem_func).parameters.keys()
                    for supply_target in supply_targets:
                        if supply_target in keys:
                            elem[supply_target] = locals()[var_name]
                            break

    # align
    if align_args:
        for name, elems in zip(
            "center  topbottom leftright topbottom leftright".split(),
            [center_arr.flatten(), top, right, bottom, left],
        ):
            if name == "center":
                df_2d_slice = row_idx, col_idx
                arr_2d_slice = np.ix_(row_idx, col_idx)
                df_1d_slice = arr_1d_slice = None
            elif name == "topbottom":
                df_2d_slice = arr_2d_slice = slice(None), col_idx
                df_1d_slice = arr_1d_slice = col_idx
            else:  # name == 'leftright':
                df_2d_slice = arr_2d_slice = row_idx, slice(None)
                df_1d_slice = arr_1d_slice = row_idx

            for elem in elems:
                for align_target in align_args:
                    if align_target in elem:
                        val = elem[align_target]
                        if isinstance(val, pd.DataFrame):
                            elem[align_target] = val.iloc[df_2d_slice]
                        elif isinstance(val, np.ndarray) and val.ndim == 2:
                            elem[align_target] = val[arr_2d_slice]
                        elif name == "center":
                            raise ValueError("Dont know how to align")
                        elif isinstance(val, pd.Series):
                            elem[align_target] = val.iloc[df_1d_slice]
                        elif isinstance(val, np.ndarray) and val.ndim == 1:
                            elem[align_target] = val[arr_1d_slice]
                        else:
                            raise ValueError("Dont know how to align")

    # adjust coords
    for name, elems in zip(
        "topbottom leftright topbottom leftright topbottom leftright".split(),
        [center_arr.flatten(), center_arr.flatten(), top, right, bottom, left],
    ):
        if name == "topbottom":
            adjust_coords_args = dict(
                spacing_group_ids=col_spacing_group_ids, spacer_sizes=col_spacer_sizes
            )
        else:
            adjust_coords_args = dict(
                spacing_group_ids=row_spacing_group_ids, spacer_sizes=row_spacer_sizes
            )
        for elem in elems:
            if elem is None or elem.get("_func") is None:
                continue
            else:
                for adjust_target in cross_plot_adjust_coord_tasks[name]:
                    if adjust_target in elem:
                        elem[adjust_target] = adjust_coords(
                            elem[adjust_target], **adjust_coords_args
                        )

    cut_dendrogram_defaults = dict(
            stop_at_cluster_level=False,
            min_height=0,
            min_cluster_size=0,
    )
    # add dendrograms to left / top
    if isinstance(col_dendrogram, dict) or col_dendrogram:
        if not isinstance(col_dendrogram, dict):
            col_dendrogram = {}
        else:
            col_dendrogram = deepcopy(col_dendrogram)
        col_dendrogram = tz.merge(cut_dendrogram_defaults, col_dendrogram)
        col_dendrogram["orientation"] = "vertical"
        # if no cluster ids are specified, use col_spacing_groups (may be None)
        if 'cluster_ids_data_order' not in col_dendrogram:
            col_dendrogram['cluster_ids_data_order'] = col_spacing_group_ids
        # if cluster ids are specified, align them if alignment mode is on
        elif col_dendrogram['cluster_ids_data_order'] is not None and align_args:
            col_dendrogram['cluster_ids_data_order'] = col_dendrogram['cluster_ids_data_order'][col_idx] 
            
        assert (
            col_order.ndim == 2
        ), "col order not linkage mat, but dendrogram requested"
        top = [
            dict(
                _func=co.plotting.cut_dendrogram,
                linkage_mat=col_order,
                spacing_groups=col_spacing_group_ids,
                spacer_size=col_spacer_sizes,
                **col_dendrogram,
            )
        ] + (top if top else [])
        top_sizes = [col_dendrogram_size] + (top_sizes if top_sizes else [])

    if isinstance(row_dendrogram, dict) or row_dendrogram:
        if not isinstance(row_dendrogram, dict):
            row_dendrogram = {}
        else:
            row_dendrogram = deepcopy(row_dendrogram)
        row_dendrogram = tz.merge(cut_dendrogram_defaults, row_dendrogram)
        # if no cluster ids are specified, use row_spacing_groups (may be None)
        if 'cluster_ids_data_order' not in row_dendrogram:
            row_dendrogram['cluster_ids_data_order'] = row_spacing_group_ids
        # if cluster ids are specified, align them if alignment mode is on
        elif align_args:
            row_dendrogram['cluster_ids_data_order'] = row_dendrogram['cluster_ids_data_order'][row_idx] 
        row_dendrogram["orientation"] = "horizontal"
        assert (
            row_order.ndim == 2
        ), "row order not linkage mat, but dendrogram requested"
        left = [
            dict(
                _func=co.plotting.cut_dendrogram,
                linkage_mat=row_order,
                spacing_groups=row_spacing_group_ids,
                spacer_size=row_spacer_sizes,
                **row_dendrogram,
            )
        ] + (left if left else [])
        left_sizes = [row_dendrogram_size] + (left_sizes if left_sizes else [])

    # get array size, init array
    def len_or_none(x):
        if x is None:
            return 0
        else:
            return len(x)

    # configure for margin ticklabels if requested
    if center_margin_ticklabels:
        xticklabels_val = True if xticklabels is None else xticklabels
        yticklabels_val = True if yticklabels is None else yticklabels
        for row_idx, col_idx in itertools.product(
            range(center_arr.shape[0]), range(center_arr.shape[1])
        ):
            elem = center_arr[row_idx, col_idx]
            elem["xticklabels"] = (
                xticklabels_val if row_idx == center_arr.shape[0] - 1 else False
            )
            elem["yticklabels"] = (
                yticklabels_val if col_idx == center_arr.shape[1] - 1 else False
            )
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
    ## between center plots
    if center_row_pad:
        row_spacer = np.empty((1, plot_arr.shape[1]), dtype=object)
        row_spacer[0, -1] = center_row_pad
        plot_arr = np.insert(
            plot_arr,
            slice(first_center_row + 1, last_center_row_p1),
            row_spacer,
            axis=0,
        )
    # adjust center_row indices to account for new rows
    last_center_row_p1 += center_arr.shape[0] - 1

    ## before and after center plots
    if pads_around_center_d["top"]:
        row_spacer = np.empty((1, plot_arr.shape[1]), dtype=object)
        row_spacer[0, -1] = pads_around_center_d["top"]
        plot_arr = np.insert(plot_arr, first_center_row, row_spacer, axis=0)
    # adjust center_row indices to account for new row
    first_center_row += 1
    last_center_row_p1 += 1

    if pads_around_center_d["bottom"]:
        row_spacer = np.empty((1, plot_arr.shape[1]), dtype=object)
        row_spacer[0, -1] = pads_around_center_d["bottom"]
        plot_arr = np.insert(plot_arr, last_center_row_p1, row_spacer, axis=0)

    # taking into account the new rows, add enlarged col spacers
    ## between center plots
    if center_col_pad:
        col_spacer = np.empty((plot_arr.shape[0], 1), dtype=object)
        col_spacer[-1, 0] = center_col_pad
        plot_arr = np.insert(
            plot_arr,
            slice(first_center_col + 1, last_center_col_p1),
            col_spacer,
            axis=1,
        )
    # adjust center col indices to account for inserted columns
    last_center_col_p1 += center_arr.shape[1] - 1

    ## left and right of center plots
    if pads_around_center_d["left"]:
        col_spacer = np.empty((plot_arr.shape[0], 1), dtype=object)
        col_spacer[-1, 0] = pads_around_center_d["left"]
        plot_arr = np.insert(plot_arr, first_center_col, col_spacer.flatten(), axis=1)
    # adjust center col indices to account for inserted column
    first_center_col += 1
    last_center_col_p1 += 1

    if pads_around_center_d["right"]:
        col_spacer = np.empty((plot_arr.shape[0], 1), dtype=object)
        col_spacer[-1, 0] = pads_around_center_d["right"]
        plot_arr = np.insert(plot_arr, last_center_col_p1, col_spacer.flatten(), axis=1)

    # Add legend Axes, rely on merging functionality of array_to_figure
    # cover a selection of the top, center and bottom rows based on legend_extent
    if legend_side is not None:
        if legend_side != "right":
            raise NotImplementedError
        legend_arr = np.empty(plot_arr.shape[0], dtype=object)
        legend_spacer = dict(_name="legend_ax", _func=spacer)
        if "top" in legend_extent:
            legend_arr[:first_center_row] = legend_spacer
        if "center" in legend_extent:
            legend_arr[first_center_row:last_center_row_p1] = legend_spacer
        if "bottom" in legend_extent:
            legend_arr[last_center_row_p1:-1] = legend_spacer
        legend_arr[-1] = legend_size
        legend_spacer_arr = np.empty(plot_arr.shape[0], dtype=object)
        legend_spacer_arr[-1] = legend_pad
        plot_arr = np.insert(plot_arr, plot_arr.shape[1] - 1, legend_spacer_arr, axis=1)
        plot_arr = np.insert(plot_arr, plot_arr.shape[1] - 1, legend_arr, axis=1)

    # Create plot, this will also collect guide specs
    res = array_to_figure(
        plot_array=plot_arr,
        figsize=figsize,
        layout_pads=layout_pads,
        constrained_layout=constrained_layout,
        merge_by_name=["legend_ax"],
    )

    # Add guides
    if legend_side is not None:
        if legend_axes_selectors is None:
            guide_spec_l = res["plot_returns_arr"].flatten()
        else:
            guide_spec_l = []
            for selector in legend_axes_selectors:
                if isinstance(selector, str):
                    # we may have a scalar ArrayElement or List[ArrayElement] under one name
                    to_add = res["plot_returns_d"][selector]
                    if isinstance(to_add, list):
                        guide_spec_l.extend(to_add)
                    else:
                        guide_spec_l.append(to_add)
                else:  # coord tuple
                    guide_spec_l.append(res["plot_returns_arr"][selector])

        add_guides(
            # Note that the order is relevant if guide_titles is not given
            # Flatten may not provide the most intuitive order
            guide_spec_l=guide_spec_l,
            ax=res["axes_d"]["legend_ax"],
            # guide titles, padding...
            **legend_args,
        )

    return res, plot_arr


def index_list_np_or_pd(o: Union[List, pd.Series, pd.DataFrame, np.ndarray],
                        idx: Array1DLike):
    """Index object with idx for list, array, series, dataframe

    Args:
        o: only 1D list
        idx:

    Returns:

    """
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return o.iloc[idx]
    elif isinstance(o, (list, np.ndarray)):
        return o[idx]
    else:
        raise TypeError()


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


def anno_axes(loc, prune_all=False, normalized=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get Axes
            if "ax" in kwargs:
                ax = kwargs["ax"]
            else:
                ax = plt.gca()

            # Remove both axis completely, or let informative axis remain
            if prune_all:
                ax.axis("off")
            else:
                if loc == "top":
                    sns.despine(ax=ax, left=True)
                    ax.tick_params(labelbottom=False, bottom=False)
                elif loc == "bottom":
                    sns.despine(ax=ax, bottom=True)
                    ax.tick_params(labelbottom=False, bottom=False)
                elif loc == "right":
                    sns.despine(ax=ax, left=True)
                    ax.tick_params(labelleft=False, left=False)
                elif loc == "left":
                    sns.despine(ax=ax, left=True)
                    ax.tick_params(labelleft=False, left=False)

            # Set x or ylim to accomodate normalized coordinates
            if normalized:
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
