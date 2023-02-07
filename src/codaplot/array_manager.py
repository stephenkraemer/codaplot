"""Use array of plotting functions to facilitate creation of complex gridspecs"""


# * Setup
import itertools
from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import codaplot as co
import codaplot.utils as coutils
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toolz as tz
from codaplot.plotting import adjust_coords
from codaplot.utils import cbar_change_style_to_inward_white_ticks
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage


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
    dpi: int = 180,
    merge_by_name: Union[bool, List[str]] = True,
    layout_pads: Optional[Dict] = None,
    constrained_layout=True,
) -> Dict:
    """Convert array of plotting instruction dicts into figure, provide access to Axes

    This is the function currently used by cross_plot

    inner array elements are None (for spacers)  or dict, the following fields have special meaning
    _func : plotting function for the axes to be created
    _name : name of the axes to be created. can the name be left out?
    _args : tuple of positional arguments passed to _func
    _supply : todo

    all other fields are passed as kwargs to _func

    details
    _func
        may be empty, in this case, an Axes is created, but no plotting action is taken
    _name
        if adjacent array elements have the same _name, and merge_by_name == True, the adjacent elements are merged. this is always done in a rectangle (ie by taking a rectangular slice across the x and y axis of the plotting array), and it is ascerted that all elements in that rectangle have the name, or are None (=spacers)

    outer array elements specify the sizes

    example


    [[{'_func': plot_stuff, '_name': 'plot1'}, none, (1, 'rel')],
     [{'_name': 'plot1}, {'_func': plot_2, '_name': 'plot2', '_args' = (2, 3), another_arg='abc}, (1, 'abs')],
     [  (1, 'abs'),   (1, 'rel'),       None]],

    this function
    - creates axes for each indicated subplot, merging adjacent names if requested
    - runs the specified plotting func if it is available
    - provides acces to the created axes via an array matched to the input plotting array
    - provides access to named plot elements via a dict with keys given by named plot elements
    - provides acces to return values of the created axes under the plot name key, which is useful to generate legends etc. by recording returns of pcolormesh etc.

    by comparison, subplot_mosaic and co
    - creates axes for each indicated subplot, merging adjacent names if requested
    - provides access to named plot elements via a dict with keys given by named plot elements


    Parameters
    ----------
    plot_array: array of dicts detailing plotting instructions
        reserved kwargs: _func, _name, _args
    merge_by_name: bool or List of names to be considered for merging
    dpi

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
    fig = plt.figure(constrained_layout=constrained_layout, figsize=figsize, dpi=dpi)
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


def add_guides(
    guide_spec_l,
    ax,
    cbar_styling_func: Callable,
    guide_titles=None,
    xpad_in=0.1,
    ypad_in=0.1,
    cbar_title_as_label=False,
    legend_kwargs=None,
) -> None:
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
               - optional: cbar_styling_func_kwargs
    guide_titles:
        if given, only guides with these titles are considered, in the given order. Multiple guides with the same title will all be printed, in indefinite order.
    ax: Axes onto which to draw the guides.
        The Axes is assumed to be completely empty.
    xpad_in: padding between guides in inch
        (final size may differ if a layout optimization such as constrained_layout is applied)
    ypad_in: padding between guides in inch (size may vary, see xpad_in)
    cbar_styling_func
        called on colorbars to modify their aesthetics
    cbar_title_as_label
        if true, draw cbar title as vertical or horizontal axis label (depending on cbar orientation), if false,
        draw cbar title as a title on top of the cbar Axes
    legend_kwargs
        passed to place_guides, will eventually be used in mlegend.Legend
    """

    if legend_kwargs is None:
        legend_kwargs = {}

    ax.axis("off")
    if guide_titles:
        selected_guide_specs_l = []
        for title in guide_titles:
            for guide_spec in guide_spec_l:
                if guide_spec is not None and guide_spec.get("title") == title:
                    selected_guide_specs_l.append(guide_spec)
    else:
        selected_guide_specs_l = guide_spec_l

    guide_placement_params_df = get_guide_placement_params(
        selected_guide_specs_l, ax, legend_kwargs
    )
    place_guides(
        ax=ax,
        placement_df=guide_placement_params_df,
        xpad_in=xpad_in,
        ypad_in=ypad_in,
        cbar_styling_func=cbar_styling_func,
        cbar_title_as_label=cbar_title_as_label,
        legend_kwargs=legend_kwargs,
    )


def get_guide_placement_params(guide_spec_l: List[Optional[Dict]], ax, legend_kwargs):
    """Get dataframe with required height and width for all guides

    Parameters
    ----------
    guide_spec_l: list of guidespec dicts
    ax: Axes to draw legend on
        legend will not be drawn by this function, Axes required to estimate guide size requirements

    Returns
    -------
     pd.DataFrame with columns: title, height, width, contents, styling_func_kwargs
         contents is a reference to the guide specification dictionary from guide_spec_l, styling_func_kwargs is a reference to the corresponding dict from guide_spec_l, or an empty dict. currently, styling_func_kwargs is only handled for colobars.
         (because its convenient to have these things together in the same frame)
    """
    # Note that 'title' is not a colorbar arg. It is removed before passing on the cbar_args
    placement_df = pd.DataFrame(
        columns=["title", "height", "width", "contents", "styling_func_kwargs"]
    )
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
                    styling_func_kwargs=guide_d.pop("styling_func_kwargs", {}),
                    height=height,
                    width=width,
                    contents=guide_d,
                ),
                ignore_index=True,
            )
        else:  # We have a standard legend

            # Create a throw-away legend object without appropriate placing parameters,
            # to get the legend size

            curr_legend_kwargs = (
                legend_kwargs
                | pd.Series(guide_d)
                .drop(
                    ["handles", "labels"]
                    + (["title"] if "title" in guide_d else [])
                    + (["_constructor"] if "_constructor" in guide_d else [])
                )
                .to_dict()
            )

            try:
                constructor = guide_d["_constructor"]
            except KeyError:
                constructor = mlegend.Legend

            l = constructor(
                ax,
                guide_d["handles"],
                guide_d["labels"],
                title=guide_d.get("title", None),
                **curr_legend_kwargs,
            )

            r = ax.figure.canvas.get_renderer()
            bbox = l.get_window_extent(r).transformed(ax.transAxes.inverted())
            placement_df = placement_df.append(
                dict(
                    title=guide_d.get("title", None),
                    # not yet considered for legends
                    styling_func_kwargs=None,
                    height=bbox.height,
                    width=bbox.width,
                    contents=guide_d,
                ),
                ignore_index=True,
            )
    return placement_df


# *** Place guides


def place_guides(
    ax: Axes,
    placement_df: pd.DataFrame,
    cbar_styling_func: Callable,
    xpad_in=0.5 / 2.54,
    ypad_in=0.5 / 2.54,
    cbar_title_as_label=False,
    legend_kwargs=None,
) -> None:
    """Pack guides into Axes

    If constrained_layout is enabled, the Axes may be shrunk. This happens after
    this function was applied, and may bring the guides too close together, or even
    make them overlap. This can be adjusted for by increasing xpad_in and ypad_in, which
    also become shrinked by constrained_layout.

    Parameters
    ----------
    ax: guide Axes, assumed to be empty
    placement_df: columns title, height, width, contents
        contents is a guide spec dictionary, see get_guide_placement_params
        for details
    cbar_styling_func
        function to change cbar aesthetics; currently the function is passed i) cbar and ii) **placement_df['styling_func_kwargs'] (if not None). Currently there is no equivalent for legend styling.
    xpad_in: padding between guides along x axis, in inch
    ypad_in: padding between guides along y axis, in inch
    cbar_title_as_label
        if true, draw cbar title as vertical or horizontal axis label (depending on cbar orientation), if false,
        draw cbar title as a title on top of the cbar Axes
    legend_kwargs
        passed to mlegend.Legend via _add_legend
    """

    # ax_width_in, ax_height_in = get_axes_dim_in(ax)
    # min_xpad_ax_coord = xpad_in / ax_width_in
    # min_ypad_ax_coord = ypad_in / ax_width_in
    min_xpad_data_coords, _ = coutils.convert_inch_to_data_coords(xpad_in, ax)
    _, min_ypad_data_coords = coutils.convert_inch_to_data_coords(ypad_in, ax)

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
            _add_legend(ax, curr_x, curr_y, row_ser, legend_kwargs)
            curr_xpad = min_xpad_data_coords
            curr_ypad = min_ypad_data_coords
        else:
            # The colorbar is added as inset axes
            # we need to add additional padding to the predefined padding to
            # account for title and tick(labels)
            add_xpad_ax_coord, add_ypad_ax_coord = _add_cbar_inset_axes(
                row_ser, ax, curr_x, curr_y, cbar_styling_func, cbar_title_as_label
            )
            curr_xpad = min_xpad_data_coords + add_xpad_ax_coord
            curr_ypad = min_ypad_data_coords + add_ypad_ax_coord

        # TODO: padding around colorbars hardcoded in axes coordinates
        curr_y = curr_y - (row_ser.height + curr_ypad)
        next_x = max(next_x, curr_x + row_ser.width + curr_xpad)


def _add_legend(ax, curr_x, curr_y, row_ser, legend_kwargs):
    """Add legend to axis
    Helper method for place_guides

    Parameters:
        kwargs:
            passed to mlegend.Legend
    """

    guide_d = row_ser["contents"]

    try:
        constructor = guide_d["_constructor"]
    except KeyError:
        constructor = mlegend.Legend

    # fmt: off
    curr_legend_kwargs = (
        legend_kwargs
        | (
            pd.Series(guide_d).drop(
            ["handles", "labels"]
            + (["title"] if "title" in guide_d else [])
            + (["_constructor"] if "_constructor" in guide_d else [])
            ).to_dict())
        | dict(
            # avoid padding between legend corner and anchor point
            borderaxespad=0,
            loc="upper left",
            bbox_to_anchor=(curr_x, curr_y),
        )
    )
    # fmt: on

    l = constructor(
        ax,
        handles=row_ser.contents["handles"],
        labels=row_ser.contents["labels"],
        title=row_ser.contents.get("title", None),
        **curr_legend_kwargs,
    )

    if constructor is mlegend.Legend:
        # https://github.com/matplotlib/matplotlib/issues/12388
        l._legend_box.align = "left"

    ax.add_artist(l)

    # # it appears that the title of the legend artist is not considered by CL at the moment
    # # add a dummy object
    # # TODO: replace quickfix for legend artist title in CL
    # title = row_ser.contents.get("title", None)
    # if title:
    #     ax.text(curr_x, curr_y, title, zorder=0, color="white")


def _add_cbar_inset_axes(
    row_ser, ax, curr_x, curr_y, cbar_styling_func, cbar_title_as_label
):
    """Add cbar to ax (as inset axes)

    Helper method for place_guides

    Parameters:
        cbar_title_as_label
            if true, draw cbar title as vertical or horizontal axis label (depending on cbar orientation), if false,
            draw cbar title as a title on top of the cbar Axes
    """
    fig: Figure = ax.figure
    cbar_ax: Axes
    ax_width_in, ax_height_in = get_axes_dim_in(ax)
    fontdict = {
        "fontsize": mpl.rcParams["legend.title_fontsize"],
        "verticalalignment": "top",
    }
    orientation = row_ser.contents.get("orientation", "vertical")

    if row_ser["title"]:
        # get dimensions, but don't place title yet (removed at the end)
        # What would be more idiomatic code? This seems suboptimal
        # Just using fontsize and padding sizes is not ideal for estimating width...
        # Could not get size of Text without adding to Axes
        # so first add text, then remove it, then add at correct coord again
        title_text = ax.text(0, 0, row_ser["title"], fontdict=fontdict)
        r = fig.canvas.get_renderer()
        title_axes_bbox = title_text.get_window_extent(r).transformed(
            ax.transAxes.inverted()
        )
        if cbar_title_as_label and orientation == "vertical":
            # assumes that title is smaller than cbar!
            title_height_axes_coord = 0
            title_width_axes_coord = 0

            if orientation == "vertical":
                cbar_label_fontsize = mpl.rcParams["ytick.labelsize"]
            else:
                cbar_label_fontsize = mpl.rcParams["xtick.labelsize"]
            additional_width_due_to_cbar_label = title_axes_bbox.height + (
                mpl.rcParams["axes.titlepad"] * cbar_label_fontsize / 72 / ax_height_in
            )

        else:
            # cbar title as Axes title or horizontal orientation (where the label acts like a title,
            # just add the bottom
            title_width_axes_coord = title_axes_bbox.width
            title_height_axes_coord = title_axes_bbox.height + (
                mpl.rcParams["legend.labelspacing"]
                * mpl.rcParams["legend.fontsize"]
                / 72
                / ax_height_in
            )
            additional_width_due_to_cbar_label = 0
        title_text.remove()

        if cbar_title_as_label:
            inset_y = curr_y - row_ser.height
        else:  # cbar title as Axes title
            inset_y = curr_y - row_ser.height - title_height_axes_coord
    else:
        title_width_axes_coord = 0
        title_height_axes_coord = 0
        additional_width_due_to_cbar_label = 0
        inset_y = curr_y - row_ser.height

    # make room for title and same padding as with legends
    cbar_ax = ax.inset_axes(
        [
            curr_x,
            inset_y,
            row_ser.width,
            row_ser.height,
        ]
    )
    cbar = fig.colorbar(**row_ser.contents, cax=cbar_ax)
    # make sure that cbar ticks are available for the cbar styling func
    fig.canvas.draw()
    cbar_styling_func(cbar, **row_ser["styling_func_kwargs"])

    if row_ser["title"]:
        if cbar_title_as_label:
            if orientation == "vertical":
                cbar_ax.set_ylabel(row_ser["title"])
            else:
                cbar_ax.set_xlabel(row_ser["title"])
        else:  # cbar title as "Axes title"
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
    if orientation == "vertical":
        first_ticklabel = cbar_ax.get_yticklabels()[0]
    else:
        first_ticklabel = cbar_ax.get_xticklabels()[0]
    first_ticklabel_axes_bbox = first_ticklabel.get_window_extent().transformed(
        ax.transAxes.inverted()
    )
    ytick_size_axes_coord = mpl.rcParams["ytick.major.size"] * 1 / 72 / ax_width_in

    if orientation == "vertical":
        # no additional y padding required
        ypad_ax_coord = title_height_axes_coord
        # we have to add additional x padding
        # either the title or the yticklabels will protrude more, select
        # the outmost boundary
        xpad_ax_coord = max(
            title_width_axes_coord - row_ser["width"],
            (
                first_ticklabel_axes_bbox.width
                + ytick_size_axes_coord
                + additional_width_due_to_cbar_label
            ),
        )
    else:
        # no additional x padding required
        if title_width_axes_coord > row_ser["width"]:
            xpad_ax_coord = title_width_axes_coord - row_ser["width"]
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


cross_plot_supply_tasks_d = dict(
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
cross_plot_adjust_coord_tasks_d = dict(topbottom=["x"], leftright=["y"])
cross_plot_align_tasks = ("data", "df", "arr")

Array1DLike = Union[List, pd.Series, np.ndarray]

# noinspection PyDefaultArgument
def cross_plot(
    # main panel
    center_plots: Union[List, np.ndarray],
    center_row_sizes: Optional[List[Tuple[float, str]]] = None,
    center_col_sizes: Optional[List[Tuple[float, str]]] = None,
    center_row_pad: Tuple[float, str] = (0.05, "rel"),
    center_col_pad: Tuple[float, str] = (0.05, "rel"),
    # ticklabels
    xticklabels: Optional[Union[List[str], bool]] = None,
    yticklabels: Optional[Union[List[str], bool]] = None,
    center_margin_ticklabels: bool = False,
    # Clustering, partitioning and dendrograms
    row_order: Optional[Union[Array1DLike, List[Array1DLike]]] = None,
    row_linkage: Union[bool, Dict, np.ndarray] = False,
    row_partitioning: Union[bool, Dict] = False,
    row_dendrogram: Optional[Union[bool, Dict]] = None,
    row_dendrogram_size: Tuple[float, str] = (1 / 2.54, "abs"),
    col_order: Optional[Union[Array1DLike, List[Array1DLike]]] = None,
    col_linkage: Union[bool, Dict, np.ndarray] = False,
    col_partitioning: Union[bool, Dict] = False,
    col_dendrogram: Optional[Union[bool, Dict]] = None,
    col_dendrogram_size: Tuple[float, str] = (1 / 2.54, "abs"),
    # spacing
    row_spacing_group_ids: Optional[Array1DLike] = None,
    row_spacer_sizes: Union[float, Array1DLike] = 0.02,
    col_spacing_group_ids: Optional[Array1DLike] = None,
    col_spacer_sizes: Union[float, Array1DLike] = 0.02,
    # Annotation panels
    top_plots: Optional[Iterable[Dict]] = None,
    top_row_sizes: Optional[List[Tuple[float, str]]] = None,
    left_plots: Optional[Iterable[Dict]] = None,
    left_col_sizes: Optional[List[Tuple[float, str]]] = None,
    bottom_plots: Optional[Iterable[Dict]] = None,
    bottom_row_sizes: Optional[List[Tuple[float, str]]] = None,
    right_plots: Optional[Iterable[Dict]] = None,
    right_row_sizes: Optional[List[Tuple[float, str]]] = None,
    # optional legend axes
    legend_side: str = "right",
    legend_extent: Tuple[str, ...] = (
        "center",
    ),  # select from 'top, 'bottom', 'center'
    legend_args: Optional[Dict] = None,
    legend_axes_selectors: Optional[List[Union[str, Tuple[int, int]]]] = None,
    legend_size: Tuple[float, str] = (0.3, "rel"),
    legend_pad: Tuple[float, str] = (0.2, "abs"),
    # plotting args
    figsize: Tuple[float, float] = (5, 5),
    constrained_layout: bool = True,
    layout_pads: Optional[Dict] = None,
    pads_around_center: Optional[List[Tuple[float, str]]] = None,
    # default plotting function
    default_plotting_func: Callable = co.plotting.heatmap,
    default_plotting_func_kwargs: Optional[Dict] = None,
    # automatic alignment, spacing-adjustment and data distribution
    supply_tasks: Optional[Dict] = cross_plot_supply_tasks_d,
    adjust_coords_tasks: Optional[Dict] = cross_plot_adjust_coord_tasks_d,
    align_args: Optional[Tuple[str, ...]] = cross_plot_align_tasks,
    cbar_styling_func=cbar_change_style_to_inward_white_ticks,
):
    # noinspection PyUnresolvedReferences
    """

    Parameters
    ----------
    center_plots: Union[List, np.ndarray]
        1d or 2d array of center plots, eg. heatmaps or scatterplots drawn
        in a grid
    center_row_sizes: optionally, one or more size spec tuples
        if one size spec tuple and several rows, all rows have the same size
        defaults to (1, 'rel')
    center_col_sizes: optionally, one or more size spec tuples
        if one size spec tuple and several rows, all rows have the same size
        defaults to (1, 'rel')
    center_row_pad: size spec tuple
        padding between center plot rows
    center_col_pad: size spec tuple
        padding between center plot cols
    center_margin_ticklabels
        if True, set xticklabel and yticklabel to True at the bottom and left edges of the center plots, and to False elsewhere
    xticklabels: array like of ticklabels (str) or bool
        passed on to center plot arg xticklabels
    yticklabels: array like of ticklabels (str) or bool
        passed on to center plot arg xticklabels
    legend_side: one of "right", (not yet implemented)
        side where legend Axes is created
    legend_extent: Tuple[str, ...], select from 'top, 'bottom', 'center'
        legend Axes is created at legend_side, and takes up the space of the given center plot areas, e.g. ('center', 'bottom') will create a legend axes next to the center and bottom plotting areas, while leaving the top plotting area empty
    legend_args:
        kwargs passed to add_guides
    legend_axes_selectors: list of axes identifiers (name or gridspec coordinate tuple)
        if not None, guides are only shown for the selected Axes
    legend_size: size spec tuple
        legend width
    legend_pad: size spec tuple
        padding between legend and center plots
    pads_around_center: list with one or four size spec tuples
        If not None, paddings are added around the center plots.
        If a single pad is given, all paddings are same size.
        Alternatively, specify for paddings [top, right, bottom, left]
    figsize
        required to be able to estimate gridspec width and height ratios,
        if absolute sizes are requested. Due to the current implementation, figsize also needs to
        be given if only relative sizes are requested.
    constrained_layout
        whether to apply constrained_layout
        Note that applying constrained layout distorts the Axes placed with
        absolute height or width, so that the absolute size is no longer guaranteed
    layout_pads: Optional[Dict] = None
        passed to fig.set_constrained_layout_pads or to Gridspec otherwise
    top_plots: Optional[Iterable[Dict]] = None
        top annotation plot specs
    top_sizes=None
        size spec tuples for top annotations, if top contains plot specs, top_sizes must be defined
    left_plots: Optional[Iterable[Dict]] = None
    left_sizes=None
    bottom_plots: Optional[Iterable[Dict]] = None
    bottom_sizes=None
    right_plots: Optional[Iterable[Dict]] = None
    right_sizes=None
    row_dendrogram: Optional[Union[Dict, bool]] = None
        if truthy, draw a row dendrogram. If non-empty dict, use dict
        as kwargs for co.plotting.cut_dendrogram.
    row_dendrogram_size=(1 / 2.54, "abs")
    col_dendrogram: Optional[Dict] = None
        if truthy, draw a col dendrogram. If non-empty dict, use dict
        as kwargs for the dendrogram function.
    col_dendrogram_size=(1 / 2.54, "abs")
    row_order: Optional[Union[Array1DLike, List[Array1DLike]]] = None
        integer index specifying row order for center plots and aligned annotations
        if all center rows have the same row order, pass one array-like
        alternatively, specify one row order array(-like) per center row and pass as list of arrays
    col_order: see row_order
    default_func=co.plotting.heatmap3
        if the plot spec dict does not contain a '_func' keyword, use the default_func to draw the plot
    default_plotting_func_kwargs: Optional[Dict] = None
        if the plot is drawn with the default func (user specified or by falling back on the default_func),
        use these kwargs as defaults arguments (which may be overriden by the plot spec)
    row_spacing_group_ids: Optional[Union[Array1DLike, List[Array1DLike]]] = None
        if all center rows have the same row order and spacing spec:
        integer array-like specifying group membership for plot spacing.
        spacers are added between rows belonging to the same group.
        Each ID is expected to be present in a single, consecutive stretch, ie a group id may not
        occur in multiple places separated by other group ids.
        if each center row has its own order and grouping, pass list of array-likes (not implemented yet)
    row_spacer_sizes: Union[float, List[float]] = 0.02
        float if same spacer size is used between all groups, else List[float] indicating
        size of spacer between each group. Size is given as fraction of axes width.
    col_spacing_group_ids=None
        see row_spacing_group_ids
    col_spacer_sizes=0.02
        see row_spacer_sizes
    row_linkage: True or dict to trigger clustering, or pass linkage matrix
        if not a linkage matrix, but truthy and row_order is None, perform hierarchical clustering with scipy.cluster.hierarchy.linkage.
        Use a dict to pass kwargs to the linkage function.
    row_partitioning: bool or dict
        if truthy and row_linkage is available, perform partitioning with dynamicCutTree.cutreeHybrid.
        Use dict to pass arguments to the partitioning function
    col_linkage: see row_linkage
    col_partitioning: see row_partitioning
    supply_tasks=cross_plot_supply_tasks
    adjust_coords_tasks=cross_plot_adjust_coord_tasks
    align_args=cross_plot_align_tasks
    cbar_styling_func
        called on colorbars to modify their aesthetics


    Spacers between row or column groups
    ------------------------------------

    **Spacers in coordinate-based plots**
    - spacers can be automatically introduced into coordinate-based plots, such as  scatter, line or bar plots
    - this can be used to introduce spacers in coordinate-based center plots (e.g. categorical scatterplots) or to align plots in the annotation panels with their targets (heatmap rows, scatterplot rows for categorical scatterplots) in the presence of spacers
    - variable names holding x or y coordinate names to be adjusted can be specified globally in adjust_coords_tasks (defaults to cross_plot_adjust_coord_tasks_d) or per plotting spec via the _adjust argument (the latter not yet implemented)
    - depending on the panel where a given plot is placed, only x or y or both coordinates are adjusted for spacers
    - spacer are introduced using adjust_coords, with additional args given by adjust_coord_args
    - Note: currently, adjust_coord_tasks defines tasks in a panel-based manner, this will be changed to a x/y-based task dict


    Automatic alignment
    -------------------
    - alignment is active if align_args is not None (default: cross_plot_align_tasks, ie alignment is active by default)
    - if alignment is active, row_spacing_group_ids and col_spacing_group_ids are  automatically aligned using row_order and col_order resp.
    - alignment is done based on the integer indices in row_order and col_order
    - Note that row_order and col_order either specify a global order for all center rows and columns, or one order per row or column respectively.
    - align_args is a list specifying arg names to be always aligned: If a plotting spec contains one of these argument names, its value will be aligned as detailed below.
    - additionally, function arguments whose value should be aligned can be specified on a per-plot basis using the _align keyword in the plot spec (not yet implemented)
    - the performed alignment depends on where the plot is drawn
        - center plots: rows and cols are aligned. Align_args are expected to point to DFs or 2D arrays, otherwise a ValueError is raised.
        - left/right annotations: rows are aligned. Align_arg values may be 2D array or DataFrame (columns will be left as is) or 1D arrays or Series
        - top/bottom annotations: columns are aligned. Allowed values: see left/right annotations.

    Passing on cross_plot args to plotting funcs
    --------------------------------------------
    cross_plot args can be passed on to plotting funcs according to
    rules specified in a dict. For an example, see the default rules dict defined in  array_manager.cross_plot_supply_tasks_d
    This dict specifies rules for the center, left+right and top+bottom plotting areas, indicating which cross_plot arguments should be passed on to plotting functions in these areas, and which arguments in the plotting functions can accept these data.
    For each area, a dict mapping crossplot args -> one or more plotting function arg names is specified.
    A crossplot arg is passed on to a plotting function, if this function
    accepts one of the specified arg names. It is an error if the function accepts several of the possible arg names.
    The crossplot arg is not passed on, if it is None or if the plotting spec
    defines a value for the corresponding arg name (i.e. plotting specs
    have precedence over crossplot arg passing)

    Guides
    ------
    - all guides are collected into one legend axes at *legend_side*
    - the size of the legend panel is controlled by *legend_extent*, *legend_size* and *legend_pad*
    - guides can be
      - colorbars for ScalarMappables (e.g. for heatmaps, scatterplots with continuous color dimension)
      - legends for any plot with legend handles and labels
    - the guides axes is drawn with *add_guides*, and legend_args are passed as kwargs if given
    - if legend_axes_selectors is given
       - only show guides for the specified axes
       - legend_axes selectors may contain names, each name indicating one more plotting axes (several axes may have the same name) or coordinates (as tuple) of axes in the final figure array. If a name points to multiple axes, each axes will still receive a separate legend.
    """

    # Implementation notes
    # --------------------
    # All plots are specified as dicts
    # Copies of these dicts are modified in place in several stages of the function.
    # For this purpose, plot elements from all plotting areas are iterated
    # over in arbitrary flattened order and the modified as needed.
    # The returned plotting array contains the modified plot specs.

    # Check and process args

    # Replace sentinel values with empty mutables

    # empty kwarg dicts
    if legend_args is None:
        legend_args = {}
    if default_plotting_func_kwargs is None:
        default_plotting_func_kwargs = {}

    # copy plotting spec dicts
    # replace plotting area=None with empty lists, to allow for iteration
    top_plots, right_plots, bottom_plots, left_plots = [
        deepcopy(x) if x is not None else []
        for x in [top_plots, right_plots, bottom_plots, left_plots]
    ]

    # TODO: breaking change: pads_around_center is now always list
    # We need to create size specs (tuples or None) as dict for [top, right, bottom, left]
    if not isinstance(pads_around_center, list):
        raise ValueError(
            f"pads_around_center must be list of tuples, given was: {pads_around_center}"
        )
    if pads_around_center is None:
        pads_around_center = [None] * 4
    elif len(pads_around_center) == 1:
        # if only one size spec tuple is given, use same size at all sides
        pads_around_center *= 4
    pads_around_center_d = dict(
        zip("top right bottom left".split(), pads_around_center)
    )

    # center is list, list of list, 1d array or 2d array
    # convert to 2d array
    center_arr = np.array(center_plots)
    if center_arr.ndim == 1:
        center_arr = center_arr[np.newaxis, :]
    # center plot size specs default to (1, 'rel')
    if center_row_sizes is None:
        center_row_sizes = [(1, "rel")] * center_arr.shape[0]
    if center_col_sizes is None:
        center_col_sizes = [(1, "rel")] * center_arr.shape[1]

    # # assemble row and col ordering information

    # row and col ordering information is either passed via the row/col_order or
    # the row/col_linkage args, which may provide a linkage matrix or request computation of it
    # because a provided/computed linkage matrix defines the row and col order, it is an
    # error of the col/row_order and col/row_linkage args are both defined

    if isinstance(row_linkage, (dict, np.ndarray)) or row_linkage:
        if row_order is not None:
            raise ValueError("row_order and row_linkage may not be specified together")
    if isinstance(col_linkage, (dict, np.ndarray)) or col_linkage:
        if col_order is not None:
            raise ValueError("col_order and col_linkage may not be specified together")

    # ## compute row linkage if necessary
    # row_linkage is either a bool or dict (to indicate whether hierarchical clustering
    if isinstance(row_linkage, np.ndarray):
        # assert that this has the dimensions of a linkage matrix
        assert row_linkage.ndim == 2
    elif row_linkage:
        if not isinstance(row_linkage, dict):
            row_linkage = {}
        row_linkage = linkage(center_arr[0, 0]["df"], **row_linkage)
    # at this point, row_linkage is either a linkage matrix or False

    # if we have a linkage matrix, use it to compute the row idx
    if isinstance(row_linkage, np.ndarray):
        row_idx = leaves_list(row_linkage)
    # else, use row_order as row_idx if possible
    elif row_order is not None:
        # row_order is given as array or Series
        assert row_order.ndim == 1
        row_idx = row_order
    else:
        row_idx = None

    # ## compute col linkage if necessary
    # col_linkage is either a bool or dict (to indicate whether hierarchical clustering
    if isinstance(col_linkage, np.ndarray):
        # assert that this has the dimensions of a linkage matrix
        assert col_linkage.ndim == 2
    elif col_linkage:
        if not isinstance(col_linkage, dict):
            col_linkage = {}
        col_linkage = linkage(center_arr[0, 0]["df"].T, **col_linkage)
    # at this point, col_linkage is either a linkage matrix or False

    # if we have a linkage matrix, use it to compute the col idx
    if isinstance(col_linkage, np.ndarray):
        col_idx = leaves_list(col_linkage)
    # else, use col_order as col_idx if possible
    elif col_order is not None:
        # col_order is given as array or Series
        assert col_order.ndim == 1
        col_idx = col_order
    else:
        col_idx = None

    # Check whether we have any information to align args
    if col_idx is None and row_idx is None:
        align_args = False

    # if align_args is not None, automatic alignment is active
    # in this case, the spacing group ids are automatically aligned
    # using row_idx and col_idx resp. (if available)
    if align_args:
        if row_idx is not None and row_spacing_group_ids is not None:
            row_spacing_group_ids = index_into_list_np_or_pd(
                row_spacing_group_ids, row_idx
            )
        if col_idx is not None and col_spacing_group_ids is not None:
            col_spacing_group_ids = index_into_list_np_or_pd(
                col_spacing_group_ids, col_idx
            )
        # spacer sizes are not aligned

    # add default func and default func kwargs
    # to all plotting spec (from all plotting areas)
    for elem in itertools.chain.from_iterable(
        it
        for it in [
            left_plots,
            top_plots,
            bottom_plots,
            right_plots,
            np.ravel(center_arr),
        ]
        if it is not None
    ):
        # supply default_func if _func kwarg is not defined
        if elem.get("_func") is None:
            elem["_func"] = default_plotting_func
        # check whether plotting func is default_func (supplied or explicitely defined)
        # if so, provide default arguments where required
        if elem["_func"] == default_plotting_func:
            # note: elem edited in place
            for default_k, default_v in default_plotting_func_kwargs.items():
                # only provide default if the argument is not defined
                # in the plotting spec dict
                if default_k in elem:
                    continue
                else:
                    elem[default_k] = default_v

    # Pass crossplot args on to plotting funcs
    # see docstring for details and cross_plot_supply_tasks_d for the default rules
    for name, elems in zip(
        "center topbottom leftright topbottom leftright".split(),
        [center_arr.flatten(), top_plots, right_plots, bottom_plots, left_plots],
    ):
        for elem in elems:
            for var_name, supply_targets in cross_plot_supply_tasks_d[name].items():
                # TODO: possible breaking change: fixed bug where var_name was quoted
                if locals().get(var_name) is None:
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

    # Align plot spec arguments if their name is specified in align_args
    # or in _align in the plot spec (not yet implemented)
    # Alignment is aware of the location of the plot (left/right, top/bottom, center) and only
    # aligns appropriate dimensions
    # See docstring for details and cross_plot_align_tasks for the default alignment targets
    for name, elems in zip(
        "center  topbottom leftright topbottom leftright".split(),
        [center_arr.flatten(), top_plots, right_plots, bottom_plots, left_plots],
    ):
        # For center plot, always align both rows and columns
        # If the value for a center plot align arg is not 2D, it is an error
        if name == "center":
            if row_idx is not None and col_idx is not None:
                df_2d_slice = row_idx, col_idx
                arr_2d_slice = np.ix_(row_idx, col_idx)
                df_1d_slice = arr_1d_slice = None
            elif row_idx is None:
                df_2d_slice = slice(None), col_idx
                arr_2d_slice = slice(None), col_idx
                df_1d_slice = arr_1d_slice = None
            elif col_idx is None:
                df_2d_slice = row_idx, slice(None)
                arr_2d_slice = row_idx, slice(None)
                df_1d_slice = arr_1d_slice = None
            else:
                # this shouldn't happen, because align_args is set to False in the absence of alignment information
                raise RuntimeError(
                    "Alignment information is not available, but it should be"
                )

        # For top/bottom or left/right plots only align columns or rows respectively
        # Accept either 2D or 1D arrays / Series
        elif name == "topbottom":
            # we cannot align anything in top/bottom without col_idx
            if col_idx is None:
                continue
            df_2d_slice = arr_2d_slice = slice(None), col_idx
            df_1d_slice = arr_1d_slice = col_idx
        else:  # name == 'leftright':
            # we cannot align anything in left/right without row_idx
            if row_idx is None:
                continue
            df_2d_slice = arr_2d_slice = row_idx, slice(None)
            df_1d_slice = arr_1d_slice = row_idx

        for elem in elems:
            # go through all argument names to be aligned
            if "_align_args" in elem:
                if elem["_align_args"] is not None:
                    curr_align_args = elem["_align_args"]
                    # need to remove _align_args before passing to array_to_figure
                    del elem["_align_args"]
                else:
                    # need to remove _align_args before passing to array_to_figure
                    del elem["_align_args"]
                    continue
            else:
                if align_args:
                    curr_align_args = align_args
                else:
                    continue
            for align_target in curr_align_args:
                # If an argument name is found, align according to position if possible
                # otherwise, raise ValueError
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

    # add spacers
    # spacers can be introduced into coordinate plots in a manner equivalent to the
    # introduction of spacers in heatmaps
    # this is used to introduce spacers in coordinate-based center plots (e.g. scatter plots)
    # or to align coordinate-based annotation plots with their targets
    # depending on the panel membership of a plot, only x or y or both coordinates are aligned
    # see docstring for details
    # these tasks are collected in adjust_coord_tasks
    # spacers are introduced using adjust_coords, with additional args given by adjust_coord_args
    # Currently, adjust_coord_tasks defines tasks in a panel-based manner, this will be changed to a x/y-based task dict
    for name, elems in zip(
        "topbottom leftright topbottom leftright topbottom leftright".split(),
        [
            center_arr.flatten(),
            center_arr.flatten(),
            top_plots,
            right_plots,
            bottom_plots,
            left_plots,
        ],
    ):
        # for top/bottom plots, only add col spacers
        # for left/right plots, only add row spacers
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
                for adjust_target in cross_plot_adjust_coord_tasks_d[name]:
                    if adjust_target in elem:
                        elem[adjust_target] = adjust_coords(
                            elem[adjust_target], **adjust_coords_args
                        )

    # add dendrograms to left / top
    # dendrogram are added using co.plotting.cut_dendrogram
    # arguments to this function can be specified by passing a dict through row_dendrogram or col_dendrogram

    # default args for cut_dendrogram
    cut_dendrogram_defaults = dict(
        stop_at_cluster_level=False, min_height=0, min_cluster_size=0
    )

    # add col dendrogram if col_dendrogram is truthy
    if isinstance(col_dendrogram, dict) or col_dendrogram:
        if not isinstance(col_dendrogram, dict):
            col_dendrogram = {}
        else:
            col_dendrogram = deepcopy(col_dendrogram)
        col_dendrogram = tz.merge(cut_dendrogram_defaults, col_dendrogram)
        col_dendrogram["orientation"] = "vertical"
        # if no cluster ids are specified, use col_spacing_groups (may be None)
        if "cluster_ids_data_order" not in col_dendrogram:
            col_dendrogram["cluster_ids_data_order"] = col_spacing_group_ids
        # if cluster ids are specified, align them if alignment mode is on
        elif col_dendrogram["cluster_ids_data_order"] is not None and align_args:
            col_dendrogram["cluster_ids_data_order"] = col_dendrogram[
                "cluster_ids_data_order"
            ][col_idx]

        assert (
            col_linkage.ndim == 2
        ), "col linkage not linkage mat, but dendrogram requested"
        top_plots = [
            dict(
                _func=co.plotting.cut_dendrogram,
                linkage_mat=col_linkage,
                spacing_groups=col_spacing_group_ids,
                spacer_size=col_spacer_sizes,
                **col_dendrogram,
            )
        ] + (top_plots if top_plots else [])
        top_row_sizes = [col_dendrogram_size] + (top_row_sizes if top_row_sizes else [])

    # add row dendrogram if row_dendrogram is truthy
    if isinstance(row_dendrogram, dict) or row_dendrogram:
        if not isinstance(row_dendrogram, dict):
            row_dendrogram = {}
        else:
            row_dendrogram = deepcopy(row_dendrogram)
        row_dendrogram = tz.merge(cut_dendrogram_defaults, row_dendrogram)
        row_dendrogram["orientation"] = "horizontal"
        # if no cluster ids are specified, use row_spacing_groups (may be None)
        if "cluster_ids_data_order" not in row_dendrogram:
            row_dendrogram["cluster_ids_data_order"] = row_spacing_group_ids
        # if cluster ids are specified, align them if alignment mode is on
        elif row_dendrogram["cluster_ids_data_order"] is not None and align_args:
            row_dendrogram["cluster_ids_data_order"] = row_dendrogram[
                "cluster_ids_data_order"
            ][row_idx]
        assert (
            row_linkage.ndim == 2
        ), "row linkage not linkage mat, but dendrogram requested"
        left_plots = [
            dict(
                _func=co.plotting.cut_dendrogram,
                linkage_mat=row_linkage,
                spacing_groups=row_spacing_group_ids,
                spacer_size=row_spacer_sizes,
                **row_dendrogram,
            )
        ] + (left_plots if left_plots else [])
        left_col_sizes = [row_dendrogram_size] + (
            left_col_sizes if left_col_sizes else []
        )

    # get array size, init array
    def len_or_none(x):
        if x is None:
            return 0
        else:
            return len(x)

    # only display ticklabels at center panel margins if requested
    # the code considers that [xy]ticklabels can take bool or list of ticklabels
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

    # Construct plot array
    n_cols = len_or_none(left_plots) + len_or_none(right_plots) + center_arr.shape[1]
    n_rows = len_or_none(top_plots) + len_or_none(bottom_plots) + center_arr.shape[0]
    # Leave space for sizes
    plot_arr = np.empty((n_rows + 1, n_cols + 1), dtype=object)

    # add ArrayElements at each side and in center, add sizes into array
    first_center_row = len(top_plots)
    first_center_col = len(left_plots)
    # last index plus 1
    last_center_row_p1 = len(top_plots) + center_arr.shape[0]
    last_center_col_p1 = len(left_plots) + center_arr.shape[1]

    # Add plots to array
    if left_plots is not None:
        plot_arr[first_center_row:last_center_row_p1, :first_center_col] = left_plots
    if right_plots is not None:
        plot_arr[
            first_center_row:last_center_row_p1, last_center_col_p1:-1
        ] = right_plots
    if top_plots is not None:
        plot_arr[:first_center_row, first_center_col:last_center_col_p1] = np.array(
            top_plots
        )[:, np.newaxis]
    if bottom_plots is not None:
        plot_arr[last_center_row_p1:-1, first_center_col:last_center_col_p1] = np.array(
            bottom_plots
        )[:, np.newaxis]
    # add center plots
    plot_arr[
        first_center_row:last_center_row_p1, first_center_col:last_center_col_p1
    ] = center_arr

    # add size specs
    height_ratios = list(
        itertools.chain.from_iterable(
            (
                x
                for x in [top_row_sizes, center_row_sizes, bottom_row_sizes]
                if x is not None
            )
        )
    ) + [None]
    width_ratios = list(
        itertools.chain.from_iterable(
            (
                x
                for x in [left_col_sizes, center_col_sizes, right_row_sizes]
                if x is not None
            )
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

    # Add guides, see docstring for details
    if legend_side is not None:
        # If no axes are selected, display guides for all of them
        if legend_axes_selectors is None:
            guide_spec_l = res["plot_returns_arr"].flatten()
        # else collect all selected axes. One name may select several axes which
        # carry that name. These axes will still receive individual legends
        else:
            # fill guide_spec_l from axes selectors
            guide_spec_l = []
            for selector in legend_axes_selectors:
                if isinstance(selector, str):
                    # we may have a scalar ArrayElement or List[ArrayElement] under one name
                    to_add = res["plot_returns_d"][selector]
                    if isinstance(to_add, list):
                        guide_spec_l.extend(to_add)
                    else:
                        guide_spec_l.append(to_add)
                else:  # coord tuple specifying one axes in final plot array
                    guide_spec_l.append(res["plot_returns_arr"][selector])

        add_guides(
            # Note that the order is relevant if guide_titles is not given
            # Flatten may not provide the most intuitive order
            guide_spec_l=guide_spec_l,
            ax=res["axes_d"]["legend_ax"],
            # guide titles, padding...
            **legend_args,
            cbar_styling_func=cbar_styling_func,
        )

    return res, plot_arr


def index_into_list_np_or_pd(
    o: Union[List, pd.Series, pd.DataFrame, np.ndarray], idx: Array1DLike
):
    """Index into object using idx - for list, array, series, dataframe

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
