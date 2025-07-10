import copy
from typing import Callable, Dict, List, Optional

import codaplot.utils as coutils
import matplotlib as mpl
import matplotlib.legend as mlegend
import pandas as pd
from codaplot.utils import cbar_change_style_to_inward_white_ticks
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.codaplot.utils import cbar_change_style_to_inward_white_ticks


def add_guides(
    guide_spec_l,
    ax,
    cbar_styling_func: Callable = cbar_change_style_to_inward_white_ticks,
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
        columns=[
            "title",
            "height",
            "width",
            "contents",
            "styling_func_kwargs",
            "cbar_ticks",
        ]
    )
    for guide_d in guide_spec_l:
        # Empty slots are allowed and should be skipped
        if guide_d is None:
            continue
        else:
            guide_d = copy.deepcopy(guide_d)
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
            tmp_df = pd.DataFrame(
                [
                    dict(
                        # remove title, it is not a cbar arg, so that the guide specification
                        # from now on only contains cbar args
                        title=guide_d.pop("title", None),
                        height=height,
                        width=width,
                        contents=guide_d,
                        styling_func_kwargs=guide_d.pop("styling_func_kwargs", {}),
                        cbar_ticks=guide_d.pop("cbar_ticks", None),
                    )
                ]
            )
            pd.testing.assert_index_equal(placement_df.columns, tmp_df.columns)
            if placement_df.empty:
                placement_df = tmp_df
            else:
                placement_df = pd.concat(
                    [placement_df, tmp_df], axis=0, ignore_index=True
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

            tmp_df = pd.DataFrame(
                [
                    dict(
                        title=guide_d.get("title", None),
                        height=bbox.height,
                        width=bbox.width,
                        contents=guide_d,
                        styling_func_kwargs=None,
                        cbar_ticks=None,
                    )
                ]
            )

            pd.testing.assert_index_equal(placement_df.columns, tmp_df.columns)

            if placement_df.empty:
                placement_df = tmp_df
            else:
                placement_df = pd.concat(
                    [placement_df, tmp_df], axis=0, ignore_index=True
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
        l._legend_box.align = "left"  # pyright: ignore[reportAttributeAccessIssue]

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
        r = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
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
    if row_ser["cbar_ticks"] is not None:
        if orientation == "vertical":
            cbar_ax.set_yticks(row_ser["cbar_ticks"])
        else:
            cbar_ax.set_xticks(row_ser["cbar_ticks"])
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
