import matplotlib.legend as mlegend
import inspect
from matplotlib.colorbar import Colorbar
import matplotlib.patches as patches
from typing import Dict, Tuple, Literal, Optional, Union, Iterable
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
from matplotlib import _api
from matplotlib.offsetbox import (
    HPacker,
    VPacker,
    DrawingArea,
    TextArea,
)

# # Unsorted


def get_subplot_adjust_hspace_wspace_params_in_inch(
    fig: Figure,
    distance_between_cols_in: float = 0,
    distance_between_rows_in: float = 0,
):
    fig_width, fig_height = fig.get_size_inches()
    # assuming plot was created with plt.subplots
    # so all axes have the same gridspec
    if isinstance(fig.axes, list):
        any_ax = fig.axes[0]
    elif isinstance(fig.axes, np.ndarray):
        any_ax = fig.axes.flatten()[0]
    else:
        raise ValueError()

    gs = any_ax.get_gridspec()  # type: ignore
    ncols = gs.ncols
    nrows = gs.nrows
    hspace = (distance_between_rows_in * nrows) / (
        fig_height - (nrows - 1) * distance_between_rows_in
    )
    wspace = (distance_between_cols_in * ncols) / (
        fig_width - (ncols - 1) * distance_between_cols_in
    )

    return dict(hspace=hspace, wspace=wspace)



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
    print("DEPRECATED, use get_text_width_height_in_x instead")
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


def get_text_width_height_in_x(
    s,
    fontsize,
    ax,
    x: Literal["inch", "data_coordinates"] = "inch",
    rotation="horizontal",
    fontfamily=None,
    draw_figure=True,
):
    r = ax.figure.canvas.get_renderer()
    if draw_figure:
        ax.figure.draw(r)
    if not s:
        raise ValueError(
            "s is an emtpy string - did you draw the figure prior to calling this function?"
        )
    # get window extent in display coordinates
    text_kwargs = dict(fontsize=fontsize)
    if fontfamily:
        text_kwargs |= dict(fontfamily=fontfamily)
    artist = ax.text(0, 0, s, **text_kwargs, rotation=rotation)
    bbox = artist.get_window_extent(renderer=r)
    transform = {
        "inch": ax.figure.dpi_scale_trans.inverted(),
        "data_coordinates": ax.transData.inverted(),
    }[x]
    transformed_bbox = bbox.transformed(transform)
    artist.remove()
    return transformed_bbox.width, transformed_bbox.height


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


def strip_all_axis(ax):
    print('DEPRECTACED, just use ax.axis("off")')
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
        labelsize=0,
        length=0,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def strip_all_axis2(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def convert_inch_to_data_coords(size, ax) -> Tuple[float, float]:
    """

    only correct for the current fig sizes and axis limits
    """

    # add a rectangle with width and height  = size
    # get data coords from bbox
    # then remove rect agaig
    rect = ax.add_patch(
        patches.Rectangle(
            xy=(0, 0),  # anchor point
            width=size,
            height=size,
            transform=ax.figure.dpi_scale_trans,
        )
    )
    r = ax.figure.canvas.get_renderer()
    bbox = rect.get_window_extent(renderer=r)
    transformed_bbox = bbox.transformed(ax.transData.inverted())
    # snippet blended transforms: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    rect.remove()
    return transformed_bbox.width, transformed_bbox.height


def get_artist_size_inch(art, fig):
    r = fig.canvas.get_renderer()
    bbox = art.get_window_extent(renderer=r)
    transformed_bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return transformed_bbox.width, transformed_bbox.height


# # Color ticklabels

# TODO: copy-pasted from plotting module, clean up


def color_ticklabels(
    axis: str, colors: Union[Iterable, Dict, pd.Series], ax: Axes
) -> None:
    """Mark tick labels with different colors

    Args:
        axis: 'x' or 'y'
        colors: either iterable of color specs of same length as number of ticklabels, or a dict label -> color
        ax: single Axes
    """

    # Implementation NOTE:
    # ax.figure.canvas.draw() caused problems when using heatmap with cross_plot
    # (the legend Axes was not shown - not sure what happened exactly)
    # I comment this part out for now - why was it necessary?
    axis = [ax.xaxis, ax.yaxis][axis == "y"]
    if isinstance(colors, (dict, pd.Series)):
        # ax.figure.canvas.draw()
        for tick in axis.get_ticklabels():  # type: ignore
            tick.set_color(colors[tick.get_text()])
    elif colors is not None:
        # we expect an iterable, with one color per ticklabel
        # ax.figure.canvas.draw()
        for tick, tick_color in zip(axis.get_ticklabels(), colors):  # type: ignore
            tick.set_color(tick_color)


# # Add aligned (and optionally colored) hierarchical ticklabels ('{outer label} | {inner label}')


def add_two_level_labels(
    outer_inner_label_df,
    label_positions,
    ax,
    axis: Literal["x", "y"],
    outer_label_colors: Optional[Dict] = None,
    inner_label_colors: Optional[Dict] = None,
    ticklabel_to_axis_margin_factor=2,
    skip_repeated_outer_labels=True,
    fontsize=6,
):
    """Add ticklabels consisting of an outer and inner label with some goodies

    ... such as coloring and skiping repeating outer labels

    text is placed as "{outer label} | {inner label}{spacer}"

    Parameters
    ----------
    outer_inner_label_df
        DF with two columns giving the outer and the inner label (in this order)
        the column names can be arbitrary
        order must be aligned with label_positions iterable
    label_positions
        in data coordinates
        the combined labels are placed on the target axis at these positions
        this does not set or check tick positions (needs to be done outside of this function)
    ax
    axis
    outer_label_colors
        if not None, color ticklabels accordingly
    inner_label_colors
        if not None, color ticklabels accordingly
    ticklabel_to_axis_margin_factor
        by default, the ticklabel is placed with a margin of two " " length to the axis
        adjust this factor here
    skip_repeated_outer_labels
        if True (default) do not repeat consecutive same outer labels


    """

    if outer_label_colors is None:
        outer_label_colors = {}
    if inner_label_colors is None:
        inner_label_colors = {}

    # this step can be very slow for many labels
    (
        max_outer_text_length_data_coords,
        max_inner_text_length_data_coords,
        approximated_bar_char_length_data_coords,
        single_spacer_width,
    ) = _get_text_length_in_target_axis_data_coords(outer_inner_label_df, ax, axis)

    # now we iterate over all elements and place the labels while
    # - adding color if requested
    # - removing consecutive labels if requested

    if axis == "x":
        labels_and_pos_from_left_to_right_or_from_top_to_bottom = zip(
            label_positions, outer_inner_label_df.iterrows()
        )
    else:
        labels_and_pos_from_left_to_right_or_from_top_to_bottom = zip(
            label_positions[::-1], outer_inner_label_df.iloc[::-1].iterrows()
        )

    current_outer_label = None
    for tick_pos, (
        _,
        (outer_label, inner_label),
    ) in labels_and_pos_from_left_to_right_or_from_top_to_bottom:

        (
            xytext_outer_label,
            xytext_bar,
            xytext_inner_label,
        ) = _get_label_positions(
            max_outer_text_length_data_coords=max_outer_text_length_data_coords,
            bar_length_data_coords=approximated_bar_char_length_data_coords,
            max_inner_text_length_data_coords=max_inner_text_length_data_coords,
            single_spacer_width=single_spacer_width,
            axis=axis,
            tick_pos=tick_pos,
            ticklabel_to_axis_margin_factor=ticklabel_to_axis_margin_factor,
        )

        if axis == "y":
            rotation = "horizontal"
            va = "center"
            ha = "left"
        else:
            rotation = "vertical"
            va = "bottom"
            ha = "center"

        shared_args = dict(
            zorder=10,
            annotation_clip=False,
            fontsize=fontsize,
            textcoords="data",
            rotation=rotation,
            va=va,
            ha=ha,
        )

        # only print consecutive same outer labels if requested
        if (not skip_repeated_outer_labels) or (not outer_label == current_outer_label):
            ax.annotate(
                text=outer_label,
                color=(
                    "black"
                    if not outer_label_colors
                    else outer_label_colors[outer_label]
                ),
                xytext=xytext_outer_label,
                xy=xytext_outer_label,
                **shared_args,
            )
            current_outer_label = outer_label

        ax.annotate(
            text="|",
            xytext=xytext_bar,
            xy=xytext_bar,
            **shared_args,
        )

        ax.annotate(
            text=inner_label,
            color=(
                "black"
                if not inner_label_colors
                else inner_label_colors[inner_label]
            ),
            xytext=xytext_inner_label,
            xy=xytext_inner_label,
            **shared_args,
        )


# ## _get_text_length_in_target_axis_data_coords


def _get_text_length_in_target_axis_data_coords(outer_inner_labels_df, ax, axis):
    """

    get maximum text length in data coordinates across
    - all outer labels
    - all innter labels

    get text length for
    - "|" char (approximated, see code)
    - " " char

    text length is defined as

    if axis == 'y'
        use ax.text(..., rotation='horizontal')
        get text width in x axis data coordinates

    if axis == 'x'
        use ax.text(..., rotation='vertical')
        get text width in y axis data coordinates
    """
    if axis == "y":
        rotation = "horizontal"
        width_height_idx = 0
    else:
        rotation = "vertical"
        width_height_idx = 1

    max_outer_text_width_data_coords = max(
        get_text_width_height_in_x(
            s=s,
            fontsize=mpl.rcParams["legend.fontsize"],
            ax=ax,
            x="data_coordinates",
            rotation=rotation,
        )[width_height_idx]
        for s in outer_inner_labels_df.iloc[:, 0]
    )

    max_inner_text_width_data_coords = max(
        get_text_width_height_in_x(
            s=s,
            fontsize=mpl.rcParams["legend.fontsize"],
            ax=ax,
            x="data_coordinates",
            rotation=rotation,
        )[width_height_idx]
        for s in outer_inner_labels_df.iloc[:, 1]
    )

    bar_width_data_coords = get_text_width_height_in_x(
        s="I",  # "|" may not give correct width
        fontsize=mpl.rcParams["legend.fontsize"],
        ax=ax,
        x="data_coordinates",
    )[width_height_idx]

    single_space_width = get_text_width_height_in_x(
        s=".",
        fontsize=mpl.rcParams["legend.fontsize"],
        ax=ax,
        x="data_coordinates",
        rotation=rotation,
    )[width_height_idx]

    return (
        max_outer_text_width_data_coords,
        max_inner_text_width_data_coords,
        bar_width_data_coords,
        single_space_width,
    )


# ## _get_label_positions


def _get_label_positions(
    max_outer_text_length_data_coords,
    bar_length_data_coords,
    max_inner_text_length_data_coords,
    single_spacer_width,
    axis,
    tick_pos,
    ticklabel_to_axis_margin_factor,
):

    x_left_or_y_bottom_inner_label = -(
        single_spacer_width * ticklabel_to_axis_margin_factor
        + max_inner_text_length_data_coords
    )
    x_left_or_y_bottom_bar = -(
        -x_left_or_y_bottom_inner_label
        + single_spacer_width
        + bar_length_data_coords
        / 2  # the bar width is relatively wide, but within this width rectangle, only a left aligned part is filled, given the impression of too wide space after the bar if not corrected
    )
    x_left_or_y_bottom_outer_label = -(
        -x_left_or_y_bottom_bar
        + single_spacer_width
        + max_outer_text_length_data_coords
    )

    xytext_outer_label = (
        x_left_or_y_bottom_outer_label,
        tick_pos,
    )
    xytext_bar = (
        x_left_or_y_bottom_bar,
        tick_pos,
    )
    xytext_inner_label = (
        x_left_or_y_bottom_inner_label,
        tick_pos,
    )

    if axis == "x":
        xytext_outer_label = xytext_outer_label[::-1]
        xytext_bar = xytext_bar[::-1]
        xytext_inner_label = xytext_inner_label[::-1]

    return (
        xytext_outer_label,
        xytext_bar,
        xytext_inner_label,
    )


LegendLocation = Literal[
    # to the right of the figure, with top edge of legend aligned to top edge of figure
    "outside right upper",
    # above the figure, with right edge of legend aligned to right edge of figure
    "outside upper right",
    # to the right of the figure, with center of legend aligned to the center of the figure
    "outside center right",
    # centered below the plot
    "outside lower center",
]

FigureLegendAlignment = Literal["left", "center", "right"]


def move_seaborn_objects_plot_legend_outside_of_figure(
    fig,
    p,
    alignment: FigureLegendAlignment = "left",
    loc: LegendLocation = "outside center right",
):
    """WIP

    NOTE this uses private attribute Plot._legend_contents and may not be stable

    based on https://stackoverflow.com/questions/75935869/how-to-move-marker-and-color-sections-of-a-figure-legend

    - this function has minimal functionality, see move_figure_legend_outside_of_figure and sns.move_legend for possible improvements, e.g. add padding

    """
    fig.legends.pop(0)
    legend_contents = p._legend_contents
    blank_handle = mpl.patches.Patch(alpha=0, linewidth=0, visible=False)  # type: ignore
    handles = []
    labels = []
    for legend_content in legend_contents:
        handles.append(blank_handle)
        handles.extend(legend_content[1])
        labels.append(legend_content[0][0])
        labels.extend(legend_content[2])
    fig.legend(handles, labels, loc=loc, alignment=alignment)
    _adjust_legend_subtitles(fig.legends[0])


# move the sub legend titles to the left such that they are aligned with the markers
def _adjust_legend_subtitles(legend):
    font_size = plt.rcParams.get("legend.title_fontsize", None)
    hpackers = legend.findobj(mpl.offsetbox.VPacker)[0].get_children()  # type: ignore
    for hpack in hpackers:
        draw_area, text_area = hpack.get_children()
        handles = draw_area.get_children()
        handle = handles[0]
        if not handles[0]._visible:
            draw_area.set_width(0)
            for text in text_area.get_children():
                if font_size is not None:
                    text.set_size(font_size)


def move_figure_legend_outside_of_figure(
    fig: mpl.figure.Figure,
    loc: LegendLocation = "outside center right",
    title: str | None = None,
    alignment: FigureLegendAlignment = "left",
    borderpad: float = 0.5,
    **kwargs,
) -> None:
    """Move the legend in fig.legends[-1] to `loc`

    *idea and code stolen from import seaborn.utils.move_legend*

    Actually: move is not possible, the legend is removed and then recreated

    currently seaborn.utils.move_legend fails with figure legends because it tries legend.remove(), resulting in NotImplementedError for figure legends

    why borderpad?
    borderaxespad does not work with constrained_layout+figure legend in mpl 3.7
    how to get spacing between axes and legends instead? => use borderpad
    borderpad is added all around the axes. we set it to 0.5 by default, then the legend can manually be moved to the edge of the figure (0.5 to the right)

    Parameters
    ----------
    kwargs
        passed to fig.legend
    title
        if None, will use original title
    """

    """
    # *idea and code stolen from import seaborn.utils.move_legend*
    Code for taking over many other legend properties when copying the legend
    can be found in seaborn.utils.move_legend
    """

    old_legend = fig.legends[-1]
    handles = old_legend.legend_handles
    labels = [t.get_text() for t in old_legend.get_texts()]
    # Extract legend properties that can be passed to the recreation method
    # (Vexingly, these don't all round-trip)
    legend_kws = inspect.signature(mpl.legend.Legend).parameters  # type: ignore
    props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}

    # get original title text object
    # it has all original font and style settings
    # change the text if required
    title_text_obj = props.pop("title")
    if title:
        title_text_obj.set_text(title)
    # set additional new kwargs on the title text object
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
    for key, val in title_kwargs.items():
        title_text_obj.set(**{key[6:]: val})
        kwargs.pop(key)

    # old_legend.remove() gives NotImplementedError
    # but it seems possible to just remove the legend from fig.legends
    del fig.legends[-1]

    del props["bbox_to_anchor"]

    new_legend: mpl.legend.Legend = fig.legend(
        handles,
        labels,
        **(
            props
            | kwargs
            | {"borderpad": borderpad, "alignment": alignment, "loc": loc}
        ),
    )
    new_legend.set_title(title_text_obj.get_text(), title_text_obj.get_fontproperties())


def cbar_change_style_to_inward_white_ticks(
    cbar: Colorbar,
    tick_width: int = 1,
    tick_length: int = 2,
    tick_params_kwargs: Optional[Dict] = None,
    remove_ticks_at_ends: Literal["always", "if_on_edge", "never"] = "if_on_edge",
    outline_is_visible=False,
):
    """Change to nicer colorbar aesthetics with good defaults

    Troubleshooting
    ---------------
    - Ticks are missing from colorbar
      - Make sure that correct ticks are available prior to calling this function (may require fig.canvas.draw()).

    Implementation note
    -------------------
    - i have not added a call to fig.canvas.draw within this function yet;
      would there be any reason to avoid this? Doing this within the function
      may avoid some problems, see Troubleshooting section.
      - I have had problems with calling fig.canvas.draw at the wrong place before,
        eg the crossplot legend was missing when I called fig.canvas.draw in color_ticklabels

    Parameters
    ----------
    cbar
        Colorbar, eg returned by plt.colorbar
    tick_params_kwargs
        passed to cbar.ax.tick_params
        overwrites the defaults
        do not pass width and length, these are controlled by dedicated args
        tick_width, tick_length
    tick_width
        cbar tick width, controlled via ax.tick_params
    tick_length
        cbar tick length, controlled via ax.tick_params
    remove_ticks_at_ends
        the outmost ticks lead to a 'sticky end' if they are inward facing and lie on the edges of the axis. This cannot really be avoided when using inward facing ticks, and is for example also present in r-ggplot2 colorbars. If 'always', the highest and lowest tick are not plotted to avoid creating sticky ends. If 'if_on_edge' the ticks are only invisible if the ticks lie on the axis edges, ie on the lower and upper axis limit. Ticks are never removed if the colorbar has "extend triangles" at the ends.
    outline_is_visible
        whether to draw spines

    """

    assert remove_ticks_at_ends in ["always", "if_on_edge", "never"]

    # use tick params to get white, inward facing ticks (default)
    tick_params_kwargs_ = dict(
        color="white",
        direction="in",
        which="both",
        axis="both",
        left=True,
        right=True,
        bottom=True,
        top=True,
    )
    if tick_params_kwargs is not None:
        tick_params_kwargs_.update(tick_params_kwargs)
        if "length" in tick_params_kwargs or "width" in tick_params_kwargs:
            print(
                "WARNING: kwarg length and/or width has been passed through tick_params_kwargs, but these args are ignored."
            )
    tick_params_kwargs_["length"] = tick_length
    tick_params_kwargs_["width"] = tick_width
    cbar.ax.tick_params(**tick_params_kwargs_)

    cbar.outline.set_visible(outline_is_visible)

    # Remove the first and last tick to avoid "sticky ends" of the colorbar due to
    # ticks at the very end of the colorbar

    if cbar.orientation == "vertical":
        ticks_objects = cbar.ax.yaxis.majorTicks
    else:
        ticks_objects = cbar.ax.xaxis.majorTicks

    # TODO: this does not handle extension to one side correctly
    if cbar.extend != "both":
        if remove_ticks_at_ends == "if_on_edge":
            ticks_arr = (
                cbar.ax.get_yticks()
                if cbar.orientation == "vertical"
                else cbar.ax.get_xticks()
            )
            ticks_axis_limits = (
                cbar.ax.get_ylim()
                if cbar.orientation == "vertical"
                else cbar.ax.get_xlim()
            )
            lowest_tick_is_on_axis_limit = np.isclose(
                ticks_arr[0], ticks_axis_limits[0]
            )
            highest_tick_is_on_axis_limit = np.isclose(
                ticks_arr[-1], ticks_axis_limits[-1]
            )
            if lowest_tick_is_on_axis_limit:
                ticks_objects[0].tick1line.set_visible(False)
                ticks_objects[0].tick2line.set_visible(False)
            if highest_tick_is_on_axis_limit:
                ticks_objects[-1].tick1line.set_visible(False)
                ticks_objects[-1].tick2line.set_visible(False)
        elif remove_ticks_at_ends == "always":
            ticks_objects[0].tick1line.set_visible(False)
            ticks_objects[0].tick2line.set_visible(False)
            ticks_objects[-1].tick1line.set_visible(False)
            ticks_objects[-1].tick2line.set_visible(False)


"""
def test_cbar_style():

    fig, ax = plt.subplots(1, 1, dpi=180, figsize=(6/2.54, 6/2.54))
    qm = ax.pcolormesh([[1, 2, 3], [1, 2, 6]], vmin=2, vmax=5)
    cbar = plt.colorbar(qm, shrink=0.5, aspect=5, ticks=[3, 4])
    fig.canvas.draw()
    cbar_change_style_to_inward_white_ticks(cbar, remove_ticks_at_ends = 'if_on_edge')

"""


def create_standalone_colorbar(
    cbar_width_axes_only: float,
    cbar_height_axes_only: float,
    cmap,
    ticks,
    vmin,
    vmax,
    png_path: str,
    title: str | None = None,
    orientation: Literal["vertical"] = "vertical",
    extend: str | None = None,
):
    """Create a stand-alone colorbar without a mappable as a figure

    NOTE: its expected behavior atm that long cbar labels are cut off

    horizontal orientation not implemened atm, should be easy to add

    Because the colobar is plotted without any underlying data, one must specify the vmin or vmax of the data explicitely

    Parameters
    ----------
    cbar_width_axes_only
        cbar width in inch without ticks, ticklabels and title
    cbar_height_axes_only
        cbar height in inch without ticks, ticklabels and title
    cmap
    ticks
    vmin
    vmax
    png_path
    title
        colorbar title is placed as axis label
    orientation
        currently only 'vertical' implemented
    cbar_kwargs=None

    """

    if orientation != "vertical":
        raise NotImplementedError()

    # %%
    fig, ax = plt.subplots(
        1,
        1,
        constrained_layout=False,
        dpi=180,
        figsize=(cbar_width_axes_only, cbar_height_axes_only),
    )

    cb = mpl.colorbar.Colorbar(
        ax=ax,
        orientation=orientation,
        cmap=cmap,
        extend=extend,
        # norm _?
        # boundaries and sequence _?
        # appear to control how many 'steps' are shown in the colorbar _???
    )

    # afg no way to get a colorbar to not only plot on yaxis between 0 and 1, but between ylim[0] and ylim[1]
    # so instead i normalize the ytick positions for the given (apparently hard to change) 0, 1 axis limits
    yticks_01_norm = (np.array(ticks) - vmin) / (vmax - vmin)
    ax.set_yticks(yticks_01_norm)
    ax.set_yticklabels(ticks)

    if title is not None:
        ax.set_ylabel(title)

    cbar_change_style_to_inward_white_ticks(cb, remove_ticks_at_ends="if_on_edge")

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path.replace(".png", ".svg"), bbox_inches="tight", pad_inches=0)
