"""Use array of plotting functions to facilitate creation of complex gridspecs"""
import itertools
from inspect import getfullargspec
from typing import Callable, Tuple, Dict, Optional

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.legend as mlegend

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
from dataclasses import dataclass, field

from IPython.display import display

# TODO: make code work with 'medium' etc. labels
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


@dataclass
class ArrayElement:
    name: str
    # without func, a placeholder Axes is created
    func: Optional[Callable] = None
    args: Tuple = tuple()
    kwargs: Dict = field(default_factory=dict)


# tests
def object_plotter(ax, a=3, b=10, title="title"):
    """Simple test function which accepts ax, and fails if b is not overwritten"""
    ax.plot(range(a), range(b), label="object")
    ax.plot(range(a), range(b), c="green", label="object_green")
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
    # TODO: add default guide title to ArrayElement, to optionally name auto-generated legends

    plot_array1 = np.array(
        [
            [
                ArrayElement(
                    "scatter1",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="green", label="green"),
                ),
                ArrayElement(
                    "scatter2",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="black", label="black"),
                ),
                None,
                (1, "abs"),
            ],
            [
                ArrayElement("plot", object_plotter, args=(), kwargs=dict(b=3)),
                ArrayElement(
                    "scatter4",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="blue", label="blue"),
                ),
                ArrayElement("legend"),
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
    res = array_to_figure(plot_array_combined, figsize=(8, 8))
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

    legends_df = create_guides(res["plot_returns"].values(), res["axes_d"]["legend"])
    fit_guides(res["axes_d"]["legend"], legends_df, 0.05)
    res['axes_d']['legend'].axis('off')

    display(res["fig"])


def array_to_figure(plot_array: np.ndarray, figsize: Tuple[float]) -> Dict:
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

    # The margins of the array contain the height and width specs as (size, 'abs'|'rel')
    # retrieve the actual plot array (inner_array) and convert the size specs
    # to relative-only ratios, given the figsize
    inner_array = plot_array[:-1, :-1]
    widths = plot_array[-1, :-1]
    width_ratios = compute_gridspec_ratios(widths, figsize[0])
    heights = plot_array[:-1, -1]
    height_ratios = compute_gridspec_ratios(heights, figsize[1])

    assert (
        np.unique(
            [elem.name for elem in inner_array.ravel() if elem is not None],
            return_counts=True,
        )[-1]
        == 1
    ).all()

    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly,
    # setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    gs = gridspec.GridSpec(
        inner_array.shape[0],
        inner_array.shape[1],
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        figure=fig,
    )

    axes_d = {}
    axes_arr = np.empty_like(inner_array)
    plot_returns = {}
    for t in itertools.product(
        range(inner_array.shape[0]), range(inner_array.shape[1])
    ):
        ge = inner_array[t]
        # Do not create Axes for Spacers (np.nan)
        if ge is None:
            continue
        # However, do create Axes for GridElements without plotting function
        ax = fig.add_subplot(gs[t])
        axes_d[ge.name] = ax
        axes_arr[t] = ax
        if ge.func is None:
            continue
        # Plotting func is given, create plot
        # Add Axes to func args if necessary
        # If ax is not in plotter args, we assume that this is a pyplot function relying on state
        plotter_args = getfullargspec(ge.func).args
        if "ax" in plotter_args:
            ge.kwargs["ax"] = ax
        plot_returns[ge.name] = ge.func(*ge.args, **ge.kwargs)

    return {
        "axes_d": axes_d,
        "axes_arr": axes_arr,
        "plot_returns": plot_returns,
        "fig": fig,
    }


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


def create_guides(guide_spec_l, ax):
    ax_height_in, ax_width_in = get_axes_dim_in(ax)
    legends_df = pd.DataFrame(columns=["title", "height", "width", "object"])
    for guide_d in guide_spec_l:
        if guide_d is None:
            continue
        # colorbar
        if "handles" not in guide_d:
            legends_df = legends_df.append(
                dict(
                    title=guide_d.get("title", None),
                    height=guide_d.get("shrink", None),
                    width=guide_d.get("aspect", None),
                    object=guide_d["mappable"],
                ),
                ignore_index=True,
            )
        else:
            elem = guide_d["handles"], guide_d["labels"]
            l = mlegend.Legend(
                ax,
                *elem,
                title=guide_d.get("title", None),
                loc="upper left",
                frameon=False,
            )
            l._legend_box.align = "left"
            width_in, height_in = get_legend_dimensions(*elem, guide_d["title"])
            legends_df = legends_df.append(
                dict(
                    title=guide_d.get("title", None),
                    height=height_in / ax_height_in,
                    width=width_in / ax_width_in,
                    object=l,
                ),
                ignore_index=True,
            )
    return legends_df


def get_axes_dim_in(ax):
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    ax_width_in, ax_height_in = bbox.width, bbox.height
    return ax_height_in, ax_width_in


def fit_guides(ax, legends_df, xpad=0.2, ypad=0.2 / 2.54):
    fig = ax.figure
    p = mpl.rcParams
    legends_df = legends_df.sort_values("height", ascending=False)
    curr_x = 0
    next_x = 0
    curr_y = 1
    ax_width_in, ax_height_in = get_axes_dim_in(ax)
    for _, row_ser in legends_df.iterrows():
        # TODO: row_ser.height may be larger 1
        if curr_y - row_ser.height < 0:
            curr_y = 1
            curr_x = next_x
        if isinstance(row_ser.object, mlegend.Legend):
            row_ser.object.set_bbox_to_anchor((curr_x, curr_y))
            ax.add_artist(row_ser.object)
            x_padding = xpad
        else:  # cbar
            cbar_ax: Axes
            cbar_ax = ax.inset_axes(
                [curr_x, curr_y - row_ser.height, row_ser.width, row_ser.height]
            )
            title_text = cbar_ax.set_title(
                row_ser["title"],
                fontdict={"fontsize": p["legend.title_fontsize"]},
                # center left right
                loc="left",
                # pad=rcParams["axes.titlepad"],
                # **kwargs: Text properties
            )

            r = fig.canvas.get_renderer()  # not necessary if figure was drawn
            title_bbox = ax.transAxes.inverted().transform(
                title_text.get_window_extent(r)
            )
            # The format is
            #        [[ left    bottom]
            #         [ right   top   ]]
            title_size_axes_coord = title_bbox[1, 0] - title_bbox[0, 0]
            fig.colorbar(row_ser.object, cax=cbar_ax)
            # draw is necessary to get yticklabels
            fig.canvas.draw()
            # assumption: all labels have same length
            bbox = cbar_ax.get_yticklabels()[0].get_window_extent()
            axes_bbox = ax.transAxes.inverted().transform(bbox)
            # The format is
            #        [[ left    bottom]
            #         [ right   top   ]]
            y_tick_label_size = axes_bbox[1, 0] - axes_bbox[0, 0]
            # add tick size
            y_tick_label_size += p["ytick.major.size"] * 1 / 72 / ax_width_in
            # add additional padding
            x_padding = max(title_size_axes_coord, y_tick_label_size) + xpad
        # TODO: padding around colorbars hardcoded in axes coordinates
        curr_y -= row_ser.height + ypad / ax_height_in
        next_x = max(next_x, curr_x + row_ser.width + x_padding)


def get_legend_dimensions(handles, labels, title=None):
    p = mpl.rcParams
    # assumption: handleheight >= 1, ie handles are at least as large as font size
    # assumption: ncol = 1
    legend_height = (
        (
            len(handles) * mpl.rcParams["legend.handleheight"]
            + 2 * mpl.rcParams["legend.borderaxespad"]
            + 2 * p["legend.borderpad"]
            + (len(handles) - 1) * p["legend.labelspacing"]
        )
        * p["legend.fontsize"]
        * 1
        / 72
    )
    legend_height += (
        0
        if title is None
        else (
            p["legend.title_fontsize"] * 1 / 72 * (sum(s == "\n" for s in title) + 1)
            + p["legend.labelspacing"] * p["legend.fontsize"] * 1 / 72
        )
    )
    legend_width = (
        (
            mpl.rcParams["legend.handlelength"]
            # TODO: take into account whether the legend is at both borders
            + 2 * mpl.rcParams["legend.borderaxespad"]
            # + 2 * p['legend.borderpad']
            + p["legend.handletextpad"]
        )
        * p["legend.fontsize"]
        * 1
        / 72
    )
    legend_width += (
        max(len(str(l)) for l in labels) * p["legend.fontsize"] * 1 / 72 * 0.5
    )
    return legend_width, legend_height


def end():
    pass
