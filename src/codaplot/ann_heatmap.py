# %%
from typing import Union, Optional, List, Dict

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
# mpl.use("Agg")  # import before pyplot import!
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage, leaves_list

import codaplot as co


# %%


# %% Heading
# * grid handling
# - space between plot axes
# - no space between anno axes

# * legend handling
#   - gridspec
#   - create categorical heatmap
#   - retrieve legend and plot on other axis

# %%


# %% new


def get_legend_dimensions(handles, labels, title=None):
    p = mpl.rcParams
    # assumption: handleheight >= 1, ie handles are at least as large as font size
    # assumption: ncol = 1
    legend_height = (
        (
            len(handles) * mpl.rcParams["legend.handleheight"]
            + 2 * mpl.rcParams["legend.borderaxespad"]
            # + 2 * p['legend.borderpad']
            + (len(handles) - 1) * p["legend.labelspacing"]
        )
        * p["legend.fontsize"]
        * 1
        / 72
    )
    legend_height += (
        0
        if title is None
        else p["legend.title_fontsize"] * 1 / 72 * (sum(s == "\n" for s in title) + 1)
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


# %%
@dataclass()
class HeatmapAnno:
    anno: Union[pd.Series, pd.DataFrame]
    name: Optional[str] = None
    cmap: Optional[str] = 'Set1'
    norm: Optional[mpl.colors.Normalize] = None

    def __post_init__(self):
        if self.name is None:
            if isinstance(self.anno, pd.Series):
                if self.anno.name is not None:
                    self.name = self.anno.name
                else:
                    raise ValueError("Name not given and anno Series not named")
            else:  # pd.DataFrame
                if self.anno.shape[1] == 1:
                    self.name = self.anno.columns[0]
                else:
                    raise ValueError(
                        "Name not given and anno DataFrame has more than one name"
                    )
        if isinstance(self.anno, pd.Series):
            self.anno = self.anno.to_frame(self.name)


# noinspection DuplicatedCode
def ann_heatmap_large():
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

    # TODO: legend.fontsize is expected in point

    #  create dummy heatmap matrix and anno vector
    df = pd.DataFrame(np.random.randn(12, 12))
    cmap = "RdBu_r"
    col_annos = [
        HeatmapAnno(pd.DataFrame({'annos1': np.repeat(["a", "b", "c", "d"], 3)}).T, name="annos1", cmap="Set1"),
        HeatmapAnno(
            pd.DataFrame(
                {"annos2": np.repeat(["var1", "var_bb", "c ccccccc", "d"], 3)}
            ).T,
            cmap="Set2",
                name='annos2',
        ),
        HeatmapAnno(pd.DataFrame({'qannos': np.arange(12)}).T, "qannos", "YlOrBr"),
    ]
    row_annos = [
        HeatmapAnno(pd.DataFrame({'qanno': [1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]}),
                    name='qanno', cmap='YlOrBr')
        ]
    row_linkage = linkage(df)
    col_linkage = linkage(df.T)
    row_dendrogram_show = True
    col_dendrogram_show = True
    cbar_shrink = 0.6
    cbar_width_in = 0.2
    figsize = (15 / 2.54, 15 / 2.54)

    row_dendrogram_width = 2
    row_annos_widths = 1
    col_annos_widths = 1
    col_dendrogram_width = 2

    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly, setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0.05, wspace=0.05)
    fig.set_size_inches(*figsize)
    gs = gridspec.GridSpec(
        5,
        4,
        width_ratios=[2, 1, 5, 10],
        height_ratios=[2, 1, 1, 2, 10],
        figure=fig,
        hspace=0,
        wspace=0,
    )

    row_start, col_start = (0, 0)

    ann_heatmap(
        cbar_shrink=cbar_shrink,
        cbar_width_in=cbar_width_in,
        cmap=cmap,
        col_annos=col_annos,
        col_dendrogram_show=col_dendrogram_show,
        col_linkage=col_linkage,
        col_start=col_start,
        df=df,
        fig=fig,
        figsize=figsize,
        gs=gs,
        row_annos=row_annos,
        row_dendrogram_show=row_dendrogram_show,
        row_linkage=row_linkage,
        row_start=row_start,
        legend_xpad=0.2,
    )
    fig
    # %%


def ann_heatmap(
    df,
    cmap,
    row_annos,
    col_annos,
    col_linkage,
    col_dendrogram_show,
    row_linkage,
    row_dendrogram_show,
    cbar_shrink,
    cbar_width_in,
    fig,
    gs,
    col_start,
    row_start,
    figsize,
    legend_xpad=0.05,
):

    p = mpl.rcParams
    # pad between axes and legend border
    p["legend.borderaxespad"] = 0
    # fractional whitspace inside legend border, still measured in font size units
    p["legend.borderpad"] = 0

    row_idx = leaves_list(row_linkage)
    col_idx = leaves_list(col_linkage)
    nrows = col_dendrogram_show + len(col_annos) + 1
    ncols = row_dendrogram_show + len(row_annos) + 2  # heatmap, legend axes
    main_row = row_start + nrows - 1
    main_col = col_start + ncols - 2
    guides = []
    if row_dendrogram_show:
        ax = fig.add_subplot(gs[main_row, col_start])
        co.plotting.dendrogram_wrapper(row_linkage, ax, orientation="left")
    if col_dendrogram_show:
        ax = fig.add_subplot(gs[row_start, main_col])
        co.plotting.dendrogram_wrapper(col_linkage, ax, orientation="top")
    if row_annos:
        for i, row_anno in enumerate(row_annos):
            ax = fig.add_subplot(gs[main_row, col_start + i + row_dendrogram_show])
            guides.append(add_anno_heatmap(ax, row_anno, is_row_anno=True))
    if col_annos:
        for i, col_anno in enumerate(col_annos):
            ax = fig.add_subplot(gs[row_start + col_dendrogram_show + i, main_col])
            guides.append(add_anno_heatmap(ax, col_anno, is_row_anno=False))
    ax = fig.add_subplot(gs[main_row, main_col])
    main_hmap_guide = co.plotting.heatmap2(
            df=df,
            ax=ax,
            pcolormesh_args=dict(cmap=cmap),
            xticklabels=True,
            yticklabels=True,
            cbar_args=dict(title='log-odds'),
            add_colorbar=False,
            title='Average methylation',
    )
    guides.append(main_hmap_guide)
    # TODO: set ylabel
    # TODO: legend_ax has a bbox for itself
    legend_ax = fig.add_subplot(gs[main_row, main_col + 1])
    legend_ax_bbox = legend_ax.get_position()
    legend_ax_width_in = figsize[0] * legend_ax_bbox.width
    legend_ax_height_in = figsize[1] * legend_ax_bbox.height
    legends_df = pd.DataFrame(columns=["title", "height", "width", "object", "priority"])
    for guide_d in guides:
        if "handles" not in guide_d:
            legends_df = legends_df.append(
                dict(
                    title=guide_d["title"],
                    height=cbar_shrink,
                    width=cbar_width_in / legend_ax_width_in,
                    object=guide_d["mappable"],
                    priority=0,
                ),
                ignore_index=True,
            )
        else:
            elem = guide_d["handles"], guide_d["labels"]
            l = Legend(legend_ax, *elem, title=guide_d["title"], loc="upper left")
            l._legend_box.align = "left"
            width_in, height_in = get_legend_dimensions(*elem, guide_d["title"])
            legends_df = legends_df.append(
                dict(
                    title=guide_d["title"],
                    height=height_in / legend_ax_height_in,
                    width=width_in / legend_ax_width_in,
                    object=l,
                    priority=0,
                ),
                ignore_index=True,
            )

    fit_guides(legend_ax, legend_ax_width_in, legends_df, xpad=legend_xpad)

    return fig


def fit_guides(legend_ax, legend_ax_width_in, legends_df, xpad=0.2):
    # import pudb.remote; pudb.remote.set_trace(term_size=(119, 49))
    fig = legend_ax.figure
    p = mpl.rcParams
    legends_df = legends_df.sort_values("height", ascending=False)
    curr_x = 0
    next_x = 0
    curr_y = 1
    for _, row_ser in legends_df.iterrows():
        # TODO: row_ser.height may be larger 1
        if curr_y - row_ser.height < 0:
            curr_y = 1
            curr_x = next_x
        if isinstance(row_ser.object, Legend):
            row_ser.object.set_bbox_to_anchor((curr_x, curr_y))
            legend_ax.add_artist(row_ser.object)
            x_padding = xpad
        else:  # cbar
            cbar_ax: Axes
            cbar_ax = legend_ax.inset_axes(
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
            title_bbox = legend_ax.transAxes.inverted().transform(
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
            axes_bbox = legend_ax.transAxes.inverted().transform(bbox)
            # The format is
            #        [[ left    bottom]
            #         [ right   top   ]]
            y_tick_label_size = axes_bbox[1, 0] - axes_bbox[0, 0]
            # add tick size
            y_tick_label_size += p["ytick.major.size"] * 1 / 72 / legend_ax_width_in
            # add additional padding
            x_padding = max(title_size_axes_coord, y_tick_label_size) + xpad
        # TODO: padding around colorbars hardcoded in axes coordinates
        curr_y -= row_ser.height + 0.025
        next_x = max(next_x, curr_x + row_ser.width + x_padding)


def add_anno_heatmap(ax, row_anno, is_row_anno):
    if not isinstance(row_anno, HeatmapAnno):
        row_anno = HeatmapAnno(*row_anno)
    # TODO: assumption: dataframe has homogeneous type
    df = row_anno.anno
    if is_numeric_dtype(row_anno.anno.iloc[:, 0]):
        guide_d = co.plotting.heatmap2(
                df=df,
                ax=ax,
                pcolormesh_args=dict(cmap=row_anno.cmap, norm=row_anno.norm),
                xticklabels=True,
                yticklabels=True,
                cbar_args=dict(),
                add_colorbar=False,
        )
        # The title should be specified as part of legend parameters in plotting function
        guide_d['title'] = row_anno.name
        return guide_d
    else:
        res = co.plotting.categorical_heatmap2(
            df,
            ax=ax,
            cmap=row_anno.cmap,
                label_stretches=False,
                heatmap_args=dict(yticklabels=True),
        )
        res['title'] = row_anno.name
        return res

def ann_heatmap_large2():

    gm = co.GridManager(

    )
