# %%
from typing import List, Union, Optional

import matplotlib as mpl
import matplotlib.colors as colors

# mpl.use("Agg")  # import before pyplot import!
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from matplotlib import pyplot as plt, patches as mpatches
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
def gridspec_handling():
    # %%
    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly, setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[5, 1],
        # height_ratios=[1, 2],
        figure=fig,
        # hspace=.1,
        # wspace=.1,
    )

    ax_hmap = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    axes = [ax_hmap, ax_legend]

    # it is not sufficient to remove the ticks and labels
    # - one must in addition set the labelsize and tick length to 0! The tick width is irrelevant
    for ax in axes:
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
    # %%

    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly, setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[5, 1],
        # height_ratios=[1, 2],
        figure=fig,
        # hspace=.1,
        # wspace=.1,
    )

    hmap_gs = gs[0, 0].subgridspec(1, 2, hspace=0.2, wspace=0.2)
    hmap_ax1 = fig.add_subplot(hmap_gs[0])
    hmap_ax2 = fig.add_subplot(hmap_gs[1])
    ax_legend = fig.add_subplot(gs[0, 1])
    axes = [hmap_ax1, hmap_ax2, ax_legend]

    # it is not sufficient to remove the ticks and labels
    # - one must in addition set the labelsize and tick length to 0! The tick width is irrelevant
    for ax in axes:
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

    # %%


def legend_handling():

    axes: List[Axes]

    fig, axes = plt.subplots(
        1, 3, gridspec_kw={"width_ratios": [5, 1, 1]}, constrained_layout=True
    )
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    h1 = axes[0].plot([1, 2, 3], [1, 2, 3], "bo-", label="a")[0]
    h2 = axes[0].plot([1, 2, 3], [1, 2, 4], "go--", label="b")[0]
    axes[1].axis("off")
    axes[1].legend(
        (h1, h2),
        ("c", "d"),
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        borderpad=0,
    )
    axes[1].tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
        labelsize=0,
        length=0,
    )
    l = Legend(
        # the axes which contains the legend
        parent=axes[1],
        handles=(h1, h2),
        labels=["l1", "l2"],
        loc="upper left",
        bbox_to_anchor=(0, 0.2),
        frameon=0,
    )
    axes[1].add_artist(l)
    l = Legend(
        # the axes which contains the legend
        parent=axes[2],
        handles=(h1, h2),
        labels=["l1", "l2"],
        loc="upper left",
        bbox_to_anchor=(0, 0.2),
        frameon=0,
    )
    axes[2].add_artist(l)
    # axes[1].legend(*axes[0].get_legend_handles_labels())


def base_clustermap():

    #  create dummy heatmap matrix and anno vector
    mat = pd.DataFrame(np.random.randn(12, 12))
    annos = pd.DataFrame(np.repeat(["a", "b", "c", "d"], 3))

    # get row and col linkage, int indices
    row_lmat = linkage(mat)
    row_idx = leaves_list(row_lmat)
    col_lmat = linkage(mat.T)
    col_idx = leaves_list(col_lmat)

    # %%
    axes: List[Axes]
    fig, axes = plt.subplots(
        3,
        3,
        gridspec_kw=dict(width_ratios=[2, 10, 5], height_ratios=[2, 1, 10]),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0.05, wspace=0.05)
    for ax in axes.ravel():
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

    import codaplot as co

    # fig does not need to be handed over
    # labels can also be taken from dataframe
    # add docstring
    co.heatmap(
        mat,
        axes[2, 1],
        fig=fig,
        cmap="RdBu_r",
        xlabel=mat.columns.to_list(),
        cbar_args=dict(shrink=0.5, aspect=30),
    )

    co.plotting.dendrogram_wrapper(row_lmat, axes[2, 0], orientation="left")

    co.plotting.dendrogram_wrapper(col_lmat, axes[0, 1], orientation="top")

    hmap_parts = co.categorical_heatmap(
        df=annos.T,
        ax=axes[1, 1],
        cmap="Set1",
        show_values=False,
        show_legend=False,
        despine=True,
    )

    # how to add title to legend
    # add title as xlabel or ylabel
    # annos.T should not be necessary
    # if legend is drawn, we still cannot get the handles and labels form it (its empty)
    #   handles, labels = axes[1, 1].get_legend_handles_labels()
    # don't make the anno axis bigger than necessary
    #   don't create an axis at all and rely on constrained layout?
    #   determine legend size beforehand?

    axes[2, 2].legend(
        hmap_parts["patches"],
        hmap_parts["levels"],
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        borderpad=0,
    )
    sns.despine(fig, left=True, bottom=True)

    # %%

    # * base heatmap: 1 col anno
    # * 2 col anno, no space between col annos
    # * 2 col anno, 2 row anno, no space between annos
    pass


# %% new


def legend_placement():

    p = mpl.rcParams
    p["legend.fontsize"] = 7 * 1 / 0.7
    # in font size units
    # details: https://matplotlib.org/3.1.1/api/legend_api.html
    p["legend.handlelength"] = 1
    p["legend.handleheight"] = 1
    # vertical space between legend entries
    p["legend.labelspacing"] = 1
    # # pad between legend handle and text
    # p["handletextpad"]
    # # pad between axes and legend border
    # p["borderaxespad"]
    # # spacing between columns
    # p["columnspacing"]
    # # fractional whitspace inside legend border, still measured in font size units
    # p["borderpad"]

    # %%
    ax: Axes
    fig, axes = plt.subplots(
        1, 2, constrained_layout=True, gridspec_kw={"width_ratios": [4, 2]}
    )
    cax = axes[0]
    qm = cax.pcolormesh(np.random.randn(3, 3), cmap="RdBu_r")
    ax = axes[1]
    figsize = (6, 2.9)
    ax.set(ylim=(0, 1), xlim=(0, 1))
    bbox = ax.get_position()
    ax_width_in = bbox.width * figsize[0]
    ax_height_in = bbox.height * figsize[1]

    patches1 = [
        mpatches.Patch(facecolor=c, edgecolor="black") for c in ["red", "green", "blue"]
    ]
    patches2 = [
        mpatches.Patch(facecolor=c, edgecolor="black")
        for c in sns.color_palette("Set1", 3)
    ]
    point_handles = [
        plt.scatter([], [], s=2, c="red"),
        plt.scatter([], [], s=2, c="blue"),
    ]
    labels = ["a", "fe", "gf", "dasdf", "casdfasdf"]
    handles = patches1 + point_handles

    l = Legend(ax, handles, labels, bbox_to_anchor=(0, 1), loc="upper left")
    legend1_width_in, legend_height_in = get_legend_dimensions(handles, labels)
    legend_height_axes_coord = legend_height_in / ax_height_in
    legend1_width_axes_coord = legend1_width_in / ax_width_in

    legend_2_width_in, legend_height_2_in = get_legend_dimensions(
        patches2, ["l1", "l2", "l3"]
    )
    legend_height_2_rel = legend_height_2_in / ax_height_in
    legend_2_width_axes_coord = legend_2_width_in / ax_width_in

    if legend_height_2_rel + legend_height_axes_coord >= 1:
        anchor = (legend1_width_axes_coord, 1)
    else:
        anchor = (0, 1 - legend_height_axes_coord)

    l2 = Legend(
        ax, patches2, ["l1", "l2", "l3"], loc="upper left", bbox_to_anchor=anchor
    )

    ax.add_artist(l)
    ax.add_artist(l2)
    shrink = 0.5
    cbar_width_in = 0.15
    cbarax = ax.inset_axes(
        [
            legend1_width_axes_coord + legend_2_width_axes_coord,
            0 + shrink / 2,
            cbar_width_in / ax_width_in,
            shrink,
        ]
    )
    fig.colorbar(qm, cax=cbarax)

    fig.set_size_inches(figsize)
    # %%


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
    cmap: Optional[str] = None
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
def ann_heatmap():

    # %%
    p = mpl.rcParams
    p["legend.fontsize"] = 8
    p["legend.title_fontsize"] = 9
    # in font size units
    # details: https://matplotlib.org/3.1.1/api/legend_api.html
    p["legend.handlelength"] = 1
    p["legend.handleheight"] = 1
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
        HeatmapAnno(pd.Series(np.repeat(["a", "b", "c", "d"], 3)), "annos1", "Set1"),
        HeatmapAnno(
            pd.DataFrame(
                {"annos2": np.repeat(["var1", "var_bb", "c ccccccc", "d"], 3)}
            ),
            cmap="Set2",
        ),
        HeatmapAnno(pd.Series(np.arange(12)), "qannos", "YlOrBr"),
    ]
    row_anno = pd.Series([1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
    row_anno.name = "row anno"
    row_annos = [(row_anno,)]
    row_linkage = linkage(df)
    col_linkage = linkage(df.T)
    row_dendrogram_show = True
    col_dendrogram_show = True
    cbar_shrink = 0.6
    cbar_width_in = 0.2
    figsize = (8, 8)

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
        width_ratios=[2, 1, 5, 2],
        height_ratios=[2, 1, 1, 2, 10],
        figure=fig,
        hspace=0,
        wspace=0,
    )

    row_start, col_start = (0, 0)

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
    qm = ax.pcolormesh(df, cmap=cmap)
    ax.set(xlabel="Xlabel")
    ax.tick_params(
        axis="both",  # x, y
        which="major",  # both minor
        length=0,
        # width=1,
        # labelsize=1,
        bottom=False,
        labelbottom=True,
        left=False,
        labelleft=False,
        rotation=90,
    )
    ax.xaxis.set_major_locator(
        MaxNLocator(
            df.shape[1],
            steps=[1, 2, 5, 10],
            integer=True,
            prune="lower",
            # lower upper both None
            # min_n_ticks=3,
            # range symmetric about zero
            # symmetric=True,
        )
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    guides.append({"qm": qm, "name": "log-odds"})

    # TODO: set ylabel

    # TODO: legend_ax has a bbox for itself
    legend_ax = fig.add_subplot(gs[main_row, main_col + 1])
    legend_ax_bbox = legend_ax.get_position()
    legend_ax_width_in = figsize[0] * legend_ax_bbox.width
    legend_ax_height_in = figsize[1] * legend_ax_bbox.height

    legends_df = pd.DataFrame(columns=["name", "height", "width", "object", "priority"])
    for guide_d in guides:
        if "qm" in guide_d:
            legends_df = legends_df.append(
                dict(
                    name=guide_d["name"],
                    height=cbar_shrink,
                    width=cbar_width_in / legend_ax_width_in,
                    object=guide_d["qm"],
                    priority=0,
                ),
                ignore_index=True,
            )
        else:
            elem = guide_d["handels"], guide_d["labels"]
            l = Legend(legend_ax, *elem, title=guide_d["name"], loc="upper left")
            l._legend_box.align = 'left'
            width_in, height_in = get_legend_dimensions(*elem, guide_d["name"])
            legends_df = legends_df.append(
                dict(
                    name=guide_d["name"],
                    height=height_in / legend_ax_height_in,
                    width=width_in / legend_ax_width_in,
                    object=l,
                    priority=0,
                ),
                ignore_index=True,
            )

    # import pudb.remote; pudb.remote.set_trace(term_size=(119, 49))

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
            x_padding = 0.2
        else:  # cbar
            cbar_ax: Axes
            cbar_ax = legend_ax.inset_axes(
                [curr_x, curr_y - row_ser.height, row_ser.width, row_ser.height]
            )
            title_text = cbar_ax.set_title(
                row_ser['name'],
                fontdict={"fontsize": p["legend.title_fontsize"]},
                # center left right
                loc="left",
                # pad=rcParams["axes.titlepad"],
                # **kwargs: Text properties
            )

            r = fig.canvas.get_renderer()  # not necessary if figure was drawn
            title_bbox = legend_ax.transAxes.inverted().transform(title_text.get_window_extent(r))
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
            x_padding = max(title_size_axes_coord, y_tick_label_size) + 0.2
        # TODO: padding around colorbars hardcoded in axes coordinates
        curr_y -= row_ser.height + 0.025
        next_x = max(next_x, curr_x + row_ser.width + x_padding)
    fig
    # %%

    # # it is not sufficient to remove the ticks and labels - one must in addition set the labelsize and tick length to 0! The tick width is irrelevant
    # for ax in axes:
    #     ax.tick_params(
    #             axis='both',
    #             which='both',
    #             bottom=False,
    #             left=False,
    #             labelbottom=False,
    #             labelleft=False,
    #             labelsize=0,
    #             length=0,
    #     )


def add_anno_heatmap(ax, row_anno, is_row_anno):
    ax.axis("off")
    if not isinstance(row_anno, HeatmapAnno):
        row_anno = HeatmapAnno(*row_anno)
    df = row_anno.anno if is_row_anno else row_anno.anno.T
    # TODO: assumption: dataframe has homogeneous type
    if is_numeric_dtype(row_anno.anno.iloc[:, 0]):
        qm = ax.pcolormesh(df, cmap=row_anno.cmap, norm=row_anno.norm)
        return dict(name=row_anno.name, qm=qm)
    else:
        res = co.plotting.categorical_heatmap(
            df,
            ax=ax,
            cmap=row_anno.cmap,
            show_values=False,
            show_legend=False,
            despine=True,
        )
        return dict(name=row_anno.name, handels=res["patches"], labels=res["levels"])


# noinspection DuplicatedCode
def reproduce_error():

    figsize = (5, 5)
    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly, setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0.05, wspace=0.05)
    fig.set_size_inches(*figsize)
    gs = gridspec.GridSpec(
        5,
        4,
        width_ratios=[2, 1, 5, 2],
        height_ratios=[2, 1, 1, 2, 10],
        figure=fig,
        hspace=0,
        wspace=0,
    )

    row_start, col_start = (0, 0)
    nrows = 5
    ncols = 4
    main_row = row_start + nrows - 1
    main_col = col_start + ncols - 2

    guides = []

    ax = fig.add_subplot(gs[main_row, col_start])
    co.plotting.dendrogram_wrapper(row_linkage, ax, orientation="left")
    ax = fig.add_subplot(gs[row_start, main_col])
    co.plotting.dendrogram_wrapper(col_linkage, ax, orientation="top")
    for i, row_anno in enumerate(row_annos):
        ax = fig.add_subplot(gs[main_row, col_start + i + row_dendrogram_show])
        guides.append(add_anno_heatmap(ax, row_anno, is_row_anno=True))
    for i, col_anno in enumerate(col_annos):
        ax = fig.add_subplot(gs[row_start + col_dendrogram_show + i, main_col])
        guides.append(add_anno_heatmap(ax, col_anno, is_row_anno=False))

    co.heatmap(
        df,
        ax=fig.add_subplot(gs[main_row, main_col]),
        fig=fig,
        cmap=cmap,
        xlabel="Xlabel",
        ylabel="Ylabel",
        cbar_args=dict(shrink=0.5, aspect=30),
    )


def mpl_get_font_size_pt():
    # assumption: medium = factor 1
    # smaller sizes: / 1.2 ** steps
    # larger size * 1.2 ** steps
    pass
