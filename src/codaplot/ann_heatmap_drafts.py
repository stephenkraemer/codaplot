from typing import List

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, gridspec as gridspec, patches as mpatches
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from scipy.cluster.hierarchy import linkage, leaves_list

import codaplot as co
from codaplot.ann_heatmap import get_legend_dimensions, add_anno_heatmap


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

def fn3():
    a = 3
    a2 = 4


    d = test(var2=a, a2)
    return d


def test(var2, var1):
    b = a * 2
    c = a2 * a
    d = a + b + c
    return d