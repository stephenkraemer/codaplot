import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

import codaplot as co
from codaplot.array_manager import cross_plot, anno_axes
from codaplot.plotting import adjust_coords


pcm_display_kwargs = dict(edgecolor="face", linewidth=0.2)

def test_cross_plot():

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

    figsize = (20 / 2.54, 20 / 2.54)
    figsize_ratio = figsize[0] / figsize[1]
    res, plot_array = cross_plot(
            figsize=figsize,
            constrained_layout=False,
            layout_pads=dict(h_pad=0, w_pad=0, hspace=0.03, wspace=0.03),
            center_margin_ticklabels=True,
            center_col_pad=(0.25 / figsize_ratio, "rel"),
            center_row_pad=(0.25, "rel"),
            align_args=False,
            pads_around_center=[
                (0.2 / 2.54, "abs"),
                (1 / 2.54, "abs"),
                (1 / 2.54, "abs"),
                (0.2 / 2.54, "abs"),
            ],
            legend_args=dict(xpad_in=0.2, guide_titles=None),
            legend_extent=["center"],
            legend_axes_selectors=["ae1", "ae2", "ae3", (4, 1)],
            center=np.array(
                    [
                        [
                            dict(
                                    _name="ae1",
                                    guide_title="1",
                                    df=df.iloc[row_order, col_order],
                                    cmap="RdBu_r",
                            ),
                            dict(
                                    _name="ae2",
                                    guide_title="2",
                                    df=df.iloc[row_order, col_order] * 10,
                                    cmap="YlOrBr",
                            ),
                        ],
                        [
                            dict(
                                    guide_title="3",
                                    df=df.iloc[row_order, col_order] * 5,
                                    cmap="RdBu_r",
                            ),
                            dict(
                                    guide_title="3",
                                    df=df.iloc[row_order, col_order] * 2,
                                    cmap="viridis",
                            ),
                        ],
                    ]
            ),
            top=[
                dict(
                        _name="ae3",
                        guide_title="Anno1",
                        df=pd.DataFrame({"anno1": col_clusters}).T.iloc[:, col_order],
                        is_categorical=True,
                        cmap="Set1",
                        guide_args=dict(),
                ),
                dict(df=pd.DataFrame({"values": np.arange(11)}).T, guide_title="anno2"),
            ],
            top_sizes=[(1 / 2.54, "abs"), (1 / 2.54, "abs")],
            left=[
                dict(
                        df=pd.DataFrame(
                                {"anno2": pd.Series(row_clusters)[row_order].astype(str)}
                        ),
                        guide_title="Anno3",
                ),
                dict(
                        _func=anno_axes(loc="left", prune_all=True)(co.plotting.frame_groups),
                        # group_ids=row_clusters[row_order],
                        direction="y",
                        colors=dict(zip([1, 2, 3], sns.color_palette("Set1", 3))),
                        linewidth=2,
                        add_labels=True,
                        labels=["1", "2", "3"],
                        label_colors=None,
                        label_groups_kwargs=dict(rotation=0),
                ),
            ],
            left_sizes=[(1 / 2.54, "abs"), (1 / 2.54, "abs")],
            right=[
                dict(
                        _func=spaced_barplot,
                        y=np.arange(0, df.shape[0]) + 0.5,
                        width=df.sum(axis=1),
                        # TODO: height not calculated automatically
                        height=0.05,
                        color="gray",
                )
            ],
            right_sizes=[(2 / 2.54, "abs")],
            # left=None,
            # left_sizes=None,
            bottom=[
                dict(
                        _func=anno_axes("bottom")(plt.bar),
                        x=co.plotting.find_stretches2(col_clusters[col_order])[1],
                        height=df.groupby(col_clusters, axis=1).sum().sum(axis=0),
                        width=0.1,
                        color="gray",
                )
            ],
            bottom_sizes=[(2.5 / 2.54, "abs")],
            row_dendrogram=True,
            col_dendrogram=True,
            row_order=row_linkage,
            col_order=col_linkage,
            row_spacing_group_ids=pd.Series(row_clusters)[row_order],
            col_spacing_group_ids=pd.Series(col_clusters)[col_order],
            row_spacer_sizes=[0.2, 0.1],
            col_spacer_sizes=[0.1, 0.2],
            default_func=co.plotting.heatmap3,
            default_func_kwargs=dict(
                    guide_args=dict(shrink=0.4, aspect=4), xticklabel_rotation=90
            ),
    )
    res["fig"].savefig("/home/stephen/temp/test.pdf")


def test_cross_plot_aligned():

    # %%
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

    @anno_axes(loc="right")
    def spaced_barplot2(df, spacing_group_ids, spacer_sizes, ax):
        ax.barh(
                y=adjust_coords(
                        np.arange(0, df.shape[0]) + 0.5,
                        spacing_group_ids=spacing_group_ids,
                        spacer_sizes=spacer_sizes,
                        ),
                width=df.sum(axis=1),
                # TODO: height not calculated automatically
                height=0.05,
                color="gray",
        )

    @anno_axes(loc="right")
    def spaced_barplot3(df, y, ax):
        ax.barh(
                y=y,
                width=df.sum(axis=1),
                # TODO: height not calculated automatically
                height=0.05,
                color="gray",
        )

    figsize = (20 / 2.54, 20 / 2.54)
    figsize_ratio = figsize[0] / figsize[1]
    res, plot_array = cross_plot(
            figsize=figsize,
            constrained_layout=False,
            layout_pads=dict(h_pad=0, w_pad=0, hspace=0.03, wspace=0.03),
            center_margin_ticklabels=True,
            center_col_pad=(0.25 / figsize_ratio, "rel"),
            center_row_pad=(0.25, "rel"),
            pads_around_center=[
                (0.2 / 2.54, "abs"),
                (1 / 2.54, "abs"),
                (1 / 2.54, "abs"),
                (0.2 / 2.54, "abs"),
            ],
            legend_args=dict(xpad_in=0.2, guide_titles=None),
            legend_extent=["center"],
            legend_axes_selectors=["ae1", "ae2", "ae3", (4, 1)],
            row_dendrogram=True,
            col_dendrogram=True,
            row_order=row_linkage,
            col_order=col_linkage,
            row_spacing_group_ids=pd.Series(row_clusters),
            col_spacing_group_ids=pd.Series(col_clusters),
            row_spacer_sizes=[0.2, 0.1],
            col_spacer_sizes=[0.1, 0.2],
            default_func=co.plotting.heatmap3,
            default_func_kwargs=dict(
                    guide_args=dict(shrink=0.4, aspect=4), xticklabel_rotation=90
            ),
            center=np.array(
                    [
                        [
                            dict(_name="ae1", guide_title="1", df=df, cmap="RdBu_r"),
                            dict(_name="ae2", guide_title="2", df=df, cmap="YlOrBr"),
                        ],
                        [
                            dict(guide_title="3", df=df, cmap="RdBu_r"),
                            dict(guide_title="3", df=df, cmap="viridis"),
                        ],
                    ]
            ),
            top=[
                dict(
                        _name="ae3",
                        guide_title="Anno1",
                        df=pd.DataFrame({"anno1": col_clusters}).T,
                        is_categorical=True,
                        cmap="Set1",
                        guide_args=dict(),
                ),
                dict(df=pd.DataFrame({"values": col_clusters}).T, guide_title="anno2"),
            ],
            top_sizes=[(1 / 2.54, "abs"), (1 / 2.54, "abs")],
            left=[
                dict(
                        df=pd.DataFrame({"anno2": pd.Series(row_clusters).astype(str)}),
                        guide_title="Anno3",
                ),
                dict(
                        _func=anno_axes(loc="left", prune_all=True)(co.plotting.frame_groups),
                        # group_ids=row_clusters[row_order],
                        direction="y",
                        colors=dict(zip([1, 2, 3], sns.color_palette("Set1", 3))),
                        linewidth=2,
                        add_labels=True,
                        labels=["1", "2", "3"],
                        label_colors=None,
                        label_groups_kwargs=dict(rotation=0),
                ),
            ],
            left_sizes=[(1 / 2.54, "abs"), (1 / 2.54, "abs")],
            right=[
                dict(_func=spaced_barplot2, df=df),
                dict(_func=spaced_barplot3, y=np.arange(df.shape[1]) + 0.5, df=df),
            ],
            right_sizes=[(2 / 2.54, "abs"), (2 / 2.54, "abs")],
    )
    res["fig"].savefig("/home/stephen/temp/test.pdf")


def test_simple_anno_heatmap():

    rng = np.random.RandomState(1234)
    row_clusters = pd.Series(np.array([2, 1, 1, 2, 3, 3, 2, 2, 3, 1, 2]))
    col_clusters = pd.Series(np.array([2, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1]))
    df = (
        pd.DataFrame(rng.randn(11, 11))
            .add(row_clusters * 2, axis=0)
            .add(col_clusters * 4, axis=1)
    )
    row_linkage = linkage(df)
    col_linkage = linkage(df.T)

    mpl.rcParams["legend.frameon"] = False

    with mpl.rc_context({"legend.title_fontsize": 7, "legend.fontsize": 7}):
        cross_plot(
                center=[dict(df=df, cmap="RdBu_r", guide_title="% Meth.", edgecolor='white')],
                center_margin_ticklabels=True,
                pads_around_center=(0.2 / 2.54, "abs"),
                figsize=(15 / 2.54, 10 / 2.54),
                constrained_layout=False,
                layout_pads=dict(wspace=0.05, hspace=0.05),
                top=[
                    dict(
                            df=pd.DataFrame({"col clusters": col_clusters}).T,
                            cmap="Set1",
                            guide_title="Col cluster",
                            is_categorical=True,
                            edgecolor='white',
                    )
                ],
                top_sizes=[(0.5 / 2.54, "abs")],
                left=[
                    dict(
                            df=pd.DataFrame({"row clusters": row_clusters}),
                            cmap="Set2",
                            guide_title="Row cluster",
                            is_categorical=True,
                            edgecolor='white1'
                    )
                ],
                left_sizes=[(0.5 / 2.54, "abs")],
                row_order=row_linkage,
                col_order=col_linkage,
                row_spacing_group_ids=row_clusters,
                col_spacing_group_ids=col_clusters,
                row_spacer_sizes=0.05,
                col_spacer_sizes=0.05,
                col_dendrogram=dict(cluster_ids_data_order=None, base_color='darkgray'),
                row_dendrogram=dict(colors="Set2", min_cluster_size=4, base_color=(.2, .2, .2)),
                default_func_kwargs=dict(guide_args=dict(shrink=0.3, aspect=8)),
        )


def test_heatmap3():

    # %%
    df = pd.DataFrame(np.random.random_integers(0, 10, (9, 9))).set_axis(
            ["Aga", "bg", "Ag", "CD", "pP", "1", "8", "3", "0"], axis=1, inplace=False
    )
    row_clusters = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    col_clusters = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1
    col_label_colors = np.array(
            sns.color_palette("Set1", np.unique(col_clusters).shape[0])
    )[col_clusters]
    # cluster_to_color = pd.Series({1: "blue", 2: "green", 3: "red"})
    # labels are always str
    label_to_color = dict(
            zip(list(df.index.astype(str)), np.repeat(["blue", "red", "black"], 3))
    )

    fig, axes = plt.subplots(
            2,
            2,
            dpi=180,
            figsize=(2.5, 2.5),
            constrained_layout=True,
            gridspec_kw=dict(height_ratios=(8, 1), width_ratios=(1, 8)),
    )
    ax = axes[0, 1]
    anno_ax = axes[1, 1]
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    # ax.imshow(pd.DataFrame(np.random.randn(9, 9)), rasterized=True)
    res = co.plotting.heatmap3(
            df=df,
            ax=ax,
            xticklabels=True,
            xticklabel_colors=col_label_colors,
            yticklabels=True,
            yticklabel_colors=label_to_color,
            pcolormesh_args=dict(**pcm_display_kwargs),
            row_spacing_group_ids=row_clusters,
            col_spacing_group_ids=col_clusters,
            row_spacer_sizes=[0.1, 0.2],
            col_spacer_sizes=[0.2, 0.1],
            show_guide=True,
            # is_categorical=True,
            # heatmap_args={},
            # cmap='Set3'
    )
    print(res)
    # ax.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True, size=0, width=0.5)

    co.plotting.label_groups(
            group_ids=col_clusters,
            ax=anno_ax,
            y=0.5,
            spacer_size=[0.2, 0.1],
            labels=np.array(["AA", "aa", "gg"]),
    )

    anno_ax.axis("off")

    co.plotting.label_groups(
            group_ids=row_clusters,
            ax=axes[0, 0],
            x=0.5,
            spacer_size=[0.1, 0.2],
            labels=np.array(["AA", "aa", "gg"]),
    )

    co.plotting.frame_groups(
            group_ids=col_clusters,
            ax=anno_ax,
            direction="x",
            colors=["black", "orange", "blue"],
            linewidth=4,
            spacer_sizes=[0.2, 0.1],
    )

    co.plotting.frame_groups(
            group_ids=row_clusters,
            ax=axes[0, 0],
            direction="y",
            colors=["black", "orange", "red"],
            linewidth=1,
            spacer_sizes=[0.1, 0.2],
    )
    axes[0, 0].axis("off")

    fig.savefig("/home/stephen/temp/test.pdf")

    # %%

    # code to mark groups in a plot
    # %%

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


'''
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
            layout_pads=dict(h_pad=0.1, w_pad=0.1, hspace=0, wspace=0),
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
            spacing_group_ids=clusters,
            spacer_sizes=0.2,
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
'''


@anno_axes("right")
def spaced_barplot(ax, **kwargs):
    ax.barh(**kwargs)