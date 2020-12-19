# -
import os
from os.path import expanduser
from pathlib import Path
import subprocess
import time
from typing import Literal

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Test data setup
# ######################################################################
# Create 'base' test data ndarray with three clusters across rows and
# three clusters across columns.
# The rows and cols of each cluster are not adjacent and need to be
# correctly arranged by applying clustering.
# Before usage, we'll likely want to add some noise to this template
from scipy.cluster.hierarchy import linkage
import sklearn.datasets

import codaplot as co
from codaplot.plotting import (
    categorical_heatmap,
    cut_dendrogram,
    grouped_rows_heatmap,
    grouped_rows_line_collections,
    grouped_rows_violin,
    spaced_heatmap,
)

# -

rows_with_three_different_levels = np.tile([20, 30, 10], (5, 10)).T
# array([[20, 20, 20, 20, 20],
#        [30, 30, 30, 30, 30],
#        [10, 10, 10, 10, 10],
#        [20, 20, 20, 20, 20],
#        ...
cols_with_three_different_levels = np.tile([5, 10, 5, 15, 10], (30, 1))
# array([[ 5, 10,  5, 15, 10],
#        [ 5, 10,  5, 15, 10],
#        ...
data = rows_with_three_different_levels + cols_with_three_different_levels
cluster_ids_row_df = pd.DataFrame(
    {"strict": np.tile([2, 3, 1], 10).T, "liberal": np.tile([1, 2, 1], 10).T}
)

n_tiles = 5000
large_data = np.tile(data, (n_tiles, 1))

rng = np.random.RandomState(1)
names = "s" + pd.Series(np.arange(5)).astype(str)
std = 1
# noinspection PyAugmentAssignment
large_data = pd.DataFrame(
    large_data + rng.randn(*large_data.shape) * std, columns=names
)
large_data_cluster_ids_arr = np.tile(cluster_ids_row_df.iloc[:, 0].values, n_tiles)

timestamp = time.strftime("%d-%m-%y_%H-%M-%S")
output_dir = Path(expanduser(f"~/temp/plotting-tests_{timestamp}"))
output_dir.mkdir(exist_ok=True)


def test_grouped_rows_violin():
    fig, axes = plt.subplots(3, 1)
    grouped_rows_violin(
        data=large_data,
        row=large_data_cluster_ids_arr,
        ax=axes,
        n_per_group=5000,
        sort=False,
        sharey=True,
    )
    fp = output_dir / "grouped-row-violin.png"
    fig.savefig(fp)
    subprocess.run(["firefox", fp])


def test_grouped_rows_line_collections():
    fig, axes = plt.subplots(3, 1, constrained_layout=True)
    grouped_rows_line_collections(
        data=large_data,
        row=large_data_cluster_ids_arr,
        ax=axes,
        show_all_xticklabels=True,
        xlabel="x-label test",
    )
    fp = output_dir / "grouped-row-line-collection.png"
    fig.savefig(fp)
    subprocess.run(["firefox", fp])


def test_cut_dendrogram_cluster_level():
    fig, ax = plt.subplots(1, 1)
    rng = np.random.RandomState(1)
    int_idx = rng.choice(large_data.shape[0], 1000)
    Z = linkage(large_data.iloc[int_idx, :])
    cut_dendrogram(
        linkage_mat=Z,
        cluster_ids_data_order=pd.Series(large_data_cluster_ids_arr[int_idx]),
        ax=ax,
        pretty=True,
    )
    fp = output_dir / "cut-dendrogram_cluster-level.png"
    fig.savefig(fp)
    subprocess.run(["firefox", fp])


def test_cut_dendrogram_inspection():
    fig, ax = plt.subplots(1, 1)
    rng = np.random.RandomState(1)
    int_idx = rng.choice(large_data.shape[0], 1000)
    Z = linkage(large_data.iloc[int_idx, :])
    cut_dendrogram(
        linkage_mat=Z,
        cluster_ids_data_order=pd.Series(large_data_cluster_ids_arr[int_idx]),
        ax=ax,
        pretty=True,
        stop_at_cluster_level=False,
        min_cluster_size=30,
    )
    fp = output_dir / "cut-dendrogram_inspection.png"
    fig.savefig(fp)
    subprocess.run(["firefox", fp])


def test_grouped_rows_heatmap():
    fig, ax = plt.subplots(1, 1)
    grouped_rows_heatmap(
        df=large_data,
        row_=large_data_cluster_ids_arr,
        fn="mean",
        cmap="YlOrBr",
        ax=ax,
        fig=fig,
    )
    fp = output_dir / "grouped-rows-heatmap.png"
    fig.savefig(fp)
    subprocess.run(["firefox", fp])


def test_spaced_heatmap(tmpdir):
    tmpdir = Path(tmpdir)

    # prepare test data
    # data with cluster ids, not ordered by cluster ids
    data, cluster_ids = sklearn.datasets.make_blobs(
        n_samples=10, n_features=10, centers=3
    )

    # data and cluster ids, sorted by cluster ids
    # *this is the input expected by the heatmap functions, for now*
    row_order = np.argsort(cluster_ids)
    row_clusters = cluster_ids[row_order] + 1
    data = data[row_order]
    df = pd.DataFrame(data)
    col_clusters = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])

    # one spacer width for all row spacers
    # one spacer width for all col spacers
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        row_spacer_size=0.02,
        col_spacer_size=0.1,
        pcolormesh_args={"vmin": -3, "vmax": 3},
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test1.png"))

    # specify sizes for all spacer elements individually
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        row_spacer_size=[0.02, 0.1],
        col_spacer_size=[0.2, 0.05],
        pcolormesh_args={"vmin": -3, "vmax": 3},
        show_col_labels=True,
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test2.png"))

    # without column clustering
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df,
        row_clusters=row_clusters,
        row_spacer_size=[0.02, 0.1],
        pcolormesh_args={"vmin": -3, "vmax": 3},
        show_row_labels=True,
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test3.png"))

    # without row clustering
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df,
        col_clusters=col_clusters,
        col_spacer_size=0.1,
        pcolormesh_args={"vmin": -3, "vmax": 3},
        show_row_labels=True,
        show_col_labels=True,
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test4.png"))

    subprocess.run(["firefox", tmpdir])

    # with string labels
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    row_clusters_str = np.array(list("aaaabbbccc"))
    qm = spaced_heatmap(
        ax=ax,
        df=df,
        row_clusters=row_clusters_str,
        col_clusters=col_clusters,
        row_spacer_size=0.02,
        col_spacer_size=0.1,
        pcolormesh_args={"vmin": -3, "vmax": 3},
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test5.png"))

    # multiindex
    df2 = df.copy()
    df2.columns = pd.MultiIndex.from_arrays(
        [list("aaaabbbccc"), np.repeat([0.1, 0.2], 5)]
    )
    df2.index = df2.columns
    ## same spacers
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df2,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        row_spacer_size=0.02,
        col_spacer_size=0.1,
        pcolormesh_args={"vmin": -3, "vmax": 3},
    )
    fig.colorbar(qm, shrink=0.4, extend="both")
    fig.savefig(tmpdir.joinpath("test6.png"))

    # colorbar
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0)
    qm = spaced_heatmap(
        ax=ax,
        df=df2,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        row_spacer_size=0.02,
        col_spacer_size=0.1,
        pcolormesh_args={"vmin": -3, "vmax": 3},
        add_colorbar=True,
        cbar_args=dict(shrink=0.9, extend="both"),
        fig=fig,
    )
    fig.savefig(tmpdir.joinpath("test7.png"))


def test_categorical_heatmap(tmpdir):
    tmpdir = str(tmpdir)

    # from int df
    data = pd.DataFrame(np.repeat(np.arange(1, 7), 10))
    fig, ax = plt.subplots(1, 1)
    categorical_heatmap(data, ax=ax)
    fig.savefig(tmpdir + "/plot1.png")

    ## with equal spacer
    fig, ax = plt.subplots(1, 1)
    categorical_heatmap(
        data,
        ax=ax,
        spaced_heatmap_args=dict(row_clusters=data.iloc[:, 0], row_spacer_size=0.1),
    )
    fig.savefig(tmpdir + "/plot2.png")

    ## with different spacers
    fig, ax = plt.subplots(1, 1)
    categorical_heatmap(
        data,
        ax=ax,
        spaced_heatmap_args=dict(
            row_clusters=data.iloc[:, 0], row_spacer_size=np.arange(5) / 100
        ),
    )
    fig.savefig(tmpdir + "/plot3.png")

    # from string
    data = pd.DataFrame(np.repeat(list("abcdef"), 10))
    fig, ax = plt.subplots(1, 1)
    categorical_heatmap(data, ax=ax)
    fig.savefig(tmpdir + "/plot4.png")

    ## with spacing
    fig, ax = plt.subplots(1, 1)
    categorical_heatmap(
        data,
        ax=ax,
        spaced_heatmap_args=dict(row_clusters=data.iloc[:, 0], row_spacer_size=0.1),
    )
    fig.savefig(tmpdir + "/plot5.png")

    subprocess.run(["firefox", tmpdir])


@pytest.mark.parametrize(
    "edge_alignment,direction",
    [
        ("outside", "y"),
        ("inside", "y"),
        ("centered", "y"),
        ("outside", "x"),
        ("inside", "x"),
        ("centered", "x"),
    ],
)
def test_frame_groups(
    tmpdir, edge_alignment: Literal["inside", "outside", "centered"], direction
):
    tmpdir = str(tmpdir)

    # %%
    spacer_sizes = 0.4
    plot_df = pd.DataFrame({"a": np.repeat([1, 2], 2000), "b": np.repeat([3, 4], 2000)})
    group_ids = plot_df["a"].to_numpy()
    heatmap_kwargs = dict(
        row_spacer_sizes=spacer_sizes,
        row_spacing_group_ids=group_ids,
    )
    if direction == "x":
        plot_df = plot_df.T
        group_ids = plot_df.loc["a"].to_numpy()
        heatmap_kwargs = dict(
            col_spacer_sizes=spacer_sizes,
            col_spacing_group_ids=group_ids,
        )

    fig, ax = plt.subplots(
        1,
        1,
        constrained_layout=False,
        figsize=(8 / 2.54, 8 / 2.54),
        dpi=1200,
    )
    co.heatmap(
        plot_df,
        **heatmap_kwargs,
        ax=ax,
        rasterized=True,
        snap=True,
    )
    # ax.axhline((1 - spacer_sizes) / 2, lw=0.3, c="red")
    # ax.axhline(1 - (1 - spacer_sizes) / 2, lw=0.3, c="red")
    # %%

    co.frame_groups(
        group_ids=group_ids,
        spacer_sizes=spacer_sizes,
        ax=ax,
        direction=direction,
        edge_alignment=edge_alignment,
        colors={1: "red", 2: "blue", 3: "green"},
        axis_off=True,
        fancy_box_kwargs=dict(alpha=0.5, clip_on=False),
        origin=(0, 0),
        add_labels=False,
        non_direction_sides_padding_axes_coord=0.02,
        direction_sides_padding_axes_coord=0.02,
    )

    # interactive development on cluster
    # png_path = '/icgc/dkfzlsdf/analysis/hs_ontogeny/notebook-data/mqgaFb6PQ5K3d2br/knk2SvgxP0BydTRK2.png'
    # svg_path = '/icgc/dkfzlsdf/analysis/hs_ontogeny/notebook-data/mqgaFb6PQ5K3d2br/knk2SvgxP0BydTRK5.svg'
    # ut.save_and_display(fig, png_path=png_path)
    # ut.dkfz_link(svg_path)
    output_dir = tmpdir + f"/edge-alignment={edge_alignment}_direction={direction}"
    os.makedirs(output_dir)
    png_path = output_dir + "/plot.png"
    svg_path = png_path.replace(".png", ".svg")

    fig.savefig(png_path)
    fig.savefig(svg_path)
    subprocess.run(["firefox", svg_path])
