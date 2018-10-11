#-
import subprocess
import time
from os.path import expanduser
from pathlib import Path

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codaplot.plotting import grouped_rows_violin, grouped_rows_line_collections, cut_dendrogram
# Test data setup
# ######################################################################
# Create 'base' test data ndarray with three clusters across rows and
# three clusters across columns.
# The rows and cols of each cluster are not adjacent and need to be
# correctly arranged by applying clustering.
# Before usage, we'll likely want to add some noise to this template
from scipy.cluster.hierarchy import linkage

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
cluster_ids_row_df = pd.DataFrame({'strict': np.tile([2, 3, 1], 10).T,
                                   'liberal':  np.tile([1, 2, 1], 10).T})

n_tiles = 5000
large_data = np.tile(data, (n_tiles, 1))

rng = np.random.RandomState(1)
names = 's' + pd.Series(np.arange(5)).astype(str)
std =  1
# noinspection PyAugmentAssignment
large_data = pd.DataFrame(large_data + rng.randn(*large_data.shape) * std,
                          columns=names)
large_data_cluster_ids_arr = np.tile(cluster_ids_row_df.iloc[:, 0].values, n_tiles)

timestamp = time.strftime('%d-%m-%y_%H-%M-%S')
output_dir = Path(expanduser(f'~/temp/plotting-tests_{timestamp}'))
output_dir.mkdir(exist_ok=True)


def test_grouped_rows_violin():
    fig, axes = plt.subplots(3, 1)
    grouped_rows_violin(data=large_data, row=large_data_cluster_ids_arr,
                        ax=axes, n_per_group=5000, sort=False, sharey=True)
    fp = output_dir / 'grouped-row-violin.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])


def test_grouped_rows_line_collections():
    fig, axes = plt.subplots(3, 1, constrained_layout=True)
    grouped_rows_line_collections(data=large_data, row=large_data_cluster_ids_arr,
                                  ax=axes, show_all_xticklabels=True,
                                  xlabel='x-label test')
    fp = output_dir / 'grouped-row-line-collection.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])

def test_cut_dendrogram_cluster_level():
    fig, ax = plt.subplots(1, 1)
    rng = np.random.RandomState(1)
    int_idx = rng.choice(large_data.shape[0], 1000)
    Z = linkage(large_data.iloc[int_idx, :])
    cut_dendrogram(linkage_mat=Z,
                   cluster_ids_data_order=pd.Series(large_data_cluster_ids_arr[int_idx]),
                   ax=ax, pretty=True)
    fp = output_dir / 'cut-dendrogram_cluster-level.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])

def test_cut_dendrogram_inspection():
    fig, ax = plt.subplots(1, 1)
    rng = np.random.RandomState(1)
    int_idx = rng.choice(large_data.shape[0], 1000)
    Z = linkage(large_data.iloc[int_idx, :])
    cut_dendrogram(linkage_mat=Z,
                   cluster_ids_data_order=pd.Series(large_data_cluster_ids_arr[int_idx]),
                   ax=ax, pretty=True,
                   stop_at_cluster_level=False,
                   min_cluster_size=30)
    fp = output_dir / 'cut-dendrogram_inspection.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])


