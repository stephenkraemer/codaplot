import time
from os.path import expanduser
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import codaplot.linkage_mat as lm

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
output_dir = Path(expanduser(f'~/temp/linkage-matrix-tests_{timestamp}'))
output_dir.mkdir(exist_ok=True)

def test_linkage_matrix():

    rng = np.random.RandomState(1)
    int_idx = rng.choice(large_data.shape[0], 1000)
    medium_size_data = large_data.iloc[int_idx, :].copy()
    dist_mat = pdist(medium_size_data)
    Z = linkage(dist_mat)

    linkage_matrix = lm.LinkageMatrix(matrix=Z, dist_mat=dist_mat,
                                      index=medium_size_data.index)

    linkage_matrix.add_dynamic_cutree_division('less strict', deepSplit=1)
    linkage_matrix.add_dynamic_cutree_division('strict', deepSplit=2)
