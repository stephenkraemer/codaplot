import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dynamicTreeCut import cutreeHybrid
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_moons
import codaplot as co
import pytest

@pytest.fixture()
def blobs():
    arr, fine_cluster_ids = make_blobs(n_samples=100, n_features=2,
                                       centers=[
                                           [1, 1],
                                           [2, 2],
                                           [3, 3],
                                           [5, 5],
                                           [6, 6],
                                           [7, 7],
                                       ],
                                       cluster_std=0.2, random_state=600)
    # scipy cluster IDs start at 0; here, 0 is reserved for unlabeled samples
    fine_cluster_ids += 1

    # Create clustering with two clusters.
    coarse_cluster_ids = np.copy(fine_cluster_ids)
    coarse_cluster_ids[coarse_cluster_ids < 4] = 1
    coarse_cluster_ids[coarse_cluster_ids >= 4] = 2
    # plt.scatter(arr[:, 0], arr[:, 1], c=coarse_cluster_ids, cmap='Set1')
    return arr, fine_cluster_ids, coarse_cluster_ids


def test_dynamic_tree_cut_with_blobs(blobs):

    arr, fine_cluster_ids, coarse_cluster_ids = blobs

    dist_mat = pdist(arr)
    Z = linkage(dist_mat)

    # make_blobs cluster ids are not going from 1 to 6 in the dendrogram order
    # adjust them in that way
    aligned_cluster_ids = pd.Series(fine_cluster_ids).replace(
            dict(zip(pd.Series(fine_cluster_ids[leaves_list(Z)]).unique(), range(1, 7)))).values
    # plt.scatter(arr[:, 0], arr[:, 1], c=aligned_cluster_ids, cmap='Set1')

    # coarse ids are already aligned
    assert np.all(pd.Series(coarse_cluster_ids[leaves_list(Z)]).unique() == [1, 2])


    # Cut using dynamicTreeCut. The cluster IDs should be in order from 1..N
    # when data are ordered in leave order of the linkage
    linkage_obj = co.Linkage(matrix=Z, dist_mat=dist_mat)
    linkage_obj.dynamic_tree_cut('coarse', minClusterSize=20, deepSplit=1)
    linkage_obj.dynamic_tree_cut('fine', minClusterSize=5, deepSplit=2)
    linkage_obj.dynamic_tree_cut('finer', minClusterSize=5, deepSplit=2.5)
    coarse_tree_cut_ids = linkage_obj.cluster_ids.df['coarse'].values
    fine_tree_cut_ids = linkage_obj.cluster_ids.df['fine'].values
    # plt.scatter(arr[:, 0], arr[:, 1], c=coarse_tree_cut_ids, cmap='Set1')
    # plt.scatter(arr[:, 0], arr[:, 1], c=fine_tree_cut_ids, cmap='Set1')
    assert np.all(coarse_cluster_ids == coarse_tree_cut_ids)
    assert np.all(aligned_cluster_ids == fine_tree_cut_ids)


def test_dynamic_tree_cut_with_moons():
    arr, cluster_ids = make_moons(n_samples=1000)
    cluster_ids += 1
    # plt.scatter(arr[:, 0], arr[:, 1], c=cluster_ids, cmap='Set1')
    dist_mat = pdist(arr)
    Z = linkage(dist_mat)
    # cluster ids already aligned when in leave order
    assert np.all(pd.Series(cluster_ids[leaves_list(Z)]).unique() == [1, 2])
    linkage_obj2 = co.Linkage(Z, dist_mat)
    linkage_obj2.dynamic_tree_cut('1', deepSplit=2)
    assert np.all(cluster_ids == linkage_obj2.cluster_ids.df['1'].values)
    # plt.scatter(arr[:, 0], arr[:, 1], c=linkage_obj2.cluster_ids.df['1'], cmap='Set1')

@pytest.mark.parametrize('usecols', [None, [0, 1]])
def test_iterative_dynamic_tree_cut_with_numeric_ids(blobs, usecols):
    arr, fine_cluster_ids, coarse_cluster_ids = blobs

    df = pd.DataFrame(arr)
    # if usecols is [0, 1] we add two more columns
    # if these columns are wrongly used despite the usecols specification,
    # the resulting cluster IDs will be wrong and the test will fail
    if usecols is not None:
        df[3] = np.repeat([30, 40], 50)
        df[4] = np.repeat([30, 40], 50)

    dist_mat = pdist(arr)
    Z = linkage(dist_mat)

    assert np.all(pd.Series(coarse_cluster_ids[leaves_list(Z)]).unique() == [1, 2])
    # that means: coarse ids happend to be aligned with the leave order

    # Cut using dynamicTreeCut. The cluster IDs should be in order from 1..N
    # when data are ordered in leave order of the linkage
    linkage_obj = co.Linkage(matrix=Z, dist_mat=dist_mat)
    linkage_obj.dynamic_tree_cut('coarse', minClusterSize=20, deepSplit=1)

    # Test that the coarse cutting is equal to the correct coarse cluster ids
    # Note that the cluster numbering in the coarse cluster IDs was already aligned
    # with the leaf ordering
    assert np.all(linkage_obj.cluster_ids.df['coarse'].values == coarse_cluster_ids)

    linkage_obj.iterative_dynamic_tree_cut(
            clustering_name='coarse', cluster_id=1,
            data=df, name='it1', treecut_args={'minClusterSize': 5,
                                               'deepSplit': 1},
            usecols=usecols,
            create_subclusters=False)

    it1_tree_cut_ids = linkage_obj.cluster_ids.df['it1'].values
    # plt.scatter(arr[:, 0], arr[:, 1], c=it1_tree_cut_ids, cmap='Set1')
    it_id_in_new_leave_order = it1_tree_cut_ids[linkage_obj.leaf_orders['it1']]

    # if we order the data in the original leave order, the cluster IDs won't
    # be sequential (ie not go from 1 to 4), but the assignment of points to
    # the same cluster will still be correct. To get sequential IDs, we start
    # from that correct assignment and now just make the IDs sequential.
    it_ids_in_orig_leave_order = _make_cluster_ids_sequential(
            it1_tree_cut_ids[leaves_list(Z)])

    true_iterative_ids = np.copy(fine_cluster_ids)
    true_iterative_ids[true_iterative_ids > 3] = 4
    true_iterative_ids = _make_cluster_ids_sequential(true_iterative_ids[linkage_obj.leaf_orders['it1']])

    assert np.all(true_iterative_ids == it_id_in_new_leave_order)
    assert np.all(true_iterative_ids == it_ids_in_orig_leave_order)

@pytest.mark.parametrize('usecols', [None, [0, 1]])
def test_iterative_dynamic_tree_cut_with_subcluster_strings(blobs, usecols):
    arr, fine_cluster_ids, coarse_cluster_ids = blobs

    df = pd.DataFrame(arr)
    # if usecols is [0, 1] we add two more columns
    # if these columns are wrongly used despite the usecols specification,
    # the resulting cluster IDs will be wrong and the test will fail
    if usecols is not None:
        df[3] = np.repeat([30, 40], 50)
        df[4] = np.repeat([30, 40], 50)

    dist_mat = pdist(arr)
    Z = linkage(dist_mat)

    linkage_obj = co.Linkage(matrix=Z, dist_mat=dist_mat)
    linkage_obj.dynamic_tree_cut('coarse', minClusterSize=20, deepSplit=1)
    assert np.all(linkage_obj.cluster_ids.df['coarse'].values == coarse_cluster_ids)
    linkage_obj.iterative_dynamic_tree_cut(
            clustering_name='coarse', cluster_id=1,
            data=df, name='it1', treecut_args={'minClusterSize': 5,
                                               'deepSplit': 1},
            usecols=usecols,
            create_subclusters=True
    )
    leaf_order_treecut_ids = np.array(linkage_obj
                                      .cluster_ids.df['it1']
                                      [linkage_obj.leaf_orders['it1']])

    # avoid bug in seaborn which converts numeric strings to floats
    renamed_str_ids = linkage_obj.cluster_ids.df['it1'].cat.rename_categories(
            {'1_1': 'a1_1', '1_2': 'a1_2',
             '1_3': 'a1_3', '2': 'a2'})
    # sns.scatterplot(arr[:, 0], arr[:, 1], hue=renamed_str_ids)

    true_iterative_ids = np.copy(fine_cluster_ids)
    true_iterative_ids[true_iterative_ids > 3] = 4
    true_iterative_ids = _make_cluster_ids_sequential(
            true_iterative_ids[linkage_obj.leaf_orders['it1']]).astype(str)
    true_iterative_ids[true_iterative_ids == '1'] = '1_1'
    true_iterative_ids[true_iterative_ids == '2'] = '1_2'
    true_iterative_ids[true_iterative_ids == '3'] = '1_3'
    true_iterative_ids[true_iterative_ids == '4'] = '2'

    assert np.all(leaf_order_treecut_ids == true_iterative_ids)


def _make_cluster_ids_sequential(arr):
    diffs = np.diff(np.insert(arr, 0, 0))
    diffs[diffs != 0] = 1
    cumsums = np.cumsum(diffs)
    return cumsums.astype(int)

def test_linkage_get_subpart():
    arr = np.array([[2, 4, 5, 10, 11, 20, 21, 22]]).T
    dist_mat = pdist(arr, metric='cityblock')
    Z = linkage(dist_mat, method='complete')
    linkage_obj = co.Linkage(matrix=Z, dist_mat=dist_mat, index=np.arange(len(arr)).astype(str))

    # test: extract coherent part of cluster (subtree)
    int_idx = [0, 1, 2]
    new_linkage_obj_labels = linkage_obj.get_subpart(labels=pd.Index(int_idx).astype(str))
    new_linkage_obj_int_idx = linkage_obj.get_subpart(int_idx = int_idx)
    assert np.all(new_linkage_obj_labels.matrix == np.array([[ 1.,  2.,  1.,  2.],
                                                             [ 0.,  3.,  3.,  3.]]))
    assert np.all(new_linkage_obj_int_idx.matrix == np.array([[ 1.,  2.,  1.,  2.],
                                                              [ 0.,  3.,  3.,  3.]]))

    # test: extract all elements
    new_linkage_obj_labels = linkage_obj.get_subpart(labels=pd.Index(range(len(arr))).astype(str))
    new_linkage_obj_int_idx = linkage_obj.get_subpart(int_idx = np.arange(len(arr)))
    assert np.all(new_linkage_obj_labels.matrix == linkage_obj.matrix)
    assert np.all(new_linkage_obj_int_idx.matrix == linkage_obj.matrix)

    # test: extract all elements, shuffled
    new_linkage_obj = linkage_obj.get_subpart(
            labels=pd.Index(np.random.permutation(np.arange(len(arr)))).astype(str))
    assert np.all(new_linkage_obj.matrix == linkage_obj.matrix)

    # test: extract random choice of elements
    int_idx = [0, 3, 6, 2, 7, 5]
    new_linkage_obj_labels = linkage_obj.get_subpart(labels=pd.Index(int_idx).astype(str))
    new_linkage_obj_int_idx = linkage_obj.get_subpart(int_idx=int_idx)
    expected_array = np.array([[3., 4., 1., 2.],
                               [5., 6., 2., 3.],
                               [0., 1., 3., 2.],
                               [2., 8., 9., 3.],
                               [7., 9., 20., 6.]])
    assert np.all(new_linkage_obj_labels.matrix == expected_array)
    assert np.all(new_linkage_obj_int_idx.matrix == expected_array)

