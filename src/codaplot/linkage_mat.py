from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dataclasses import dataclass
from dynamicTreeCut import cutreeHybrid
from more_itertools import ilen, unique_justseen
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from codaplot.cluster_ids import ClusterIDs


@dataclass
class Linkage:

    matrix: np.ndarray
    dist_mat: Optional[np.ndarray] = None
    index: Optional[pd.MultiIndex] = None
    cluster_ids: Optional[ClusterIDs] = None
    leaf_orders: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.leaf_order = leaves_list(self.matrix)  # pylint: disable=W0201
        self._df = None

    def dynamic_tree_cut(self, name: Optional[str] = None, **kwargs) -> None:
        if name is None:
            name = self._dict_to_compact_str(kwargs)
        cutree_result = cutreeHybrid(self.matrix, self.dist_mat, **kwargs)
        # cutree cluster ids are not sequential with respect to the leave order
        # given by Z, i.e. in a leave-ordered heatmap, the cluster id sequence
        # may be 3, 1, 5, ...
        data_order_ids = pd.Series(cutree_result['labels'], index=self.index)
        unique_labels = data_order_ids.unique()
        n_clusters = len(unique_labels)

        leave_order_ids = data_order_ids.iloc[self.leaf_order]

        # assert that elements with the same cluster ids form uniterrupted blocks
        # when in leave order
        assert ilen(unique_justseen(leave_order_ids)) == n_clusters

        cluster_ids_random_to_sequential_mapping = dict(
                zip(leave_order_ids.unique(), np.arange(unique_labels.min(), n_clusters + unique_labels.min())))
        data_ordered_ids_relabeled = data_order_ids.map(cluster_ids_random_to_sequential_mapping)

        if self.cluster_ids is None:
            self.cluster_ids = ClusterIDs(
                    df=pd.DataFrame(
                            {name: data_ordered_ids_relabeled}, index=self.index)
            )
            assert self.leaf_orders is None
            self.leaf_orders = pd.DataFrame({name: self.leaf_order})
        else:
            assert name not in self.cluster_ids.df.columns
            self.cluster_ids.df[name] = data_ordered_ids_relabeled
            if self.leaf_orders is not None:
                assert name not in self.leaf_orders.columns
                self.leaf_orders[name] = self.leaf_order
            else:
                self.leaf_orders = pd.DataFrame({name: self.leaf_order})


    @staticmethod
    def _dict_to_compact_str(d: Dict[str, Any]) -> str:
        return ','.join(f'{k}={v}' for k, v in d.items())

    @property
    def df(self):
        if self._df is None:
            # noinspection PyAttributeOutsideInit
            self._df = pd.DataFrame(
                    self.matrix,
                    columns=['left_child', 'right_child', 'height', 'n_elements']).astype(
                    dtype={'left_child': int, 'right_child': int,
                           'height': float, 'n_elements': int})
        return self._df

    def get_link_cluster_ids(self, clustering_name: str) -> np.ndarray:
        """ Assign cluster ids to links defined in a linkage matrix

        - cluster ids are assigned to links, not to branches of links
        - links with children in more than one cluster are assigned id -1
        """

        assert self.cluster_ids is not None
        # TODO: add test
        n_leaves = self.matrix.shape[0] + 1

        link_cluster_ids = -2 * np.ones(self.matrix.shape[0])

        # noinspection PyShadowingNames
        def get_cluster_ids(obs_idx, Z, n, link_cluster_ids, leave_cluster_ids):
            if obs_idx < n:
                return leave_cluster_ids[obs_idx]
            link_idx = obs_idx - n
            if link_cluster_ids[link_idx] != -2:
                return link_cluster_ids[link_idx]
            left_cluster_id = get_cluster_ids(int(Z[link_idx, 0]), Z, n,
                                              link_cluster_ids, leave_cluster_ids)
            right_cluster_id = get_cluster_ids(int(Z[link_idx, 1]), Z, n,
                                               link_cluster_ids, leave_cluster_ids)
            if left_cluster_id == right_cluster_id:
                link_cluster_ids[link_idx] = left_cluster_id
                return left_cluster_id
            link_cluster_ids[link_idx] = -1
            return -1
        get_cluster_ids(self.matrix.shape[0] * 2, self.matrix, n_leaves, link_cluster_ids,
                        self.cluster_ids.df[clustering_name])

        return link_cluster_ids


    def iterative_dynamic_tree_cut(self, clustering_name, cluster_id,
                                   data: pd.DataFrame, name: Optional[str] = None,
                                   treecut_args: Optional[Dict] = None,
                                   metric='euclidean', method='average',
                                   usecols = None,
                                   create_subclusters: bool = True,
                                   ) -> None:
        """Iteratively cut a cluster from an existing partitioning

        Currently, the only option is to compute a new linkage for the data
        in the cluster and use this to compute the new partitioning within the cluster.
        For this purpose, the columns use for the pdist calculation can be specified.

        In the future, we may add the possibility to reuse a part of the original linkage
        and cut this part again with finer parameters than in the original clustering.

        Args:
            clustering_name: base clustering, where one cluster is selected for
                iterative refinement
            cluster_id: the cluster selected for refinement
            data: the original dataframe underlying the clustering
            name: name of the new resulting clustering
            treecut_args: passed to cutreeHybrid
            metric: passed to scipy.pdist
            method: passed to scipy.linkage
            usecols: features used for clustering. List of column labels
                for label-based indexing.
            create_subclusters: passed to ClusterIDs.split
        """

        assert isinstance(data, pd.DataFrame)
        assert isinstance(self.cluster_ids, ClusterIDs)

        # TODO: add test
        if treecut_args is None:
            treecut_args = {}
        if usecols is None:
            usecols = slice(None)
        else:
            assert isinstance(usecols, list)
        if name is None:
            name = f'iterative_{cluster_id}_{self._dict_to_compact_str(treecut_args)}'

        is_in_split_cluster = self.cluster_ids.df[clustering_name] == cluster_id

        sub_data = data.loc[is_in_split_cluster, usecols]
        dist_mat = pdist(sub_data, metric=metric)
        Z = linkage(dist_mat, method=method)
        inner_linkage_mat = Linkage(Z, dist_mat, index=sub_data.index)
        inner_linkage_mat.dynamic_tree_cut(name=name, **treecut_args)
        # mainly to make mypy happy
        assert isinstance(inner_linkage_mat.cluster_ids, ClusterIDs)
        self.cluster_ids.split(name=clustering_name, new_name=name,
                               spec={cluster_id: inner_linkage_mat.cluster_ids.df[name]},
                               create_subclusters=create_subclusters)
        new_leaf_order = self.leaf_order.copy()

        # TODO: add assert
        cluster_data_int_indices = np.arange(len(data))[is_in_split_cluster]
        clustered_data_ii_ordered = cluster_data_int_indices[inner_linkage_mat.leaf_order]
        new_leaf_order[np.isin(new_leaf_order, cluster_data_int_indices)] = clustered_data_ii_ordered

        if self.leaf_orders is None:
            self.leaf_orders = pd.DataFrame({name: new_leaf_order})
        else:
            assert name not in self.leaf_orders.columns
            self.leaf_orders[name] = new_leaf_order
