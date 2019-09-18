from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from dynamicTreeCut import cutreeHybrid
from more_itertools import ilen, unique_justseen
from scipy.cluster.hierarchy import leaves_list, linkage, cophenet
from scipy.spatial.distance import pdist, squareform

from codaplot.cluster_ids import ClusterIDs


@dataclass
class Linkage:

    matrix: np.ndarray = field(repr=False)
    dist_mat: Optional[np.ndarray] = field(repr=False, default=None)
    index: Optional[pd.MultiIndex] = field(default=None, repr=False)
    cluster_ids: Optional[ClusterIDs] = None
    leaf_orders: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.leaf_order = leaves_list(self.matrix)  # pylint: disable=W0201
        self._df = None

    def dynamic_tree_cut(self, name: Optional[str] = None, **kwargs) -> None:
        if name is None:
            name = self._dict_to_compact_str(kwargs)
        cutree_result = cutreeHybrid(self.matrix, self.dist_mat, **kwargs)

        # TODO: add test for cases with and without missing cluster labels
        # cutree cluster ids are not sequential with respect to the leave order
        # given by Z, i.e. in a leave-ordered heatmap, the cluster id sequence
        # may be 3, 1, 5, ...
        data_order_ids = pd.Series(cutree_result['labels'], index=self.index)
        unique_labels = data_order_ids.unique()
        n_clusters = len(unique_labels)
        if 0 in unique_labels:
            n_clusters -= 1

        has_cluster_label = data_order_ids > 0
        int_idx_no_cluster_label = np.arange(len(data_order_ids))[~has_cluster_label]
        leaf_order_labelled_only = self.leaf_order[
            ~np.isin(self.leaf_order, int_idx_no_cluster_label)]
        labeled_leave_order_cluster_ids = data_order_ids.iloc[leaf_order_labelled_only]

        # assert that elements with the same cluster ids form uniterrupted blocks
        # when in leave order
        assert ilen(unique_justseen(labeled_leave_order_cluster_ids)) == n_clusters

        cluster_ids_random_to_sequential_mapping = dict(
                zip(labeled_leave_order_cluster_ids.unique(),
                    np.arange(1, n_clusters + 1)))
        cluster_ids_random_to_sequential_mapping[0] = 0
        data_ordered_ids_relabeled = data_order_ids.map(
                cluster_ids_random_to_sequential_mapping)

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
        # TODO: add test

        assert self.cluster_ids is not None
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


    def get_subpart(self, labels: Optional[Union[pd.Index, pd.MultiIndex]] = None,
                    int_idx: Optional[np.ndarray] = None):
        """Linkage with original cophenetic distances for a subset of observations

        Must specify either labels or int_idx, but not both.

        Args:
            labels: passed to .loc to retrieve observations based on their index
                labels. Typically Index or MultiIndex. Linkage.index must be defined
                if labels are passed. Does not need to be sorted.
            int_idx: integer index of the observations in the subset. Does not need
                to be sorted.

        Background:
            This works by retrieving the cophenetic distances from the full linkage.
            A new linkage is constructed based on the cophenetic distance matrix
            for only the selected observations.
        """

        assert (labels is not None) + (int_idx is not None) == 1, \
            'Either labels or int_idx must specified (mutually exclusive)'
        if labels is not None:
            assert self.index is not None
            int_idx = np.squeeze(pd.DataFrame(np.arange(len(self.index)), index=self.index)
                                 .loc[labels].sort_index().values.T)
        else:
            int_idx = np.sort(int_idx)

        cophenetic_dist_mat = cophenet(self.matrix)
        cophenetic_dist_mat_square = squareform(cophenetic_dist_mat)
        sampled_dist_mat = cophenetic_dist_mat_square[int_idx, :][:, int_idx]
        # Note: many other linkage methods would also work, because the cophenetic
        # distances are the same between each points in two clusters which are about
        # to be joined. Selection of average linkage is arbitrary.
        sampled_Z = linkage(squareform(sampled_dist_mat), method='average')
        # maybe in the future: could also add distance matrix and index again
        return Linkage(matrix=sampled_Z)

    def get_subpart_leaf_order(self, label_idx=None, int_idx=None):
        if (label_idx is None) + (int_idx is None) != 1:
            print('Either label_idx or int_idx, but not both, must be specified')
        if label_idx is not None:
            running_int_ser = pd.Series(np.arange(0, self.cluster_ids.df.shape[0]),
                                        index=self.cluster_ids.df.index)
            int_idx = running_int_ser.loc[label_idx].values
        return np.array([i for i in self.leaf_order if i in int_idx])

    def get_proportionally_sampled_leaf_order(self, n, random_state=123):
        return self.get_subpart_leaf_order(
                int_idx=np.random.RandomState(random_state)
                    .choice(self.cluster_ids.df.shape[0], n)
        )

    def get_equal_sampled_leaf_order(self, **kwargs):
        sampled_idx = self.cluster_ids.sample_equal(**kwargs).df.index
        return self.get_subpart_leaf_order(label_idx=sampled_idx)



