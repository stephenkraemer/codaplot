from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from dataclasses import dataclass
from dynamicTreeCut import cutreeHybrid
from more_itertools import ilen, unique_justseen
from scipy.cluster.hierarchy import leaves_list

from codaplot.cluster_ids import ClusterIDs


@dataclass
class LinkageMatrix:

    matrix: np.ndarray
    dist_mat: Optional[np.ndarray] = None
    index: Optional[pd.MultiIndex] = None
    cluster_ids: Optional[ClusterIDs] = None

    def __post_init__(self):
        self.leaves_list = leaves_list(self.matrix)  # pylint: disable=W0201
        self._df = None

    def add_dynamic_cutree_division(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = self._dict_to_compact_str(kwargs)
        cutree_result = cutreeHybrid(self.matrix, self.dist_mat, **kwargs)
        # cutree cluster ids are not sequential with respect to the leave order
        # given by Z, i.e. in a leave-ordered heatmap, the cluster id sequence
        # may be 3, 1, 5, ...
        data_order_ids = pd.Series(cutree_result['labels'], index=self.index)
        unique_labels = data_order_ids.unique()
        n_clusters = len(unique_labels)

        leave_order_ids = data_order_ids.iloc[self.leaves_list]

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
        else:
            assert name not in self.cluster_ids.df.columns
            self.cluster_ids.df[name] = data_ordered_ids_relabeled

        return name

    @staticmethod
    def _dict_to_compact_str(d: Dict[str, Any]) -> str:
        return ','.join(f'{k}={v}' for k, v in d.items())

    @property
    def df(self):
        # TODO: add test
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

