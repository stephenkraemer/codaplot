from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from dataclasses import dataclass
from dpcontracts import invariant
from dynamicTreeCut import cutreeHybrid
from more_itertools import ilen, unique_justseen
from scipy.cluster.hierarchy import leaves_list



def data_ordered_cluster_ids_contract(df):
    assert not isinstance(df.columns, pd.MultiIndex)
    assert not df.columns.duplicated().any()
    return True
@invariant('cluster_ids format', lambda inst: (
        data_ordered_cluster_ids_contract(inst.data_order_cluster_ids)
        if inst.data_order_cluster_ids is not None else True))
@dataclass
class LinkageMatrix:

    Z: np.ndarray
    dist_mat: np.ndarray
    index: pd.MultiIndex
    data_order_cluster_ids: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.leaves_list = leaves_list(self.Z)

    def add_dynamic_cutree_division(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = self._dict_to_compact_str(kwargs)
        cutree_result = cutreeHybrid(self.Z, self.dist_mat, **kwargs)
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

        if self.data_order_cluster_ids is None:
            self.data_order_cluster_ids = pd.DataFrame(
                    {name: data_ordered_ids_relabeled}, index=self.index)
        else:
            assert name not in self.data_order_cluster_ids.columns
            self.data_order_cluster_ids[name] = data_ordered_ids_relabeled

        return name

    def merge_clusters(self, old_name, groups: List[List[int]],
                       new_name: Optional[str] = None) -> pd.Series:
        """

        Args:
            groups: cluster groups must be consecutive. For example:
                [[3, 4], [8, 9 10]], but not [[3, 4], [8, 10]]

        Returns:
            Series with merged cluster ids. The cluster ids are adapted
            to be consecutive again. If this is applied to cluster ids which
            are chosen to be consecutive when leave-ordered, the new cluster ids
            will still be consecutive when leave-ordered. This also holds if the
            input cluster ids are data-ordered.
        """

        # groups must be consecutive
        groups_sorted = [sorted(x) for x in groups]
        assert all([all(np.array(x) == np.arange(len(x)) + x[0]) for x in groups_sorted])

        if self.data_order_cluster_ids is None:
            raise ValueError('need to calculate cluster ids first')
        assert old_name in self.data_order_cluster_ids.columns

        if new_name is None:
            new_name = old_name + '_merged_' + '_'.join(['-'.join(str(i) for i in x) for x in groups])

        d = {}
        for x in groups_sorted:
            new_id = x[0]
            for elem in x:
                d[elem] = new_id

        merged_ids = self.data_order_cluster_ids[old_name].replace(d)
        n_clusters = merged_ids.nunique()
        merged_ids = merged_ids.replace(dict(zip(sorted(merged_ids.unique()), range(1, n_clusters + 1))))

        # assert that cluster ids are consecutive
        assert np.all(np.sort(merged_ids.unique()) == np.arange(1, n_clusters + 1))
        # assert that cluster ids are still consecutive in leave order
        assert ilen(unique_justseen(merged_ids.iloc[self.leaves_list])) == n_clusters

        self.data_order_cluster_ids[new_name] = merged_ids

        return new_name


    def split(self, cluster_id):
        pass

    @staticmethod
    def _dict_to_compact_str(d: Dict[str, Any]) -> str:
        return ','.join(f'{k}={v}' for k, v in d.items())
