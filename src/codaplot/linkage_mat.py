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

    @staticmethod
    def _dict_to_compact_str(d: Dict[str, Any]) -> str:
        return ','.join(f'{k}={v}' for k, v in d.items())
