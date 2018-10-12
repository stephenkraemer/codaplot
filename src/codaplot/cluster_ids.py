from itertools import chain

import pandas as pd
from pandas.api.types import is_integer_dtype, is_categorical_dtype
import numpy as np
from typing import List, Optional, Union, Dict, Iterable
from dataclasses import dataclass
from toolz import merge


@dataclass
class ClusterIDs:
    df: pd.DataFrame
    order: Optional[np.ndarray] = None

    def merge(self, name: str, new_name: str,
              spec: List[List[Union[int, str]]]):
        """Merge cluster IDs

        Args:
            name: name of the input cluster IDs (must be column in
                df
            new_name: name for the new set of Cluster IDs
            spec: Which clusters to merge, e.g.
                [[3, 4], [4, 9]] or [['1', '2'], ['4_1', '4_2']]

        Note that for hierarchical clustering, it is important to respect
        the linkage matrix / dendrogram structure. Merging clusters across
        dendrogram boundaries does not make sense and will interfere with some
        downstreama analyses, e.g. with plotting aligned and cut dendrograms.
        """

        if is_integer_dtype(self.df[name]):
            has_int_clust_ids = True
        elif is_categorical_dtype(self.df[name]):
            has_int_clust_ids = False
        else:
            raise TypeError()

        if has_int_clust_ids:
            groups_sorted = [sorted(x) for x in spec]
        else:
            groups_sorted = [sorted(list_of_ids, key=lambda x: [int(i) for i in x.split('_')])  # type: ignore
                             for list_of_ids in spec]

        identity_mapping = {cluster_id: cluster_id for cluster_id in self.df[name].unique()}

        # Every cluster within a merge group gets assigned the lowest cluster ID
        # in the merge group
        merge_group_to_lowest_group_id_mapping = {}
        for x in groups_sorted:
            new_id = x[0]
            for elem in x:
                merge_group_to_lowest_group_id_mapping[elem] = new_id

        # construct full mapping which combines identiy mappings and mappings
        # which replace a cluster ID with the lowest ID from the merge group
        full_lowest_group_id_mapping = merge(identity_mapping, merge_group_to_lowest_group_id_mapping)

        unique_merged_ids = pd.Series(np.unique(list(full_lowest_group_id_mapping.values())))
        if has_int_clust_ids:
            merged_ids_numeric_df = pd.DataFrame({0: unique_merged_ids})
        else:
            merged_ids_numeric_df = unique_merged_ids.str.split('_', expand=True).replace({None: '1'}).astype('i8')

        if merged_ids_numeric_df.shape[1] == 2:
            merged_ids_numeric_df[0] = self._make_subsequent(merged_ids_numeric_df[0])
            merged_ids_numeric_df[1] = pd.Series(list(chain(
                    *merged_ids_numeric_df[1]
                        .groupby(merged_ids_numeric_df[0])
                        .apply(self._make_subsequent))), dtype='object')
            subsequent_ids = merged_ids_numeric_df.apply(
                    lambda ser: ser[~ser.isnull()].astype(str).str.cat(sep='_'), axis=1)
        else:
            subsequent_ids = self._make_subsequent(merged_ids_numeric_df[0])
            if not has_int_clust_ids:
                subsequent_ids = subsequent_ids.astype(str)
        remaining_cluster_ids_to_subsequent_id_mapping = dict(
                zip(unique_merged_ids, subsequent_ids))
        full_mapping = {cluster_id: remaining_cluster_ids_to_subsequent_id_mapping[
            full_lowest_group_id_mapping[cluster_id]]
                        for cluster_id in full_lowest_group_id_mapping.keys()}

        if has_int_clust_ids:
            self.df[new_name] = self.df[name].replace(full_mapping)
        else:
            ordered_categories = sorted(list(subsequent_ids),
                                        key=lambda x: [int(i) for i in x.split('_')])
            self.df[new_name] = (pd.Categorical(
                    self.df[name]
                        .replace(full_mapping), ordered=True, categories=ordered_categories))

        return self.df[new_name]

    def split(self, name: str, new_name: str,
              spec: Dict[Union[int, str], Iterable], create_subclusters: bool = True) -> None:
        """Split cluster into new clusters or into subclusters

        Args:
            name: Name of the input clustering
            new_name: Name of the resulting split clustering
            spec: maps IDs of clusters to be split to the new subcluster IDs for the
                elements in the cluster (must be given in the data order, i.e. they
                will not be aligned)
            create_subclusters: if True, splitting is carried out by appending
                the subcluster ids to the original cluster id, joined with an
                underscore. If False, the original cluster id is discarded and
                replaced with separate IDs for each subcluster. All following cluster
                IDs are incremented so that the final clustering has subsequent
                cluster IDs
        """

        if create_subclusters:
            split_ids = self.df[name].astype(str)

            for cluster_id, new_clusters in spec.items():
                new_clusters = np.array(new_clusters).astype(str)
                cluster_id = str(cluster_id)
                split_ids[split_ids == cluster_id] = (
                     split_ids[split_ids == cluster_id] + '_' + new_clusters)
            categories = list(sorted(split_ids.unique(), key=lambda x: x.split('_')))
            split_ids = pd.Series(pd.Categorical(split_ids, ordered=True,
                                                     categories=categories))
            self.df[new_name] = split_ids
        else:
            if not is_integer_dtype(self.df[name]):
                raise NotImplementedError
            sorted_cluster_ids = np.unique(self.df[name])
            sorted_spec_keys = list(sorted(spec.keys()))
            orig_id_new_id_ser = pd.Series(np.arange(1, len(sorted_cluster_ids) + 1),
                                   index=np.arange(1, len(sorted_cluster_ids) + 1))
            self.df[new_name] = self.df[name].copy()
            for cluster_id in sorted_spec_keys:
                n_ids = len(np.unique(spec[cluster_id]))
                self.df[new_name].loc[self.df[new_name] > orig_id_new_id_ser[cluster_id]] += n_ids - 1
                orig_id_new_id_ser[(orig_id_new_id_ser[cluster_id]+1):] += n_ids - 1
                self.df[new_name].loc[self.df[new_name] == orig_id_new_id_ser[cluster_id]] = (
                        np.array(spec[cluster_id]) + orig_id_new_id_ser[cluster_id] - 1)

    def mask(self, name: str, new_name: str, spec: List[Union[int, str]]):
        """Mask cluster IDs to mark regions without cluster assignment

        Masked values are replaced by -1, either as integer or as category
        depending on the dtype of the input cluster IDs.

        Currently, the remaining cluster IDs are not modified to be
        again subsequent.
        """
        new_ids = self.df[name].copy()
        if is_categorical_dtype(new_ids):
            new_ids = new_ids.cat.set_categories(['-1'] + list(new_ids.cat.categories))
            new_ids.loc[new_ids.isin(spec)] = '-1'
            new_ids = new_ids.cat.remove_unused_categories()
        else:
            new_ids.loc[new_ids.isin(spec)] = -1
        self.df[new_name] = new_ids


    @staticmethod
    def _make_subsequent(ser: pd.Series):
        if ser.shape[0] == 1:
            return [None]
        diffs = np.diff(np.insert(ser.values, 0, 0))
        diffs[diffs > 1] = 1
        cumsums = np.cumsum(diffs)
        return cumsums.astype(int)


    # def merge_clusters(self, name, groups_to_merge: List[List[int]],
    #                    new_name: Optional[str] = None):
    #     """ Merge groups of clusters
    #
    #     Args:
    #         groups_to_merge: cluster groups must be consecutive. For example:
    #             [[3, 4], [8, 9 10]], but not [[3, 4], [8, 10]]
    #
    #      The cluster ids are adapted to be consecutive again.
    #      If this is applied to cluster ids which
    #         are chosen to be consecutive when leave-ordered, the new cluster ids
    #         will still be consecutive when leave-ordered. This also holds if the
    #         input cluster ids are data-ordered.
    #     """
    #
    #     # groups must be consecutive
    #     groups_sorted = [sorted(x) for x in groups_to_merge]
    #     assert all([all(np.array(x) == np.arange(len(x)) + x[0]) for x in groups_sorted])
    #
    #     assert name in self.df.columns
    #
    #     if new_name is None:
    #         new_name = (name + '_merged_' + '_'.join(
    #                 ['-'.join(str(i) for i in x) for x in groups_to_merge]))
    #
    #
    #
    #     # assert that cluster ids are consecutive
    #     assert np.all(np.sort(merged_ids.unique()) == np.arange(1, n_clusters + 1))
    #     if self.order is not None:
    #         # assert that cluster ids are still consecutive in leave order
    #         assert ilen(unique_justseen(merged_ids.iloc[self.leaves_list])) == n_clusters
    #
    #     self.df[new_name] = merged_ids
    #
    #     return new_name




# def df_contract(inst: ClusterIDs):
    # categoricals are strings of the form \d(_\d+)?, e.g. 1_1, 1_2, 2
    # cluster ids are always subsequent, ie no gaps
    # state = (all(inst.df.apply(is_categorical) | inst.df.apply(is_integer))
             # )
