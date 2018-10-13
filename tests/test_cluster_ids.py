import pandas as pd
import pytest
from pandas.testing import assert_series_equal
import numpy as np
from pandas.util.testing import assert_frame_equal

import codaplot.cluster_ids as ci


def test_merge():

    cluster_ids_df = pd.DataFrame({'clustering1': pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_3', '3_4', '4', '5', '6'], 3), ordered=True),
                                   'clustering2': pd.Categorical(np.tile(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 3), ordered=True),
                                   'clustering3': pd.Series(np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
                                   })
    cluster_ids = ci.ClusterIDs(df = cluster_ids_df)
    cluster_ids.merge('clustering1', new_name='split1', spec=[['3_3', '3_2'], ['5', '4']])
    cluster_ids.merge('clustering2', new_name='split2', spec=[['3', '4'], ['2', '6']])
    cluster_ids.merge('clustering3', new_name='split3', spec=[[3, 4], [2, 6]])

    exp_split1 = pd.Series(pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_2', '3_3', '4', '4', '5'], 3), ordered=True,
                                                   categories=['1', '2', '3_1', '3_2', '3_3', '4', '5']), name='split1')
    assert_series_equal(cluster_ids.df['split1'], exp_split1)

    exp_split2 = pd.Series(pd.Categorical(np.tile(['1', '2', '3', '3', '4', '2', '5', '6', '7'], 3), ordered=True), name='split2')
    assert_series_equal(cluster_ids.df['split2'], exp_split2)

    exp_split3 = pd.Series(np.tile([1, 2, 3, 3, 4, 2, 5, 6, 7], 3), name='split3')
    assert_series_equal(cluster_ids.df['split3'], exp_split3)

def test_split():
    cluster_ids_df = pd.DataFrame({'clustering1': pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_3', '3_4', '4', '5', '6'], 3), ordered=True),
                                   'clustering2': pd.Categorical(np.tile(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 3), ordered=True),
                                   'clustering3': pd.Series(np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
                                   })
    cluster_ids = ci.ClusterIDs(df = cluster_ids_df)


    cluster_ids.split(name='clustering1', new_name='split1',
                      spec={'1': ['1', '2', '2'],
                            '2': ['1', '2', '2']})

    exp_split_1 = cluster_ids_df['clustering1'].copy().astype(str)
    exp_split_1.loc[exp_split_1 == '1'] = ['1_1', '1_2', '1_2']
    exp_split_1.loc[exp_split_1 == '2'] = ['2_1', '2_2', '2_2']
    exp_split_1 = pd.Series(pd.Categorical(exp_split_1, ordered=True),
                            name='split1')
    assert_series_equal(exp_split_1, cluster_ids.df['split1'])

    cluster_ids.split(name='clustering3', new_name='split2',
                      spec={1: [1, 2, 2],
                            6: [2, 2, 1]},
                      create_subclusters=False)
    exp_split_3 = pd.Series([1, 3, 4, 5, 6, 8, 9, 10, 11,
                             2, 3, 4, 5, 6, 8, 9, 10, 11,
                             2, 3, 4, 5, 6, 7, 9, 10, 11], name='split2')
    assert_series_equal(cluster_ids.df['split2'], exp_split_3)

    cluster_ids.split(name='clustering3', new_name='split3',
                      spec={1: [1, 2, 2],
                            6: [2, 2, 1]},
                      create_subclusters=True)
    exp_split_3 = cluster_ids_df['clustering3'].copy().astype(str)
    exp_split_3.loc[exp_split_3 == '1'] = ['1_1', '1_2', '1_2']
    exp_split_3.loc[exp_split_3 == '6'] = ['6_2', '6_2', '6_1']
    exp_split_3 = pd.Series(pd.Categorical(exp_split_3, ordered=True),
                            name='split3')
    assert_series_equal(cluster_ids.df['split3'], exp_split_3)


def test_discard():
    cluster_ids_df = pd.DataFrame({'clustering1': pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_3', '3_4', '4', '5', '6'], 3), ordered=True),
                                   'clustering2': pd.Categorical(np.tile(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 3), ordered=True),
                                   'clustering3': pd.Series(np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
                                   })
    cluster_ids = ci.ClusterIDs(df = cluster_ids_df)

    # note that currently the resulting cluster ids are not subsequent

    cluster_ids.mask(name='clustering3', new_name='discard3', spec=[3, 5])
    expected_discard3 = pd.Series(np.tile([1, 2, -1, 4, -1, 6, 7, 8, 9], 3), name='discard3')
    assert_series_equal(expected_discard3, cluster_ids.df['discard3'])

    cluster_ids.mask(name='clustering1', new_name='discard1', spec=['3_2', '4'])
    expected_discard1 = pd.Series(pd.Categorical(
            np.tile(['1', '2', '3_1', '-1', '3_3', '3_4', '-1', '5', '6'], 3), ordered=True),
            name='discard1')
    assert_series_equal(expected_discard1, cluster_ids.df['discard1'])



def test_sample_proportional():

    cluster_ids_df = pd.DataFrame({'clustering1': pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_3', '3_4', '4', '5', '6'], 3), ordered=True),
                                   'clustering2': pd.Categorical(np.tile(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 3), ordered=True),
                                   'clustering3': pd.Series(np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
                                   })
    cluster_ids = ci.ClusterIDs(df = cluster_ids_df)

    for name in ['clustering1', 'clustering2', 'clustering3']:
        sampled_cluster_ids = cluster_ids.sample_proportional(
                n_total=11, name=name,
                random_state=1, min_cluster_size=1)
        rng = np.random.RandomState(1)
        exp_cluster_id_df = pd.concat([
            (cluster_ids_df
             .groupby(name, group_keys=False)
             .apply(lambda df: df.sample(frac=11/27, random_state=rng))
             ),
            cluster_ids_df.sample(2, random_state=rng)], axis=0).sort_index()
        assert ci.ClusterIDs(exp_cluster_id_df) == sampled_cluster_ids


    for name in ['clustering1', 'clustering2', 'clustering3']:
        sampled_cluster_ids = cluster_ids.sample_proportional(
                frac=0.3, name=name, random_state=1, min_cluster_size=1)
        rng = np.random.RandomState(1)
        exp_cluster_ids = ci.ClusterIDs(cluster_ids_df
                                        .groupby(name, group_keys=False)
                                        .apply(lambda df: df.sample(frac=0.3, random_state=rng))
                                        .sort_index())
        assert exp_cluster_ids == sampled_cluster_ids

    # fail if n_total and frac are given at the same time
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_proportional('clustering1', n_total=4, frac=0.1)

    # fail if n_total or frac are defined such that at least one cluster falls below
    # min_cluster_size
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_proportional(
                'clustering1', n_total=4, random_state=1, min_cluster_size=1)
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_proportional(
                'clustering2', frac=0.1, random_state=1, min_cluster_size=1)
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_proportional(
                'clustering3', frac=0.33, random_state=1, min_cluster_size=2)


def test_sample_equal():
    cluster_ids_df = pd.DataFrame({'clustering1': pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_3', '3_4', '4', '5', '6'], 3), ordered=True),
                                   'clustering2': pd.Categorical(np.tile(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 3), ordered=True),
                                   'clustering3': pd.Series(np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
                                   })
    cluster_ids = ci.ClusterIDs(df = cluster_ids_df)

    def assert_sample_equal(sampled_cluster_ids, cluster_ids,
                            name, n_per_cluster, n_total=None):
        rng = np.random.RandomState(1)
        expected_df = (cluster_ids.df
                       .groupby(name, group_keys=False)
                       .apply(lambda df: df.sample(n=n_per_cluster, random_state=rng))
                       .sort_index()
                       )
        if n_total is not None:
            # if n_total is defined, the result must have exactly n_total rows
            # in the example used here, we need to add additional rows
            # to achieve this. The sample_equal function could also handle the converse case
            expected_df = pd.concat(
                    [expected_df,
                     cluster_ids.df.sample(n=n_total - len(expected_df),
                                           random_state=rng)],
                    axis=0)
            expected_df = expected_df.sort_index()
        assert ci.ClusterIDs(expected_df) == sampled_cluster_ids

    sampled_cluster_ids = cluster_ids.sample_equal(
            n_per_cluster=2, random_state=1,
            name='clustering2', strict=True)
    assert_sample_equal(sampled_cluster_ids, cluster_ids,
                        n_per_cluster=2, name='clustering2')

    # Sample with n_total. This will require adding additional elements from
    # individual clusters, which is hardcoded in the assert_sample_equal function
    sampled_cluster_ids = cluster_ids.sample_equal(
            n_total=20, random_state=1,
            name='clustering3', strict=True)
    assert_sample_equal(sampled_cluster_ids, cluster_ids,
                        name='clustering1', n_per_cluster=2,
                        n_total=20)

    # raise if n_per_cluster and n_total are given at the same time
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_equal('clustering3', n_per_cluster=2, n_total=10)

    # raise if any cluster has size < n_per_cluster
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_equal('clustering3', n_per_cluster=4,
                                     random_state=1, strict=True)

    # raise if n_total > len(data)
    with pytest.raises(ValueError):
        _ = cluster_ids.sample_equal(
                n_total=40, random_state=2,
                name='clustering1', strict=False)
