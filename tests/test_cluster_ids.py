import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np
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
