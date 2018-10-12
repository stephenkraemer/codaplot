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
    cluster_ids.merge('clustering1', new_name='split1', groups_to_merge=[['3_3', '3_2'], ['5', '4']])
    cluster_ids.merge('clustering2', new_name='split2', groups_to_merge=[['3', '4'], ['2', '6']])
    cluster_ids.merge('clustering3', new_name='split3', groups_to_merge=[[3, 4], [2, 6]])

    exp_split1 = pd.Series(pd.Categorical(np.tile(['1', '2', '3_1', '3_2', '3_2', '3_3', '4', '4', '5'], 3), ordered=True,
                                                   categories=['1', '2', '3_1', '3_2', '3_3', '4', '5']), name='split1')
    assert_series_equal(cluster_ids.df['split1'], exp_split1)

    exp_split2 = pd.Series(pd.Categorical(np.tile(['1', '2', '3', '3', '4', '2', '5', '6', '7'], 3), ordered=True), name='split2')
    assert_series_equal(cluster_ids.df['split2'], exp_split2)

    exp_split3 = pd.Series(np.tile([1, 2, 3, 3, 4, 2, 5, 6, 7], 3), name='split3')
    assert_series_equal(cluster_ids.df['split3'], exp_split3)
