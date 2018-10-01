#-
import pytest
import numpy as np
import pandas as pd

import complex_heatmap as ch
#-

rng = np.random.RandomState(1)
data = np.tile([5, 10], (5, 5)).T + np.tile(np.arange(0, 20, 4), (10, 1))
names = 's' + pd.Series(np.arange(5)).astype(str)
df1 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df2 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df3 = pd.DataFrame(rng.randn(10, 5) * 3 + 1, columns=names)

complex_heatmap_list = ch.ComplexHeatmapList(
        [
            ch.ComplexHeatmap(df1,
                              is_main=True,
                              cluster_use_cols = ['s1', 's2'],
                              cmap_sequential='YlOrBr',
                              cluster_cols = False,
                              col_show_list = ['s1', 's2', 's3'],
                              col_dendrogram_height=1/2.54,
                              row_dendrogram_width=1/2.54,
                              col_dendrogram_show=False,
                              title='dtype1'
                              ),
            ch.ComplexHeatmap(df2, cmap_sequential='Blues',
                              title='dtype2'
                              ),
            ch.ComplexHeatmap(df3, cmap_norm='midpoint',
                              title='dtype3'
                              ),
        ],
        figsize=(12/2.54, 6/2.54),
        dpi=360
)
fig = complex_heatmap_list.plot()
#-
