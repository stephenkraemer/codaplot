#-
import pytest
import numpy as np
import pandas as pd

import complex_heatmap as ch
#-

rng = np.random.RandomState(1)
df1 = pd.DataFrame(rng.randn(10, 5) + np.tile([5, 10], 5)[:, np.newaxis])
df2 = pd.DataFrame(rng.randn(10, 5)+ np.tile([2, 4], 5)[:, np.newaxis])

complex_heatmap_list = ch.ComplexHeatmapList(
        [
            ch.ComplexHeatmap(df1, is_main=True),
            ch.ComplexHeatmap(df2)
        ]
)
fig = complex_heatmap_list.plot()
#-
