n_cores = 12
 
# isort: off
import os
num_threads = str(n_cores)
# these need to be set prior to numpy import
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import numpy as np
# isort: on
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.style as mplstyle
import pandas as pd
import seaborn as sns
import subprocess
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from IPython.display import display
from collections import defaultdict
from joblib import delayed, Parallel

import codaplot as co
import codaplot.utils as coutils

from codaplot.new_api.gap_scale import compute_breakpoints_and_offsets, groups_t, gaps_t
 

def heatmap(
    data: pd.DataFrame | np.ndarray,
    ax: Axes,
    x_groups,
    x_gaps,
    y_groups,
    y_gaps,
    norm,
    vmin,
    vmax,
    cmap='rocket',
    **kwargs,
):

    # %%
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    rgba_array = scalar_mappable.to_rgba(data.values)

    x_bps, _ = compute_breakpoints_and_offsets(x_groups, x_gaps)
    y_bps, _ = compute_breakpoints_and_offsets(y_groups, y_gaps)

    n_rows, n_cols = data.shape
    rgba_array = np.insert(rgba_array, x_bps, np.ones((n_rows, len(x_bps), 4)), axis=1)
    rgba_array = np.insert(rgba_array, y_bps, np.ones((len(y_bps), n_cols + len(x_bps), 4)), axis=0)

    x = np.arange(n_cols + 1, dtype='f4')
    x[x_bps] -= 0.00001
    x = np.insert(x, x_bps + 1, x_bps)

    y = np.arange(n_rows + 1, dtype='f4')
    y[y_bps] -= 0.00001
    y = np.insert(y, y_bps + 1, y_bps)

    ax.pcolormesh(x, y, rgba_array)
    # %%



