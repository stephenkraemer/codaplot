from matplotlib import pyplot as plt
import numpy as np
from src.codaplot.new_api.heatmap import heatmap
from codaplot.new_api.gap_scale import compute_breakpoints_and_offsets

import pandas as pd


def test_heatmap():

    # %%
    data = pd.DataFrame(np.random.randn(10, 5))
    x_groups = [1, 1, 2, 2, 3]
    y_groups = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    x_gaps = [0.5, 3]
    y_gaps = [0.5, 1, 2]

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(2, 2))
    fig.get_layout_engine().set(hspace=0, wspace=0, h_pad=0, w_pad=0)
    ax.set_xscale("gap_scale_x", groups=x_groups, gap_sizes=x_gaps)
    ax.set_yscale("gap_scale_y", groups=y_groups, gap_sizes=y_gaps)

    heatmap(
        data,
        ax=ax,
        x_groups=x_groups,
        x_gaps=x_gaps,
        y_gaps=y_gaps,
        y_groups=y_groups,
        vmin=-1,
        vmax=1,
    )

    ax.scatter(2.5, 5.5, c="red")
    ax.set_xticks(np.arange(0.5, 5.5))
    ax.set_xticklabels([f"tf{i}" for i in np.arange(5)])
    fig
    # %%
