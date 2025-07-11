import time
from matplotlib import pyplot as plt
import numpy as np
from src.codaplot.new_api.heatmap import heatmap

from src.codaplot.new_api.legend import add_guides

import pandas as pd

print("reloaded test_heatmap")


def test_heatmap():

    # %%
    data = pd.DataFrame(np.random.randn(3_000, 5_000))
    x_groups = np.repeat([1, 2, 3, 4], 1250)
    y_groups = np.repeat(["a", "b", "c"], 1000)
    x_gaps = [100, 100, 100]
    y_gaps = np.array([50, 50]) * 10

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 4))
    ax.set_xscale("gap_scale", groups=x_groups, gap_sizes=x_gaps)
    ax.set_yscale("gap_scale", groups=y_groups, gap_sizes=y_gaps)

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

    plt.close()

    fig.savefig("/home/stephen/projects/codaplot/src/codaplot/new_api/test1.png")
    # %%


def test_heatmap_normal():

    # %%
    data = pd.DataFrame(np.random.randn(3_000, 5_000))

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 4))

    heatmap(
        data,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    plt.close()

    fig.savefig("/home/stephen/projects/codaplot/src/codaplot/new_api/test2.png")
    # %%


def test_heatmap_small():
    # %%
    data = pd.DataFrame(np.random.randn(10, 500))
    x_groups = np.repeat([1, 2, 3, 4], 125)
    y_groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4])
    x_gaps = [40, 40, 40]
    y_gaps = [0.5, 1, 2]

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(2, 2))
    ax.set_xscale("gap_scale", groups=x_groups, gap_sizes=x_gaps)
    ax.set_yscale("gap_scale", groups=y_groups, gap_sizes=y_gaps)

    qm = heatmap(
        data,
        ax=ax,
        x_groups=x_groups,
        x_gaps=x_gaps,
        y_gaps=y_gaps,
        y_groups=y_groups,
        vmin=-1,
        vmax=1,
        palette="RdBu_r",
    )

    # fig.colorbar(qm['mappable'], ax=ax)
    # ax.scatter(2.5, 5.5, c="red")
    # ax.set_xticks(np.arange(0.5, 5.5))
    # ax.set_xticklabels([f"tf{i}" for i in np.arange(5)])
    # %%


def test_heatmap_categorical():

    # %%
    data = pd.DataFrame(
        {"a": ["a", "b", "c"], "b": ["a", "d", "d"], "c": ["a", "d", "e"]}
    )
    x_groups = [1, 1, 2]
    y_groups = [1, 2, 2]
    x_gaps = 1
    y_gaps = 1

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(2, 2))
    ax.set_xscale("gap_scale", groups=x_groups, gap_sizes=x_gaps)
    ax.set_yscale("gap_scale", groups=y_groups, gap_sizes=y_gaps)

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
    # %%


def test_complex_heatmap():

    # %%
    data = pd.DataFrame(
        {"a": ["a", "b", "c"], "b": ["a", "d", "d"], "c": ["a", "d", "e"]}
    )
    x_groups = [1, 1, 2]
    y_groups = [1, 2, 2]
    x_gaps = 1
    y_gaps = 1

    fig, axes = plt.subplots(
        1, 2, layout="none", figsize=(10 / 2.54, 5 / 2.54), width_ratios=(1, 1)
    )
    cat_plot_ax = axes[0]
    cont_plot_ax = axes[1]
    cat_plot_ax.set_xscale("gap_scale", groups=x_groups, gap_sizes=x_gaps)
    cat_plot_ax.set_yscale("gap_scale", groups=y_groups, gap_sizes=y_gaps)

    guide_d_1 = heatmap(
        data,
        ax=cat_plot_ax,
        x_groups=x_groups,
        x_gaps=x_gaps,
        y_gaps=y_gaps,
        y_groups=y_groups,
        vmin=-1,
        vmax=1,
        show_guide=False,
        guide_title="Legend",
    )

    guide_d_2 = heatmap(
        data=np.random.randn(3, 3),
        ax=cont_plot_ax,
        show_guide=False,
        guide_title="Cbar",
    )

    add_guides(guide_spec_l=[guide_d_1, guide_d_2], fig=fig)

    plt.close("all")
    fig

    # %%


if __name__ == "__main__":

    # %%
    t1 = time.time()
    test_heatmap()
    print(time.time() - t1)
    # %%
    t1 = time.time()
    test_heatmap_normal()
    print(time.time() - t1)
    # %%
