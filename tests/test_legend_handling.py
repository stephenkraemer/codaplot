import codaplot.utils as coutils
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
import codaplot as co
import codaplot.utils as coutils
import seaborn as sns
import mouse_hema_meth.styling as mhstyle


def test_legend():

# %%
    mpl.rcParams.update(mhstyle.paper_context)

    # create figure
    fig, ax = plt.subplots(1, 1, dpi=180, figsize=(3, 3))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # add patches with meaningful height and width
    patches = []
    heights = np.linspace(0, 0.03, 5)
    for i, curr_height in enumerate(heights):
        curr_width = curr_height * 2
        patches.append(
            mpatches.Rectangle(
                xy=(0.2, i * 0.1),
                width=curr_width,
                height=curr_height,
                facecolor="black",
                edgecolor=None,
                linewidth=0,
                zorder=1,
                clip_on=False,
            )
        )
        ax.add_patch(patches[-1])

    # create list of pseudopatches at meaningful sizes, here this is not necessary,
    # we just use the patches created above; in real life, new patches at characteristic values
    # have to be created

    leg = co.create_legend_for_norm_size_patches(
        ax=ax,
        handles=patches,
        labels=list("ABCDE"),
        legend_kwargs=dict(
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            labelspacing=1,
            # handletextpad=0,
            # borderaxespad=0,
            # borderpad=0,
            # numpoints=None,
            # frameon=True,
            # mode='expand',
        ),
    )

    ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
# %%
