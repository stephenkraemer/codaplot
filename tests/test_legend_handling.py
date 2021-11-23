import os
import matplotlib.legend as mlegend
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
import mouse_hema_meth.utils as ut


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
    heights = np.linspace(0, 0.1, 5)
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
        loc="upper left",
        bbox_to_anchor=(0.75, 0.75),
        labelspacing=1,
        # handletextpad=0,
        # borderaxespad=0,
        # borderpad=0,
        # numpoints=None,
        # frameon=True,
        # mode='expand',
    )

    ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
# %%

def test_cl_with_legend():

    # %%
    mpl.rcParams.update(mhstyle.paper_context)

    # create figure
    fig, ax = plt.subplots(1, 1, dpi=180, figsize=(3, 3), constrained_layout=True)
    # fig.subplots_adjust(0, 0, 1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # add patches with meaningful height and width
    patches = []
    heights = np.linspace(0, 0.1, 5)
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
    # leg = mlegend.Legend(
    # leg = ax.legend(
        ax,
        handles=patches,
        labels=list("ABCDE"),
        loc="upper left",
        bbox_to_anchor=(1, 1),
        labelspacing=1,
        title=None,
        # handletextpad=0,
        # borderaxespad=0,
        # borderpad=0,
        # numpoints=None,
        # frameon=True,
        # mode='expand',
    )
    # leg.get_in_layout()
    # leg.get_clip_on()
    leg.set_clip_on(False)

    # ax.legend_ = leg
    # ax.legend_._remove_method = ax._remove_legend
    ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
    # %%

