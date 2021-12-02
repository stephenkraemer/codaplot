# # Imports

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

# # Known problems (see tests)

# - CL can distort the rectangle patches


# # Tests

# ## Works: anything without CL

# ### Legend within axes, no CL: works

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
    heights = np.linspace(0, 0.05, 5)
    for i, curr_height in enumerate(heights):
        curr_width = curr_height * 2
        patches.append(
            mpatches.Rectangle(
                xy=(0.2, i * 0.2),
                width=curr_width,
                height=curr_height,
                # change long and short side
                # width=curr_height,
                # height=curr_width,
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
        bbox_to_anchor=(0.5, 1),
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

# ### Legend on separate axes, no CL: works

def test_legend_without_cl_on_second_axes():

    # %%
    mpl.rcParams.update(mhstyle.paper_context)

    # create figure
    fig, (ax, legend_ax) = plt.subplots(
        1, 2, dpi=180, figsize=(3, 3), gridspec_kw=dict(width_ratios=(3, 1))
    )
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # add patches with meaningful height and width
    patches = []
    heights = np.linspace(0, 0.05, 5)
    for i, curr_height in enumerate(heights):
        curr_width = curr_height * 2
        patches.append(
            mpatches.Rectangle(
                xy=(0.2, i * 0.2),
                width=curr_width,
                height=curr_height,
                # change long and short side
                # width=curr_height,
                # height=curr_width,
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
        ax=legend_ax,
        handles=patches,
        labels=list("ABCDE"),
        loc="upper left",
        bbox_to_anchor=(0.5, 1),
        labelspacing=1,
        # handletextpad=0,
        # borderaxespad=0,
        # borderpad=0,
        # numpoints=None,
        # frameon=True,
        # mode='expand',
    )

    legend_ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
    # %%

# ## Fails: anything with CL
# ### Legend on axes, but plotted outside axes, with CL: legend becomes distorted

# not super clearly visible in this example, but its true - i checked in inkscape

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
    heights = np.linspace(0, 0.02, 3)
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
        labels=list("ABCDE") * 3,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        labelspacing=1,
        title="a",
        # handletextpad=0,
        # borderaxespad=0,
        # borderpad=0,
        # numpoints=None,
        # frameon=True,
        # mode='expand',
    )
    # leg.get_in_layout()
    # leg.get_clip_on()
    # leg.set_clip_on(False)
    # leg._legend_box.align = 'right'

    # ax.legend_ = leg
    # ax.legend_._remove_method = ax._remove_legend
    ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
    # %%

# ### Legend on second axes, reaching outside of axes, so that CL resizes the axes: distorted

def test_cl_with_legend_on_second_axes():
    # legend is placed using CL, but the boxes are distorted!

    # %%
    mpl.rcParams.update(mhstyle.paper_context)

    # create figure
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        dpi=180,
        figsize=(3, 3),
        constrained_layout=True, gridspec_kw=dict(width_ratios=(2, 1))
    )
    # fig.subplots_adjust(0, 0, 1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # add patches with meaningful height and width
    patches = []
    heights = np.linspace(0, 0.2, 3)
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
        legend_ax,
        handles=patches,
        labels=list("ABCDE") * 3,
        loc="upper left",
        bbox_to_anchor=(0.5, 1),
        labelspacing=1,
        title="W\n1\n2\n3\n4",
        # handletextpad=0,
        # borderaxespad=0,
        # borderpad=0,
        # numpoints=None,
        # frameon=True,
        # mode='expand',
    )
    # leg.get_in_layout()
    # leg.get_clip_on()
    # leg.set_clip_on(False)

    # THIS MESSES THE TITLE ALIGNMENT UP ATM
    # leg._legend_box.align = 'left'

    # ax.legend_ = leg
    # ax.legend_._remove_method = ax._remove_legend
    legend_ax.add_artist(leg)

    fig.savefig(
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/notebook-figures/browser.svg"
    )
    # %%




