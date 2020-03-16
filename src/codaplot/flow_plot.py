import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from typing import Optional


def flow_plot(
    df: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: str = "Set1",
    min_perc_left=0,
    min_perc_right=0,
    min_n=1,
) -> None:
    """Flow plot

    Parameters
    ----------
    df
        two columns, first column is left part of flow plot, second column is right part
    ax
        Axes to be plotted on
    cmap
        colors for flow lines are assigned based on left categories, using this cmap
    min_perc_left
        combis (left category, right category) are only considered if
        n_features in combi / n_features with left_category > min_perc_left
    min_perc_right
        see min_perc_left
    min_n
        combis (left category, right category) are only considered if
        n_features in combi > min_n
    """

    df = df.copy()
    df.columns = ["left_categories", "right_categories"]

    # filter for clusters of interest
    # add spacers
    # add left and right heatmaps
    # consider normalizing left *and* right cats to the same size at the same time
    # backlog
    # - stratify/facet clusters by compartment, or by degree of pan-ness, or by the number of dmrs clusters a cpg cluster is connected to

    filtered_df = _filter_infrequent_category_combis(
        df, min_n=min_n, min_perc_left=min_perc_left, min_perc_right=min_perc_right
    )

    left_cat_curr_pos_ser, right_cat_curr_pos_ser = _get_category_start_positions(filtered_df)

    left_color_d = dict(
        zip(
            left_cat_curr_pos_ser.keys(),
            sns.color_palette(cmap, len(left_cat_curr_pos_ser)),
        )
    )

    # this basic sigmoid shape (logistic function) is scaled and shifted to produce the different flow lines
    # the fraction at the beginning and end of the line which is relatively flat can be
    # adjusted by setting the range of x (-8, 8) -> relatively long flat parts, could also
    # be (-6, 6) for example. See shape of logistic function:
    # https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg
    x = np.linspace(-8, 8, 1000)
    sigmoid_base_line = 1 / (1 + np.exp(-x))

    # loop over all combinations of (left_category, right_category) and get the number
    # of elements belonging to each combination
    for (left_cat, right_cat), combi_size in (
        filtered_df.groupby([filtered_df.iloc[:, 0], filtered_df.iloc[:, 1]])
        .size()
        .iteritems()
    ):
        # note that right start may be > or < left_start
        left_start = left_cat_curr_pos_ser[left_cat]
        right_start = right_cat_curr_pos_ser[right_cat]
        left_end = left_start + combi_size
        right_end = right_start + combi_size
        # plot fill_between
        # lower boundary for fill_between
        # intercept + sigmoid_base_line * scale
        # scale < 0 flips the sigmoid curve
        sigmoid_line_low = left_start + sigmoid_base_line * (right_start - left_start)
        # upper boundary for fill between
        sigmoid_line_upp = sigmoid_line_low + (left_end - left_start)
        ax.fill_between(
            x,
            sigmoid_line_low,
            sigmoid_line_upp,
            color=left_color_d[left_cat],
            alpha=0.2,
        )
        # update current positions for the active categories
        left_cat_curr_pos_ser[left_cat] = left_end + 1
        right_cat_curr_pos_ser[right_cat] = right_end + 1


def _get_category_start_positions(filtered_df):
    # get start positions for left categories, each element is one unit-size high
    # e.g. the first column of filtered_df may be: [1, 1, 1, 2, 2, 2, 3, 3, 3]
    # Then we do
    left_cat_curr_pos_ser = (
        filtered_df.groupby(filtered_df.iloc[:, 0], sort=True).size().cumsum().add(1)
    )
    # Series(1: 4, 2: 7, 3:10)
    left_cat_curr_pos_ser.iloc[1:] = left_cat_curr_pos_ser.iloc[0:-1].to_numpy()
    # Series(1: 4, 2: 4, 3: 7)
    left_cat_curr_pos_ser.iloc[0] = 0
    # Series(1: 0, 2: 4, 3:7)
    # get start positions for right categories, see above for code explanation
    right_cat_curr_pos_ser = (
        filtered_df.groupby(filtered_df.iloc[:, 1], sort=True).size().cumsum().add(1)
    )
    right_cat_curr_pos_ser.iloc[1:] = right_cat_curr_pos_ser.iloc[0:-1].to_numpy()
    right_cat_curr_pos_ser.iloc[0] = 0
    return left_cat_curr_pos_ser, right_cat_curr_pos_ser


def _filter_infrequent_category_combis(df, min_perc_left, min_perc_right, min_n):
    # prior to calculating plot coordinates, filter out combis (left_category, right_category) which
    # are too infrequent to be shown based on min_perc_left, min_perc_right, min_n
    left_cat_sizes = df["left_categories"].value_counts()
    right_cat_sizes = df["right_categories"].value_counts()
    filtered_df = df.groupby([df.iloc[:, 0], df.iloc[:, 1]]).filter(
        lambda df: df.shape[0] > min_n
        and df.shape[0] / left_cat_sizes[df.name[0]] > min_perc_left
        and df.shape[0] / right_cat_sizes[df.name[1]] > min_perc_right
    )
    return filtered_df
