import pandas as pd
import numpy as np
import seaborn as sns
from IPython.core.display import display
from matplotlib.axes import Axes
from typing import Optional, List, Tuple
import codaplot.plotting


def flow_plot(
    df: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: str = "Set1",
    min_perc_left=0,
    min_perc_right=0,
    min_n=1,
    return_combi_stats=False,
    allow_losing_categories=False,
    add_spacers=True,
    spacer_size: float = 0.05,
    equal_sizes=False,
) -> Tuple:
    """Flow plot

    Notes
    -----

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
    return_combi_stats
        if True, return n, perc_left, perc_right for all combis
        This is meant to facilitate choosing appropriate values for the filtering criteria
    allow_losing_categories
        raise ValueError if any category is lost due to cardinality filtering, unless this
        is set to True
    add_spacers
        add spacers between clusters of left and right categories
    spacer_size
        fraction of Axes used for a single spacer.
        Note that currently only a single float is allowed.
    equal_sizes
        if False (default), the left and right categories are displayed proportional to their
        cardinality, otherwise, all categories will get the same fraction of the axis
    """

    df = df.copy()
    df.columns = ["left_category", "right_category"]

    # filter for clusters of interest
    # add spacers
    # add left and right heatmaps
    # consider normalizing left *and* right cats to the same size at the same time
    # backlog
    # - stratify/facet clusters by compartment, or by degree of pan-ness, or by the number of dmrs clusters a cpg cluster is connected to

    res = filter_infrequent_category_combis(
        df,
        min_n=min_n,
        min_perc_left=min_perc_left,
        min_perc_right=min_perc_right,
        allow_losing_categories=allow_losing_categories,
        return_combi_stats=return_combi_stats
    )
    if return_combi_stats:
        filtered_df, combi_stats = res
    else:
        filtered_df = res[0]
        combi_stats = None

    left_cat_curr_pos_ser, right_cat_curr_pos_ser = _get_category_start_positions(
        filtered_df, equal_sizes
    )

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

    fill_between_vars_l: List[pd.Series] = []
    left_cardinalities = filtered_df["left_category"].value_counts()
    right_cardinalities = filtered_df["right_category"].value_counts()
    left_nunique, right_nunique = filtered_df.nunique()
    left_cat_size = 1 / left_nunique
    right_cat_size = 1 / right_nunique

    # loop over all combinations of (left_category, right_category) and get the number
    # of elements belonging to each combination
    for (left_cat, right_cat), combi_size in (
        filtered_df.groupby(["left_category", "right_category"]).size().iteritems()
    ):
        # note that right start may be > or < left_start
        # note that for normalized cluster sizes (not yet implemented),
        # left_upper - left_lower ! = right_upper - right_lower
        # so we need to record both
        left_lower = left_cat_curr_pos_ser[left_cat]
        right_lower = right_cat_curr_pos_ser[right_cat]
        if equal_sizes:
            left_upper = (
                left_lower + combi_size / left_cardinalities[left_cat] * left_cat_size
            )
            right_upper = (
                right_lower
                + combi_size / right_cardinalities[right_cat] * right_cat_size
            )
        else:
            left_upper = left_lower + combi_size
            right_upper = right_lower + combi_size
        # update current positions for the active categories
        left_cat_curr_pos_ser[left_cat] = left_upper
        right_cat_curr_pos_ser[right_cat] = right_upper
        fill_between_vars_l.append(
            pd.Series(
                dict(
                    left_category=left_cat,
                    right_category=right_cat,
                    left_lower=left_lower,
                    left_upper=left_upper,
                    right_lower=right_lower,
                    right_upper=right_upper,
                )
            )
        )
    fill_between_vars_df = pd.DataFrame(fill_between_vars_l)

    # display(fill_between_vars_df)

    if add_spacers:
        for curr_category, coord_var in zip(
            ["left_category", "left_category", "right_category", "right_category"],
            ["left_lower", "left_upper", "right_lower", "right_upper"],
        ):
            if equal_sizes:
                curr_nunique = (
                    left_nunique if coord_var.startswith("left") else right_nunique
                )
                spacer_coords = 1 / curr_nunique * np.arange(1, curr_nunique)
                fill_between_vars_df[coord_var] = _fn(
                    x=fill_between_vars_df[coord_var],
                    right_open=coord_var.endswith("lower"),
                    spacer_coords=spacer_coords,
                    spacer_size=spacer_size,
                )
            else:
                # the coordinates are for unit-sized elements
                fill_between_vars_df[coord_var] = codaplot.plotting.adjust_coords(
                    coords=fill_between_vars_df[coord_var],
                    spacing_group_ids=filtered_df[curr_category]
                    .sort_values()
                    .to_numpy(),
                    spacer_sizes=spacer_size,
                    right_open=coord_var.endswith("lower"),
                )

    # display(fill_between_vars_df)
    for _unused, var_ser in fill_between_vars_df.iterrows():
        # plot fill_between
        # lower boundary for fill_between
        # intercept + sigmoid_base_line * scale
        # scale < 0 flips the sigmoid curve
        sigmoid_line_low = var_ser.left_lower + sigmoid_base_line * (
            var_ser.right_lower - var_ser.left_lower
        )
        # upper boundary for fill between
        height_left = var_ser.left_upper - var_ser.left_lower
        height_right = var_ser.right_upper - var_ser.right_lower
        # a linear slope will lead to overlapping flow lines at the left side of the plot,
        # which is confusing
        # if you still want to try it:
        # slope = np.arange(x.shape[0]) * (height_right - height_left) / x.shape[0]
        slope = sigmoid_base_line * (height_right - height_left)
        sigmoid_line_upp = sigmoid_line_low + height_left + slope
        ax.fill_between(
            x,
            sigmoid_line_low,
            sigmoid_line_upp,
            color=left_color_d[var_ser.left_category],
            alpha=0.2,
        )

    if equal_sizes or add_spacers:
        ax.set_ylim(0, 1)

    ax.tick_params(
        axis="both",  # x, y
        which="both",  # both minor
        # length=1,
        # width=1,
        # labelsize=1,
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        # rotation=90,
        # colors : color
        #     Changes the tick color and the label color to the same value:
        #     mpl color spec.
        # bottom, top, left, right : bool
        #     Whether to draw the respective ticks.
        # labelbottom, labeltop, labelleft, labelright : bool
        #     Whether to draw the respective tick labels.
        # labelrotation : float
        #     Tick label rotation
    )

    sns.despine(ax=ax, bottom=True, left=True)

    if combi_stats is not None:
        return ax, combi_stats
    else:
        return (ax, )


def _get_category_start_positions(
    df: pd.DataFrame, equal_sizes: bool
) -> Tuple[pd.Series, pd.Series]:
    """Get start positions for left and right categories

    If the size of the categories in the plot is proportional to their cardinality
    (equal_sizes == False), return coordinates for unit-sized elements.
    Otherwise, (equal_sizes == True) the start coordinates are trivially given by
    the number of categories

    Parameters
    ----------
    df: columns left_category right_category
    equal_sizes: see flow_plot doc

    Returns
    -------

    """
    # get start positions for left categories, each element is one unit-size high
    # e.g. the first column of filtered_df may be: [1, 1, 1, 2, 2, 2, 3, 3, 3]
    # Then we do
    if not equal_sizes:
        left_cat_curr_pos_ser = df.groupby(df.iloc[:, 0], sort=True).size().cumsum()
        # Series(1: 3, 2: 6, 3:9)
        left_cat_curr_pos_ser.iloc[1:] = left_cat_curr_pos_ser.iloc[0:-1].to_numpy()
        # Series(1: 3, 2: 3, 3: 6)
        left_cat_curr_pos_ser.iloc[0] = 0
        # Series(1: 0, 2: 3, 3:6)
        # get start positions for right categories, see above for code explanation
        right_cat_curr_pos_ser = df.groupby(df.iloc[:, 1], sort=True).size().cumsum()
        right_cat_curr_pos_ser.iloc[1:] = right_cat_curr_pos_ser.iloc[0:-1].to_numpy()
        right_cat_curr_pos_ser.iloc[0] = 0
    else:
        unique_left, unique_right = df.nunique()
        left_cat_curr_pos_ser = pd.Series(
            1 / unique_left * np.arange(unique_left),
            index=df["left_category"].sort_values().unique(),
        )
        right_cat_curr_pos_ser = pd.Series(
            1 / unique_right * np.arange(unique_right),
            index=df["right_category"].sort_values().unique(),
        )

    return left_cat_curr_pos_ser, right_cat_curr_pos_ser


def filter_infrequent_category_combis(
    df, min_perc_left, min_perc_right, min_n, allow_losing_categories=False,
        return_combi_stats=False
) -> Tuple[pd.DataFrame, ...]:
    """Filter out infrequent category combinations from left_category, right category df

    filter out combis (left_category, right_category) which
    are too infrequent to be shown based on min_perc_left, min_perc_right, min_n

    Parameters
    ----------
    df
    min_perc_left
    min_perc_right
    min_n
    return_combi_stats
        if True, return n, perc_left, perc_right for all combis
        This is meant to facilitate choosing appropriate values for the filtering criteria


    Returns
    -------

    Raises
    ------
    ValueError
        if any category is lost completely, because this often messes up plot alignment
        unless allow_losing_categories = True


    """

    left_cat_sizes = df["left_category"].value_counts()
    right_cat_sizes = df["right_category"].value_counts()
    filtered_df = df.groupby([df.iloc[:, 0], df.iloc[:, 1]]).filter(
        lambda df: df.shape[0] > min_n
        and df.shape[0] / left_cat_sizes[df.name[0]] > min_perc_left
        and df.shape[0] / right_cat_sizes[df.name[1]] > min_perc_right
    )

    nunique = filtered_df.nunique()
    if not allow_losing_categories:
        if (
            left_cat_sizes.shape[0] != nunique["left_category"]
            or right_cat_sizes.shape[0] != nunique["right_category"]
        ):
            raise ValueError(
                "With these filtering criteria, you complete lose at least one category"
            )

    if return_combi_stats:
        combi_stats = df.groupby(['left_category', 'right_category']).apply(
                lambda df: pd.Series(dict(
                        n= df.shape[0],
                        perc_left= df.shape[0] / left_cat_sizes[df.name[0]],
                        perc_right= df.shape[0] / right_cat_sizes[df.name[1]],
                ))
        )
        res = (filtered_df, combi_stats)
    else:
        res = (filtered_df, )

    return res


def _fn(x, right_open, spacer_coords, spacer_size):
    """

    Notes
    -----
    This code relies on a quick ad-hoc solution for possible float precision issues when
    comparing values of x against spacer_coords. If the flow_plot appears to have categories
    with blown up sizes (ie not all category sizes are equal), have a look at the code
    and check whether the float comparison in searchsorted is the problem

    Parameters
    ----------
    x
    right_open
    spacer_coords
    spacer_size

    Returns
    -------

    """
    # because x and spacer_coords are floats, relying on np.searchsorted(side='left'|'right') does not work reliably on its own
    # to be able to use /side/, we add/subtract (depending on right_open) a small shift to x so that comparison of x against the spacer sites is not influenced by float precision
    # this is an ad-hoc solution - may need to improve later

    shift = 1e-3
    if right_open:
        x += shift
    else:
        x -= shift

    spacer_coords = np.round(spacer_coords, 3)
    idx = np.searchsorted(spacer_coords, x, "right" if right_open else "left")
    total_spacer_size = spacer_size * spacer_coords.shape[0]
    spazer_size_cumsum = spacer_size * np.arange(spacer_coords.shape[0] + 1)
    return x * (1 - total_spacer_size) + spazer_size_cumsum[idx]
