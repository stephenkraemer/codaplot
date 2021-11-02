import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib as mpl
import codaplot as co
import codaplot.utils as coutils
import seaborn as sns
import pyranges as pr
from ncls import NCLS
from sorted_nearest import annotate_clusters

print("reloaded dodge.py")


def dodge_intervals_horizontally(starts, ends, round_to=7, slack=0):
    """

    1-based intervals, cluster half width 10.5, cluster center at 30
    to retain cluster center at 30, fist intervals would need to start at 19.5 and last interval would need to and at 40.5
    if integers are forced through rounding, we arrive at start = 20 and end = 41, with center = 30.5
    for half width 10.6 we would round to start = 19, end = 41 and get cluster center 30
    for half width 10.3 we would round to start = 20, end = 40 and get cluster center 30
    """

    # %%
    assert isinstance(starts, np.ndarray)
    assert isinstance(ends, np.ndarray)
    assert ends.dtype == starts.dtype
    if starts.dtype in [np.int32, np.int64]:
        round_to = 0

    # annotate_clusters requires int input
    if round_to > 0:
        start_ints = (starts.round(round_to) * 10 ** round_to).astype("i8")
        end_ints = (ends.round(round_to) * 10 ** round_to).astype("i8")
        slack_int = np.int64(slack * 10 ** round_to)
    else:
        start_ints = starts
        end_ints = ends
        slack_int = slack

    # %%
    intervals = pd.DataFrame(
        dict(
            starts=start_ints,
            ends=end_ints,
            length=end_ints - start_ints,
        )
    )

    # sorting is required for annotated clusters (I think)
    pd.testing.assert_frame_equal(intervals, intervals.sort_values(["starts", "ends"]))

    reduced_intervals = intervals.assign(starts_min=start_ints, ends_max=end_ints)

    while True:

        cluster_ids = annotate_clusters(
            starts=reduced_intervals["starts"].to_numpy(),
            ends=reduced_intervals["ends"].to_numpy(),
            slack=slack,
        )

        if not pd.Series(cluster_ids).value_counts().gt(1).any():
            break

        reduced_intervals = (
            reduced_intervals.groupby(cluster_ids, group_keys=False)
            .apply(_agg_clusters, slack=slack_int)
            .reset_index(drop=True)
        )
    # %%

    intervals_ncls = NCLS(
        starts=start_ints, ends=end_ints, ids=np.arange(start_ints.shape[0])
    )
    rh_idx, lh_idx = intervals_ncls.all_overlaps_both(
        reduced_intervals["starts"].to_numpy(),
        reduced_intervals["ends"].to_numpy(),
        reduced_intervals.index.to_numpy(),
    )

    shifted_intervals = intervals.groupby(rh_idx, group_keys=False).apply(
        _spread_intervals, reduced_intervals=reduced_intervals, slack=slack_int
    )

    return (
        shifted_intervals["starts"].to_numpy() / 10 ** round_to,
        shifted_intervals["ends"].to_numpy() / 10 ** round_to,
    )


def _agg_clusters(group_df, slack):
    if group_df.shape[0] > 1:
        # original start and end without slack to calculate cluster center
        start = group_df["starts_min"].min()
        end = group_df["ends_max"].max()
        center = start + (end - start) / 2
        width = (group_df["length"] + slack).sum() - slack  # no slack for last interval
        half_width = width / 2
        return pd.DataFrame(
            dict(
                starts=np.round([center - half_width], 0).astype("i4"),
                ends=np.round([center + half_width], 0).astype("i4"),
                length=width,
                starts_min=start,
                ends_max=end,
            )
        )
    return group_df


def _spread_intervals(group_df, reduced_intervals, slack):
    if group_df.shape[0] > 1:
        cluster_start, cluster_end = reduced_intervals.iloc[group_df.name][
            ["starts", "ends"]
        ]
        l = np.concatenate([[0], (group_df["length"] + slack).cumsum()[:-1]])
        group_df["starts"] = cluster_start + l
        group_df["ends"] = group_df["starts"] + group_df["length"]
    return group_df
