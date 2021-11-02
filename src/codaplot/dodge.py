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


def dodge_intervals_horizontally(starts, ends, max_precision=9, slack=0):
    """

    1-based intervals, cluster half width 10.5, cluster center at 30
    to retain cluster center at 30, fist intervals would need to start at 19.5 and last interval would need to and at 40.5
    if integers are forced through rounding, we arrive at start = 20 and end = 41, with center = 30.5
    for half width 10.6 we would round to start = 19, end = 41 and get cluster center 30
    for half width 10.3 we would round to start = 20, end = 40 and get cluster center 30
    """

    # jj snippets
    # it appears that annotate_clusters (which we'll used later on) implicitely casts to int32,
    # which can lead to wrong results if an int overflow occurs
    # make sure that starts and ends are small enough even after scaling floats to int (which again is required for annotate_clusters)

    max_int32_value = np.iinfo('i4').max
    max_int32_precision = np.floor(np.log10(np.iinfo("i4").max))
    assert max_precision <= max_int32_precision
    # get maximum precision of integer part
    max_int_prec = np.int32(np.floor(np.log10(starts)).max())
    scale = max_precision - max_int_prec

    # %%
    assert isinstance(starts, np.ndarray)
    assert isinstance(ends, np.ndarray)
    assert ends.dtype == starts.dtype
    # annotate_clusters requires int input
    orig_dtype = starts.dtype
    assert ends.dtype == orig_dtype

    if starts.dtype == ends.dtype == np.int32:
        starts_i4 = starts
        ends_i4 = ends
        if np.can_cast(slack, "i4"):
            slack_i4 = np.int32(slack)
        else:
            raise ValueError
    elif starts.dtype == np.int64:
        assert (
            np.can_cast(starts, np.int32)
            and np.can_cast(ends, np.int32)
            and np.can_cast(slack, np.int32)
        )
        starts_i4 = starts.astype("i4")
        ends_i4 = ends.astype("i4")
        slack_i4 = np.int32(slack)
    elif starts.dtype in [np.float32, np.float64]:
        # float32 and float64 overflow warnings work,
        # so we should pick up overflows here
        start_floats = starts.round(scale) * 10 ** scale
        end_floats = ends.round(scale) * 10 ** scale
        slack_float = slack.round(scale) * 10 ** scale
        # int32 and int64 overflow warnings fail in multiple use cases
        # and i dont understand when
        # so test explicitly
        if (
            (start_floats < max_int32_value).all()
            and (end_floats < max_int32_value).all()
            and (slack_float < max_int32_value)
        ):
            starts_i4 = start_floats.astype("i4")
            ends_i4 = end_floats.astype("i4")
            slack_i4 = slack_float.astype("i4")
        else:
            raise ValueError
    else:
        raise TypeError

    # %%
    intervals = pd.DataFrame(
        dict(
            starts=starts_i4,
            ends=ends_i4,
            length=ends_i4 - starts_i4,
        )
    )

    # sorting is required for annotated clusters (I think)
    pd.testing.assert_frame_equal(intervals, intervals.sort_values(["starts", "ends"]))

    reduced_intervals = intervals.assign(starts_min=starts_i4, ends_max=ends_i4)

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
            .apply(_agg_clusters, slack=slack_i4)
            .reset_index(drop=True)
        )
    # %%

    intervals_ncls = NCLS(
        starts=starts_i4, ends=ends_i4, ids=np.arange(starts_i4.shape[0])
    )
    rh_idx, lh_idx = intervals_ncls.all_overlaps_both(
        reduced_intervals["starts"].to_numpy(),
        reduced_intervals["ends"].to_numpy(),
        reduced_intervals.index.to_numpy(),
    )

    shifted_intervals = intervals.groupby(rh_idx, group_keys=False).apply(
        _spread_intervals, reduced_intervals=reduced_intervals, slack=slack_i4
    )

    # during _agg_clusters, starts and ends are cast to int64 by pandas groupby.apply
    if orig_dtype == np.int32:

        assert shifted_intervals["starts"].lt(max_int32_value).all()
        assert shifted_intervals["ends"].lt(max_int32_value).all()
        return (
            shifted_intervals["starts"].to_numpy().astype('i4'),
            shifted_intervals["ends"].to_numpy().astype('i4'),
        )
    elif orig_dtype == np.int64:
        assert shifted_intervals[["starts", "ends"]].dtypes.eq(np.int64).all()
        return (
            shifted_intervals["starts"].to_numpy(),
            shifted_intervals["ends"].to_numpy(),
        )
    else:  # float
        # int64 can always be cast back to float
        return (
            (shifted_intervals["starts"].to_numpy() / 10 ** scale)
            .astype(orig_dtype)(shifted_intervals["ends"].to_numpy() / 10 ** scale)
            .astype(orig_dtype),
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
