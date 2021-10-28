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

from codaplot.dodge import dodge_intervals_horizontally


def test_dodge_intervals_int():

    intervals_int = pd.DataFrame(
        dict(
            starts=[10, 30, 40, 80],
            ends=[25, 50, 60, 95],
        )
    )

    # with slack = 2
    # cluster 1: 0, 1, 2
    # so 4th interval should remain unchanged
    # cluster center = 35
    # cluster width = 55 + 4 (slack) = 59
    # start first interval = 35 - 59/2 = 5.5, rounded = 6
    # end last interval = 35 + 59/2 = 64.5, rounded = 65
    # rounded center = 6 + 59/2 = 35.5

    res = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy(),
        ends=intervals_int["ends"].to_numpy(),
        # round_to=3,
        slack=2,
    )

    intervals_int["starts_shifted"] = res[0]
    intervals_int["ends_shifted"] = res[1]

    intervals_int

    assert intervals_int.iloc[0]['starts_shifted'] == 6
    assert intervals_int.iloc[2]['ends_shifted'] == 65
    assert (intervals_int.iloc[3][['starts', 'ends']].to_numpy() == intervals_int.iloc[3][['starts_shifted', 'ends_shifted']].to_numpy()).all()
    # assert that lengths haven't changed
    assert intervals_int.eval('ends - starts == ends_shifted - starts_shifted').all()


def test_dodge_intervals_float():

    intervals_int = pd.DataFrame(
        dict(
            starts=[10, 30, 40, 80],
            ends=[25, 50, 60, 95],
        )
    )

    # with slack = 2
    # cluster 1: 0, 1, 2
    # so 4th interval should remain unchanged
    # cluster center = 35
    # cluster width = 55 + 4 (slack) = 59
    # start first interval = 35 - 59/2 = 5.5, rounded = 6
    # end last interval = 35 + 59/2 = 64.5, rounded = 65
    # rounded center = 6 + 59/2 = 35.5

    starts_int, ends_int = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy(),
        ends=intervals_int["ends"].to_numpy(),
        # round_to=3,
        slack=2,
    )

    starts_float, ends_float = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy() / 1200,
        ends=intervals_int["ends"].to_numpy() / 1200,
        round_to=7,
        slack=2/1200,
    )

    # we fall below 0.5 to 0.49999; add small increment to round up at values which are numerically close to 0.499
    assert ((starts_float * 1200 + 0.01).round(0).astype('i8') == starts_int).all()
    assert ((ends_float * 1200 + 0.01).round(0).astype('i8') == ends_int).all()

    # in general, the round_to has a suprisingly large effect, should investigate

    starts_float, ends_float = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy() / 1200,
        ends=intervals_int["ends"].to_numpy() / 1200,
        round_to=5,
        slack=2/1200,
    )
    starts_float * 1200

    starts_float, ends_float = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy() / 1200,
        ends=intervals_int["ends"].to_numpy() / 1200,
        round_to=8,
        slack=2/1200,
    )
    starts_float * 1200

    # and its really bad if informative, non periodic decimal places are cut

    starts_float, ends_float = dodge_intervals_horizontally(
        starts=intervals_int["starts"].to_numpy() / 1200,
        ends=intervals_int["ends"].to_numpy() / 1200,
        round_to=2,
        slack=2/1200,
    )
    starts_float * 1200

