import numpy as np
import pandas as pd
import pytest

from codaplot.new_api.gap_scale import compute_breakpoints_and_offsets


def test_1d_scalar_gap():
    # groups with two breaks: at index 2 and 4
    groups = ["a", "a", "b", "b", "c", "c"]
    expected_bps = np.array([2, 4])
    expected_offs = np.array([0.5, 0.5])
    bps, offs = compute_breakpoints_and_offsets(groups, gaps=0.5)
    assert np.array_equal(bps, expected_bps)
    assert np.allclose(offs, expected_offs)


def test_1d_list_gap():
    groups = [
        "v",
        "v",
        "x",
        "y",
        "y",
        "z",
    ]
    # break at 1 (x->y) and at 3 (y->z)
    expected_bps = np.array([2, 3, 5])
    gap_sizes = [0.2, 0.1, 0.2]
    bps, offs = compute_breakpoints_and_offsets(groups, gaps=gap_sizes)
    assert np.array_equal(bps, expected_bps)
    assert np.allclose(offs, gap_sizes)


def test_1d_dict_gap():
    groups = ["a", "b", "b", "c"]
    # breaks: 1(a->b), 3(b->c)
    expected_bps = np.array([1, 3])
    gap_dict = {"a": 0.2, "b": 0.3}
    bps, offs = compute_breakpoints_and_offsets(groups, gaps=gap_dict)
    assert np.array_equal(bps, expected_bps)
    # first break uses a->b => key 'a', second break b->c => key 'b'
    assert np.allclose(offs, np.array([0.2, 0.3]))


def test_dataframe_sequence_gap():
    df = pd.DataFrame(
        {"A": [1, 1, 2, 2, 3, 3, 3], "B": ["a", "b", "c", "c", "c", "c", "e"]}
    )
    # level-1 breaks at index 2, 4
    # level-2 breaks at index 1, 6
    expected_bps = np.array([1, 2, 4, 6])
    # using sequence gap_sizes [0.1, 0.2]
    bps, offs = compute_breakpoints_and_offsets(df, gaps=[0.3, 0.1])
    assert np.array_equal(bps, expected_bps)
    # offs should map level-2 gaps at positions 1,3 have size 0.2; level-1 at 2,4 size 0.1
    expected_offs = np.array([0.1, 0.3, 0.3, 0.1])
    assert np.allclose(offs, expected_offs)


def test_dataframe_dict_gap():

    df = pd.DataFrame({"A": [1, 1, 2, 2], "B": ["x", "x", "y", "z"]})
    # possible breaks at 2 (1->2), 3 (y->z)
    gap_dict = {(1, "x"): 0.05, (2, "y"): 0.15}
    bps, offs = compute_breakpoints_and_offsets(df, gaps=gap_dict)
    assert np.array_equal(bps, np.array([2, 3]))
    # breakpoint at 2 uses key (1,'x'), at 3 uses (2,'y')
    assert np.allclose(offs, np.array([0.05, 0.15]))


def test_no_breaks_returns_empty():

    groups = ["a", "a", "a"]
    bps, offs = compute_breakpoints_and_offsets(groups, gaps=1.0)
    assert bps.size == 0
    assert offs.size == 0


def test_invalid_gap_sizes_length():

    groups = ["a", "b", "c"]
    with pytest.raises(ValueError):
        compute_breakpoints_and_offsets(groups, gaps=[0.1])
