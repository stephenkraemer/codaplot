"""Add gaps to the x and y axis to create whitespace between axis domains

This allows creating plots with spaced-out elements, e.g. heatmaps with gaps
between clusters, with standard matplotlib artists.
"""

print('reloaded gap_scale')

import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
from typing import Any, Dict, List, Sequence, Tuple, Union

from matplotlib.transforms import Transform, IdentityTransform
from matplotlib.scale import ScaleBase, register_scale

groups_t = Union[Sequence[Any], pd.Series, pd.DataFrame, np.ndarray]
gaps_t = Union[float, Sequence[float], Dict[Any, float], np.ndarray]


def compute_breakpoints_and_offsets(
    groups: groups_t, gaps: gaps_t
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integer breakpoints and corresponding offsets for gapped scales.

    Parameters
    ----------
    groups : 1D sequence-like or pandas DataFrame
        If sequence-like, length N, with values defining group membership for positions 0..N-1.
        If DataFrame, each column defines a nested grouping level.
    gap_sizes : scalar, sequence, or dict
        - If groups is 1D:
          * scalar: same gap size for all breaks
          * sequence of length num_breaks: per-break gap sizes
          * dict mapping group value -> gap size (applied when leaving that value)
        - If groups is DataFrame:
          * sequence of length num_columns: gap size per column-level
          * dict mapping full-group-tuple -> gap size (applied after last row of that tuple)

    Returns
    -------
    breakpoints : np.ndarray of int
        Positions in original coordinates where gaps should be inserted.
    offsets : np.ndarray of float
        Corresponding sizes of each gap.
    """

    breakpoints: np.ndarray | list[float]
    offsets: np.ndarray | list[float]

    # Case 1: flat sequence of groups
    if not isinstance(groups, (pd.DataFrame,)):
        seq = np.array(groups)
        N = len(seq)
        # find breakpoints where group changes
        breakpoints = np.where(seq[1:] != seq[:-1])[0] + 1
        M = len(breakpoints)
        # determine offsets
        if isinstance(gaps, numbers.Number):
            offsets = np.array([gaps] * M, dtype="f4")
        elif isinstance(gaps, (Sequence, pd.Series, np.ndarray)) and not isinstance(
            gaps, (str, bytes)
        ):
            if len(gaps) != M:
                raise ValueError(f"gap_sizes length {len(gaps)} != num_breaks {M}")
            offsets = np.array(gaps)
        elif isinstance(gaps, dict):
            offsets = []
            for i in breakpoints:
                key = seq[i - 1]
                try:
                    offsets.append(gaps[key])
                except KeyError:
                    raise KeyError(f"No gap size for group '{key}' at breakpoint {i}")
        else:
            raise TypeError("gap_sizes type not understood for 1D groups")
        return np.array(breakpoints, dtype=int), np.array(offsets, dtype=float)

    # Case 2: DataFrame of nested groups
    df = pd.DataFrame(groups).reset_index(drop=True)
    n_levels = df.shape[1]
    used = set()
    breakpoints = []
    offsets = []

    # pre-check gap_sizes
    if isinstance(gaps, (Sequence, np.ndarray)) and not isinstance(gaps, dict):
        if len(gaps) != n_levels:
            raise ValueError(
                f"For DataFrame groups, need {n_levels} gap_sizes, got {len(gaps)}"
            )
        level_sizes: Dict[int, float] = {lvl: float(sz) for lvl, sz in enumerate(gaps)}
    elif isinstance(gaps, dict):
        dict_sizes = gaps
    elif isinstance(gaps, float):
        # broadcast scalar to all levels
        level_sizes = {lvl: gaps for lvl in range(n_levels)}
    else:
        raise TypeError("gap_sizes type not understood for DataFrame groups")

    # iterate levels left-to-right
    for lvl in range(n_levels):
        col = df.columns[lvl]
        vals = df[col].values
        # find changes
        changes = np.where(vals[1:] != vals[:-1])[0] + 1
        for idx in changes:
            if idx in used:
                continue
            # record this breakpoint
            breakpoints.append(idx)
            used.add(idx)
            # select offset size
            if isinstance(gaps, dict):
                # key is full tuple of row at idx-1
                key = tuple(df.iloc[idx - 1].tolist())
                try:
                    sz = float(dict_sizes[key])  # type: ignore
                except KeyError:
                    raise KeyError(
                        f"No gap size for key {key} at level {lvl}, row {idx}"
                    )
            else:
                sz = level_sizes[lvl]  # type: ignore
            offsets.append(sz)

    # sort by breakpoint position
    order = np.argsort(breakpoints)
    bps_sorted = np.array([breakpoints[i] for i in order], dtype=int)
    offs_sorted = np.array([offsets[i] for i in order], dtype=float)

    return bps_sorted, offs_sorted


class GapTransform(Transform):
    """
    Piecewise-linear Transform that shifts any x (or y) above each breakpoint
    by the cumulative sum of offsets.

    breakpoints and offsets must be sorted by increasing breakpoint.
    """

    input_dims = 1  # type: ignore
    output_dims = 1  # type: ignore
    has_inverse = True  # type: ignore

    def __init__(self, breakpoints: np.ndarray, offsets: np.ndarray):
        super().__init__()
        self.bps = np.asarray(breakpoints, dtype=int)
        self.offs = np.asarray(offsets, dtype=float)

    def transform_non_affine(self, values):
        x = np.asanyarray(values)
        if self.bps.size == 0:
            return x
        shifts = np.sum(self.offs * (x[..., None] >= self.bps), axis=-1)
        return x + shifts

    def inverted(self):
        return InvertedGapTransform(self.bps, self.offs)


class InvertedGapTransform(Transform):
    """
    Inverse of GapTransform: subtracts cumulative offsets in one vectorized step.

    We precompute shifted breakpoints = bp + sum(offs before that bp),
    then for each x, any shifted_bp > x contributes its offset to the backshift.
    """

    input_dims = 1  # type: ignore
    output_dims = 1  # type: ignore
    has_inverse = True  # type: ignore

    def __init__(self, breakpoints: np.ndarray, offsets: np.ndarray):
        super().__init__()
        bps = np.asarray(breakpoints, dtype=int)
        self.offs = np.asarray(offsets, dtype=float)
        # compute sum of offsets before each breakpoint
        cum_prev = np.concatenate((np.array([0.0]), np.cumsum(self.offs[:-1])))
        # shifted breakpoints
        self.shifted_bps = bps + cum_prev

    def transform_non_affine(self, values):
        x = np.asanyarray(values)
        if self.shifted_bps.size == 0:
            return x
        backshifts = np.sum(self.offs * (x[..., None] <= self.shifted_bps), axis=-1)
        return x - backshifts

    def inverted(self):
        # Inverse of inverse is forward transform with original bps and offs
        return GapTransform(self.shifted_bps, self.offs)


class GapScale(ScaleBase):
    name = "gap_scale"

    def __init__(self, axis, *, groups, gap_sizes):
        super().__init__(axis)
        self.groups = groups
        self.gap_sizes = gap_sizes

    def get_transform(self):
        bps, offs = compute_breakpoints_and_offsets(self.groups, self.gap_sizes)
        if bps.size == 0:
            return IdentityTransform()
        return GapTransform(bps, offs)

    def set_default_locators_and_formatters(self, axis):
        pass

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax


register_scale(GapScale)
