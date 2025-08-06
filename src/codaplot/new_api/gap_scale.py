"""Add gaps to the x and y axis to create whitespace between axis domains

This allows creating plots with spaced-out elements, e.g. heatmaps with gaps
between clusters, with standard matplotlib artists.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
import numbers
from typing import Any, Dict, List, Sequence, Tuple, Union

groups_t = Union[Sequence[Any], pd.Series, pd.DataFrame]
gaps_t = Union[float, Sequence[float], Dict[Any, float]]


def compute_breakpoints_and_offsets(
    groups: groups_t,
    gaps: gaps_t
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
    # Helper to classify gap_sizes
    def is_sequence(obj):
        return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))

    # Case 1: flat sequence of groups
    if not isinstance(groups, (pd.DataFrame,)):
        seq = list(groups)
        N = len(seq)
        # find breakpoints where group changes
        bps = [i for i in range(1, N) if seq[i] != seq[i - 1]]
        M = len(bps)
        # determine offsets
        if isinstance(gaps, numbers.Number):
            offs = [float(gaps)] * M
        elif is_sequence(gaps):
            if len(gaps) != M:
                raise ValueError(f"gap_sizes length {len(gaps)} != num_breaks {M}")
            offs = list(map(float, gaps))
        elif isinstance(gaps, dict):
            offs = []
            for i in bps:
                key = seq[i - 1]
                try:
                    offs.append(float(gaps[key]))
                except KeyError:
                    raise KeyError(f"No gap size for group '{key}' at breakpoint {i}")
        else:
            raise TypeError("gap_sizes type not understood for 1D groups")
        return np.array(bps, dtype=int), np.array(offs, dtype=float)

    # Case 2: DataFrame of nested groups
    df = pd.DataFrame(groups).reset_index(drop=True)
    n_rows, n_levels = df.shape
    used = set()
    bps: List[int] = []
    offs: List[float] = []

    # pre-check gap_sizes
    if isinstance(gaps, Sequence) and not isinstance(gaps, dict):
        if len(gaps) != n_levels:
            raise ValueError(f"For DataFrame groups, need {n_levels} gap_sizes, got {len(gaps)}")
        level_sizes: Dict[int, float] = {lvl: float(sz) for lvl, sz in enumerate(gaps)}
    elif isinstance(gaps, dict):
        dict_sizes = gaps
    elif isinstance(gaps, numbers.Number):
        # broadcast scalar to all levels
        level_sizes = {lvl: float(gaps) for lvl in range(n_levels)}
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
            bps.append(idx)
            used.add(idx)
            # select offset size
            if isinstance(gaps, dict):
                # key is full tuple of row at idx-1
                key = tuple(df.iloc[idx - 1].tolist())
                try:
                    sz = float(dict_sizes[key])
                except KeyError:
                    raise KeyError(f"No gap size for key {key} at level {lvl}, row {idx}")
            else:
                sz = level_sizes[lvl]
            offs.append(sz)

    # sort by breakpoint position
    order = np.argsort(bps)
    bps = np.array([bps[i] for i in order], dtype=int)
    offs = np.array([offs[i] for i in order], dtype=float)

    return bps, offs

# %%


from matplotlib.transforms import Transform, IdentityTransform
from matplotlib.scale import ScaleBase, register_scale



class GapTransform(Transform):
    """
    Piecewise-linear Transform that shifts any x (or y) above each breakpoint
    by the cumulative sum of offsets.

    breakpoints and offsets must be sorted by increasing breakpoint.
    """
    input_dims = output_dims = 1
    has_inverse = True

    def __init__(self, breakpoints: np.ndarray, offsets: np.ndarray):
        super().__init__()
        self.bps = np.asarray(breakpoints, dtype=int)
        self.offs = np.asarray(offsets, dtype=float)

    def transform_non_affine(self, a):
        x = np.asanyarray(a)
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
    input_dims = output_dims = 1
    has_inverse = True

    def __init__(self, breakpoints: np.ndarray, offsets: np.ndarray):
        super().__init__()
        bps = np.asarray(breakpoints, dtype=int)
        self.offs = np.asarray(offsets, dtype=float)
        # compute sum of offsets before each breakpoint
        cum_prev = np.concatenate((np.array([0.0]), np.cumsum(self.offs[:-1])))
        # shifted breakpoints
        self.shifted_bps = bps + cum_prev

    def transform_non_affine(self, a):
        x = np.asanyarray(a)
        if self.shifted_bps.size == 0:
            return x
        backshifts = np.sum(self.offs * (x[..., None] <= self.shifted_bps), axis=-1)
        return x - backshifts

    def inverted(self):
        # Inverse of inverse is forward transform with original bps and offs
        return GapTransform(self.shifted_bps, self.offs)


class _BaseGapScale(ScaleBase):
    """
    Base class for gap scales on X or Y axis.
    Subclasses must set `name`.
    """
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


class GapScaleX(_BaseGapScale):
    name = 'gap_scale_x'


class GapScaleY(_BaseGapScale):
    name = 'gap_scale_y'


# register the new scales
register_scale(GapScaleX)
register_scale(GapScaleY)

# %%

# fig, ax = plt.subplots()
# ax.set_xscale('gap_scale_x', groups=['a', 'a', 'b', 'b'], gap_sizes=0.5)
# ax.plot([0, 1], [0, 1])
# ax.plot([2, 3], [2, 3])
# ax.grid()
# fig.savefig('/home/stephekr/projects/codaplot/src/codaplot/new_api/test.pdf')