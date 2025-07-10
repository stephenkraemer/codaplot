print("reloaded heatmap 3")

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
import pandas as pd
import numpy as np
from matplotlib.axes import Axes

from codaplot.new_api.gap_scale import compute_breakpoints_and_offsets, groups_t, gaps_t
from matplotlib.colors import Normalize
from matplotlib.colors import Colormap

pd.set_option('future.no_silent_downcasting', True)


def heatmap(
    data: pd.DataFrame | np.ndarray,
    ax: Axes,
    x_groups: groups_t | None = None,
    x_gaps: gaps_t | None = None,
    y_groups: groups_t | None = None,
    y_gaps: gaps_t | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    palette: str | Colormap | dict | list | None = None,
    na_color: str | tuple = 'white',
    **kwargs,
) -> dict:

    # %%

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise TypeError('data must be DataFrame or np.ndarray')

    data = pd.DataFrame(data)

    is_categorical = (
        isinstance(data.dtypes.iloc[0], pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(data.dtypes.iloc[0])
        or pd.api.types.is_string_dtype(data.dtypes.iloc[0])
    )

    if palette is None:
        if is_categorical:
            palette = 'Set1'
        else:
            palette = 'viridis'

    if is_categorical:
        levels = _find_ordered_levels(data)
        color_d = _generate_color_mapping(palette, na_color, levels, len(levels))
        plot_arr = _df_to_rgba_vectorized(data=data, color_d=color_d, na_color=na_color, levels=levels)
    else:
        cmap = palette
        if norm is None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[reportAttributeAccessIssue]
        plot_arr = scalar_mappable.to_rgba(data)

    if x_groups is not None or y_groups is not None:
        plot_arr, x, y = _insert_gaps(plot_arr, x_groups, x_gaps, y_groups, y_gaps)
    else:
        x = np.arange(plot_arr.shape[1])
        y = np.arange(plot_arr.shape[0])

    ax.pcolormesh(x, y, plot_arr, **kwargs)

    if not is_categorical:
        return {'mappable': scalar_mappable}  # pyright: ignore[reportPossiblyUnboundVariable]
    return {}

def _insert_gaps(plot_arr, x_groups, x_gaps, y_groups, y_gaps):
    n_rows, n_cols, _ = plot_arr.shape
    if x_groups is not None:
        if x_gaps is None:
            raise ValueError("If x_groups is specified, x_gaps must also be specified")
        x_bps, _ = compute_breakpoints_and_offsets(x_groups, x_gaps)
        plot_arr = np.insert(
            plot_arr, x_bps, np.ones((n_rows, len(x_bps), 4)), axis=1
        )
        x = np.arange(n_cols + 1, dtype="f4")
        # if we subtract too much, for heatmaps with many elements (e.g. 5000)
        # the spacer column may have no effect (i.e. one just sees a stretched data columns in tests)
        x[x_bps] -= 0.001
        x = np.insert(x, x_bps + 1, x_bps)
        n_cols += len(x_bps)
    else:
        x = np.arange(n_cols + 1)
    if y_groups is not None:
        if y_gaps is None:
            raise ValueError("If y_groups is specified, y_gaps must also be specified")
        y_bps, _ = compute_breakpoints_and_offsets(y_groups, y_gaps)
        plot_arr = np.insert(
            plot_arr, y_bps, np.ones((len(y_bps), n_cols, 4)), axis=0
        )
        y = np.arange(n_rows + 1, dtype="f4")
        y[y_bps] -= 0.001
        y = np.insert(y, y_bps + 1, y_bps)
        n_rows += len(y_bps)
    else:
        y = np.arange(n_rows + 1)
    return plot_arr,x,y

def _generate_color_mapping(palette, na_color, levels, n_levels) -> dict:
    """Generate dict mapping each level to a color, and mapping null to na_color"""
    color_mapping: dict
    if isinstance(palette, str):
        color_l = get_first_n_colors(palette, n_levels)
        color_mapping = dict(zip(levels, color_l))
        # for a list of colors, cycle through them multiple times if necessary
    elif isinstance(palette, (list, np.ndarray, tuple)):
        color_l = (
                np.ceil(n_levels / len(palette)).astype(int) * palette
            )[:n_levels]
        color_mapping = dict(zip(levels, color_l))
    elif isinstance(palette, dict):
        assert set(palette.keys()) == set(levels)
        color_mapping = palette
    elif isinstance(palette, pd.Series):
        assert set(palette.index) == set(levels)
        color_mapping = palette.to_dict()
    else:
        raise TypeError()
        
    nullable_categorical_markers = [
            None,     # Python’s null object
            np.nan,   # NumPy “not a number” (float) marker
            pd.NaT,   # pandas “Not a Time” for datetime/timedelta
            pd.NA ,    # pandas’ new generic missing value for nullable dtypes
        ]
    if any(x in color_mapping.keys() for x in nullable_categorical_markers):
        raise ValueError('NA color must be specified with argument na_color, not as part of palette')
    for null_value in nullable_categorical_markers:
        color_mapping[null_value] = na_color
    return color_mapping

def _find_ordered_levels(data: pd.DataFrame) -> np.ndarray:
    """Returns ordered levels, either in category order or in sort order otherwise"""
    if isinstance(data.dtypes.iloc[0], pd.CategoricalDtype):
            # level order is given by CategoricalDtype or by sort order
            # note that categories cannot be null, but the dataframe may still
            # contain null values
        levels = data.dtypes.iloc[0].categories.values
    else:
        uni = pd.unique(data.values.ravel())
        uni = uni[~pd.isnull(uni)]
        levels = np.sort(uni)
    return levels


def get_first_n_colors(palette_name: str, n: int):
    cmap = plt.get_cmap(palette_name)         # e.g. 'tab20'
    try:
        # most categorical maps (tab10, tab20, etc.) expose a .colors list
        return cmap.colors[:n]  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        # fallback for continuous maps: sample N evenly‐spaced points
        return [cmap(i/(n-1)) for i in range(n)]


def _df_to_rgba_vectorized(data: pd.DataFrame,
                          color_d: dict,
                          levels: np.ndarray,
                          na_color: str | tuple = 'none') -> np.ndarray:
    

    if isinstance(data.dtypes.iloc[0], pd.CategoricalDtype):
        codes = data.iloc[:, 0].cat.codes
    else:
        code_map = {cat: i for i, cat in enumerate(levels)}


        codes = (
            data.replace(code_map)
            .astype(int)
            # default code for na in Categoricals is -1
            .fillna(-1)
             .values
        )

    # build RGBA lookup table
    lut = np.empty((len(levels) + 1, 4), dtype=float)
    for i, cat in enumerate(levels):
        col = color_d[cat]
        # normalize if tuple of 0–255 ints
        if isinstance(col, tuple) and max(col) > 1:
            col = tuple(c/255.0 for c in col)
        lut[i] = to_rgba(col)
    if isinstance(na_color, tuple) and max(na_color) > 1:
        na_color = tuple(c/255.0 for c in na_color)
    lut[-1] = to_rgba(na_color)

    # fancy‐index
    rgba = lut[codes]   # shape (H, W, 4)
    return rgba
