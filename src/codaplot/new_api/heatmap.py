from copy import deepcopy
from typing import Any
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from matplotlib.axes import Axes

from codaplot.new_api.gap_scale import compute_breakpoints_and_offsets, groups_t, gaps_t
from codaplot.utils import cbar_change_style_to_inward_white_ticks, despine
from matplotlib.colors import Normalize
from matplotlib.colors import Colormap

pd.set_option("future.no_silent_downcasting", True)


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
    na_color: str | tuple = "white",
    show_guide: bool = True,
    guide_args: dict | None = None,
    guide_title: str | None = None,
    guide_ax: Axes | None = None,
    cbar_ticks: np.ndarray | list | None = None,
    cbar_styling_func=cbar_change_style_to_inward_white_ticks,
    cbar_styling_func_kwargs: dict | None = None,
    **kwargs,
) -> dict:

    palette, cbar_styling_func_kwargs, df, is_categorical, guide_args_copy = (
        _handle_inputs(data, palette, guide_args, guide_title, cbar_styling_func_kwargs)
    )

    # General aesthetics
    # ax.tick_params(length=0, which="both", axis="both")
    despine(ax, left=True, bottom=True)

    if is_categorical:
        levels, patches = _create_categorical_heatmap(
            ax,
            x_groups,
            x_gaps,
            y_groups,
            y_gaps,
            palette,
            na_color,
            show_guide,
            guide_title,
            kwargs,
            df,
            guide_args_copy,
        )

        return guide_args_copy | {
            "title": guide_title,
            "handles": patches,
            "labels": levels,
        }

    scalar_mappable, cbar = _create_continuous_heatmap(
        ax,
        x_groups,
        x_gaps,
        y_groups,
        y_gaps,
        norm,
        vmin,
        vmax,
        palette,
        show_guide,
        guide_title,
        guide_ax,
        cbar_ticks,
        cbar_styling_func,
        cbar_styling_func_kwargs,
        kwargs,
        df,
        guide_args_copy,
    )

    return guide_args_copy | {
        # "cbar": cbar,
        "mappable": scalar_mappable,
        "title": guide_title,
        "styling_func_kwargs": cbar_styling_func_kwargs,
        "cbar_ticks": cbar_ticks,
    }


def _create_continuous_heatmap(
    ax,
    x_groups,
    x_gaps,
    y_groups,
    y_gaps,
    norm,
    vmin,
    vmax,
    palette,
    show_guide,
    guide_title,
    guide_ax,
    cbar_ticks,
    cbar_styling_func,
    cbar_styling_func_kwargs,
    kwargs,
    df,
    guide_args_copy,
):

    cmap = palette
    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[reportAttributeAccessIssue]
    plot_arr = scalar_mappable.to_rgba(df)
    plot_arr, x, y = _insert_gaps_and_generate_coordinates(
        x_groups, x_gaps, y_groups, y_gaps, plot_arr
    )
    ax.pcolormesh(x, y, plot_arr, **kwargs)

    cbar = None
    if show_guide:
        cbar = ax.figure.colorbar(scalar_mappable, ax=ax, cax=guide_ax, **guide_args_copy)  # type: ignore
        if cbar_ticks is not None:
            cbar.ax.set_yticks(cbar_ticks)
        ax.figure.canvas.draw()
        cbar_styling_func(cbar, **cbar_styling_func_kwargs)
        if guide_title is not None:
            cbar.set_label(guide_title)

    return scalar_mappable, cbar


def _create_categorical_heatmap(
    ax,
    x_groups,
    x_gaps,
    y_groups,
    y_gaps,
    palette,
    na_color,
    show_guide,
    guide_title,
    kwargs,
    df,
    guide_args_copy,
):
    levels = _find_ordered_levels(df)
    color_d = _generate_color_mapping(palette, na_color, levels, len(levels))
    plot_arr = _df_to_rgba_vectorized(
        data=df, color_d=color_d, na_color=na_color, levels=levels
    )
    plot_arr, x, y = _insert_gaps_and_generate_coordinates(
        x_groups, x_gaps, y_groups, y_gaps, plot_arr
    )
    ax.pcolormesh(x, y, plot_arr, **kwargs)

    patches = [Patch(facecolor=color_d[l]) for l in levels]
    if show_guide:
        ax.legend(
            handles=patches,
            labels=list(levels),
            title=guide_title,
            **guide_args_copy,
        )

    return levels, patches


def _insert_gaps_and_generate_coordinates(x_groups, x_gaps, y_groups, y_gaps, plot_arr):
    if x_groups is not None or y_groups is not None:
        plot_arr, x, y = _insert_gaps(plot_arr, x_groups, x_gaps, y_groups, y_gaps)
    else:
        x = np.arange(plot_arr.shape[1])
        y = np.arange(plot_arr.shape[0])
    return plot_arr, x, y


def _handle_inputs(data, palette, guide_args, guide_title, cbar_styling_func_kwargs):
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise TypeError("data must be DataFrame or np.ndarray")

    df = pd.DataFrame(data)

    if cbar_styling_func_kwargs is None:
        cbar_styling_func_kwargs = {}

    is_categorical = (
        isinstance(df.dtypes.iloc[0], pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df.dtypes.iloc[0])
        or pd.api.types.is_string_dtype(df.dtypes.iloc[0])
    )

    guide_args_copy: dict[str, Any]
    if guide_args is None:
        if is_categorical:
            guide_args_copy = dict(
                loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0
            )
        else:
            guide_args_copy = {}
    else:
        guide_args_copy = deepcopy(guide_args)

    if palette is None:
        if is_categorical:
            palette = "Set1"
        else:
            palette = "viridis"
    return palette, cbar_styling_func_kwargs, df, is_categorical, guide_args_copy


def _insert_gaps(plot_arr, x_groups, x_gaps, y_groups, y_gaps):
    n_rows, n_cols, _ = plot_arr.shape
    if x_groups is not None:
        if x_gaps is None:
            raise ValueError("If x_groups is specified, x_gaps must also be specified")
        x_bps, _ = compute_breakpoints_and_offsets(x_groups, x_gaps)
        plot_arr = np.insert(plot_arr, x_bps, np.ones((n_rows, len(x_bps), 4)), axis=1)
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
        plot_arr = np.insert(plot_arr, y_bps, np.ones((len(y_bps), n_cols, 4)), axis=0)
        y = np.arange(n_rows + 1, dtype="f4")
        y[y_bps] -= 0.001
        y = np.insert(y, y_bps + 1, y_bps)
        n_rows += len(y_bps)
    else:
        y = np.arange(n_rows + 1)
    return plot_arr, x, y


def _generate_color_mapping(palette, na_color, levels, n_levels) -> dict:
    """Generate dict mapping each level to a color, and mapping null to na_color"""
    color_mapping: dict
    if isinstance(palette, str):
        color_l = get_first_n_colors(palette, n_levels)
        color_mapping = dict(zip(levels, color_l))
        # for a list of colors, cycle through them multiple times if necessary
    elif isinstance(palette, (list, np.ndarray, tuple)):
        color_l = (np.ceil(n_levels / len(palette)).astype(int) * palette)[:n_levels]
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
        None,  # Python’s null object
        np.nan,  # NumPy “not a number” (float) marker
        pd.NaT,  # pandas “Not a Time” for datetime/timedelta
        pd.NA,  # pandas’ new generic missing value for nullable dtypes
    ]
    if any(x in color_mapping.keys() for x in nullable_categorical_markers):
        raise ValueError(
            "NA color must be specified with argument na_color, not as part of palette"
        )
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
    cmap = plt.get_cmap(palette_name)  # e.g. 'tab20'
    try:
        # most categorical maps (tab10, tab20, etc.) expose a .colors list
        return cmap.colors[:n]  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        # fallback for continuous maps: sample N evenly‐spaced points
        return [cmap(i / (n - 1)) for i in range(n)]


def _df_to_rgba_vectorized(
    data: pd.DataFrame,
    color_d: dict,
    levels: np.ndarray,
    na_color: str | tuple = "none",
) -> np.ndarray:

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
            col = tuple(c / 255.0 for c in col)
        lut[i] = to_rgba(col)
    if isinstance(na_color, tuple) and max(na_color) > 1:
        na_color = tuple(c / 255.0 for c in na_color)
    lut[-1] = to_rgba(na_color)

    # fancy‐index
    rgba = lut[codes]  # shape (H, W, 4)
    return rgba
