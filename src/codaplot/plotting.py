""" Currently: messy collection of plotting functions

Heatmaps
--------

- different heatmap functions co-exist in this module
- only one heatmap function should be used: heatmap
  - both for categorical and numeric heatmaps
  - both for dense and spaced heatmaps
  - heatmap internally calls heatmap2 and spaced_heatmap2. these functions should
    not be public and should not be used directly.

- this module also houses dendrogram functionality

"""
import warnings
from codaplot.utils import cbar_change_style_to_inward_white_ticks
from copy import deepcopy
from dataclasses import dataclass
from itertools import product, zip_longest
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    Literal,
)

import codaplot as co
import matplotlib as mpl
import matplotlib.patches as patches
import itertools
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
import toolz as tz
from codaplot.cluster_ids import ClusterIDs
from codaplot.linkage_mat import Linkage
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.groupby import GroupBy
from scipy.cluster.hierarchy import dendrogram, leaves_list

CMAP_DICT = dict(
    divergent_meth_heatmap=plt.get_cmap("RdBu_r"),
    sequential_meth_heatmap=plt.get_cmap("YlOrBr"),
    cluster_id_to_color=dict(zip(np.arange(1, 101), sns.color_palette("Set1", 100))),
)
CMAP_DICT["cluster_id_to_color"][-1] = (0, 0, 0)
CMAP_DICT["cluster_id_to_color"][0] = (0, 0, 0)


def remove_axis(ax: Axes, y=True, x=True):
    """Remove spines, ticks, labels, ..."""
    if y:
        ax.set(yticks=[], yticklabels=[])
        sns.despine(ax=ax, left=True)
    if x:
        ax.set(xticks=[], xticklabels=[])
        sns.despine(ax=ax, bottom=True)


def dendrogram_wrapper(linkage_mat, ax: Axes, orientation: str):
    """Wrapper around scipy dendrogram - nicer plot

    Despines plot and removes ticks and labels
    """
    dendrogram(
        linkage_mat,
        ax=ax,
        color_threshold=-1,
        above_threshold_color="black",
        orientation=orientation,
    )
    ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
    sns.despine(ax=ax, bottom=True, left=True)


def categorical_heatmap(
    df: pd.DataFrame,
    ax: Axes,
    cmap: str = "Set1",
    colors: Optional[List] = None,
    show_values=True,
    show_legend=False,
    legend_ax: Optional[Axes] = None,
    legend_args: Optional[Dict[str, Any]] = None,
    despine=True,
    show_yticklabels=False,
    show_xticklabels=False,
    row_clusters=None,
    col_clusters=None,
    spaced_heatmap_args=None,
    title=None,
):
    """Categorical heatmap

    Args:
        df: Colors will be assigned to df values in sorting order.
            Columns must have homogeneous types.
        cmap: a cmap name that sns.color_palette understands; ignored if
              colors are given
        colors: List of color specs
        legend_args: if None, defaults to dict(bbox_to_anchor=(-1, 0, 1, 1), loc='best'),
            which is meant to place the legend outside of the categorical heatmap to the left.
            This will need to be adjusted depending on the label length and the width
            of the heatmap
        row_clusters, col_clusters: passed to spaced heatmap. these arguments must be given outside of space_heatmap_args (to allow alignment in dynamic grid manager)


    Does not accept NA values.
    """

    if df.isna().any().any():
        raise ValueError("NA values not allowed")

    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError("All columns must have the same dtype")

    is_categorical = df.dtypes.iloc[0].name == "category"

    if is_categorical:
        levels = df.dtypes.iloc[0].categories.values
    else:
        levels = np.unique(df.values)
    n_levels = len(levels)

    # Colors are repeatedly tiled until all levels are covered
    if colors is None:
        color_list = sns.color_palette(cmap, n_levels)
    else:
        # tile colors to get n_levels color list
        color_list = (np.ceil(n_levels / len(colors)).astype(int) * colors)[:n_levels]
    # noinspection PyUnresolvedReferences
    cmap = mpl.colors.ListedColormap(color_list)

    # Get integer codes matrix for pcolormesh, ie levels are represented by
    # increasing integers according to the level ordering
    if is_categorical:
        codes_df = df.apply(lambda ser: ser.cat.codes, axis=0)
    else:
        value_to_code = {value: code for code, value in enumerate(levels)}
        codes_df = df.replace(value_to_code)

    if row_clusters is not None or col_clusters is not None:
        pargs = spaced_heatmap_args.get("pcolormesh_args", {})
        if "cmap" in pargs:
            print(
                "Warning: cmap is defined via the categorical heatmap argument, not via pcolormesh_args"
            )
        pargs.update(cmap=cmap)
        qm = spaced_heatmap(
            ax,
            codes_df,
            row_clusters=row_clusters,
            col_clusters=col_clusters,
            **spaced_heatmap_args,
        )
    else:
        qm = ax.pcolormesh(codes_df, cmap=cmap)

    if despine:
        sns.despine(ax=ax, bottom=True, left=True)
    if not show_xticklabels:
        ax.set(xticks=[], xticklabels=[])
    else:
        plt.setp(ax.get_xticklabels(), rotation=90)
    if not show_yticklabels:
        ax.set(yticks=[], yticklabels=[])

    # Create dummy patches for legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="black") for c in color_list]
    if show_legend:
        if legend_args is None:
            legend_args = dict(bbox_to_anchor=(-1, 0, 1, 1), loc="best")
        if legend_ax is not None:
            legend_ax.legend(patches, levels, **legend_args)
        else:
            ax.legend(patches, levels, **legend_args)

    if show_values:
        for i in range(df.shape[1]):
            # note: categorical_series.values is not a numpy array
            # any series has __array__ method - so instead of values
            # attribute, use np.array
            y, s = find_stretches(np.array(df.iloc[:, i]))
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va="center", ha="center")

    if title is not None:
        ax.set_title(title)

    return {"quadmesh": qm, "patches": patches, "levels": levels}


def categorical_heatmap2(
    df: pd.DataFrame,
    ax: Axes,
    cmap: Optional[Union[str, List]] = "Set1",
    label_stretches=True,
    legend=False,
    legend_ax: Optional[Axes] = None,
    legend_args: Optional[Dict] = None,
    heatmap_args: Optional[Dict] = None,
    spaced_heatmap_args: Optional[Dict] = None,
) -> Dict:
    """Categorical heatmap

    Args:
        df: Colors will be assigned to df values in sorting order.
            Columns must have homogeneous types. NA values are not allowed.
        ax: heatmap Axes
        cmap: either a cmap name that sns.color_palette understands or a list of color specs acting as color palette
        label_stretches: put centered value labels in each stretch of consecutive values
        legend: whether to add the legend
        legend_ax: optional: second axes where legend should be drawn (instead of adding it to ax)
        legend_args: passed to ax.legend
        heatmap_args: if not None, heatmap will be drawn with co.heatmap
        spaced_heatmap_args: if not None, heatmap will be drawn with co.spaced_heatmap

    Returns:
        Dict with values required for drawing the legend
    """

    # 1. Process input data
    # 2. Pass to either heatmap or spaced_heatmap
    # 3. add specific features of categorical heatmaps
    #    3.1 add stretch labels
    #    3.2 add Legend of Rectangle patches

    # Check inputs
    if df.isna().any().any():
        raise ValueError("NA values not allowed")
    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError("All columns must have the same dtype")
    if (spaced_heatmap_args is not None) + (heatmap_args is not None) != 1:
        raise ValueError("Either spaced_heatmap_args or heatmap_args must be specified")
    if legend_args is None:
        legend_args = {}
    assert isinstance(cmap, (str, list))

    # Process data: turn categories into consecutive integers (codes_df)
    # Tile colors to get vector mapping each category to its color (cmap_listmap)
    is_categorical = df.dtypes.iloc[0].name == "category"
    if is_categorical:
        levels = df.dtypes.iloc[0].categories.values
    else:
        levels = np.unique(df.values)
    n_levels = len(levels)

    # Colors are repeatedly tiled until all levels are covered
    # They may be specified as color palette names or list of colors
    if isinstance(cmap, str):
        color_list = sns.color_palette(cmap, n_levels)
    else:  # List of colors
        color_list = (np.ceil(n_levels / len(cmap)).astype(int) * cmap)[:n_levels]
    # noinspection PyUnresolvedReferences
    cmap_listmap = mpl.colors.ListedColormap(color_list)

    # Get integer codes matrix for pcolormesh, ie levels are represented by
    # increasing integers according to the level ordering
    if is_categorical:
        codes_df = df.apply(lambda ser: ser.cat.codes, axis=0)
    else:
        value_to_code = {value: code for code, value in enumerate(levels)}
        codes_df = df.replace(value_to_code)

    # Call either heatmap or spaced_heatmap on codes_df with cmap_listmap
    fn = heatmap2 if heatmap_args is not None else spaced_heatmap2
    kwargs = heatmap_args if heatmap_args is not None else spaced_heatmap_args
    # continous colorbar for categorical heatmap makes no sense
    kwargs["add_colorbar"] = False
    if "pcolormesh_args" not in kwargs:
        kwargs["pcolormesh_args"] = {}
    elif "cmap" in kwargs["pcolormesh_args"]:
        print(
            "Warning: cmap is defined via the categorical heatmap argument, not via pcolormesh_args"
        )
    kwargs["pcolormesh_args"]["cmap"] = cmap_listmap
    _ = fn(ax=ax, df=codes_df, **kwargs)

    # Some specific features of categorical heatmaps at the end:

    # Legend with proxy artists
    patches = [mpatches.Patch(facecolor=c, edgecolor="black") for c in color_list]
    if legend:
        if legend_args is None:
            legend_args = dict(bbox_to_anchor=(-1, 0, 1, 1), loc="best")
        if legend_ax is not None:
            legend_ax.legend(patches, levels, **legend_args)
        else:
            ax.legend(patches, levels, **legend_args)

    # Label stretches
    if label_stretches:
        for i in range(df.shape[1]):
            # note: categorical_series.values is not a numpy array
            # any series has __array__ method - so instead of values
            # attribute, use np.array
            y, s = find_stretches(np.array(df.iloc[:, i]))
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va="center", ha="center")

    return {"handles": patches, "labels": levels, **legend_args}



def heatmap(
    df: pd.DataFrame,
    ax: Axes,
    is_categorical: bool = False,
    categorical_colors: Optional[List] = None,
    xticklabels: Union[bool, List[str]] = None,
    xticklabel_rotation=90,
    xticklabel_colors: Optional[Union[Iterable, Dict]] = None,
    xticklabel_side="bottom",
    xlabel: Optional[str] = None,
    yticklabels: Union[bool, List[str]] = None,
    yticklabel_colors: Optional[Union[Iterable, Dict]] = None,
    yticklabel_side="right",
    ylabel: Optional[str] = None,
    show_guide=False,
    guide_args: Optional[Dict] = None,
    guide_ax: Optional[Axes] = None,
    guide_title: Optional[str] = None,
    title: Optional[str] = None,
    annotate: Union[str, bool] = False,
    row_spacing_group_ids: Union[np.ndarray, pd.Series] = None,
    col_spacing_group_ids: Union[np.ndarray, pd.Series] = None,
    row_spacer_sizes: Union[float, Iterable[float]] = 0.01,
    col_spacer_sizes: Union[float, Iterable[float]] = 0.01,
    frame_spaced_elements=False,
    frame_kwargs=None,
    cbar_styling_func=cbar_change_style_to_inward_white_ticks,
    cbar_styling_func_kwargs: Optional[Dict] = None,
    categorical_legend_patch_kwargs=None,
    **kwargs,
) -> Dict:
    """General heatmap

    Args:
        df
            Colors will be assigned to df values in sorting order.
            Columns must have homogeneous types. NA values are not allowed.
        ax
            heatmap Axes
        xticklabel_colors
            either iterable of color specs of same length as number of ticklabels, or a dict label -> color. Currenlty does not work with series, just dict.
        yticklabel_colors
            either iterable of color specs of same length as number of ticklabels, or a dict label -> color. Currenlty does not work with series, just dict.
        kwargs
            pcolormesh args
        annotate: False, 'stretches'
            'stretches' currently has a bug
        categorical_legend_patch_kwargs
            passed to Patches created as legend keys for categorical heatmaps
            if None, defaults to dict(linewidth=0)

    Returns:
        Dict with values required for drawing the legend

    Implementation notes:
    - I had to comment out figure.canvas.draw() in color_ticklabels, because this
      line caused the legend not to be drawn in cross_plot. Haven't understood/investigated
      yet. If ticklabels are not colored in the future, this would be the first place
      to troubleshoot.
    - I have no added a fig.canvas.draw prior to the cbar styling func call; this may cause new trouble with missing legends in cross_plot down the road, let's see what happens
    """

    if annotate == "stretches":
        raise ValueError('annotate == "stretches" currently not working')

    if guide_args is None:
        guide_args_copy = {}
    else:
        guide_args_copy = deepcopy(guide_args)

    if kwargs is None:
        pcolormesh_args = {}
    else:
        pcolormesh_args = deepcopy(kwargs)

    if cbar_styling_func_kwargs is None:
        cbar_styling_func_kwargs = {}

    if categorical_legend_patch_kwargs is None:
        categorical_legend_patch_kwargs = dict(linewidth=0)

    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError("All columns must have the same dtype")

    is_categorical = is_categorical or df.dtypes[0].name in ["category", "object"]

    if annotate and not is_categorical:
        raise NotImplementedError()

    if is_categorical:
        if categorical_colors is None:
            categorical_colors = pcolormesh_args.get("cmap", "Set1")
        categorical_colors_listmap, codes_df, levels = _get_categorical_codes_colors(
            categorical_colors, df
        )
        pcolormesh_args["cmap"] = categorical_colors_listmap
        df = codes_df

    if isinstance(xticklabel_colors, pd.Series) or isinstance(
        yticklabel_colors, pd.Series
    ):
        raise ValueError("Ticklabel colors currently only work with dicts")

    shared_args = dict(
        df=df,
        ax=ax,
        pcolormesh_args=pcolormesh_args,
        xticklabels=xticklabels,
        xticklabel_rotation=xticklabel_rotation,
        yticklabels=yticklabels,
        # For categorical, no colorbar, legend is added later on
    )

    if row_spacing_group_ids is not None or col_spacing_group_ids is not None:
        # we need a spaced heatmap
        qm = spaced_heatmap2(
            **shared_args,
            row_clusters=row_spacing_group_ids,
            col_clusters=col_spacing_group_ids,
            row_spacer_size=row_spacer_sizes,
            col_spacer_size=col_spacer_sizes,
            frame_spaced_elements=frame_spaced_elements,
            frame_kwargs=frame_kwargs,
        )
    else:
        # normal heatmap
        qm = heatmap2(**shared_args)

    # the qm is only necessary and allowed if we use a colorbar,
    # ie if the heatmap is not categorical
    if not is_categorical:
        guide_args_copy["mappable"] = qm

    # Axis and tick labels
    if xticklabels:
        if xticklabel_side == "bottom":
            ax.tick_params(labelbottom=True, labeltop=False)
        elif xticklabel_side == "top":
            ax.tick_params(labelbottom=False, labeltop=True)
        else:
            raise ValueError("Unknown xticklabel_side")
    else:
        ax.tick_params(labelbottom=False, labeltop=False)

    if yticklabels:
        if yticklabel_side == "right":
            ax.tick_params(labelright=True, labelleft=False)
        elif yticklabel_side == "left":
            ax.tick_params(labelright=False, labelleft=True)
        else:
            raise ValueError("Unknown yticklabel_side")
    else:
        ax.tick_params(labelright=False, labelleft=False)

    # Color ticklabels
    if xticklabels:
        color_ticklabels("x", xticklabel_colors, ax)
    if yticklabels:
        color_ticklabels("y", yticklabel_colors, ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        if yticklabel_side == "right":
            ax.yaxis.set_label_position("right")
        else:
            ax.yaxis.set_label_position("left")
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # General aesthetics
    ax.tick_params(length=0, which="both", axis="both")
    sns.despine(ax=ax, bottom=True, left=True)

    # add colorbar for continuous data (if requested)
    if not is_categorical and show_guide:
        cb = ax.figure.colorbar(ax=ax, **guide_args_copy)
        # cb.outline.set_linewidth(0)
        # cb.ax.tick_params(length=0, which="both", axis="both")
        ax.figure.canvas.draw()
        cbar_styling_func(cb, **cbar_styling_func_kwargs)
        if guide_title is not None:
            cb.set_label(guide_title)

    # Done with continous data, return the result
    if not is_categorical:
        # return guide_args_copy, it will be used as legend spec, therefore add title
        guide_args_copy["title"] = guide_title
        guide_args_copy["styling_func_kwargs"] = cbar_styling_func_kwargs
        return guide_args_copy

    # This is a cateogrical heatmap, take care of the legend
    # Unlike fig.colorbar, Legend takes a title arg
    guide_args_copy["title"] = guide_title

    # Legend with proxy artists
    # noinspection PyUnboundLocalVariable
    patches = [mpatches.Patch(facecolor=c) for c in categorical_colors_listmap.colors]
    if show_guide:
        if guide_ax is not None:
            # noinspection PyUnboundLocalVariable
            guide_ax.legend(
                patches,
                levels,
                **guide_args_copy,
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
            )
        else:
            # noinspection PyUnboundLocalVariable
            ax.legend(
                patches,
                levels,
                **guide_args_copy,
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
            )

    # Label stretches
    if annotate == "stretches":
        for i in range(df.shape[1]):
            # note: categorical_series.values is not a numpy array
            # any series has __array__ method - so instead of values
            # attribute, use np.array
            y, s = find_stretches(np.array(df.iloc[:, i]))
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va="center", ha="center")

    return {"handles": patches, "labels": levels, **guide_args_copy}


def color_ticklabels(axis: str, colors: Union[Iterable, Dict], ax: Axes) -> None:
    """Mark tick labels with different colors

    Args:
        axis: 'x' or 'y'
        colors: either iterable of color specs of same length as number of ticklabels, or a dict label -> color
        ax: single Axes
    """

    # Implementation NOTE:
    # ax.figure.canvas.draw() caused problems when using heatmap with cross_plot
    # (the legend Axes was not shown - not sure what happened exactly)
    # I comment this part out for now - why was it necessary?
    axis = [ax.xaxis, ax.yaxis][axis == "y"]
    if isinstance(colors, dict):
        # ax.figure.canvas.draw()
        for tick in axis.get_ticklabels():
            tick.set_color(colors[tick.get_text()])
    elif colors is not None:
        # we expect an iterable, with one color per ticklabel
        # ax.figure.canvas.draw()
        for tick, tick_color in zip(axis.get_ticklabels(), colors):
            tick.set_color(tick_color)


def _get_categorical_codes_colors(categorical_colors, df):
    # Check inputs
    if df.isna().any().any():
        raise ValueError("NA values not allowed")
    # Process data: turn categories into consecutive integers (codes_df)
    # Tile colors to get vector mapping each category to its color (cmap_listmap)
    is_categorical = df.dtypes.iloc[0].name == "category"
    if is_categorical:
        levels = df.dtypes.iloc[0].categories.values
    else:
        levels = np.unique(df.values)
    n_levels = len(levels)
    # Colors are repeatedly tiled until all levels are covered
    # They may be specified as color palette names or list of colors
    if isinstance(categorical_colors, str):
        color_list = sns.color_palette(categorical_colors, n_levels)
    else:  # List of colors
        color_list = (
            np.ceil(n_levels / len(categorical_colors)).astype(int) * categorical_colors
        )[:n_levels]
    # noinspection PyUnresolvedReferences
    categorical_colors_listmap = mpl.colors.ListedColormap(color_list)
    # Get integer codes matrix for pcolormesh, ie levels are represented by
    # increasing integers according to the level ordering
    if is_categorical:
        codes_df = df.apply(lambda ser: ser.cat.codes, axis=0)
    else:
        value_to_code = {value: code for code, value in enumerate(levels)}
        codes_df = df.replace(value_to_code)
    return categorical_colors_listmap, codes_df, levels


# Note: jit compilation does not seem to provide speed-ups. Haven't
# checked it out yet.
# @numba.jit(nopython=True)
def find_stretches(arr):
    """Find stretches of equal values in ndarray"""
    assert isinstance(arr, np.ndarray)
    stretch_value = arr[0]
    start = 0
    end = 1
    marks = np.empty(arr.shape, dtype=np.float64)
    values = np.empty_like(arr)
    mark_i = 0
    for value in arr[1:]:
        if value == stretch_value:
            end += 1
        else:
            marks[mark_i] = start + (end - start) / 2
            values[mark_i] = stretch_value
            start = end
            end += 1
            stretch_value = value
            mark_i += 1
    # add last stretch
    marks[mark_i] = start + (end - start) / 2
    values[mark_i] = stretch_value
    return marks[: (mark_i + 1)], values[: (mark_i + 1)]


# Note: jit compilation does not seem to provide speed-ups. Haven't
# checked it out yet.
# @numba.jit(nopython=True)
def find_stretches2(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find stretches of equal values in ndarray

    Each element is assumed to have unit size in data coordinates

    Returns multiple arrays
        - boundaries,  stretches go from 0:b[0], b[0]:b[1]
        - middle points: coordinate of middle of stretches
        - values: values for each stretch
    """
    arr = np.array(arr)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    stretch_value = arr[0]
    start = 0
    end = 1
    middle_marks = np.empty(arr.shape, dtype=np.float64)
    end_marks = np.empty(arr.shape, dtype=np.float64)
    values = np.empty_like(arr)
    mark_i = 0
    for value in arr[1:]:
        if value == stretch_value:
            end += 1
        else:
            end_marks[mark_i] = end
            middle_marks[mark_i] = start + (end - start) / 2
            values[mark_i] = stretch_value
            start = end
            end += 1
            stretch_value = value
            mark_i += 1
    # add last stretch
    middle_marks[mark_i] = start + (end - start) / 2
    end_marks[mark_i] = end
    values[mark_i] = stretch_value
    return (
        end_marks[: (mark_i + 1)],
        middle_marks[: (mark_i + 1)],
        values[: (mark_i + 1)],
    )


# From matplotlib docs:
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# TODO:
# - figure does not need to be given
# - document
#   - cbar is added to ax
#   - if labels are not given, take them from the dataframe
# noinspection PyIncorrectDocstring
def heatmap_depr(
    df: pd.DataFrame,
    ax: Axes,
    fig: Figure,
    cmap: str,
    # midpoint_normalize: bool =  False,
    col_labels_show: bool = True,
    row_labels_show: bool = False,
    tick_length=0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    row_labels_rotation=90,
    add_colorbar=True,
    cbar_args: Optional[Dict] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """Simple heatmap plotter

    Args:
        kwargs: passed to pcolormesh. E.g. to pass a normalization
    """

    # if midpoint_normalize:
    #     norm: Optional[mpl.colors.Normalize] = MidpointNormalize(
    #             vmin=df.min().min(), vmax=df.max().max(), midpoint=0)
    # else:
    #     norm = None

    # qm = ax.pcolormesh(df, cmap=cmap, norm=norm, **kwargs)

    if cbar_args is None:
        cbar_args = {}

    qm = ax.pcolormesh(df, cmap=cmap, **kwargs)

    ax.tick_params(length=tick_length, which="both", axis="both")

    if col_labels_show:
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_xticklabels(df.columns, rotation=row_labels_rotation)
    else:
        ax.set_xticks([])
        ax.tick_params(length=0, which="both", axis="x")

    if row_labels_show:
        ax.set_yticks(np.arange(df.shape[0]) + 0.5)
        ax.set_yticklabels(df.index)
    else:
        ax.set_yticks([])
        ax.tick_params(length=0, which="both", axis="y")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if add_colorbar:
        cb = fig.colorbar(qm, ax=ax, **cbar_args)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(length=3.5, width=0.5, which="both", axis="both")


def simple_line(ax):
    ax.plot([1, 2, 3, 4, 5])


def cluster_size_plot(
    cluster_ids: pd.Series,
    ax: Axes,
    bar_height=0.6,
    xlabel=None,
    invert_xaxis=False,
    cmap=None,
    color="gray",
):
    """Horizontal cluster size barplot

    Currently, this assumes that the cluster IDs can be sorted directly
    to get the correct ordering desired in the plot. This means that
    either:
    1. The cluster IDs are numerical OR
    2. The cluster IDs are ordered categorical OR
    3. The cluster IDs are strings, which can be ordered correctly *by
       alphabetic sorting*

    Args:
        cluster_ids: index does not have to be the same as for the dfs
            used in a ClusterProfile plot. This may change in the future.
            Best practice is to use the same index for the data and the cluster
            IDs.
        bar_height: between 0 and 1
        xlabel: x-Axis label for the plot
        invert_xaxis: will invert xaxis
    """
    cluster_sizes = cluster_ids.value_counts()
    cluster_sizes.sort_index(inplace=True)
    if cmap is not None:
        color = sns.color_palette(cmap, len(cluster_sizes))
    else:
        color = color
    ax.barh(
        y=np.arange(0.5, cluster_sizes.shape[0]),
        width=cluster_sizes.values,
        height=bar_height,
        color=color,
    )
    ax.set_ylim(0, cluster_sizes.shape[0])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    remove_axis(ax, x=False)
    if invert_xaxis:
        ax.invert_xaxis()


def col_agg_plot(df: pd.DataFrame, fn: Callable, ax: Axes, xlabel=None):
    """Plot aggregate statistic as line plot

    Args:
        fn: function which will be passed to pd.DataFrame.agg
    """
    agg_stat = df.agg(fn)
    ax.plot(np.arange(0.5, df.shape[1]), agg_stat.values)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def row_group_agg_plot(
    data: pd.DataFrame,
    fn,
    row: Optional[Union[str, Sequence]],
    ax: List[Axes],
    show_all_xticklabels=False,
    xlabel=None,
    ylabel=None,
    sharey=True,
    ylim=None,
    hlines=None,
    plot_args: Optional[Dict] = None,
    hlines_args: Optional[Dict] = None,
):

    axes = ax
    used_plot_args = dict(marker="o", linestyle="-", color="black", linewidth=None)
    if plot_args is not None:
        used_plot_args.update(plot_args)

    if isinstance(row, str):
        if row in data:
            levels = data[row].unique()
        elif row in data.index.names:
            levels = data.index.get_level_values(row).unique()
        else:
            raise ValueError()
    else:
        levels = pd.Series(row).unique()

    agg_values = data.groupby(row).agg(fn).loc[levels, :]
    if sharey and ylim is None:
        ylim = compute_padded_ylim(agg_values)

    ncol = data.shape[1]
    x = np.arange(0.5, ncol)
    xlim = (0, ncol)
    for curr_ax, (group_name, agg_row) in zip(axes[::-1], agg_values.iterrows()):
        curr_ax.plot(x, agg_row.values, **used_plot_args)
        curr_ax.set_xlim(xlim)
        if ylim is not None:
            curr_ax.set_ylim(ylim)
        if not show_all_xticklabels:
            curr_ax.set(xticks=[], xticklabels=[], xlabel="")
        else:
            curr_ax.set_xticks(x)
            curr_ax.set_xticklabels(data.columns, rotation=90)
            curr_ax.set_xlabel("")
        curr_ylabel = (
            str(group_name) if not ylabel else str(group_name) + "\n\n" + ylabel
        )
        curr_ax.set_ylabel(curr_ylabel)

        if hlines:
            for h_line in hlines:
                curr_ax.axhline(h_line, **hlines_args)
        sns.despine(ax=curr_ax)

    # Add xticks and xticklabels to the bottom Axes
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(data.columns, rotation=90)
    axes[-1].set_xlabel(xlabel if xlabel is not None else "")


def compute_padded_ylim(df):
    ymax = np.max(df.values)
    pad = abs(0.1 * ymax)
    padded_ymax = ymax + pad
    if ymax < 0 < padded_ymax:
        padded_ymax = 0
    ymin = np.min(df.values)
    padded_ymin = ymin - pad
    if ymin > 0 > padded_ymin:
        padded_ymin = 0
    ylim = (padded_ymin, padded_ymax)
    return ylim


def grouped_rows_violin(
    data: pd.DataFrame,
    row: Union[str, Iterable],
    ax: List[Axes],
    n_per_group=2000,
    sort=False,
    sharey=False,
    ylim=None,
    show_all_xticklabels=False,
    xlabel=None,
) -> None:
    # axes list called ax because currently only ax kwarg recognized by Grid
    # currently, axes are given in top down order, but we want to fill them
    # in bottom up order (aka the pcolormesh plotting direction)
    axes = ax[::-1]
    grouped: GroupBy = data.groupby(row)
    levels = get_groupby_levels_in_order_of_appearance(data, row)
    if sort:
        levels = np.sort(levels)
    if sharey and ylim is None:
        ylim = compute_padded_ylim(data)

    # seaborn violin places center of first violin above x=0
    xticks = np.arange(0, data.shape[1])
    xticklabels = data.columns.values
    for curr_ax, curr_level in zip(axes, levels):
        group_df: pd.DataFrame = sample_n(grouped.get_group(curr_level), n_per_group)
        group_df.columns.name = "x"
        long_group_df = group_df.stack().to_frame("y").reset_index()
        sns.violinplot(x="x", y="y", data=long_group_df, ax=curr_ax)
        if ylim:
            curr_ax.set_ylim(ylim)
        curr_ax.set_xlabel("")
        if not show_all_xticklabels:
            curr_ax.set(xticks=[], xticklabels=[], xlabel="")
        else:
            # sns.violinplot already sets the same labels (data.columns.values),
            # but does not rotate them
            curr_ax.set_xticklabels(xticklabels, rotation=90)
            curr_ax.set_xlabel("")

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=90)
    axes[0].set_xlabel("" if xlabel is None else xlabel)


def grouped_rows_line_collections(
    data: pd.DataFrame,
    row: Union[str, Iterable],
    ax: List[Axes],
    n_per_group=2000,
    sort=False,
    sharey=False,
    ylim=None,
    show_all_xticklabels=False,
    xlabel=None,
    alpha=0.01,
):
    axes = ax[::-1]
    grouped: GroupBy = data.groupby(row)
    levels = get_groupby_levels_in_order_of_appearance(data, row)
    if sort:
        levels = np.sort(levels)
    if ylim is not None:
        shared_ylim = ylim
    elif sharey:
        shared_ylim = compute_padded_ylim(data)
    else:
        shared_ylim = None

    xticks = np.arange(0.5, data.shape[1])
    xticklabels = data.columns.values

    for curr_ax, curr_level in zip(axes, levels):
        group_df: pd.DataFrame = sample_n(grouped.get_group(curr_level), n_per_group)
        segments = np.zeros(group_df.shape + (2,))
        segments[:, :, 1] = group_df.values
        segments[:, :, 0] = np.arange(0.5, group_df.shape[1])
        line_collection = LineCollection(segments, color="black", alpha=alpha)
        curr_ax.add_collection(line_collection)

        # need to set plot limits, they will not autoscale
        curr_ax.set_xlim(0, data.shape[1])
        if shared_ylim is None:
            curr_ylim = compute_padded_ylim(group_df)
        else:
            curr_ylim = shared_ylim
        curr_ax.set_ylim(curr_ylim)

        if show_all_xticklabels:
            curr_ax.set_xticks(xticks)
            curr_ax.set_xticklabels(xticklabels, rotation=90)
            curr_ax.set_xlabel("")
        else:
            curr_ax.set(xticks=[], xticklabels=[], xlabel="")

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=90)
    axes[0].set_xlabel(xlabel if xlabel is not None else "")


def sample_n(df, n):
    if df.shape[0] < 50:
        print("Warning: less than 50 elements for calculating violin plot")
    if df.shape[0] <= n:
        return df
    return df.sample(n=n)


def get_groupby_levels_in_order_of_appearance(df, groupby_var) -> np.ndarray:
    if isinstance(groupby_var, str):
        if groupby_var in df:
            levels = df[groupby_var].unique()
        elif groupby_var in df.index.names:
            levels = df.index.get_level_values(groupby_var).unique()
        else:
            raise ValueError(f"Can not find {groupby_var} in df")
    else:
        levels = pd.Series(groupby_var).unique()
    return levels


@dataclass
class CutDendrogram:
    """Plot dendrogram with link filtering, coloring and downsampling"""

    Z: np.ndarray
    ax: Axes
    cluster_ids_data_order: Optional[pd.Series] = None
    # spacing groups do not need to match cluster ids, eg a spacing group may
    # contain multiple clusters
    # in data order
    spacing_groups: Optional[np.ndarray] = None
    spacer_size: Optional[float] = 0.02
    pretty: bool = True
    orientation: str = "horizontal"

    def __post_init__(self):
        self._args_processed = False

    def plot_links_until_clusters(self, color="black"):

        if not self._args_processed:
            self._process_args()

        def get_cluster_ids(obs_idx, link_cluster_ids, leave_cluster_ids):
            n_obs = leave_cluster_ids.shape[0]
            if obs_idx < n_obs:
                return leave_cluster_ids[obs_idx]
            link_idx = obs_idx - n_obs
            return link_cluster_ids[link_idx]

        for i in range(-1, -self.Z.shape[0] - 1, -1):
            if self.link_cluster_ids[i] != -1:
                continue
            x, y = self.xcoords.iloc[i, :].copy(), self.ycoords.iloc[i, :].copy()
            left_child_idx, right_child_idx = self.Z_df[
                ["left_child", "right_child"]
            ].iloc[i]
            left_cluster_id = get_cluster_ids(
                left_child_idx, self.link_cluster_ids, self.cluster_ids_data_order
            )
            right_cluster_id = get_cluster_ids(
                right_child_idx, self.link_cluster_ids, self.cluster_ids_data_order
            )
            if left_cluster_id != -1:
                y["ylow_left"] = 0
            if right_cluster_id != -1:
                y["ylow_right"] = 0

            if self.orientation == "horizontal":
                self.ax.plot(y, x, color=color)
            else:
                self.ax.plot(x, y, color=color)

        self._plot_post_processing()

    def plot_links_with_cluster_inspection(
        self,
        show_cluster_points: bool = False,
        point_params: Optional[Dict] = None,
        min_cluster_size: Optional[int] = None,
        min_height: str = "auto",
        colors: Union[str, List, Dict] = "Set1",
        no_member_color: [Tuple[int], str] = "black",
    ):
        """

        Args:
            show_cluster_points:
            point_params:
            min_cluster_size:
            min_height:
            colors: either name of cmap, list of color specs or dict cluster_id -> color spec; should not contain cluster id -1 (no membership), this is handled by no_member_color; list of color specs must have one entry per cluster id
            no_member_color: color for links which connect clusters (which are not a member in a single cluster)

        Returns:

        """

        if not self._args_processed:
            self._process_args()

        if point_params is None:
            point_params = {"s": 2, "marker": "x", "c": "black"}
        if min_height == "auto":
            min_height = self._linkage_estimate_min_height()
        if min_cluster_size is None:
            min_cluster_size = self.n_leaves * 0.02

        # Annotate each link with its color
        # 1. we want a dict cluster_id -> color spec
        sorted_unique_cluster_ids = np.setdiff1d(np.unique(self.link_cluster_ids), [-1])
        n_clusters = len(sorted_unique_cluster_ids)
        if isinstance(colors, dict):
            # assert that colors dict contains all cluster ids and nothing else
            assert np.all(np.sort(list(colors.keys())) == sorted_unique_cluster_ids)
            clusterid_color_d = colors
        elif isinstance(colors, str):
            clusterid_color_d = dict(
                zip(sorted_unique_cluster_ids, sns.color_palette(colors, n_clusters))
            )
        elif isinstance(colors, list):
            assert len(colors) == n_clusters
            clusterid_color_d = dict(zip(sorted_unique_cluster_ids, colors))
        else:
            raise TypeError("colors has inappropriate type")
        # 2. links without cluster id (annotate with id -1) get no_member_color
        clusterid_color_d[-1] = no_member_color
        # 3. map link_cluster_ids to their colors using the color dict
        link_colors = pd.Series(self.link_cluster_ids).map(clusterid_color_d)

        # cluster_ids_ser, link_colors = linkage_get_link_cluster_ids(Z, cluster_ids_ser)
        # ys = []
        # _linkage_get_child_y_coords(Z, ys, Z.shape[0] * 2, 3)
        point_xs = []
        point_ys = []
        for i in range(self.Z.shape[0]):
            x = self.xcoords.loc[i, :].copy()
            y = self.ycoords.loc[i, :].copy()

            if self.Z[i, 3] < min_cluster_size:
                continue
            if y["yhigh1"] < min_height:
                continue

            try:
                left_child_size = self.Z[int(self.Z[i, 0]) - self.n_leaves, 3]
            except IndexError:
                left_child_size = 1
            try:
                right_child_size = self.Z[int(self.Z[i, 1]) - self.n_leaves, 3]
            except IndexError:
                right_child_size = 1

            if (
                left_child_size < min_cluster_size
                and right_child_size < min_cluster_size
            ):
                y["ylow_left"] = 0
                y["ylow_right"] = 0
                if self.orientation == "vertical":
                    self.ax.plot(x, y, color=link_colors[i])
                else:
                    self.ax.plot(y, x, color=link_colors[i])

            else:
                if y["ylow_left"] < min_height or left_child_size < min_cluster_size:
                    y["ylow_left"] = 0

                    curr_point_x = x["xleft1"]
                    obs_idx = self.Z[i, 0]

                    # if obs_idx > self.Z.shape[0]:
                    lookup_idx = int(obs_idx - (self.Z.shape[0] + 1))
                    point_ys.append(self.Z[lookup_idx, 2])
                    point_xs.append(curr_point_x)
                    # curr_size = self.Z[lookup_idx, 3]
                    curr_ys: List[float] = []
                    curr_ys = self._linkage_get_child_y_coords(curr_ys, obs_idx, 8)
                    point_ys += curr_ys
                    point_xs += [curr_point_x] * len(curr_ys)
                    # else:
                    #     curr_size = 1

                if y["ylow_right"] < min_height or right_child_size < min_cluster_size:
                    y["ylow_right"] = 0
                    curr_point_x = x["xright1"]
                    obs_idx = self.Z[i, 1]
                    if obs_idx > self.Z.shape[0]:
                        lookup_idx = int(obs_idx - (self.Z.shape[0] + 1))
                        point_ys.append(self.Z[lookup_idx, 2])
                        point_xs.append(curr_point_x)
                        curr_ys = []
                        curr_ys = self._linkage_get_child_y_coords(curr_ys, obs_idx, 4)
                        point_ys += curr_ys
                        point_xs += [curr_point_x] * len(curr_ys)

                if self.orientation == "vertical":
                    self.ax.plot(x, y, color=link_colors[i])
                else:
                    self.ax.plot(y, x, color=link_colors[i])

            if show_cluster_points:
                if self.orientation == "vertical":
                    self.ax.scatter(point_xs, point_ys, **point_params, zorder=20000)
                else:
                    self.ax.scatter(point_ys, point_xs, **point_params, zorder=20000)

        self._plot_post_processing()

    def plot_all_links(self, color="black"):
        if not self._args_processed:
            self._process_args()
        for i in range(self.Z.shape[0]):
            x = self.xcoords.loc[i, :].copy()
            y = self.ycoords.loc[i, :].copy()
            if self.orientation == "vertical":
                self.ax.plot(x, y, color=color)
            else:
                self.ax.plot(y, x, color=color)
        self._plot_post_processing()

    def _process_args(self):
        self.n_leaves = self.Z.shape[0] + 1
        if self.cluster_ids_data_order is not None:
            cluster_ids = ClusterIDs(
                self.cluster_ids_data_order.to_frame("clustering1")
            )
        else:
            cluster_ids = None
        linkage_mat = Linkage(self.Z, cluster_ids=cluster_ids)
        self.Z_df = linkage_mat.df
        self.xcoords, self.ycoords = self._linkage_get_coord_dfs()
        if self.cluster_ids_data_order is not None:
            self.link_cluster_ids = linkage_mat.get_link_cluster_ids("clustering1")
            self.Z_df["cluster_ids"] = self.link_cluster_ids

        assert self.orientation in ["horizontal", "vertical"]

        if self.orientation == "horizontal":
            self.ax.invert_xaxis()
            #     self.ax.invert_yaxis()
            self.ax.margins(0)

        self._args_processed = True

    def _plot_post_processing(self):
        """Orientation, axis limits, despine..."""

        # Set axis limits
        if self.spacing_groups is not None:
            # the min and max links will be a little larger and smaller than 0 and 1 resp.
            # for correct alignment, set correct 0,1  limits
            if self.orientation == "horizontal":
                self.ax.set_ylim(0, 1)
            else:
                self.ax.set_xlim(0, 1)
        else:
            # Final xcoord is in middle of last column, need to add 5 to get end of the column
            # (because xcoord is in 'column integer coordinates' * 10, its returned like that by the scipy dendrogram function
            data_axis_right_lim = self.xcoords.max().max() + 5
            xcoord_axis = "x" if self.orientation == "vertical" else "y"
            self.ax.set(**{f"{xcoord_axis}lim": [0, data_axis_right_lim]})

        # margins
        # data axis margin should be zero for correct alignment with heatmap
        # distance axis margin should be sufficient to not cut the top dendrogram link,
        # so needs to be > 0, because the top dendrogram link is plotted directly
        # on the axis limit value, and the line is centered at the coordinate
        # (ie half of the line lies beyond the coordinate)

        # leave distance_axis margin as is for now, it appears to be set correctly automatically
        if self.orientation == "horizontal":
            data_axis = "y"
            # distance_axis = 'x'
        else:
            data_axis = "x"
            # distance_axis = 'y'
        self.ax.margins(**{data_axis: 0})

        if self.pretty:
            self.ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
            sns.despine(ax=self.ax, left=True, bottom=True)

    def _linkage_estimate_min_height(self):
        min_cluster_height = (
            self.Z_df.groupby("cluster_ids")["height"].max().drop((-1.0)).min()
        )
        return 0.7 * min_cluster_height

    def _linkage_get_child_y_coords(self, ys, obs_idx, n_children):
        n = self.Z.shape[0] + 1
        n_children -= 1
        lookup_idx = int(obs_idx - n)
        left_obs_idx = int(self.Z[lookup_idx, 0])
        if left_obs_idx > n:
            ys.append(self.Z[left_obs_idx - n, 2])
            if n_children > 0:
                _ = self._linkage_get_child_y_coords(ys, obs_idx, n_children)
        right_obs_idx = int(self.Z[lookup_idx, 1])
        if right_obs_idx > n:
            ys.append(self.Z[right_obs_idx - n, 2])
            if n_children > 0:
                _ = self._linkage_get_child_y_coords(ys, right_obs_idx, n_children)
        return ys

    def _linkage_get_coord_dfs(self):
        dendrogram_dict = dendrogram(self.Z, no_plot=True)
        dcoords = pd.DataFrame.from_records(dendrogram_dict["dcoord"])
        dcoords.columns = ["ylow_left", "yhigh1", "yhigh2", "ylow_right"]
        dcoords = dcoords.sort_values("yhigh1")
        icoords = pd.DataFrame.from_records(dendrogram_dict["icoord"])
        icoords.columns = ["xleft1", "xleft2", "xright1", "xright2"]
        icoords = icoords.loc[dcoords.index, :]
        # x coords are observation coords
        xcoords = icoords.reset_index(drop=True)
        if self.spacing_groups is not None:
            # xcoords are in units of row number * 10,
            # eg a link aligned with row 9 would be at x = 95 (in the middle of the row)
            # there are two coords, each in duplicate cols
            # only one of the cols contains the outmost x coordinate, and therefore
            # the information about max_coord required for correct adjustment for spacers
            # thus, max_coord must be calculated and passed to all column applies
            xcoords = xcoords.divide(10).apply(
                adjust_coords,
                spacing_group_ids=self.spacing_groups[leaves_list(self.Z)],
                spacer_sizes=self.spacer_size,
            )
        # y coords are link height coords
        ycoords = dcoords.reset_index(drop=True)
        return xcoords, ycoords


def cut_dendrogram(
    linkage_mat: np.ndarray,
    cluster_ids_data_order: pd.Series,
    ax: Axes,
    colors: Union[str, List, Dict] = "Set1",
    base_color: [Tuple[int], str] = "black",
    spacing_groups: Optional[np.ndarray] = None,
    spacer_size: float = 0.02,
    pretty: bool = True,
    stop_at_cluster_level=True,
    orientation: str = "horizontal",
    show_cluster_points: bool = False,
    point_params: Optional[Dict] = None,
    min_cluster_size: Optional[int] = None,
    min_height: str = "auto",
):
    cut_dendrogram = CutDendrogram(
        Z=linkage_mat,
        cluster_ids_data_order=cluster_ids_data_order,
        ax=ax,
        pretty=pretty,
        orientation=orientation,
        spacing_groups=spacing_groups,
        spacer_size=spacer_size,
    )
    if cluster_ids_data_order is None:
        cut_dendrogram.plot_all_links(color=base_color)
    elif stop_at_cluster_level:
        cut_dendrogram.plot_links_until_clusters(color=base_color)
    else:
        cut_dendrogram.plot_links_with_cluster_inspection(
            show_cluster_points=show_cluster_points,
            point_params=point_params,
            min_cluster_size=min_cluster_size,
            min_height=min_height,
            colors=colors,
            no_member_color=base_color,
        )


def grouped_rows_heatmap(
    df: pd.DataFrame,
    row_: Union[str, Iterable],
    fn: Union[str, Callable],
    cmap: str,
    ax: Axes,
    fig=Figure,
    sort=False,
    **kwargs,
):
    agg_df = df.groupby(row_).agg(fn)
    levels = get_groupby_levels_in_order_of_appearance(df, row_)
    if sort:
        levels = np.sort(levels)
    agg_df = agg_df.loc[levels, :]
    heatmap_depr(df=agg_df, ax=ax, fig=fig, cmap=cmap, **kwargs)

    # xticks = np.arange(0.5, df.shape[1])
    # xticklabels = df.columns.values
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels, rotation=90)
    # ax.set_xlabel(xlabel if xlabel is not None else '')


# noinspection PyUnboundLocalVariable
def spaced_heatmap(
    ax,
    df,
    row_clusters: Union[np.ndarray, pd.Series] = None,
    col_clusters: Union[np.ndarray, pd.Series] = None,
    row_spacer_size: Union[float, Iterable[float]] = 0.01,
    col_spacer_size: Union[float, Iterable[float]] = 0.01,
    pcolormesh_args=None,
    show_row_labels=False,
    show_col_labels=False,
    rotate_col_labels=True,
    add_colorbar=False,
    cbar_args=None,
    fig=None,
    title=None,
    frame_spaced_elements=False,
    frame_kwargs=None,
) -> mpl.collections.QuadMesh:
    """Plot pcolormesh with spacers between groups of rows and columns

    Spacer size can be specified for each spacer individually, or with
    one size for all row spacers and one size for all col spacers.

    To control the colormap, specify pcolormesh_args, (cmap, vmin, vmax, norm)
    if neither vmin+vmax or norm is specified, vmin and vmax are the 0.02 and 0.98 quantiles of the data.
    to avoid vmin/vmax determination, you can also set vmin=None, vmax=None (eg for a categorical heatmap)

    Args:
        ax: Axes to plot on
        df: pd.DataFrame, index and columns will be used for labeling
        row_clusters: one id per row, specifies which columns to group together. Each group must consist of a single, consecutive stretch of rows. Any dtype is allowed.
        col_clusters: same as row_clusters
        row_spacer_size: if float, specifies the size for a single spacer, which will be used for each individual spacer.  If List[float] specifies size for each spacer in order.
            Size is given as fraction of the Axes width.
        col_spacer_size: Same as row_spacer_size, size is given as fraction of the Axis height
        pcolormesh_args: if the colormapping is not specified via either vmin+vmax or norm, the 0.02 and 0.98 percentiles of df will be used as vmin and vmax.
        fig: if add_colorbar == True, fig must specified, otherwise it is ignored
        frame_spaced_elements: if True, draw a rectangle around each individual
            heatmap. This is only implemented for row-clusters only at the moment (experimental!) and is ignored in other cases
        frame_kwargs: passed to patches.Rectangle

    Returns:
        quadmesh, for use in colorbar plotting etc.
    """

    # Argument checking and pre-processing
    if row_clusters is None and col_clusters is None:
        raise TypeError("You must specify at least one of {row_cluster, col_cluster}")
    elif row_clusters is not None and not isinstance(
        row_clusters, (pd.Series, np.ndarray)
    ):
        raise TypeError(
            "Cluster ids must be given as array or series, not e.g. DataFrame"
        )
    elif col_clusters is not None and not isinstance(
        col_clusters, (pd.Series, np.ndarray)
    ):
        raise TypeError(
            "Cluster ids must be given as array or series, not e.g. DataFrame"
        )
    if pcolormesh_args is None:
        pcolormesh_args = {}
    if add_colorbar:
        if cbar_args is None:
            cbar_args = dict(shrink=0.4, aspect=20)
        if fig is None:
            raise ValueError("If add_colorbars == True, fig must be specified")

    # if df has a multiindex, flatten it to get labels usable for plotting
    if isinstance(df.index, pd.MultiIndex):
        df.index = [" | ".join(str(s) for s in t) for t in df.index]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(s) for s in t) for t in df.columns]

    if row_clusters is not None:
        (
            row_cluster_ids,
            row_cluster_unique_idx,
            row_cluster_nelem_per_cluster,
        ) = np.unique(row_clusters, return_counts=True, return_index=True)
        row_cluster_id_order = np.argsort(row_cluster_unique_idx)
        row_cluster_nelem_per_cluster = row_cluster_nelem_per_cluster[
            row_cluster_id_order
        ]
        row_cluster_ids = row_cluster_ids[row_cluster_id_order]
        row_cluster_cumsum_nelem_prev_clusters = np.insert(
            np.cumsum(row_cluster_nelem_per_cluster), 0, 0
        )
        row_cluster_nclusters = len(row_cluster_ids)

        if isinstance(row_spacer_size, float):
            # all row spacers have the same size
            # the total fraction of the Axes height reserved for row spacers is then:
            total_row_spacer_frac = row_spacer_size * (row_cluster_nclusters - 1)
            # a vector with the size for each individual row spacer, after the last cluster, there is a row spacer with '0' width
            row_spacers_with_0_end = np.repeat(
                (row_spacer_size, 0), (row_cluster_nclusters - 1, 1)
            )
        else:
            # there is one size per spacer, assert the correct length of the iterable
            assert len(row_spacer_size) == row_cluster_nclusters - 1
            # the total fraction of the Axes height reserved for row spacers is then:
            total_row_spacer_frac = np.sum(row_spacer_size)
            # a vector with the size for each individual row spacer, after the last cluster, there is a row spacer with '0' width
            row_spacers_with_0_end = np.append(row_spacer_size, 0)

        # get the start and end positions for each quadmesh block (space in between will be left 'unplotted' to create the spacers)
        # Fraction of all elements contained in the different clusters
        row_cluster_rel_size_for_each_cluster = row_cluster_nelem_per_cluster / np.sum(
            row_cluster_nelem_per_cluster
        )
        # Compute the fraction of axes occupied by each cluster, together with its subsequent spacer
        row_cluster_axes_fraction_for_each_cluster = (
            1 - total_row_spacer_frac
        ) * row_cluster_rel_size_for_each_cluster
        row_cluster_axes_fraction_for_each_cluster_with_spacer = (
            row_cluster_axes_fraction_for_each_cluster + row_spacers_with_0_end
        )
        # Now we can calculate the start and end of each quadmesh block
        row_cluster_start_axis_fraction = np.cumsum(
            np.insert(row_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
        )[:-1]
        row_cluster_end_axis_fraction = (
            np.cumsum(
                np.insert(row_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
            )[1:]
            - row_spacers_with_0_end
        )

    # see analogous row_spacer code block for comments
    if col_clusters is not None:
        (
            col_cluster_ids,
            col_cluster_unique_idx,
            col_cluster_nelem_per_cluster,
        ) = np.unique(col_clusters, return_counts=True, return_index=True)
        col_cluster_id_order = np.argsort(col_cluster_unique_idx)
        col_cluster_nelem_per_cluster = col_cluster_nelem_per_cluster[
            col_cluster_id_order
        ]
        col_cluster_ids = col_cluster_ids[col_cluster_id_order]
        col_cluster_cumsum_nelem_prev_clusters = np.insert(
            np.cumsum(col_cluster_nelem_per_cluster), 0, 0
        )
        col_cluster_nclusters = len(col_cluster_ids)
        if isinstance(col_spacer_size, float):
            total_col_spacer_frac = col_spacer_size * (col_cluster_nclusters - 1)
            col_spacers_with_0_end = np.repeat(
                (col_spacer_size, 0), (col_cluster_nclusters - 1, 1)
            )
        else:
            assert len(col_spacer_size) == col_cluster_nclusters - 1
            total_col_spacer_frac = np.sum(col_spacer_size)
            col_spacers_with_0_end = np.append(col_spacer_size, 0)

        # see analogous row cluster handling code above for comments
        col_cluster_rel_size_for_each_cluster = col_cluster_nelem_per_cluster / np.sum(
            col_cluster_nelem_per_cluster
        )
        col_cluster_axes_fraction_for_each_cluster = (
            1 - total_col_spacer_frac
        ) * col_cluster_rel_size_for_each_cluster
        col_cluster_axes_fraction_for_each_cluster_with_spacer = (
            col_cluster_axes_fraction_for_each_cluster + col_spacers_with_0_end
        )
        col_cluster_start_axis_fraction = np.cumsum(
            np.insert(col_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
        )[:-1]
        col_cluster_end_axis_fraction = (
            np.cumsum(
                np.insert(col_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
            )[1:]
            - col_spacers_with_0_end
        )

    # we work with fractions of the Axes height and width, so we set the limits to:
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # Prepare colormapping - all pcolormesh blocks must use the same colormapping
    # The user can guarantee this by specifying norm or vmin AND vmax in the pcolormesh_args
    # Otherwise, we determine vmin and vmax as:
    if not (
        "norm" in pcolormesh_args
        or ("vmin" in pcolormesh_args and "vmax" in pcolormesh_args)
    ):
        pcolormesh_args["vmin"] = np.quantile(df, 0.02)
        pcolormesh_args["vmax"] = np.quantile(df, 1 - 0.02)

    # Plot all colormesh blocks, and do some defensive assertions
    col_widths = []  # for asserting equal column width
    row_heights = []  # for asserting equal row heights
    x_ticks = []
    x_ticklabels = []
    y_ticks = []
    y_ticklabels = []

    if row_clusters is not None and col_clusters is not None:
        for row_cluster_idx, col_cluster_idx in product(
            range(row_cluster_nclusters), range(col_cluster_nclusters)
        ):
            # Retrieve the data for the current pcolormesh block
            dataview = df.iloc[
                row_cluster_cumsum_nelem_prev_clusters[
                    row_cluster_idx
                ] : row_cluster_cumsum_nelem_prev_clusters[row_cluster_idx + 1],
                col_cluster_cumsum_nelem_prev_clusters[
                    col_cluster_idx
                ] : col_cluster_cumsum_nelem_prev_clusters[col_cluster_idx + 1],
            ]

            # Create meshgrid for pcolormesh
            # create 1D x and y arrays, then use np.meshgrid to get X and Y for pcolormesh(X, Y)
            x = np.linspace(
                col_cluster_start_axis_fraction[col_cluster_idx],
                col_cluster_end_axis_fraction[col_cluster_idx],
                col_cluster_nelem_per_cluster[col_cluster_idx] + 1,
            )
            col_widths.append(np.ediff1d(x))
            if row_cluster_idx == 0:
                # for the bottom row, add x-ticks in the middle of each column
                x_ticks.append(
                    (x + (x[1] - x[0]) / 2)[:-1]
                )  # no tick after the end of this colormesh block
                x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(
                row_cluster_start_axis_fraction[row_cluster_idx],
                row_cluster_end_axis_fraction[row_cluster_idx],
                row_cluster_nelem_per_cluster[row_cluster_idx] + 1,
            )
            row_heights.append(np.ediff1d(y))
            if col_cluster_idx == 0:
                y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
                y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)

            # Plot, color mapping is controlled via pcolormesh_args
            quadmesh = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)

        assert np.all(np.isclose(np.concatenate(row_heights), row_heights[0][0]))
        assert np.all(np.isclose(np.concatenate(col_widths), col_widths[0][0]))
    elif row_clusters is not None:
        # see row + col grouping code above for comments
        for row_cluster_idx in range(row_cluster_nclusters):
            dataview = df.iloc[
                row_cluster_cumsum_nelem_prev_clusters[
                    row_cluster_idx
                ] : row_cluster_cumsum_nelem_prev_clusters[row_cluster_idx + 1],
                :,
            ]
            x = np.linspace(0, 1, df.shape[1] + 1)
            if row_cluster_idx == 0:
                x_ticks.append(
                    (x + (x[1] - x[0]) / 2)[:-1]
                )  # no tick after the end of this colormesh block
                x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(
                row_cluster_start_axis_fraction[row_cluster_idx],
                row_cluster_end_axis_fraction[row_cluster_idx],
                row_cluster_nelem_per_cluster[row_cluster_idx] + 1,
            )
            row_heights.append(np.ediff1d(y))
            y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
            y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)
            quadmesh = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)
        assert np.all(np.isclose(np.concatenate(row_heights), row_heights[0][0]))
        # TEST: surround the row clusters with black framing lines
        if frame_spaced_elements:
            for row_cluster_idx in range(row_cluster_nclusters):
                y0, y1 = (
                    row_cluster_start_axis_fraction[row_cluster_idx],
                    row_cluster_end_axis_fraction[row_cluster_idx],
                )
                rect = patches.Rectangle(
                    (0, y0), width=1, height=y1 - y0, fill=None, **frame_kwargs
                )
                ax.add_patch(rect)
    elif col_clusters is not None:
        for col_cluster_idx in range(col_cluster_nclusters):
            dataview = df.iloc[
                :,
                col_cluster_cumsum_nelem_prev_clusters[
                    col_cluster_idx
                ] : col_cluster_cumsum_nelem_prev_clusters[col_cluster_idx + 1],
            ]
            x = np.linspace(
                col_cluster_start_axis_fraction[col_cluster_idx],
                col_cluster_end_axis_fraction[col_cluster_idx],
                col_cluster_nelem_per_cluster[col_cluster_idx] + 1,
            )
            col_widths.append(np.ediff1d(x))
            x_ticks.append(
                (x + (x[1] - x[0]) / 2)[:-1]
            )  # no tick after the end of this colormesh block
            x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(0, 1, df.shape[0] + 1)
            if col_cluster_idx == 0:
                y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
                y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)
            quadmesh = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)
        assert np.all(np.isclose(np.concatenate(col_widths), col_widths[0][0]))

    # Ticks and beautify
    if show_row_labels:
        ax.set(yticks=np.concatenate(y_ticks), yticklabels=np.concatenate(y_ticklabels))
    else:
        ax.set(yticks=[])
    if show_col_labels:
        ax.set_xticks(np.concatenate(x_ticks))
        if rotate_col_labels:
            ax.set_xticklabels(np.concatenate(x_ticklabels), rotation=90)
        else:
            ax.set_xticklabels(np.concatenate(x_ticklabels))
    else:
        ax.set(xticks=[])

    if title:
        ax.set_title(title)

    ax.tick_params(length=0, which="both", axis="both")
    sns.despine(ax=ax, bottom=True, left=True)

    if add_colorbar:
        cb = fig.colorbar(quadmesh, ax=ax, **cbar_args)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(length=3.5, width=0.5, which="both", axis="both")

    # noinspection PyUnboundLocalVariable
    return quadmesh


# noinspection PyUnboundLocalVariable
def spaced_heatmap2(
    df: pd.DataFrame,
    ax: Axes,
    pcolormesh_args=None,
    xticklabels: Union[bool, List[str]] = None,
    xticklabel_rotation=90,
    yticklabels: Union[bool, List[str]] = None,
    row_clusters: Union[np.ndarray, pd.Series] = None,
    col_clusters: Union[np.ndarray, pd.Series] = None,
    row_spacer_size: Union[float, Iterable[float]] = 0.01,
    col_spacer_size: Union[float, Iterable[float]] = 0.01,
    frame_spaced_elements=False,
    frame_kwargs=None,
) -> Dict:
    """Plot pcolormesh with spacers between groups of rows and columns

    Spacer size can be specified for each spacer individually, or with
    one size for all row spacers and one size for all col spacers.

    Notes
    -----
    This functions plots all subheatmaps with individual calls to ax.pcolormesh
    To guarantee that all subheatmaps have the same color mapping, vmin is set to df.min().min()
    and vmax is set to df.max().max() if the user does not pass norm, vmin, vmax through
    pcolormesh_args.


    Parameters
    ----------
    df
        pd.DataFrame, index and columns will be used for labeling
    ax
        Axes to plot on
    row_clusters
        one id per row, specifies which columns to group together. Each group must consist of a single, consecutive stretch of rows. Any dtype is allowed.
    col_clusters
        same as row_clusters
    row_spacer_size
        if float, specifies the size for a single spacer, which will be used for each individual spacer.  If List[float] specifies size for each spacer in order.
        Size is given as fraction of the Axes width.
    col_spacer_size
        Same as row_spacer_size, size is given as fraction of the Axis height
    pcolormesh_args
        dict passed to ax.pcolormesh
    frame_spaced_elements
        if True, draw a rectangle around each individual
        heatmap. This is only implemented for row-clusters only at the moment (experimental!) and is ignored in other cases
    frame_kwargs
        passed to patches.Rectangle

    Returns
    -------
    quadmesh
        for use in colorbar plotting etc.
    """

    # Troubleshooting
    # ---------------
    # - for dataframes with multiple columns and different presence of categories
    #   (e.g. partitionings with different ranks), larger partitionings can all receive
    #   the same color if there is a problem with vmin and vmax. For categorical data,
    #   vmin/vmax should be the minimum and maximum category (the default if vmin/vmax
    #   are not passed through pcolormesh_args).

    if pcolormesh_args is None:
        pcolormesh_args = {}

    # TODO-refactor: these checks should be moved to the mean heatmap function
    # Argument checking and pre-processing
    if row_clusters is None and col_clusters is None:
        raise TypeError("You must specify at least one of {row_cluster, col_cluster}")
    elif row_clusters is not None and not isinstance(
        row_clusters, (pd.Series, np.ndarray)
    ):
        raise TypeError(
            "Cluster ids must be given as array or series, not e.g. DataFrame"
        )
    elif col_clusters is not None and not isinstance(
        col_clusters, (pd.Series, np.ndarray)
    ):
        raise TypeError(
            "Cluster ids must be given as array or series, not e.g. DataFrame"
        )
    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError("All columns must have the same dtype")

    # currently ticklabels must be stored as flat dataframe indices
    # if df has a multiindex, flatten it to get labels usable for plotting
    if xticklabels:
        if isinstance(xticklabels, list):
            df.index = xticklabels
        else:
            df.index = index_to_labels(df.index)
    else:
        ax.tick_params(labelbottom=False)
    if yticklabels:
        if isinstance(yticklabels, list):
            df.columns = yticklabels
        else:
            df.columns = index_to_labels(df.columns)
    else:
        ax.tick_params(labelleft=False)

    if row_clusters is not None:
        (
            row_cluster_ids,
            row_cluster_unique_idx,
            row_cluster_nelem_per_cluster,
        ) = np.unique(row_clusters, return_counts=True, return_index=True)
        row_cluster_id_order = np.argsort(row_cluster_unique_idx)
        row_cluster_nelem_per_cluster = row_cluster_nelem_per_cluster[
            row_cluster_id_order
        ]
        row_cluster_ids = row_cluster_ids[row_cluster_id_order]
        row_cluster_cumsum_nelem_prev_clusters = np.insert(
            np.cumsum(row_cluster_nelem_per_cluster), 0, 0
        )
        row_cluster_nclusters = len(row_cluster_ids)

        if isinstance(row_spacer_size, float):
            # all row spacers have the same size
            # the total fraction of the Axes height reserved for row spacers is then:
            total_row_spacer_frac = row_spacer_size * (row_cluster_nclusters - 1)
            # a vector with the size for each individual row spacer, after the last cluster, there is a row spacer with '0' width
            row_spacers_with_0_end = np.repeat(
                (row_spacer_size, 0), (row_cluster_nclusters - 1, 1)
            )
        else:
            # there is one size per spacer, assert the correct length of the iterable
            assert len(row_spacer_size) == row_cluster_nclusters - 1
            # the total fraction of the Axes height reserved for row spacers is then:
            total_row_spacer_frac = np.sum(row_spacer_size)
            # a vector with the size for each individual row spacer, after the last cluster, there is a row spacer with '0' width
            row_spacers_with_0_end = np.append(row_spacer_size, 0)

        # get the start and end positions for each quadmesh block (space in between will be left 'unplotted' to create the spacers)
        # Fraction of all elements contained in the different clusters
        row_cluster_rel_size_for_each_cluster = row_cluster_nelem_per_cluster / np.sum(
            row_cluster_nelem_per_cluster
        )
        # Compute the fraction of axes occupied by each cluster, together with its subsequent spacer
        row_cluster_axes_fraction_for_each_cluster = (
            1 - total_row_spacer_frac
        ) * row_cluster_rel_size_for_each_cluster
        row_cluster_axes_fraction_for_each_cluster_with_spacer = (
            row_cluster_axes_fraction_for_each_cluster + row_spacers_with_0_end
        )
        # Now we can calculate the start and end of each quadmesh block
        row_cluster_start_axis_fraction = np.cumsum(
            np.insert(row_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
        )[:-1]
        row_cluster_end_axis_fraction = (
            np.cumsum(
                np.insert(row_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
            )[1:]
            - row_spacers_with_0_end
        )

    # see analogous row_spacer code block for comments
    if col_clusters is not None:
        (
            col_cluster_ids,
            col_cluster_unique_idx,
            col_cluster_nelem_per_cluster,
        ) = np.unique(col_clusters, return_counts=True, return_index=True)
        col_cluster_id_order = np.argsort(col_cluster_unique_idx)
        col_cluster_nelem_per_cluster = col_cluster_nelem_per_cluster[
            col_cluster_id_order
        ]
        col_cluster_ids = col_cluster_ids[col_cluster_id_order]
        col_cluster_cumsum_nelem_prev_clusters = np.insert(
            np.cumsum(col_cluster_nelem_per_cluster), 0, 0
        )
        col_cluster_nclusters = len(col_cluster_ids)
        if isinstance(col_spacer_size, float):
            total_col_spacer_frac = col_spacer_size * (col_cluster_nclusters - 1)
            col_spacers_with_0_end = np.repeat(
                (col_spacer_size, 0), (col_cluster_nclusters - 1, 1)
            )
        else:
            assert len(col_spacer_size) == col_cluster_nclusters - 1
            total_col_spacer_frac = np.sum(col_spacer_size)
            col_spacers_with_0_end = np.append(col_spacer_size, 0)

        # see analogous row cluster handling code above for comments
        col_cluster_rel_size_for_each_cluster = col_cluster_nelem_per_cluster / np.sum(
            col_cluster_nelem_per_cluster
        )
        col_cluster_axes_fraction_for_each_cluster = (
            1 - total_col_spacer_frac
        ) * col_cluster_rel_size_for_each_cluster
        col_cluster_axes_fraction_for_each_cluster_with_spacer = (
            col_cluster_axes_fraction_for_each_cluster + col_spacers_with_0_end
        )
        col_cluster_start_axis_fraction = np.cumsum(
            np.insert(col_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
        )[:-1]
        col_cluster_end_axis_fraction = (
            np.cumsum(
                np.insert(col_cluster_axes_fraction_for_each_cluster_with_spacer, 0, 0)
            )[1:]
            - col_spacers_with_0_end
        )

    # we work with fractions of the Axes height and width, so we set the limits to:
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # Prepare colormapping - all pcolormesh blocks must use the same colormapping
    if not "norm" in pcolormesh_args:
        if not "vmin" in pcolormesh_args:
            pcolormesh_args["vmin"] = df.min().min()
        if not "vmax" in pcolormesh_args:
            pcolormesh_args["vmax"] = df.max().max()

    # Plot all colormesh blocks, and do some defensive assertions
    col_widths = []  # for asserting equal column width
    row_heights = []  # for asserting equal row heights
    x_ticks = []
    x_ticklabels = []
    y_ticks = []
    y_ticklabels = []

    if row_clusters is not None and col_clusters is not None:
        for row_cluster_idx, col_cluster_idx in product(
            range(row_cluster_nclusters), range(col_cluster_nclusters)
        ):
            # Retrieve the data for the current pcolormesh block
            dataview = df.iloc[
                row_cluster_cumsum_nelem_prev_clusters[
                    row_cluster_idx
                ] : row_cluster_cumsum_nelem_prev_clusters[row_cluster_idx + 1],
                col_cluster_cumsum_nelem_prev_clusters[
                    col_cluster_idx
                ] : col_cluster_cumsum_nelem_prev_clusters[col_cluster_idx + 1],
            ]

            # Create meshgrid for pcolormesh
            # create 1D x and y arrays, then use np.meshgrid to get X and Y for pcolormesh(X, Y)
            x = np.linspace(
                col_cluster_start_axis_fraction[col_cluster_idx],
                col_cluster_end_axis_fraction[col_cluster_idx],
                col_cluster_nelem_per_cluster[col_cluster_idx] + 1,
            )
            col_widths.append(np.ediff1d(x))
            if row_cluster_idx == 0:
                # for the bottom row, add x-ticks in the middle of each column
                x_ticks.append(
                    (x + (x[1] - x[0]) / 2)[:-1]
                )  # no tick after the end of this colormesh block
                x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(
                row_cluster_start_axis_fraction[row_cluster_idx],
                row_cluster_end_axis_fraction[row_cluster_idx],
                row_cluster_nelem_per_cluster[row_cluster_idx] + 1,
            )
            row_heights.append(np.ediff1d(y))
            if col_cluster_idx == 0:
                y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
                y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)

            # Plot, color mapping is controlled via pcolormesh_args
            qm = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)

        assert np.all(np.isclose(np.concatenate(row_heights), row_heights[0][0]))
        assert np.all(np.isclose(np.concatenate(col_widths), col_widths[0][0]))
    elif row_clusters is not None:
        # see row + col grouping code above for comments
        for row_cluster_idx in range(row_cluster_nclusters):
            dataview = df.iloc[
                row_cluster_cumsum_nelem_prev_clusters[
                    row_cluster_idx
                ] : row_cluster_cumsum_nelem_prev_clusters[row_cluster_idx + 1],
                :,
            ]
            x = np.linspace(0, 1, df.shape[1] + 1)
            if row_cluster_idx == 0:
                x_ticks.append(
                    (x + (x[1] - x[0]) / 2)[:-1]
                )  # no tick after the end of this colormesh block
                x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(
                row_cluster_start_axis_fraction[row_cluster_idx],
                row_cluster_end_axis_fraction[row_cluster_idx],
                row_cluster_nelem_per_cluster[row_cluster_idx] + 1,
            )
            row_heights.append(np.ediff1d(y))
            y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
            y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)
            qm = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)
        assert np.all(np.isclose(np.concatenate(row_heights), row_heights[0][0]))
        # TEST: surround the row clusters with black framing lines
        if frame_spaced_elements:
            for row_cluster_idx in range(row_cluster_nclusters):
                y0, y1 = (
                    row_cluster_start_axis_fraction[row_cluster_idx],
                    row_cluster_end_axis_fraction[row_cluster_idx],
                )
                rect = patches.Rectangle(
                    (0, y0), width=1, height=y1 - y0, fill=None, **frame_kwargs
                )
                ax.add_patch(rect)
    elif col_clusters is not None:
        for col_cluster_idx in range(col_cluster_nclusters):
            dataview = df.iloc[
                :,
                col_cluster_cumsum_nelem_prev_clusters[
                    col_cluster_idx
                ] : col_cluster_cumsum_nelem_prev_clusters[col_cluster_idx + 1],
            ]
            x = np.linspace(
                col_cluster_start_axis_fraction[col_cluster_idx],
                col_cluster_end_axis_fraction[col_cluster_idx],
                col_cluster_nelem_per_cluster[col_cluster_idx] + 1,
            )
            col_widths.append(np.ediff1d(x))
            x_ticks.append(
                (x + (x[1] - x[0]) / 2)[:-1]
            )  # no tick after the end of this colormesh block
            x_ticklabels.append(dataview.columns.tolist())
            y = np.linspace(0, 1, df.shape[0] + 1)
            if col_cluster_idx == 0:
                y_ticks.append((y + (y[1] - y[0]) / 2)[:-1])
                y_ticklabels.append(dataview.index.tolist())
            X, Y = np.meshgrid(x, y)
            qm = ax.pcolormesh(X, Y, dataview, **pcolormesh_args)
        assert np.all(np.isclose(np.concatenate(col_widths), col_widths[0][0]))

    # Ticks and beautify
    if yticklabels:
        ax.set_yticks(np.concatenate(y_ticks))
        ax.set_yticklabels(np.concatenate(y_ticklabels), va="center")
    if xticklabels:
        ax.set_xticks(np.concatenate(x_ticks))
        ax.set_xticklabels(
            np.concatenate(x_ticklabels), ha="center", rotation=xticklabel_rotation
        )

    return qm


def index_to_labels(index):
    if isinstance(index, pd.MultiIndex):
        return [" | ".join(str(s) for s in t) for t in index]
    else:
        return list(index)


def heatmap2(
    df: pd.DataFrame,
    ax: Axes,
    pcolormesh_args=None,
    xticklabels: Union[bool, List[str]] = None,
    xticklabel_rotation=90,
    yticklabels: Union[bool, List[str]] = None,
) -> Dict:

    if pcolormesh_args is None:
        pcolormesh_args = {}

    qm = ax.pcolormesh(df, **pcolormesh_args)

    if xticklabels:
        if not isinstance(xticklabels, list):
            xticklabels = df.columns
        # the default locator will likely not set ticks at each column
        # centering with label alignment fails for large cells
        ax.set_xticks(np.arange(0, df.shape[1]) + 0.5)
        ax.set_xticklabels(xticklabels, ha="center", rotation=xticklabel_rotation)

    if yticklabels:
        if not isinstance(yticklabels, list):
            yticklabels = df.index
        # the default locator will likely not set ticks at each column
        # centering with label alignment fails for large cells
        ax.set_yticks(np.arange(0, df.shape[0]) + 0.5)
        ax.set_yticklabels(yticklabels, va="center")

    return qm


def adjust_coords(
    coords: Union[np.ndarray, pd.Series],
    spacing_group_ids: np.ndarray,
    spacer_sizes: Union[float, List[float]],
    right_open: bool = True,
) -> np.ndarray:
    """Adjust data coordinates to account for spacers

    This function is meant for plots containing or annotating elements of unit size (in data coordinates), e.g. heatmap rows or columns. The spacing_group_ids indicate where spacers should be introduced between groups of elements. For consistent handling of such spaced plots, the original unit size coordinates are transformed to the interval [0, 1]

    Parameters
    ----------
    coords
        x or y coordinates, for a plot with unit size elements. May be numerical or categorical. Categorical coordinates are translated to unit size coordinates.
    spacing_group_ids
        spacing group indicators, one per element, eg [1, 1, 1, 2, 2, 2], given in plotting order (e.g. in order after the clustering which was applied to a heatmap, not in the order before clustering).
    spacer_sizes
        as used by cross_plot
    right_open
        how to treat coordinates equal to a spacing point. Say you have a heatmap with 9 rows and three clusters, with spacing_group_ids = cluster_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3]. The function will calculate that spacers occur at coordinates 3 and 6 (on axis with limits [0, n_rows], ie prior to normalization of all coordiantes to the [0, 1] interval. If right open = True, the spacers are introduced such that data points [0, 3[, [3, 6[, [6, 9] are separated by spacers. If coords contains a value == 3, this value is moved directly *after* the first spacer. Conversely, if right_open == False, the value would be considered to be directly *before* the first spacer.

    Returns:
        Adjusted coordinates \in [0, 1]
    """
    # Possible improvements
    # - what happens with unsorted coordinates?
    # - does this allow multiple occurences of same coordinate?

    # Coords may be numeric or categorical
    # Categorical coordinates need to be translated to unit-sized numeric coordinates
    assert isinstance(
        coords, (np.ndarray, pd.Series)
    )  # list would fail is_numeric_dtype_test
    coords = np.array(coords)
    assert coords.ndim == 1
    if not is_numeric_dtype(coords):
        # We assume that for categorical data, the y coord should be in the middle of a bin of size 1 for each element
        numeric_coords = np.arange(len(coords)) + 0.5
    else:
        numeric_coords = coords

    thresholds, *_ = find_stretches2(spacing_group_ids)
    thresholds = thresholds[:-1]

    side = "right" if right_open else "left"
    idx = np.searchsorted(thresholds, coords, side)

    # if a scalar spacer_size is given, convert to array of spacers
    if isinstance(spacer_sizes, float):
        spacer_sizes = np.repeat(spacer_sizes, len(thresholds))
    total_spacer_size = np.sum(spacer_sizes)
    # We look up the spacer value to be added to coords from the cumsum of spacers
    # starting with 0 spacer for elements before the first spacer
    spacer_size_cumsum = np.insert(np.cumsum(spacer_sizes), 0, 0)

    # Scale coordinates
    scaled_coords = (
        # first scale y_coords to [0, 1]
        numeric_coords
        / len(spacing_group_ids)
        # then scale them to the space available after reducing the spacers, eg. [0, 0.8]
        * (1 - total_spacer_size)
    )

    # add one or more spacers according to searchsorted index
    adjusted_coords = scaled_coords + spacer_size_cumsum[idx]

    return adjusted_coords


def label_groups(
    group_ids: np.ndarray,
    ax: Axes,
    x: Optional[float] = None,
    y: Optional[float] = None,
    spacer_size: Optional[Union[float, List[float]]] = None,
    labels: Optional[Union[Dict, Iterable]] = None,
    colors: Optional[Union[Dict, Iterable, str, Tuple[float]]] = None,
    **kwargs,
):
    """Add text labels in the middle of stretches of group ids

    - Meant for plot with unit-sized elements, such as heatmap rows and columns.
    - Addition of spacers leads to transformation of coordinates to interval [0, 1]
    - sets limits along the directional axis, either (0, 1) when spacers are used, else (0, len(group_ids)

    Args:
        group_ids: one id per element
        ax: Axes to plto on
        x: if given, plot text labels in y direction, at the fixed x
        y: if given, plot text labels in x direction, at the fixed y
        spacer_size: if given, coordinates are adjusted for spacers between the groups
        labels: labels for groups, either iterable with one element per group occurence, or dict group_id -> label
        colors: either iterable of color specs of same length as number of groups, or a dict group id -> color
        kwargs: passed to ax.text, note that ha and va default to 'center' and rotation to 90 (for text along y axis) if no kwargs are passed

    Returns:

    """

    assert (x is not None) + (y is not None) == 1, "Specify either x OR y"

    # ha and va default to center, rotation defaults to 90 if text goes along x axis
    kwargs = tz.merge(dict(ha="center", va="center"), kwargs)
    if x and "rotation" not in kwargs:
        kwargs["rotation"] = 90

    bounds, mids, values = find_stretches2(group_ids)
    # values: one value per consecutive group id stretch
    # in the absence of labels arg, use values as final_labels
    # if labels is a dict, look up final_labels based on values
    final_labels = get_list_of_stretch_annos(stretch_ids=values, annos=labels)

    # Get final colors, if colors is None, do not use colors
    # First, convert scalar color spec to list of repeated colorspec, if given
    if isinstance(colors, (str, tuple)):
        # can't use np.repeat or tile with tuples
        colors = [colors] * len(values)
    if colors is not None:
        # the function returns array of stretch ids if annos=None, so we handle that case separately
        final_colors = get_list_of_stretch_annos(stretch_ids=values, annos=colors)
    else:
        # for iteration, create array of None objects
        final_colors = np.repeat(None, len(values))

    # adjust coordinates to account for spacers if necessary
    # in this case, the axis limits of the direction axis must be set to 0, 1
    # otherwise, they are set to (0, len(group_ids))
    if spacer_size:
        coords = co.adjust_coords(mids, group_ids, spacer_sizes=spacer_size)
        if x:
            ax.set_ylim(0, 1)
        else:
            ax.set_xlim(0, 1)
    else:
        coords = mids
        if x:
            ax.set_ylim(0, len(group_ids))
        else:
            ax.set_xlim(0, len(group_ids))

    # Place labels in middle of stretches, at the fixed x or y coordinate
    # Place along direction without fixed value (x or y will be None)
    for final_label, coord, final_color in zip(final_labels, coords, final_colors):
        # Final color supersedes constan color in kwargs, if its present
        used_color = kwargs.pop("color", None) if final_color is None else final_color
        t = (coord, y) if y else (x, coord)
        ax.text(*t, final_label, color=used_color, **kwargs)


def get_list_of_stretch_annos(
    stretch_ids: Iterable, annos: Optional[Union[Iterable, Dict]]
) -> np.ndarray:
    """Create array with one annotation value per stretch

    Args:
        annos: iterable of annotation values, or dict mapping stretch ids to annotation values. If None, the stretch_ids will be used as annotations
        stretch_ids: unique value contained in each stretch

    Returns:
          array with one annotation per stretch
    """

    if isinstance(annos, dict):
        annos_ser = pd.Series(stretch_ids).map(annos)
        assert annos_ser.notnull().all()
        return annos_ser.array

    if annos is not None:
        # else: annos are iterable
        assert len(annos) == len(stretch_ids)
        annos = np.array(annos)
        assert annos.ndim == 1
        return annos

    else:
        return np.array(stretch_ids)


def frame_groups(
    group_ids: np.ndarray,
    ax: Axes,
    direction: str,
    # only dict functional at the moment
    # colors: Optional[Union[Iterable, Dict, str, Tuple[float]]] = None,
    colors: Optional[Dict] = None,
    spacer_sizes: Optional[Union[float, List[float]]] = None,
    edge_alignment: Literal["outside", "inside", "centered"] = "inside",
    origin=(0, 0),
    add_labels: bool = False,
    labels: Optional[Union[Dict, Iterable]] = None,
    label_groups_kwargs: Optional[Dict] = None,
    label_colors: Optional[Union[Dict, Iterable, str, Tuple[float]]] = None,
    axis_off: bool = True,
    fancy_box_kwargs: Optional[Dict] = None,
    direction_sides_padding_axes_coord: float = 0,
    non_direction_sides_padding_axes_coord: float = 0,
    size: Optional[int] = None,
    **kwargs,
) -> None:
    """Frame groups of elements

    - Meant for plot with unit-sized elements, such as heatmap rows and columns.
    - Addition of spacers leads to transformation of coordinates to interval [0, 1]
    - sets limits along the directional axis, either (0, 1) when spacers are used, else (0, len(group_ids)


    Args:
        ax: single Axes
        group_ids: array of group ids
        spacer_sizes: if given, coordinates are adjusted for spacers between the groups
        direction: axis along which to place the patches ('x' or 'y')
        edge_alignment:
            wether to draw edges outside of the rectangle patch covering the
            group elements, or fully inside this rectangle patch. Note: both choices avoid
            the default mpl behavior of placing the edge centered on the rectangle patch.
        origin: origin of first rectangle patch
        colors
            *Note*: it seems that colors currently only works with a dict; other possible values may fail, including None
            either iterable of color specs of same length as number of ticklabels, or a dict label -> color
        size:
            DEPRECATED extent of rectangle patch along the unused axis (constant for all patches)
        **kwargs:
            DEPRECATED
            passed to mpatches.FancyBboxPatch. Default args: dict((fill = False, boxstyle = mpatches.BoxStyle("Round", pad=0), edgecolor = 'black'). Edgecolor is ignored if colors is passed.
        add_labels: if True, call label_groups(..., labels, **label_groups_kwargs)
        labels: passed to label_groups
        label_groups_kwargs: passed to label_groups
        label_colors
            allows defining label colors independent of frame colors (if not given, frame colors
            will be used as label colors). Not sure if currently functional for all/any arg type.
        axis_off: if True, remove spines and turn of ticks and ticklabels
        fancy_box_kwargs
            passed to mpatches.FancyBboxPatch. Default args: dict((fill = False, boxstyle = mpatches.BoxStyle("Round", pad=0), edgecolor = 'black'). Edgecolor is ignored if colors is passed.
        direction_sides_padding_axes_coord
            different params for direction and non direction sides: these are axes coords,
            and the axes size in inch may be quite different for x,y, making two different parameters
            easiest quick and dirty implementation. axes coords are also the unit for the spacers,
            which may need to be considered when settings this parameter
        non_direction_sides_padding_axes_coord
    """

    if size is not None:
        warnings.warn("size arg deprecated", DeprecationWarning)
    # constant, assumes that axes limits can be controlled by this function and set to approx. (0,1)
    # - exact limits will depend on the padding
    non_direction_patch_size = 1

    if not isinstance(colors, dict):
        raise ValueError(
            'Other types than dict for arg "colors" may currently not work, including None'
        )
    assert edge_alignment in ["inside", "outside", "centered"]
    if kwargs:
        warnings.warn("passing **kwargs is deprecated", category=DeprecationWarning)

    if fancy_box_kwargs is None:
        fancy_box_kwargs = {}
    defaults = dict(
        fill=False, boxstyle=mpatches.BoxStyle("Round", pad=0), edgecolor="black"
    )
    fancy_box_kwargs = tz.merge(defaults, fancy_box_kwargs)

    if axis_off:
        ax.axis("off")

    bounds, mids, values = find_stretches2(group_ids)

    # We need colors as an iterable, matching the group stretches - or empty
    # If colors is given, edgecolor specification is ignored
    if isinstance(colors, dict):
        colors = pd.Series(colors)[values].to_numpy()
    if colors is not None:
        if isinstance(colors, (tuple, str)):
            # can't use np.repeat or tile with tuple elements
            colors = np.array([colors] * len(values))
        fancy_box_kwargs.pop("edgecolor")
    else:
        colors = np.array([])

    # if spacers are used, axis limits along the directional axis must be set to (0, 1)
    # in the other case, one will usually want (0, len(group_ids))
    ax.set(**{f"{direction}lim": (0, 1) if spacer_sizes else (0, len(group_ids))})

    # Get starts and ends of the patches, adjusting for spacers if necessary
    if spacer_sizes:
        adjust_bounds = adjust_coords(
            coords=bounds[:-1],
            spacing_group_ids=group_ids,
            spacer_sizes=spacer_sizes,
            right_open=True,
        )
        starts = np.insert(adjust_bounds, 0, 0)
    else:
        starts = np.insert(bounds[:-1], 0, 0)
    if spacer_sizes:
        ends = adjust_coords(
            coords=bounds,
            spacing_group_ids=group_ids,
            spacer_sizes=spacer_sizes,
            right_open=False,
        )
    else:
        ends = bounds

    # Add patches
    # ------------
    # Patches are specified as (x, y), width, height; x, y is the origin
    # The edges of patches in matplotlib are always centered on x, x + height, y, y + width, ie
    # half of the line is to the left and right of the coordinate
    # to manipulate this, we use some transformation magic to adjust
    # the patch size such that the edgelines have the desired alignment (inside, outside or centered)

    # Transformations
    # 1. bring coordinates to display coords
    # 2. Add offset in inch
    # 3. bring back to axes coord

    # we prepare transformations to shift points left, right, up and down by half a line width
    # line width in inch is retrieved as follows:
    # Get line width in inch, for use with dpi_scale_trans
    linewidth_in = (
        (fancy_box_kwargs.get("linewidth") or mpl.rcParams["patch.linewidth"]) * 1 / 72
    )

    # Get transformations
    y_up_shift = (
        ax.transData
        + mtransforms.ScaledTranslation(
            *(0, linewidth_in / 2), ax.figure.dpi_scale_trans
        )
        + ax.transData.inverted()
    )
    y_down_shift = (
        ax.transData
        + mtransforms.ScaledTranslation(
            *(0, -linewidth_in / 2), ax.figure.dpi_scale_trans
        )
        + ax.transData.inverted()
    )
    x_left_shift = (
        ax.transData
        + mtransforms.ScaledTranslation(
            *(-linewidth_in / 2, 0), ax.figure.dpi_scale_trans
        )
        + ax.transData.inverted()
    )
    x_right_shift = (
        ax.transData
        + mtransforms.ScaledTranslation(
            *(linewidth_in / 2, 0), ax.figure.dpi_scale_trans
        )
        + ax.transData.inverted()
    )

    # Add patches iteratively
    # use transforms according to edge_alignment and direction
    # finally, add padding if desired
    if edge_alignment == "outside" and direction == "y":
        direction_dim_start_shift = y_down_shift
        direction_dim_end_shift = y_up_shift
        other_dim_start_shift = x_left_shift
        other_dim_end_shift = x_right_shift
    elif edge_alignment == "outside" and direction == "x":
        direction_dim_start_shift = x_left_shift
        direction_dim_end_shift = x_right_shift
        other_dim_start_shift = y_down_shift
        other_dim_end_shift = y_up_shift
    elif edge_alignment == "inside" and direction == "y":
        direction_dim_start_shift = y_up_shift
        direction_dim_end_shift = y_down_shift
        other_dim_start_shift = x_right_shift
        other_dim_end_shift = x_left_shift
    elif edge_alignment == "inside" and direction == "x":
        direction_dim_start_shift = x_right_shift
        direction_dim_end_shift = x_left_shift
        other_dim_start_shift = y_up_shift
        other_dim_end_shift = y_down_shift

    slice_direction = 1 if direction == "x" else -1
    for color, start, end in zip_longest(colors, starts, ends):
        start: float
        end: float
        if edge_alignment != "centered":
            # if colors is empty, color will always be none
            if direction == "x":
                # noinspection PyUnboundLocalVariable
                curr_origin = other_dim_start_shift.transform(
                    direction_dim_start_shift.transform((start, origin[1]))
                )
                curr_origin = (
                    curr_origin[0] - direction_sides_padding_axes_coord,
                    curr_origin[1]
                    # - direction_sides_padding_axes_coord
                    - non_direction_sides_padding_axes_coord,
                )
                # noinspection PyUnboundLocalVariable
                end = direction_dim_end_shift.transform((end, 0))[0]
                end = end + direction_sides_padding_axes_coord
                patch_size_in_direction = end - curr_origin[0]
                # noinspection PyUnboundLocalVariable
                patch_size_other_dim = (
                    other_dim_end_shift.transform((non_direction_patch_size, 0))[0]
                    - curr_origin[1]
                )
                patch_size_other_dim = (
                    patch_size_other_dim
                    # + direction_sides_padding_axes_coord
                    + non_direction_sides_padding_axes_coord
                )
            else:
                curr_origin = other_dim_start_shift.transform(
                    direction_dim_start_shift.transform((origin[0], start))
                )
                curr_origin = (
                    curr_origin[0]
                    # - direction_sides_padding_axes_coord
                    - non_direction_sides_padding_axes_coord,
                    curr_origin[1] - direction_sides_padding_axes_coord,
                )
                end = direction_dim_end_shift.transform((0, end))[1]
                end = end + direction_sides_padding_axes_coord
                patch_size_in_direction = end - curr_origin[1]
                patch_size_other_dim = (
                    other_dim_end_shift.transform((non_direction_patch_size, 0))[0]
                    - curr_origin[0]
                )
                patch_size_other_dim = (
                    patch_size_other_dim
                    # + direction_sides_padding_axes_coord
                    + non_direction_sides_padding_axes_coord
                )
        elif edge_alignment == "centered" and direction == "y":
            curr_origin = (origin[0], start)
            curr_origin = (
                curr_origin[0]
                # - direction_sides_padding_axes_coord
                - non_direction_sides_padding_axes_coord,
                curr_origin[1] - direction_sides_padding_axes_coord,
            )
            end = end + direction_sides_padding_axes_coord
            patch_size_in_direction = end - curr_origin[1]
            patch_size_other_dim = non_direction_patch_size
            patch_size_other_dim = (
                patch_size_other_dim
                # + 2 * direction_sides_padding_axes_coord
                + 2 * non_direction_sides_padding_axes_coord
            )
        elif edge_alignment == "centered" and direction == "x":
            curr_origin = (start, origin[0])
            curr_origin = (
                curr_origin[0] - direction_sides_padding_axes_coord,
                curr_origin[1]
                # - direction_sides_padding_axes_coord
                - non_direction_sides_padding_axes_coord,
            )
            end = end + direction_sides_padding_axes_coord
            patch_size_in_direction = end - curr_origin[0]
            patch_size_other_dim = non_direction_patch_size
            patch_size_other_dim = (
                patch_size_other_dim
                # + 2 * direction_sides_padding_axes_coord
                + 2 * non_direction_sides_padding_axes_coord
            )
        else:
            raise ValueError()

        # noinspection PyTypeChecker
        patch = mpatches.FancyBboxPatch(
            cast(Tuple[float, float], curr_origin),
            *(patch_size_in_direction, patch_size_other_dim)[::slice_direction],
            # use color as edgecolor if defined; otherwise, kwargs will contain an edgecolor kwarg
            **dict(edgecolor=color) if color else {},
            **kwargs,
            **fancy_box_kwargs,
        )
        ax.add_artist(patch)

    if add_labels:
        if not label_groups_kwargs:
            label_groups_kwargs = {}
        label_groups(
            group_ids=group_ids,
            ax=ax,
            x=0.5 if direction == "y" else None,
            y=0.5 if direction == "x" else None,
            spacer_size=spacer_sizes,
            labels=labels,
            colors=colors if label_colors is None else label_colors,
            **label_groups_kwargs,
        )

    other_dim = "x" if direction == "y" else "x"
    # increase axis limits to include the patch+edge. quick+dirty code which only
    # works with the standard frame-around-labels situation, and only *kinda*, since
    # half of the edge lines will be outside the axis
    # patch_size_other_dim is taken from the last loop iteration, it should anyway always be the same
    # also quite q+d
    # noinspection PyUnboundLocalVariable
    ax.set(**{f"{other_dim}lim": (0, patch_size_other_dim)})


def add_cluster_anno_bubbles(
    cluster_ids_ser,
    cluster_colors_d: Union[Dict, pd.Series],
    cluster_to_anno_lines_dol,
    anno_side: Literal['top', 'bottom', 'left', 'right'],
    ax,
    n_tiers=2,
    fontsize=None,
    connector_lw=1,
    spacer_between_subsequent_bubbles=0.1,
    spacer_between_bubble_tiers=0.1,
    boxstyle="Round,pad=0,rounding_size=0.015",
    spacing_group_ids=None,
    spacer_sizes=None,
    cluster_order=None,
) -> None:
    """Annotate heatmap with 'bubbles' aligned with clusters (considering spacers)

    Usage notes
    -----------
    - How to make boxes fit around the text
      - adjust the axes size outside of the plotting function
      - control the size of the boxes by adjusting the spacer_between_subsequent_bubbles and the spacer_between_bubble_tiers

    Implementation notes
    --------------------
    - currently assumes that all clusters have the same size

    Parameters
    ----------
    cluster_ids_ser
        cluster_ids of rows in plotting order
    cluster_to_anno_lines_l
        dict mapping cluster id -> List[Str, ...], e.g. {1: ['gene1', 'gene2']}
    annotated_axis
        whether bubbles should be plotted along the x or along the y axis of the heatmap
    fontsize
        defaults to {annotated_axis}tick.labelsize
    spacer_size
        if None, no spacing, set {annotated_axis}_lim to (0, n_clusters)
        else, apply specing, set {annotated_axis}_lim to (0, 1)
    spacer_between_subsequent_bubbles
        spacer between two subsequent bubbles in one row/col in kind-of"percent". Bubbles are centered on the cluster they annotate and have width cluster_width * n_cols * (1 - spacer_between_subsequent_bubbles) when annotating an x axis, and analogously when annotating a y axis
    spacer_between_bubble_tiers
        spacer between rows/cols of bubbles in kind of percent. For x axis annotation, bubbles are centered at y=0.5, 1.5, and have width (1 - spacer_between_bubble_tiers)
    spacing_group_ids
        in plotting order, used for co.plotting.adjust_coords
    cluster_order
        optional, not necessary when cluster_ids_ser is categorical. always pass in ascending order, if annotated_axis='y', this will be flipped automatically atm
    n_tiers
        number of rows/cols of bubbles. within each row, the bubble size along the annotated axis is cluster_size * n_tiers * spacer_between_subsequent_bubbles


    """

    if anno_side in ['top', 'bottom']:
        annotated_axis = 'x'
    else:
        annotated_axis = 'y'

    if fontsize is None:
        fontsize = mpl.rcParams[f"{annotated_axis}tick.labelsize"]

    if cluster_order is None:
        try:
            cluster_order = cluster_ids_ser.cat.categories
        except AttributeError:
            raise ValueError()
    cluster_sizes = cluster_ids_ser.value_counts()
    common_cluster_size = cluster_sizes[0]
    assert cluster_sizes.eq(common_cluster_size).all()

    # calculate limits for annotated axis == 'x'
    # NOTE: cannot add margin to avoid clipping bubble edgelines, because this would break alignment with the heatmap annotated by this plot. this is better solved eg with a spacer in a cross_plot
    ylim = (0, n_tiers)
    if spacing_group_ids is not None:
        assert spacer_sizes is not None
        xlim = (0, 1)
    else:
        xlim = (0, cluster_ids_ser.shape[0])
    if annotated_axis == "y":
        ylim, xlim = xlim, ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    bubble_centers = (
        np.arange(len(cluster_sizes)) * common_cluster_size + common_cluster_size * 0.5
    )
    bubble_centers_adjusted = adjust_coords(
        coords=bubble_centers,
        spacing_group_ids=spacing_group_ids,
        spacer_sizes=spacer_sizes,
    )

    common_cluster_sizes_after_spacer_transformation = np.ediff1d(
        adjust_coords(
            coords=np.array([0, 749.999, 750, 1499.999]),
            spacing_group_ids=spacing_group_ids,
            spacer_sizes=spacer_sizes,
            right_open=True,
        )
    )
    common_cluster_size_transformed = common_cluster_sizes_after_spacer_transformation[
        0
    ]
    assert np.isclose(
        common_cluster_size_transformed,
        common_cluster_sizes_after_spacer_transformation[-1],
    )

    width_in_anno_dir = (
        common_cluster_size_transformed
        * n_tiers
        * (1 - spacer_between_subsequent_bubbles)
    )
    height_in_anno_dir = 1 - spacer_between_bubble_tiers

    # bubbler_spacer_size_data_coords_x = (
    # patch_width * bubbler_spacer_size_perc_x
    # )

    for cluster_id, bubble_center, col_idx in zip(
        cluster_order, bubble_centers_adjusted, itertools.cycle(np.arange(n_tiers))
    ):

        # calculate limits for annotated axis == 'x'
        x = bubble_center
        y = 0.5 + col_idx
        xy = (
            bubble_center - width_in_anno_dir / 2,
            # - common_cluster_size_transformed / 2
            # + bubbler_spacer_size_data_coords_x / 2,
            y - height_in_anno_dir / 2,
        )
        patch_width = width_in_anno_dir
        patch_height = height_in_anno_dir
        # width = common_cluster_size_transformed - bubbler_spacer_size_data_coords_x
        if annotated_axis == "y":
            y, x = x, y
            xy = xy[::-1]
            patch_width, patch_height = patch_height, patch_width
        # print(pd.Series(dict(cluster_id=cluster_id, x=x, y=y, patch_width=patch_width, patch_height=patch_height, xy=xy)))

        # where to aligned anno line text on the non-annotated axis
        text = "\n".join(cluster_to_anno_lines_dol[cluster_id])
        ax.text(x=x, y=y, s=text, va="center", ha="center", zorder=3, fontsize=fontsize)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                xy=xy,
                width=patch_width,
                height=patch_height,
                # color=cluster_colors_d[cluster_id],
                # alpha=alpha,
                # color=None,
                # color overwrite both
                # color='white',
                facecolor="white",
                zorder=2,
                clip_on=False,
                # boxstyle=mpatches.BoxStyle.Round(pad=0.1),
                boxstyle=boxstyle,
                # edgecolor=None,
                edgecolor=cluster_colors_d[cluster_id],
                linewidth=1.5,
            )
        )

        k = 1
        line_endpoint = {
            "top": -k,
            "bottom": n_tiers + k,
            "left": n_tiers + k,
            "right": -k,
        }[anno_side]
        if anno_side in ['top', 'bottom']:
            args = (
            [x, x],
            [y, line_endpoint])
        else:
            args = ([x, line_endpoint], [y, y])
        ax.plot(
            *args,
            c=cluster_colors_d[cluster_id],
            linewidth=connector_lw,
            # linewidth=0.5,
            # linestyle="--",
            zorder=1,
        )

        ax.axis("off")
    if annotated_axis == "y":
        ax.invert_yaxis()
