#-
from abc import ABC
from functools import partial
from typing import Optional, List, Tuple, Union, Callable, Dict, Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib.axes import Axes  # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

mpl.use('Agg') # import before pyplot import!
import seaborn as sns

from complex_heatmap.dynamic_grid import GridManager, GridElement, Spacer

MixedGrid = List[List[Union['ClusterProfilePlotPanel', GridElement]]]
#-

class ClusterProfilePlotPanel(ABC):
    align_vars: List[str] = []
    supply_vars: Dict[str, str] = []
    plotter: Callable = None
    row_deco: bool = True
    col_deco: bool = True

    def __init__(self, panel_width=1, panel_kind='rel', name=None, **kwargs):
        self.kwargs = kwargs
        self.panel_width = panel_width
        self.panel_kind = panel_kind
        self.name = name
        if not isinstance(type(self).__dict__['plotter'], staticmethod):
            raise TypeError('Error in class definition: '
                            'plotter must be a static method')

    def align_and_supply(self, cluster_profile_plot):

        for target, value in self.supply_vars.items():
            if target not in self.kwargs:
                self.kwargs[target] = getattr(cluster_profile_plot, value)

        for var in self.align_vars:
            row_slice = cluster_profile_plot.row_int_idx
            if row_slice is None:
                row_slice = slice(None)
            col_slice = cluster_profile_plot.col_int_idx
            if col_slice is None:
                col_slice = slice(None)
            self.kwargs[var] = (self.kwargs[var]
                                .iloc[row_slice, col_slice]
                                .copy()
                                )

    def plot(self):
        self.plotter(**self.kwargs)
#-

@dataclass
class ClusterProfilePlot:
    """
    general kwargs for all plotting functions

    Syntactic sugar to add dendrograms
    row_dendrogram: bool = False
    col_dendrogram: bool = False
    fig_args: Optional[Dict[str, Any]] = None
    """
    main_df: pd.DataFrame
    supp_dfs: Optional[List[pd.DataFrame]] = None
    cluster_ids: Optional[pd.DataFrame] = None
    row_linkage: Optional[np.ndarray] = None
    col_linkage: Optional[np.ndarray] = None
    col_int_idx: Optional[np.ndarray] = None
    row_int_idx: Optional[np.ndarray] = None

    def cluster_rows(self, method='average', metric='euclidean',
                     usecols: Optional[List[str]] = None,
                     ) -> 'ClusterProfilePlot':
        """Cluster rows using hierarchical clustering

        Args:
            usecols: only use these columns to compute the clustering

        Uses scipy linkage function.
        """
        if usecols is None:
            df = self.main_df
        else:
            df = self.main_df.loc[:, usecols]
        self.row_linkage = linkage(df, metric=metric, method=method)
        self.row_int_idx = leaves_list(self.row_linkage)
        return self

    def cluster_cols(self, method='average', metric='euclidean',
                     userows: Optional[List[str]] = None,
                     ) -> 'ClusterProfilePlot':
        """Cluster columns using hierarchical clustering

        Args:
            userows: only use these rows to compute the clustering

        Uses scipy linkage function.
        """
        if userows is None:
            df = self.main_df
        else:
            df = self.main_df.loc[userows, :]
        self.col_linkage = linkage(df.T, metric=metric, method=method)
        self.col_int_idx = leaves_list(self.col_linkage)
        return self

    def plot_grid(self, grid: MixedGrid,
                  figsize: Tuple[float, float],
                  height_ratios: Optional[List[Tuple[float, str]]] = None,
                  h_pad = 1/72, w_pad = 1/72, hspace=1/72, wspace=1/72,
                  row_dendrogram: bool = False, col_dendrogram: bool = False,
                  row_annotation: Optional[pd.DataFrame] = None,
                  row_anno_heatmap_args: Optional[Dict[str, Any]] = None,
                  row_anno_col_width: float = 0.6/2.54,
                  fig_args: Optional[Dict[str, Any]] = None
                  ):
        """Create grid of plots with an optionally underlying clustering

        Row and column dendrograms and annotations can be added automatically
        or be placed in the grid for finer control.

        If the clustering of the data was computed using the GridManager
        cluster_{cols,rows} methods or if the respective linkage
        matrices are passed from outside, plots can optionally be aligned based on the
        clustering.

        Plots are aligned if they are given as ClusterProfilePlotPanel
        subclass with variables to be aligned indicated in the align_vars
        class variable.

        Args:
            figsize: must be specified so that GridManager can compute
                the appropriate width and height ratios. Actually, this is only
                necessary if some widths or heights are given with absolute
                size. Therefore, figsize may become optional in the future,
                for cases where only relative sizes are required.
            height_ratios: in the form [(1, 'abs'), (2, 'rel'), (1, 'rel')]
                if omitted, all rows are plotted with equal heights
        """

        # noinspection PyUnusedLocal
        if height_ratios is None:
            height_ratios = [(1, 'rel') for unused_row in grid]

        processed_grid = self.convert_panel_element_to_grid_element(grid)

        # Create a GridManager from the 'base grid', then use the GridManager
        # grid manipulation functions to add the decoration rows and columns

        # noinspection PyUnusedLocal
        gm = GridManager(processed_grid, height_ratios=height_ratios,
                         figsize=figsize, fig_args=fig_args,
                         h_pad=h_pad, w_pad=w_pad, hspace=hspace, wspace=wspace)

        self.add_row_decoration(gm, row_anno_col_width, row_anno_heatmap_args,
                                row_annotation, row_dendrogram)

        self.add_column_decoration(col_dendrogram, gm)


        return gm

    def add_column_decoration(self, col_dendrogram, gm):
        """Add column dendrogram if required

        Column annotation heatmaps will be implemented soon.
        """
        if col_dendrogram:
            col_dendro_ge = GridElement('col_dendrogram', plotter=dendrogram_wrapper,
                                        linkage_mat=self.col_linkage,
                                        orientation='top',
                                        tags=['no_row_dendrogram']
                                        )
            col_dendro_only_cols = [i for i, ge in enumerate(gm.grid[0])
                                    if not 'no_col_dendrogram' in ge.tags]
            gm.insert_matched_row(0, col_dendro_ge, height=(1 / 2.54, 'abs'),
                                  only_cols=col_dendro_only_cols)

    def add_row_decoration(self, gm, row_anno_col_width, row_anno_heatmap_args,
                           row_annotation, row_dendrogram):
        """Add row annotation and dendrogram if required"""
        if row_annotation is not None:
            assert isinstance(row_annotation, pd.DataFrame)
            if self.row_int_idx is not None:
                row_annotation = row_annotation.iloc[self.row_int_idx, :]
            row_anno_width_kind = 'abs'
            row_anno_col = []
            row_anno_partial_constructor = partial(
                    GridElement, width=row_anno_col_width, kind=row_anno_width_kind,
                    tags=['no_col_dendrogram'],
                    plotter=categorical_heatmap,
                    df=row_annotation, **row_anno_heatmap_args)
        if row_dendrogram:
            row_dendrogram_width = 1 / 2.54
            row_dendrogram_width_kind = 'abs'
            row_dendro_col = []
            row_dendro_ge_partial_constructor = partial(
                    GridElement,
                    width=row_dendrogram_width, kind=row_dendrogram_width_kind,
                    plotter=dendrogram_wrapper,
                    linkage_mat=self.row_linkage, orientation='left',
                    tags=['no_col_dendrogram'])
        if row_dendrogram or row_annotation is not None:
            for row_idx, row in enumerate(gm.grid):
                for row_grid_element in row:
                    if row_grid_element.name.startswith('spacer'):
                        continue
                    if 'no_row_dendrogram' in row_grid_element.tags:
                        if row_annotation is not None:
                            row_anno_col.append(Spacer(width=row_anno_col_width,
                                                       kind=row_anno_width_kind))
                        if row_dendrogram:
                            row_dendro_col.append(Spacer(width=row_dendrogram_width,
                                                         kind=row_dendrogram_width_kind))
                        break
                    else:
                        if row_annotation is not None:
                            row_anno_col.append(row_anno_partial_constructor(
                                    name=row_grid_element.name + '_row-anno'
                            ))
                        if row_dendrogram:
                            row_dendro_col.append(row_dendro_ge_partial_constructor(
                                    name=row_grid_element.name + '_row-dendrogram'
                            ))
                        break
                else:
                    raise ValueError('There is a row which only consists of Spacers.'
                                     'Cant insert dendrograms'
                                     '- please remove this row and try again')

            if row_annotation is not None:
                gm.prepend_col_from_sequence(row_anno_col)
            if row_dendrogram:
                gm.prepend_col_from_sequence(row_dendro_col)

    def convert_panel_element_to_grid_element(self, grid: MixedGrid):
        """Convert PanelElement in grid into GridElement

        Grid may contain PanelElement or GridElement instances.

        PanelElement instances use their align and supply methods, before
        their relevant attributes and their arguments are transferred to a GridElement.
        """
        # noinspection PyUnusedLocal
        processed_grid: List[List[GridElement]] = [[[] for unused in range(len(row))] for row in grid]
        for row_idx, row in enumerate(grid):
            for col_idx, panel_or_grid_element in enumerate(row):
                if isinstance(panel_or_grid_element, GridElement):
                    processed_grid[row_idx][col_idx] = panel_or_grid_element
                else:
                    # This is a PanelElement and we need to convert it into a
                    # GridElement
                    panel_or_grid_element.align_and_supply(self)
                    tags = []
                    if not panel_or_grid_element.row_deco:
                        tags.append('no_row_dendrogram')
                    if not panel_or_grid_element.col_deco:
                        tags.append('no_col_dendrogram')
                    if panel_or_grid_element.name is None:
                        name = f'Unnamed_{row_idx}-{col_idx}'
                    else:
                        name = panel_or_grid_element.name
                        if name.startswith('Unnamed_'):
                            raise ValueError('Element names starting with'
                                             '"Unnamed_" are reserved for internal use.')
                    processed_grid[row_idx][col_idx] = GridElement(name=name,
                                                                   plotter=panel_or_grid_element.plotter,
                                                                   width=panel_or_grid_element.panel_width,
                                                                   kind=panel_or_grid_element.panel_kind,
                                                                   tags=tags,
                                                                   **panel_or_grid_element.kwargs)
        return processed_grid


    @staticmethod
    def _determine_linkage(main_hmap):
        if main_hmap.col_linkage_matrix is not None:
            if not isinstance(main_hmap.col_linkage_matrix, np.ndarray):
                raise TypeError('User supplied column linkage matrix, '
                                'but the linkage matrix is not an NDarray')
            col_Z = main_hmap.col_linkage_matrix
        elif main_hmap.cluster_cols:
            col_Z = linkage(main_hmap.df.T, method=main_hmap.cluster_cols_method,
                            metric=main_hmap.cluster_cols_metric)
        else:
            col_Z = None
        if main_hmap.row_linkage_matrix is not None:
            if not isinstance(main_hmap.row_linkage_matrix, np.ndarray):
                raise TypeError('User-supplied row linkage matrix, '
                                'but the linkage matrix is not an NDarray')
            row_Z = main_hmap.row_linkage_matrix
        elif main_hmap.cluster_rows:
            if main_hmap.cluster_use_cols is None:
                use_cols_slice = slice(None)
            else:
                assert isinstance(main_hmap.cluster_use_cols, list)
                use_cols_slice = main_hmap.cluster_use_cols
            row_Z = linkage(main_hmap.df.loc[:, use_cols_slice], method=main_hmap.cluster_rows_method,
                            metric=main_hmap.cluster_rows_metric)
        else:
            row_Z = None
        return col_Z, row_Z


def dendrogram_wrapper(linkage_mat, ax: Axes, orientation: str):
    """Wrapper around scipy dendrogram - nicer plot

    Despines plot and removes ticks and labels
    """
    dendrogram(linkage_mat, ax=ax, color_threshold=-1,
               above_threshold_color='black', orientation=orientation)
    ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
    sns.despine(ax=ax, bottom=True, left=True)


def categorical_heatmap(df: pd.DataFrame, ax: Axes,
                        cmap: str = 'Set1', colors: List = None,
                        show_values = True,
                        show_legend = False,
                        legend_ax: Optional[Axes] = None,
                        despine = True,
                        show_yticklabels=False,
                        show_xticklabels=False,
                        ):
    """Categorical heatmap

    Args:
        df: Colors will be assigned to df values in sorting order.
            Columns must have homogeneous types.
        cmap: a cmap name that sns.color_palette understands; ignored if
              colors are given
        colors: List of color specs

    Does not accept NA values.
    """

    if df.isna().any().any():
        raise ValueError('NA values not allowed')

    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError('All columns must have the same dtype')

    is_categorical = df.dtypes.iloc[0].name == 'category'

    if is_categorical:
        levels = df.dtypes.iloc[0].categories.values
    else:
        levels = np.unique(df.values)
    n_levels = len(levels)

    if colors is None:
        colors = sns.color_palette(cmap, n_levels)
    else:
        # tile colors to get n_levels color list
        colors = (np.ceil(n_levels / len(colors)).astype(int) * colors)[:n_levels]
    cmap = mpl.colors.ListedColormap(colors)


    # Get integer codes matrix for pcolormesh, ie levels are represented by
    # increasing integers according to the level ordering
    if is_categorical:
        codes_df = df.apply(lambda ser: ser.cat.codes, axis=0)
    else:
        value_to_code = {value: code for code, value in enumerate(levels)}
        codes_df = df.replace(value_to_code)

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
    patches = [mpatches.Patch(facecolor=c, edgecolor='black') for c in colors]
    if show_legend:
        if legend_ax is not None:
            legend_ax.legend(patches, levels)
        else:
            ax.legend(patches, levels)

    if show_values:
        for i in range(df.shape[1]):
            # note: categorical_series.values is not a numpy array
            # any series has __array__ method - so instead of values
            # attribute, use np.array
            y, s = find_stretches(np.array(df.iloc[:, i]))
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va='center', ha='center')

    return {'quadmesh': qm, 'patches': patches, 'levels': levels}


# Note: jit compilation does not seem to provide speed-ups. Haven't
# checked it out yet.
numba.jit(nopython=True)
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
            marks[mark_i] = start + (end - start)/2
            values[mark_i] = stretch_value
            start = end
            end += 1
            stretch_value = value
            mark_i += 1
    # add last stretch
    marks[mark_i] = start + (end - start)/2
    values[mark_i] = stretch_value
    return marks[:(mark_i + 1)], values[:(mark_i + 1)]


# From matplotlib docs:
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def heatmap(df: pd.DataFrame,
            ax: Axes, fig: Figure,
            cmap: str,
            midpoint_normalize: bool =  False,
            col_labels_show: bool = True,
            row_labels_show: bool = False,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            ):

    if midpoint_normalize:
        norm = MidpointNormalize(vmin=df.min().min(),
                                 vmax=df.max().max(),
                                 midpoint=0)
    else:
        norm = None

    qm = ax.pcolormesh(df, cmap=cmap, norm=norm)

    if col_labels_show:
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90)

    if row_labels_show:
        ax.set_yticklabels(df.index)
    else:
        ax.set_yticks([])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.colorbar(qm, ax=ax)

class Heatmap(ClusterProfilePlotPanel):
    align_vars = ['df']
    supply_vars = {'df': 'main_df'}
    plotter = staticmethod(heatmap)


def simple_line(ax):
    ax.plot([1, 2, 3, 4, 5])
class SimpleLine(ClusterProfilePlotPanel):
    align_vars = []
    supply_vars = {}
    row_deco = False
    col_deco = False
    plotter = staticmethod(simple_line)


def cluster_size_plot(cluster_ids: pd.Series, ax: Axes,
                      bar_height = 0.6, xlabel=None):
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
    """
    cluster_sizes = cluster_ids.value_counts()
    cluster_sizes.sort_index(inplace=True)
    ax.barh(y=np.arange(0.5, cluster_sizes.shape[0]),
            width=cluster_sizes.values,
            height=bar_height)
    ax.set_ylim(0, cluster_sizes.shape[0])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    remove_axis(ax, x=False)

class ClusterSizePlot(ClusterProfilePlotPanel):
    """Wrapper for cluster_size_plot

    No alignment is performed. The cluster_ids are supplied if necessary
    and possible.
    """
    align_vars = []
    supply_vars = {'cluster_ids': 'cluster_ids'}
    row_deco = False
    col_deco = False
    plotter = staticmethod(cluster_size_plot)

def col_agg_plot(df: pd.DataFrame, fn: Callable, ax: Axes,
                 xlabel=None):
    """Plot aggregate statistic as line plot

    Args:
        fn: function which will be passed to pd.DataFrame.agg
    """
    agg_stat = df.agg(fn)
    ax.plot(np.arange(0.5, df.shape[1]),
            agg_stat.values)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


class ColAggPlot(ClusterProfilePlotPanel):
    """Wrapper around col_agg_plot"""
    align_vars = ['df']
    supply_vars = {'df': 'main_df'}
    row_deco = False
    col_deco = True
    plotter = staticmethod(col_agg_plot)


def remove_axis(ax: Axes, y=True, x=True):
    if y:
        ax.set(yticks=[], yticklabels=[])
        sns.despine(ax=ax, left=True)
    if x:
        ax.set(xticks=[], xticklabels=[])
        sns.despine(ax=ax, bottom=True)
