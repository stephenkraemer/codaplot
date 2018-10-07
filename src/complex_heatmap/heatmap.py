#-
from abc import ABC
from inspect import getfullargspec
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

from complex_heatmap.dynamic_grid import GridManager, GridElement
#-

# From matplotlib docs:



# def align(vars: List[str]):
#     def decorate(f):
#         @wraps(f)
#         def wrapper(cluster_profile_plot, **kwargs):
#             for var in vars:
#                 kwargs[var] = (kwargs[var]
#                                 .iloc[slice(cluster_profile_plot.row_int_idx),
#                                       slice(cluster_profile_plot.col_int_idx)]
#                                 .copy()
#                                 )
#             return f(**kwargs)
#         return wrapper
#     return decorate
#
# aligned_heatmap = align(['df'])
#
# def delayed(f):
#     @wraps(f)
#     def wrapped(*args, **kwargs):
#         def delayed():
#             return f(*args, **kwargs)
#         return delayed
#     return wrapped
#
# def f1(a, b, c):
#     return a + b + c
#
#-
class ClusterProfilePlotPanel(ABC):
    align_vars: List[str] = []
    supply_vars: Dict[str, str] = []
    plotter: Callable = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert isinstance(type(self).__dict__['plotter'], staticmethod), \
        'plotter must be a static method'

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

class SnsHeatmap(ClusterProfilePlotPanel):
    align_vars = ['data']
    plotter = staticmethod(sns.heatmap)
#
#
# h = SnsHeatmap(x=[1, 2, 3, 1])
# h.plot()
#
# def align_data(f):
#     class NoName(Plot):
#         plotter = staticmethod(f)
#         align_vars = ['data']
#     return NoName
#
# def align(f, align_vars = None, supply_vars = None):
#     class NoName(Plot):
#         plotter = staticmethod(f)
#         align_vars = align_vars
#         supply_vars = supply_vars
#     return NoName
#
#
# a1 = align_data(fn)
# a1(x = [1, 2]).plot()
# a2 = align_data(lambda x, y: x / y)
# a1.plot(1, 2, 3)
# a2.plot(1, 3)
#
#
# res = delayed(f1)
# res2 = res(a=1, b=2, c=3)

@dataclass
class UnusedHeatmap:
    df: pd.DataFrame
    title: Optional[str] = None
    cmap: Optional[str] = None
    cmap_class: str = 'auto'
    cmap_divergent: str = 'RdBu_r'
    cmap_sequential: str = 'YlOrBr'
    cmap_qualitative: str = 'Set1'
    cmap_norm: Optional[Union[str, mpl.colors.Normalize]] = None
    is_main: bool = False
    cluster_cols: bool = True
    cluster_use_cols: Optional[List[str]] = None
    cluster_cols_method: str = 'complete'
    cluster_cols_metric: str = 'euclidean'
    col_linkage_matrix: Optional[np.ndarray] = None
    col_dendrogram_height: float = 2.0
    col_dendrogram_show: bool = True
    col_labels_show: bool = True
    cluster_rows: bool = True
    row_anno: pd.DataFrame = None
    row_anno_width_per_col: float = 1/2.54
    cluster_rows_method: str = 'complete'
    cluster_rows_metric: str = 'euclidean'
    row_linkage_matrix: Optional[np.ndarray] = None
    row_dendrogram_width: float = 2.0
    row_dendrogram_show: bool = True
    row_labels_show: bool = False
    col_show_list: Optional[List[str]] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    rel_width: int = 1
    rel_height: int = 1

    def __post_init__(self):
        if self.col_show_list is not None and self.col_dendrogram_show:
            raise NotImplementedError()
        if self.row_anno is not None:
            assert isinstance(self.row_anno, pd.DataFrame)
            self.row_anno_width = self.row_anno.shape[1] * self.row_anno_width_per_col

    def plot(self) -> Figure:
        raise NotImplementedError()

class Plotter:
    def __init__(self, plotter: Callable, align_targets: Optional[List[str]], **kwargs):
        assert 'ax' in getfullargspec(plotter).args
        self.plotter = plotter
        self.align_targets = align_targets
        self.kwargs = kwargs
    def plot(self, ax, idx):
        if self.align_targets is not None:
            for align_target in self.align_targets:
                if not self.kwargs[align_target].index.equals(idx):
                    self.kwargs[align_target] = self.kwargs[align_target].loc[idx]
        self.plotter(ax=ax, **self.kwargs)


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
    row_anno: Optional[List[Plotter]] = None
    col_anno: Optional[List[Plotter]] = None
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

    def plot_grid(self, old_grid: List[List['ClusterProfilePlotPanel']],
                  figsize: Tuple[float, float],
                  h_pad = 1/72, w_pad = 1/72, hspace=1/72, wspace=1/72,
                  row_dendrogram = False, col_dendrogram = False,
                  row_annotation: Optional[pd.DataFrame] = None,
                  row_anno_heatmap_args: Optional[Dict[str, Any]] = None,
                  row_anno_col_width: float = 0.6/2.54,
                  fig_args: Optional[Dict[str, Any]] = None
                  ):
        """Create grid of plots with an optionally underlying clustering

        Row and column dendrograms and annotations can be added automatically
        or be placed in the grid for finer control.

        If the clustering of the data was computed using the GridManager
        cluster_{cols,rows} methods or by passing the respective linkage
        matrices from outside, plots can optionally be aligned based on the
        clustering.

        Plots are aligned if they are given as ClusterProfilePlotPanel
        subclass with variables to be aligned indicated in the align_vars
        class variable.
        """

        # noinspection PyUnusedLocal
        height_ratios = [(1, 'rel') for unused_row in old_grid]

        # noinspection PyUnusedLocal
        new_grid: List[List[GridElement]] = [[[] for unused in range(len(row))] for row in old_grid]
        for row_idx, row in enumerate(old_grid):
            for col_idx, panel_element in enumerate(row):
                panel_element.align_and_supply(self)
                new_grid[row_idx][col_idx] = GridElement(f'{row_idx}_{col_idx}',
                                                         plotter=panel_element.plotter,
                                                         **panel_element.kwargs)

        # noinspection PyUnusedLocal
        gm = GridManager(new_grid, height_ratios=height_ratios,
                         figsize=figsize, fig_args=fig_args,
                         h_pad=h_pad, w_pad=w_pad, hspace=hspace, wspace=wspace)

        if row_annotation is not None:
            assert isinstance(row_annotation, pd.DataFrame)
            if self.row_int_idx is not None:
                row_annotation = row_annotation.iloc[self.row_int_idx, :]
            anno_ge = GridElement('row_anno', width=row_anno_col_width, kind='abs',
                                  tags=['no_col_dendrogram'],
                                  plotter=categorical_heatmap,
                                  df=row_annotation, **row_anno_heatmap_args)
            row_annotation_only_rows = [row_idx for row_idx, row in enumerate(gm.grid)
                                        if not 'no_row_dendrogram' in row[0].tags]
            gm.prepend_col(anno_ge, only_rows=row_annotation_only_rows)

        if row_dendrogram:
            row_dendro_ge = GridElement('row_dendrogram', width=1/2.54,
                                        kind='abs', plotter=dendrogram_wrapper,
                                        linkage_mat=self.row_linkage,
                                        orientation='left',
                                        tags=['no_col_dendrogram'])
            row_dendro_only_rows = []
            for row_idx, row in enumerate(gm.grid):
                if not 'no_row_dendrogram' in row[0].tags:
                    row_dendro_only_rows.append(row_idx)
            gm.prepend_col(row_dendro_ge, only_rows=row_dendro_only_rows)

        if col_dendrogram:
            col_dendro_ge = GridElement('col_dendrogram', plotter=dendrogram_wrapper,
                                        linkage_mat=self.col_linkage,
                                        orientation='top',
                                        tags=['no_row_dendrogram']
                                        )
            col_dendro_only_cols = [i for i, ge in enumerate(gm.grid[0])
                                    if not 'no_col_dendrogram' in ge.tags]
            gm.insert_matched_row(0, col_dendro_ge, height=(1/2.54, 'abs'),
                                  only_cols=col_dendro_only_cols)


        return gm



    # def unused_plot(self):
    #
    #     main_hmap = [hmap for hmap in self.hmaps if hmap.is_main]
    #     assert len(main_hmap) == 1
    #     main_hmap = main_hmap[0]
    #     main_hmap.title += '\n(main)'
    #     show_cols = main_hmap.col_show_list
    #
    #     # GridManager currently requires names for each GridElement
    #     assert all(hmap.title is not None for hmap in self.hmaps)
    #
    #     col_Z, row_Z = self._determine_linkage(main_hmap)
    #
    #     heatmap_row = [
    #         GridElement(hmap.title, 1, plotter=self.heatmap,
    #                     hmap=hmap, col_Z=col_Z, row_Z=row_Z, show_cols=show_cols)
    #         for hmap in self.hmaps]
    #
    #     heights = [(1, 'rel')]
    #     col_dendro_row = None
    #     if col_Z is not None and main_hmap.col_dendrogram_show:
    #         col_dendro_row = [GridElement(f'col_dendro_{i}', 1, 'rel',
    #                                       self.dendrogram, row_Z, orientation='top')
    #                           for i in range(len(self.hmaps))]
    #         heights.insert(0, (main_hmap.col_dendrogram_height, 'abs'))
    #     if main_hmap.row_anno is not None:
    #         heatmap_row.insert(0, GridElement(
    #                 'row_anno', main_hmap.row_anno_width, 'abs',
    #                 self._anno_bar, anno_df=main_hmap.row_anno
    #         ))
    #         if col_dendro_row is not None:
    #             col_dendro_row.insert(0,
    #                   GridElement('spacer2', main_hmap.row_anno_width, 'abs'))
    #     if row_Z is not None and main_hmap.row_dendrogram_show:
    #         heatmap_row.insert(0, GridElement(
    #                 'row_dendrogram', main_hmap.row_dendrogram_width, 'abs',
    #                 self.dendrogram, row_Z, orientation='left')
    #                            )
    #         if col_dendro_row is not None:
    #             col_dendro_row.insert(0,
    #                                   GridElement('spacer1',main_hmap.row_dendrogram_width, 'abs'))
    #
    #     if col_dendro_row is not None:
    #         grid = [col_dendro_row, heatmap_row]
    #     else:
    #         grid = [heatmap_row]
    #     gm = GridManager(grid, height_ratios=heights, figsize=self.figsize)
    #
    #     title_axes = gm.axes_list[0][-len(self.hmaps):]
    #
    #     for ax, hmap in zip(title_axes, self.hmaps):
    #         if hmap.title:
    #             ax.set_title(hmap.title)
    #
    #     return gm.fig


    @staticmethod
    def _anno_bar(anno_df, ax):
        pass

        # listed_colormap = ListedColormap(
        #         pd.Series(CMAP_DICT['cluster_id_to_color'])
        #             .loc[sorted(main_cluster_ids_ser.unique())])
        # ax.pcolormesh(C, cmap=listed_colormap)
        # ax.set(yticks=[], yticklabels=[])
        #
        # if other_cluster_ids_df is not None:
        #     ax.set_xticklabels(C.columns, rotation=90)
        # else:
        #     ax.set(xticks=[], xticklabels=[])
        #
        # value_counts = main_cluster_ids_ser.value_counts().sort_index()
        # cluster_labels_ys = (value_counts.cumsum() - value_counts / 2).round().astype(int)
        # cluster_labels_xs = [0.5] * cluster_labels_ys.shape[0]
        # cluster_labels_text = main_cluster_ids_ser.unique().astype(str)
        #
        # for x,y,t in zip(cluster_labels_xs, cluster_labels_ys, cluster_labels_text):
        #     ax.text(x,y,t,
        #             horizontalalignment='center',
        #             verticalalignment='center')
        #
        # # ax.set_xlim([-2, 2])
        # # ax.set_ylim([0, 1000])
        # sns.despine(ax=ax, left=True, bottom=True)

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

    # def compute_grid_ratios(self, ratios: List[Tuple[float, str]]) -> np.ndarray:
    #     width_float, width_anno = zip(*ratios)
    #     width_float = np.array(width_float)
    #     width_anno = np.array(width_anno)
    #     absolute_width = width_float[width_anno == 'absolute'].sum()
    #     remaining_figsize = self.figsize[0] - absolute_width
    #     rel_widths= width_float[width_anno == 'relative']
    #     width_float[width_anno == 'relative'] = rel_widths / rel_widths.sum() * remaining_figsize
    #     return width_float


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
            y, s = find_stretches(df.iloc[:, i].values)
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va='center', ha='center')

    return {'quadmesh': qm, 'patches': patches, 'levels': levels}


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


def agg_line(df: pd.DataFrame, gs_tuple, ax: Axes, fig: Figure, cluster_ids: pd.Series, fn: Callable,
             sharey=True, ylim=None):
    agg_values = df.groupby(cluster_ids).agg(fn)
    if sharey and ylim is None:
        ymax = np.max(agg_values.values)
        pad = abs(0.1 * ymax)
        padded_ymax = ymax + pad
        if ymax < 0 < padded_ymax:
            padded_ymax = 0
        ymin = np.min(agg_values.values)
        padded_ymin = ymin - pad
        if ymin > 0 > padded_ymin:
            padded_ymin = 0
        ylim = (padded_ymin, padded_ymax)
    n_clusters = agg_values.shape[0]
    gssub = gs_tuple.subgridspec(n_clusters, 1)
    ax.remove()
    for row_idx, (cluster_id, row_ser) in enumerate(agg_values.iterrows()):
        ax = fig.add_subplot(gssub[row_idx, 0])
        ax.set(xticks=[], xticklabels=[])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(row_ser.values, marker='.', linestyle='-')
    ax.set(xticks=[1, 2, 3], xticklabels=[1, 2, 3], xlabel='test')
    # ax.plot([1, 2, 3], label='inline label')
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

class AggLine(ClusterProfilePlotPanel):
    align_vars = ['df']
    supply_vars = {'df': 'main_df'}
    plotter = staticmethod(agg_line)



