#-
from typing import Optional, List, Tuple, Union, Callable
import matplotlib.patches as mpatches

import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib.axes import Axes  # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
from pandas.api.types import is_numeric_dtype
from scipy._lib.decorator import getfullargspec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

mpl.use('Agg') # import before pyplot import!
import seaborn as sns

from complex_heatmap.dynamic_grid import GridManager, GridElement
#-

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

@dataclass
class ComplexHeatmap:
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

class AnnoPlot:
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
class ComplexHeatmapList:
    hmaps: List[ComplexHeatmap]
    figsize: Tuple[float, float] = (15/2.54, 15/2.54)
    dpi: int =  180

    def plot(self):

        main_hmap = [hmap for hmap in self.hmaps if hmap.is_main]
        assert len(main_hmap) == 1
        main_hmap = main_hmap[0]
        main_hmap.title += '\n(main)'
        show_cols = main_hmap.col_show_list

        # GridManager currently requires names for each GridElement
        assert all(hmap.title is not None for hmap in self.hmaps)

        col_Z, row_Z = self._determine_linkage(main_hmap)

        heatmap_row = [
            GridElement(hmap.title, 1, plotter=self.heatmap,
                        hmap=hmap, col_Z=col_Z, row_Z=row_Z, show_cols=show_cols)
            for hmap in self.hmaps]

        heights = [(1, 'rel')]
        col_dendro_row = None
        if col_Z is not None and main_hmap.col_dendrogram_show:
            col_dendro_row = [GridElement(f'col_dendro_{i}', 1, 'rel',
                                          self.dendrogram, row_Z, orientation='top')
                              for i in range(len(self.hmaps))]
            heights.insert(0, (main_hmap.col_dendrogram_height, 'abs'))
        if main_hmap.row_anno is not None:
            heatmap_row.insert(0, GridElement(
                    'row_anno', main_hmap.row_anno_width, 'abs',
                    self._anno_bar, anno_df=main_hmap.row_anno
            ))
            if col_dendro_row is not None:
                col_dendro_row.insert(0,
                      GridElement('spacer2', main_hmap.row_anno_width, 'abs'))
        if row_Z is not None and main_hmap.row_dendrogram_show:
            heatmap_row.insert(0, GridElement(
                    'row_dendrogram', main_hmap.row_dendrogram_width, 'abs',
                    self.dendrogram, row_Z, orientation='left')
                               )
            if col_dendro_row is not None:
                col_dendro_row.insert(0,
                                      GridElement('spacer1',main_hmap.row_dendrogram_width, 'abs'))

        if col_dendro_row is not None:
            grid = [col_dendro_row, heatmap_row]
        else:
            grid = [heatmap_row]
        gm = GridManager(grid, height_ratios=heights, figsize=self.figsize)

        title_axes = gm.axes_list[0][-len(self.hmaps):]

        for ax, hmap in zip(title_axes, self.hmaps):
            if hmap.title:
                ax.set_title(hmap.title)

        return gm.fig


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

    def compute_grid_ratios(self, ratios: List[Tuple[float, str]]) -> np.ndarray:
        width_float, width_anno = zip(*ratios)
        width_float = np.array(width_float)
        width_anno = np.array(width_anno)
        absolute_width = width_float[width_anno == 'absolute'].sum()
        remaining_figsize = self.figsize[0] - absolute_width
        rel_widths= width_float[width_anno == 'relative']
        width_float[width_anno == 'relative'] = rel_widths / rel_widths.sum() * remaining_figsize
        return width_float

    @staticmethod
    def dendrogram(Z, ax: Axes, orientation: str):
        dendrogram(Z, ax=ax, color_threshold=-1,
                   above_threshold_color='black', orientation=orientation)
        ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
        sns.despine(ax=ax, bottom=True, left=True)


    @staticmethod
    def heatmap(ax: Axes, fig: Figure, hmap: ComplexHeatmap,
                col_Z=None, row_Z=None, show_cols=None):

        if col_Z is not None:
            col_idx = leaves_list(col_Z)
            if show_cols:
                col_idx = [i for i in col_idx if hmap.df.columns[i] in show_cols]
        else:
            if show_cols:
                col_idx = [i for i in range(hmap.df.shape[1]) if hmap.df.columns[i] in show_cols]
            else:
                col_idx = slice(None)
        if row_Z is not None:
            row_idx = leaves_list(row_Z)
        else:
            row_idx = slice(None)

        if hmap.cmap is not None:
            cmap = hmap.cmap
        elif hmap.cmap_class == 'auto':
            if is_numeric_dtype(hmap.df.values):
                if hmap.df.min().min() < 0 < hmap.df.max().max():
                    cmap = hmap.cmap_divergent
                else:
                    cmap = hmap.cmap_sequential
            else:
                cmap = hmap.cmap_qualitative
        elif hmap.cmap_class == 'sequential':
            cmap = hmap.cmap_sequential
        elif hmap.cmap_class == 'divergent':
            cmap = hmap.cmap_divergent
        elif hmap.cmap_class == 'qualitative':
            cmap = hmap.cmap_qualitative
        else:
            raise ValueError('Unknown cmap_class')

        plot_df = hmap.df.iloc[row_idx, col_idx]

        norm = None
        if hmap.cmap_norm == 'midpoint':
            norm = MidpointNormalize(vmin=hmap.df.min().min(),
                                     vmax=hmap.df.max().max(),
                                     midpoint=0)
            print('using midpoint norm')

        qm = ax.pcolormesh(plot_df, cmap=cmap, norm=norm)

        if hmap.col_labels_show:
            ax.set_xticks(np.arange(plot_df.shape[1]) + 0.5)
            ax.set_xticklabels(plot_df.columns, rotation=90)

        if hmap.row_labels_show:
            ax.set_yticklabels(plot_df.index)
        else:
            ax.set_yticks([])

        fig.colorbar(qm, ax=ax, shrink=0.7, orientation='horizontal', aspect=5)


def categorical_heatmap(df: pd.DataFrame, ax: Axes,
                        cmap: str = None, colors: List = None,
                        show_values = True,
                        show_legend = True,
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
            y, s = find_stretches(df.iloc[:, i])
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s))

    return {'quadmesh': qm, 'patches': patches, 'levels': levels}


numba.jit(nopython=True)
def find_stretches(arr):
    """Find stretches """
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



