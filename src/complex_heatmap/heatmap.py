#-
from typing import Optional, List, Tuple, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from dataclasses import dataclass

import matplotlib as mpl
from matplotlib.axes import Axes # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

mpl.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
    cluster_cols_method: str = 'complete'
    cluster_cols_metric: str = 'euclidean'
    col_linkage_matrix: Optional[np.ndarray] = None
    col_dendrogram_height: float = 2.0
    col_dendrogram_show: bool = True
    col_labels_show: bool = True
    cluster_rows: bool = True
    cluster_rows_method: str = 'complete'
    cluster_rows_metric: str = 'euclidean'
    row_linkage_matrix: Optional[np.ndarray] = None
    row_dendrogram_width: float = 2.0
    row_dendrogram_show: bool = True
    row_labels_show: bool = False
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    rel_width: int = 1
    rel_height: int = 1

    def plot(self) -> Figure:
        return Figure

@dataclass
class ComplexHeatmapList:
    hmaps: List[ComplexHeatmap]
    figsize: Tuple[float, float] = (15/2.54, 15/2.54)

    def plot(self):
        width_ratios = np.array([hmap.rel_width for hmap in self.hmaps])
        height_ratios = np.array([1])

        main_hmap = [hmap for hmap in self.hmaps if hmap.is_main]
        assert len(main_hmap) == 1
        main_hmap = main_hmap[0]

        if main_hmap.cluster_cols and main_hmap.col_linkage_matrix is None:
            col_Z = linkage(main_hmap.df.T, method=main_hmap.cluster_cols_method,
                            metric=main_hmap.cluster_cols_metric)
        else:
            col_Z = main_hmap.col_linkage_matrix

        if main_hmap.cluster_rows and main_hmap.row_linkage_matrix is None:
            row_Z = linkage(main_hmap.df, method=main_hmap.cluster_rows_method,
                            metric=main_hmap.cluster_rows_metric)
        else:
            row_Z = main_hmap.row_linkage_matrix

        if main_hmap.cluster_cols and main_hmap.col_dendrogram_show:
            norm_heights = height_ratios / np.sum(height_ratios)
            height_ratios = np.concatenate([
                np.array([main_hmap.col_dendrogram_height]),
                norm_heights * (self.figsize[1] - main_hmap.col_dendrogram_height)])
            heatmap_row = 1
            print(height_ratios)
        else:
            heatmap_row = 0
        if main_hmap.cluster_rows and main_hmap.row_dendrogram_show:
            norm_widths = width_ratios / np.sum(width_ratios)
            width_ratios = np.concatenate([
                np.array([main_hmap.row_dendrogram_width]),
                norm_widths * (self.figsize[0] - main_hmap.row_dendrogram_width)])
            heatmap_col_start = 1
            print(width_ratios)
        else:
            heatmap_col_start = 0

        fig = plt.figure(constrained_layout=True, figsize=self.figsize)
        gs = gridspec.GridSpec(ncols=len(width_ratios), nrows=len(height_ratios),
                               width_ratios=width_ratios,
                               height_ratios=height_ratios,
                               figure=fig,
                               hspace=0, wspace=0)

        hmap_axes = [fig.add_subplot(gs[heatmap_row, heatmap_col_start + i])
                        for i in range(len(self.hmaps))]
        if main_hmap.row_dendrogram_show:
            ax_row_dendro = fig.add_subplot(gs[heatmap_row, 0])
        if main_hmap.col_dendrogram_show:
            col_dendro_axes = [fig.add_subplot(gs[heatmap_row - 1, heatmap_col_start + i])
                                               for i in range(len(self.hmaps))]

        for hmap_ax, hmap in zip(hmap_axes, self.hmaps):
            self.heatmap(ax=hmap_ax, fig=fig, hmap=hmap, col_Z=col_Z, row_Z=row_Z)

        if main_hmap.col_dendrogram_show:
            self.dendrogram(col_Z, col_dendro_axes, orientation='top')

        if main_hmap.row_dendrogram_show:
            self.dendrogram(row_Z, ax_row_dendro, orientation='left')

        if main_hmap.col_dendrogram_show:
            title_axes = col_dendro_axes
        else:
            title_axes = hmap_axes
        for ax, hmap in zip(title_axes, self.hmaps):
            if hmap.title:
                ax.set_title(hmap.title)


        return fig


    def dendrogram(self, Z, axes: Union[Axes, List[Axes]], orientation):
        kwargs = dict(
                color_threshold=-1,
                above_threshold_color='black',
                orientation=orientation,
        )
        if isinstance(axes, list):
            for ax in axes:
                dendrogram(Z, ax=ax, **kwargs)
                ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
                sns.despine(ax=ax, bottom=True, left=True)
        else:
            dendrogram(Z, ax=axes, **kwargs)
            axes.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
            sns.despine(ax=axes, bottom=True, left=True)


    def heatmap(self, ax: Axes, fig: Figure, hmap: ComplexHeatmap, col_Z=None, row_Z=None):

        if col_Z is not None:
            col_idx = leaves_list(col_Z)
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
                if hmap.df.min().min() < 0 and hmap.df.max().max() > 0:
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
            ax.set_xticklabels(plot_df.columns)

        if hmap.row_labels_show:
            ax.set_yticklabels(plot_df.index)
        else:
            ax.set_yticks([])

        fig.colorbar(qm, ax=ax, shrink=0.7, orientation='horizontal', aspect=5)





