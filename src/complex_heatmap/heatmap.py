#-
from typing import Optional, List, Tuple, Union

import pandas as pd
import numpy as np
from dataclasses import dataclass

import matplotlib
from matplotlib.axes import Axes # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

matplotlib.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#-


@dataclass
class ComplexHeatmap:
    df: pd.DataFrame
    cmap: Optional[str] = None
    cmap_divergent: str = 'RdBu_r'
    cmap_sequential: str = 'YlOrBr'
    is_main: bool = False
    cluster_cols: bool = True
    cluster_cols_method: str = 'complete'
    cluster_cols_metric: str = 'euclidean'
    col_linkage_matrix: Optional[np.ndarray] = None
    col_dendrogram_height_cm = 2
    col_dendrogram_show: bool = True
    cluster_rows: bool = True
    cluster_rows_method: str = 'complete'
    cluster_rows_metric: str = 'euclidean'
    row_linkage_matrix: Optional[np.ndarray] = None
    row_dendrogram_width_cm = 2
    row_dendrogram_show: bool = True
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
        height_ratios = np.array([hmap.rel_height for hmap in self.hmaps])

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
            height_ratios = np.concatenate([np.array([main_hmap.col_dendrogram_height_cm]),
                                     norm_heights * self.figsize[1]])
            heatmap_row = 1
        else:
            heatmap_row = 0
        if main_hmap.cluster_rows and main_hmap.row_dendrogram_show:
            norm_widths = width_ratios / np.sum(width_ratios)
            width_ratios = np.concatenate([np.array([main_hmap.col_dendrogram_height_cm]),
                                      norm_widths * self.figsize[1]])
            heatmap_col_start = 1
        else:
            heatmap_col_start = 0

        fig = plt.figure(constrained_layout=True, figsize=self.figsize)
        gs = gridspec.GridSpec(ncols=len(width_ratios), nrows=len(height_ratios),
                               width_ratios=width_ratios,
                               height_ratios=height_ratios,
                               figure=fig,
                               hspace=0, wspace=0)

        heatmap_axes = [fig.add_subplot(gs[heatmap_row, heatmap_col_start + i])
                        for i in range(len(self.hmaps))]
        if main_hmap.row_dendrogram_show:
            ax_row_dendro = fig.add_subplot(gs[heatmap_row, 0])
        if main_hmap.col_dendrogram_show:
            col_dendro_axes = [fig.add_subplot(gs[heatmap_row - 1, heatmap_col_start + i])
                                               for i in range(len(self.hmaps))]

        for hmap_ax, hmap in zip(heatmap_axes, self.hmaps):
            self.heatmap(ax=hmap_ax, hmap=hmap, col_Z=col_Z, row_Z=row_Z)

        if main_hmap.col_dendrogram_show:
            self.dendrogram(col_Z, col_dendro_axes, orientation='top')

        if main_hmap.row_dendrogram_show:
            self.dendrogram(row_Z, ax_row_dendro, orientation='left')

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




    def heatmap(self, ax, hmap: ComplexHeatmap, col_Z=None, row_Z=None):

        if col_Z is not None:
            col_idx = leaves_list(col_Z)
        else:
            col_idx = slice(None)
        if row_Z is not None:
            row_idx = leaves_list(row_Z)
        else:
            row_idx = slice(None)

        ax.pcolormesh(hmap.df.loc[row_idx, col_idx])




