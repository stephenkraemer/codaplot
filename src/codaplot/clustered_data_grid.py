#-
from abc import ABC
from functools import partial
from itertools import chain
from typing import Optional, List, Tuple, Union, Dict, Any, Type

import matplotlib as mpl

mpl.use('Agg') # import before pyplot import!

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, leaves_list

from codaplot.dynamic_grid import GridManager, GridElement, Spacer, FacetedGridElement
from codaplot.plotting import (
    categorical_heatmap, dendrogram_wrapper, heatmap, simple_line,
    cluster_size_plot, col_agg_plot, row_group_agg_plot,
    grouped_rows_violin,
    grouped_rows_line_collections,
)

MixedGrid = List[List[Union['ClusteredDataGridElement', GridElement]]]
#-

class ClusteredDataGridElement(ABC):
    align_vars: List[str] = []
    supply_vars: Dict[str, str] = {}
    align_maybe: List[str] = []
    plotter: Optional[staticmethod] = None
    row_deco: bool = True
    col_deco: bool = True

    def __init__(self, panel_width=1, panel_kind='rel', name=None, **kwargs):
        self.kwargs = kwargs
        self.panel_width = panel_width
        self.panel_kind = panel_kind
        self.name = name
        plotter = type(self).__dict__['plotter']
        if plotter is not None and not isinstance(plotter, staticmethod):
            raise TypeError('Error in class definition: '
                            'plotter must be a static method')

    def align_and_supply(self, cluster_profile_plot):

        for target, value in self.supply_vars.items():
            if target not in self.kwargs:
                self.kwargs[target] = getattr(cluster_profile_plot, value)

        row_slice = cluster_profile_plot.row_int_idx
        if row_slice is None:
            row_slice = slice(None)
        col_slice = cluster_profile_plot.col_int_idx
        if col_slice is None:
            col_slice = slice(None)

        for var in self.align_vars:
            self.kwargs[var] = (self.kwargs[var]
                                .iloc[row_slice, col_slice]
                                .copy()
                                )

        for var in self.align_maybe:
            var_value = self.kwargs[var]
            if isinstance(var_value, pd.DataFrame):
                self.kwargs[var] = (var_value
                                    .iloc[row_slice, col_slice]
                                    .copy()
                                    )
            elif isinstance(var_value, pd.Series):
                self.kwargs[var] = (var_value
                                    .iloc[row_slice]
                                    .copy()
                                    )

    def plot(self):
        self.plotter(**self.kwargs)
#-

# Note: in principle, this class can also be useful to plot data which
# need to be aligned with an index with no relation to any clustering
# If this use case becomes relevant, it may make sense to provide to
# different classes: ClusteredDataGrid and AlignedDataGridPlot
# They could both inherit from a DataGridPlot super class
# The ClusteredDataGrid would define methods for clustering and
# plotting dendrograms, which are not necessary in the AlignedDataGridPlot

@dataclass
class ClusteredDataGrid:
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
                     ) -> 'ClusteredDataGrid':
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
                     ) -> 'ClusteredDataGrid':
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

        Plots are aligned if they are given as ClusteredDataGridElement
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

        if height_ratios is None:
            # noinspection PyUnusedLocal
            height_ratios = [(1, 'rel') for unused_row in grid]

        processed_grid = self._convert_panel_element_to_grid_element(grid)

        # Create a GridManager from the 'base grid', then use the GridManager
        # grid manipulation functions to add the decoration rows and columns

        # noinspection PyUnusedLocal
        gm = GridManager(processed_grid, height_ratios=height_ratios,
                         figsize=figsize,
                         fig_args={} if fig_args is None else fig_args,
                         h_pad=h_pad, w_pad=w_pad, hspace=hspace, wspace=wspace)

        self._add_row_decoration(gm, row_anno_col_width, row_anno_heatmap_args,
                                 row_annotation, row_dendrogram)

        self._add_column_decoration(col_dendrogram, gm)


        return gm

    def _add_column_decoration(self, col_dendrogram, gm):
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

    def _add_row_decoration(self, gm, row_anno_col_width,
                            row_anno_heatmap_args: Optional[Dict],
                            row_annotation, row_dendrogram):
        """Add row annotation and dendrogram if required"""

        if row_annotation is not None:
            if row_anno_heatmap_args is None:
                row_anno_heatmap_args = {}
            assert isinstance(row_annotation, pd.DataFrame)
            if self.row_int_idx is not None:
                row_annotation = row_annotation.iloc[self.row_int_idx, :]
            row_anno_width_kind = 'abs'
            row_anno_col: List[GridElement] = []
            row_anno_partial_constructor = partial(
                    GridElement, width=row_anno_col_width, kind=row_anno_width_kind,
                    tags=['no_col_dendrogram'],
                    plotter=categorical_heatmap,
                    df=row_annotation, **row_anno_heatmap_args)
        if row_dendrogram:
            row_dendrogram_width = 1 / 2.54
            row_dendrogram_width_kind = 'abs'
            row_dendro_col: List[GridElement] = []
            row_dendro_ge_partial_constructor = partial(
                    GridElement,
                    width=row_dendrogram_width, kind=row_dendrogram_width_kind,
                    plotter=dendrogram_wrapper,
                    linkage_mat=self.row_linkage, orientation='left',
                    tags=['no_col_dendrogram'])
        if row_dendrogram or row_annotation is not None:
            for row in gm.grid:
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

    def _convert_panel_element_to_grid_element(self, grid: MixedGrid):
        """Convert PanelElement in grid into GridElement

        Grid may contain PanelElement or GridElement instances.

        PanelElement instances use their align and supply methods, before
        their relevant attributes and their arguments are transferred to a GridElement.
        """

        # noinspection PyUnusedLocal
        processed_grid: List[List[Optional[GridElement]]] = [
            [None for unused in range(len(row))] for row in grid]
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
                    # noinspection PyUnusedLocal
                    klass: Type[GridElement]
                    if 'row' in panel_or_grid_element.kwargs:
                        klass = FacetedGridElement
                    else:
                        klass = GridElement
                    processed_grid[row_idx][col_idx] = klass(  # type: ignore
                            name=name,
                            plotter=panel_or_grid_element.plotter,
                            width=panel_or_grid_element.panel_width,
                            kind=panel_or_grid_element.panel_kind,
                            tags=tags,
                            **panel_or_grid_element.kwargs)
        assert all(x is not None for x in chain(*processed_grid))
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


# Ready-made DataGridplotElements
# ###############################

class Heatmap(ClusteredDataGridElement):
    align_vars = ['df']
    supply_vars = {'df': 'main_df'}
    plotter = staticmethod(heatmap)


class SimpleLine(ClusteredDataGridElement):
    align_vars: List[str] = []
    supply_vars: Dict[str, str] = {}
    row_deco = False
    col_deco = False
    plotter = staticmethod(simple_line)


class ClusterSizePlot(ClusteredDataGridElement):
    """Wrapper for cluster_size_plot

    No alignment is performed. The cluster_ids are supplied if necessary
    and possible.
    """
    align_vars: List[str] = []
    supply_vars = {'cluster_ids': 'cluster_ids'}
    row_deco = True
    col_deco = False
    plotter = staticmethod(cluster_size_plot)


class ColAggPlot(ClusteredDataGridElement):
    """Wrapper around col_agg_plot"""
    align_vars = ['df']
    supply_vars = {'df': 'main_df'}
    row_deco = False
    col_deco = True
    plotter = staticmethod(col_agg_plot)


class RowGroupAggPlot(ClusteredDataGridElement):
    row_deco = False
    col_deco = True
    plotter = staticmethod(row_group_agg_plot)
    align_vars = ['data']
    align_maybe = ['row']
    supply_vars = {'data': 'main_df'}


class Violin(ClusteredDataGridElement):
    row_deco = False
    col_deco = True
    align_vars = ['data']
    align_maybe = ['row']
    supply_vars = {'data': 'main_df'}
    plotter = staticmethod(grouped_rows_violin)

class MultiLine(ClusteredDataGridElement):
    row_deco = False
    col_deco = True
    align_vars = ['data']
    align_maybe = ['row']
    supply_vars = {'data': 'main_df'}
    plotter = staticmethod(grouped_rows_line_collections)

