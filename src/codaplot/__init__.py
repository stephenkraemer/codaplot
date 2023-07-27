# fmt: off
# ruff: noqa

import codaplot.plotting

from .array_manager import cross_plot, anno_axes, spacer, array_to_figure, add_guides, compute_gridspec_ratios

__all__ = ['cross_plot', 'anno_axes', 'spacer', 'array_to_figure', 'add_guides', 'compute_gridspec_ratios']

from .clustered_data_grid import ClusteredDataGrid

__all__ += ['ClusteredDataGrid']

from .plotting import categorical_heatmap, find_stretches, spaced_heatmap, heatmap_depr, heatmap, frame_groups, label_groups, add_cluster_anno_bubbles, adjust_coords, cbar_change_style_to_inward_white_ticks

__all__ += ['categorical_heatmap', 'find_stretches', 'spaced_heatmap', 'heatmap_depr', 'heatmap', 'frame_groups', 'label_groups', 'add_cluster_anno_bubbles', 'adjust_coords', 'cbar_change_style_to_inward_white_ticks']

from .jointplot import jointplot

__all__ += ['jointplot']

from .flow_plot import flow_plot

__all__ += ['flow_plot']

from .cluster_homogeneity import cluster_homogeneity_multiline_plot

__all__ += ['cluster_homogeneity_multiline_plot']

from .dodge import dodge_intervals_horizontally

__all__ += ['dodge_intervals_horizontally']

from .legend import create_legend_for_norm_size_patches

__all__ += ['create_legend_for_norm_size_patches']

from .clustered_data_grid import (
    Heatmap, ClusterSizePlot, ColAggPlot, RowGroupAggPlot,
    Violin, MultiLine, Dendrogram, AggHeatmap, SpacedHeatmap,
    CategoricalHeatmap, CategoricalColumnAnnotationHeatmap)

__all__ += [
    'Heatmap', 'ClusterSizePlot', 'ColAggPlot', 'RowGroupAggPlot',
    'Violin', 'MultiLine', 'Dendrogram', 'AggHeatmap', 'SpacedHeatmap',
    'CategoricalHeatmap', 'CategoricalColumnAnnotationHeatmap']

from .dynamic_grid import GridManager, GridElement

__all__ += ['GridManager', 'GridElement']

from .linkage_mat import Linkage

__all__ += ['Linkage']

from .cluster_ids import ClusterIDs

__all__ += ['ClusterIDs']

import codaplot.utils
