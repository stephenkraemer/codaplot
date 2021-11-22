from .array_manager import cross_plot, anno_axes, spacer, array_to_figure, add_guides
from .clustered_data_grid import ClusteredDataGrid
from .plotting import categorical_heatmap, find_stretches, spaced_heatmap, heatmap_depr, heatmap, frame_groups, label_groups, add_cluster_anno_bubbles, adjust_coords, cbar_change_style_to_inward_white_ticks
from .flow_plot import flow_plot
from .dodge import dodge_intervals_horizontally
from .legend import create_legend_for_norm_size_patches

from .clustered_data_grid import (
    Heatmap, ClusterSizePlot, ColAggPlot, RowGroupAggPlot,
    Violin, MultiLine, Dendrogram, AggHeatmap, SpacedHeatmap,
    CategoricalHeatmap, CategoricalColumnAnnotationHeatmap)
from .dynamic_grid import GridManager, GridElement
from .linkage_mat import Linkage
from .cluster_ids import ClusterIDs

import codaplot.utils
