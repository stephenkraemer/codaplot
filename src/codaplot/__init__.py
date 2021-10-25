from .array_manager import cross_plot, anno_axes, spacer, array_to_figure
from .clustered_data_grid import ClusteredDataGrid
from .plotting import categorical_heatmap, find_stretches, spaced_heatmap, heatmap_depr, heatmap, frame_groups, label_groups, add_cluster_anno_bubbles, adjust_coords
from .flow_plot import flow_plot

from .clustered_data_grid import (
    Heatmap, ClusterSizePlot, ColAggPlot, RowGroupAggPlot,
    Violin, MultiLine, Dendrogram, AggHeatmap, SpacedHeatmap,
    CategoricalHeatmap, CategoricalColumnAnnotationHeatmap)
from .dynamic_grid import GridManager, GridElement
from .linkage_mat import Linkage
from .cluster_ids import ClusterIDs
