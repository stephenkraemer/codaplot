# # Setup

# ## Imports

from typing import Union, Any

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ## Types

colname_or_df_opt = Optional[Union[str, pd.DataFrame]]
str_or_num_t = Union[str, int, float]
str_or_num_t_opt = Optional[Union[str, int, float]]
cmap_t = Optional[str]
norm_t_opt = Optional[str]
kwds_t = Optional[Dict[str, Any]]
arr_or_list_opt = Optional[Union[np.ndarray, list]]


# # Shared docstrings

vmin = "p1.5 for 1.5th percentile"
vmax = "p99.5 for percentile"

# # Heatmap

patch_types = [
    'rectangle',
    'circle',
    'halfcircle',
    'triangle',
    ]
patch_t = Union[str, mpatches.Patch]

bbox_to_anchor_values = ['lower_left', 'upper_left', 'upper_right', 'lower_right', 'center']
bbox_to_anchor_values_t = str


class Heatmap:
    """
    """

    def __init__(
        self,
        # Aesthetic mappings
        color: colname_or_df_opt = None,
        size: colname_or_df_opt = None,
        data: Optional[pd.DataFrame] = None,
        is_categorical: bool = False,
        # Patch
        patch: patch_t = "rectangle",
        marker_color_vmin: str_or_num_t_opt = None,
        marker_color_vmax: str_or_num_t_opt = None,
        marker_color_vcenter: str_or_num_t_opt = None,
        marker_color_norm: norm_t_opt = None,
        marker_color_map: cmap_t = None,
        marker_size_vmin: str_or_num_t_opt = None,
        marker_size_vmax: str_or_num_t_opt = None,
        marker_size_vcenter: str_or_num_t_opt = None,
        marker_size_norm: norm_t_opt = None,
        # How to draw the patch into the field
        field_max_fraction: float = 1.0,
        field_loc: Tuple[float, float] = (0.5, 0.5),
        field_bbox_to_anchor: bbox_to_anchor_values_t = "center",
        # Guides
        guides_color_title: Optional[str] = None,
        guides_color_cbar_kwds: kwds_t = None,
        guides_color_legend_kwds: kwds_t = None,
        guides_size_title: Optional[str] = None,
        guides_size_legend_kwds: kwds_t = None,
        guides_ax: Optional[Axes] = None,
        guides_add: bool = True,
        # Spacing
        spacing_row_ids: arr_or_list_opt = None,
        spacing_row_gap_sizes: arr_or_list_opt = None,
        spacing_col_ids: arr_or_list_opt = None,
        spacing_col_gap_sizes: arr_or_list_opt = None,
    ) -> None:
        pass

    def draw(self, ax: Axes):
        pass

# # Patch drawing functions

def add_full_field_rectangle_patch(
        height,
        width,
        color,
        field_x_left,
        field_y_low,
        field_width,
        field_height,
        bbox_to_anchor,
        loc,
        ):
    pass

def add_size_variable_rectangle_patch(
        height,
        width,
        color,
        field_x_left,
        field_y_low,
        field_width,
        field_height,
        bbox_to_anchor,
        loc,
        ):
    pass

def add_circle_patch(
        height,
        width,
        color,
        field_x_left,
        field_y_low,
        field_width,
        field_height,
        bbox_to_anchor,
        loc,
        ):
    # ignore bbox_to_anchor, loc
    # print warning message if not None
    pass

def add_triangle_patch(
        height,
        width,
        orientation,
        color,
        field_x_left,
        field_y_low,
        field_width,
        field_height,
        bbox_to_anchor,
        loc,
        ):
    pass
