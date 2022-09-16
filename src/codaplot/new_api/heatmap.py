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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mouse_hema_meth.utils as ut
from mouse_hema_meth.utils import cm

# ## Types

colname_or_df_opt = Optional[Union[str, pd.DataFrame]]
str_or_num_t = Union[str, int, float]
str_or_num_t_opt = Optional[Union[str, int, float]]
cmap_t = Optional[str]
norm_t_opt = Optional[str]
mesh_t_opt = Union[
    Mesh,
    Literal[
        "pcolormesh",
        "pcolormesh_plus",
        "rectangle_mesh",
        "circle_mesh",
        "triangle_mesh",
        "multimark_mesh",
    ],
]
kwds_t = Optional[Dict[str, Any]]
arr_or_list_opt = Optional[Union[np.ndarray, list]]


# # Shared doc

vmin = "p1.5 for 1.5th percentile"
vmax = "p99.5 for percentile"

# # Heatmap


class Heatmap:
    """

    - manage drawing of a single Mesh onto an Axes
      - define the rectangular areas where the Mesh may draw its marks
        - while considering spacing
        - while consider placement restrictions
      - make the Mesh draw into this area
        - pass arguments to mesh
        - run Mesh
        - collect guide information returned by Mesh
      - add one or more guides
        - pass guide related arguments to co.legend.draw_guides

    """

    def __init__(
        self,
        # marker properties mapping
        color: colname_or_df_opt = None,
        size: colname_or_df_opt = None,
        data: Optional[pd.DataFrame] = None,
        is_categorical: bool = False,
        # Marker
        marker_mesh: mesh_t_opt = "pcolormesh",
        marker_max_field_fraction: float = 1.0,
        marker_loc: Tuple[float, float] = (0.5, 0.5),
        marker_bbox_to_anchor: Literal["upper_left", "center", "lower_right"] = "center",
        marker_kwargs: kwds_t = None,
        # guides
        guides_color_title: Optional[str] = None,
        guides_color_cbar_kwds: kwds_t = None,
        guides_color_legend_kwds: kwds_t = None,
        guides_size_title: Optional[str] = None,
        guides_size_legend_kwds: kwds_t = None,
        guides_ax: Optional[Axes] = None,
        guides_add: bool = True,
        # spacing
        spacing_row_ids: arr_or_list_opt = None,
        spacing_row_gap_sizes: arr_or_list_opt = None,
        spacing_col_ids: arr_or_list_opt = None,
        spacing_col_gap_sizes: arr_or_list_opt = None,
    ) -> None:
        pass

    def draw(self, ax: Axes):
        pass


# # Meshes


# %% [markdown]
# ## PcolorMeshPlus


class Mesh:
    pass


class ColorMesh(Mesh):
    def __init__(
        self,
        color_min: str_or_num_t_opt = None,
        color_max: str_or_num_t_opt = None,
        color_map: cmap_t = None,
        color_norm: norm_t_opt = None,
    ) -> None:
        pass


class SizeMesh(Mesh):
    def __init__(
        self,
        size_min: str_or_num_t_opt = None,
        size_max: str_or_num_t_opt = None,
        size_norm: norm_t_opt = None,
    ) -> None:
        pass


class PcolorMeshPlus(ColorMesh):
    pass


# ## CircleMesh


class CircleMesh(ColorMesh, SizeMesh):
    pass


# ## RectangleMesh


class RectangleMesh(ColorMesh, SizeMesh):
    pass


# ## TriangleMesh


class TriangleMesh(ColorMesh, SizeMesh):
    pass


# ## MultiMarkMesh


class MultiMarkMesh:
    pass


# # End
