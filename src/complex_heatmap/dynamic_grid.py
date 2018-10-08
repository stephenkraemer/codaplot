from collections import defaultdict
from copy import deepcopy
from inspect import getfullargspec
from typing import Optional, List, Tuple, Callable, Any, Dict, DefaultDict

import matplotlib.gridspec as gridspec
# mpl.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from dataclasses import field
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from complex_heatmap.utils import warn


class GridElement:
    """Represents Axes to be created in a plot grid

    To create a grid of plots, GridElements are arranged in a matrix,
    specified as a list of lists (like a numpy array).
    This matrix (called grid in the following) is passed to a GridManager
    instance, which computes the corresponding GridSpec and creates a
    Figure with the corresponding Axes.

    Attributes:
        name: The Axes will be available under GridManager.axes_dict[name]
        width: relative or absolute (in inches) width of the Axes (see kind)
        kind: 'rel' or 'abs', specifies which kind of width is given
        plotter: optionally, a function may be given. This function should have
            an ax argument, a fig argument, or both. These arguments must be visible
            by inspecting the function signature. Such a plotting function can
            then automatically be called by the GridManager, which provides the
            correct Axes to the ax argument, as well as a reference to the Figure
            instance if necessary.
        tags: Arbitrary tags are possible. The GridManager checks for the following
              tags
              - no_col_dendrogram: if this plot is in the top row, no row decoration
                    will be added before it
              - no_row_dendrogram: if this plot is in the first or last column, no
              column decoration will be added before resp. after it

    """
    def __init__(self,
                 name: str,
                 width: float = 1,
                 kind: str = 'rel',
                 plotter: Optional[Callable] = None,
                 tags: Optional[List[str]] = None,
                 *args: List[Any],
                 **kwargs: Dict[Any, Any],
                 ):
        self.name = name
        self.width = width
        self.kind = kind
        self.plotter = plotter
        self.args = args
        self.kwargs = kwargs
        self.tags = [] if tags is None else tags
GE = GridElement


# noinspection PyMissingConstructor
class Spacer(GridElement):
    """Spacer is a GridElement which creates an empty Axes"""

    # The name of a GridElement can be repeated across rows to tell GridManager
    # that a GridElement stretches across multiple rows. To avoid merging spacers
    # across multiple rows, we give each spacer a unique name. This is achieved
    # by counting the Spacer instantiations
    instantiation_count = [0]

    def __init__(self, width: float = 1, kind: str = 'rel'):
        self.instantiation_count[0] += 1
        # IMPORTANT: We rely on Spacer names starting with 'spacer' in this
        # package. Don't change the prefix.
        self.name = f'spacer_{self.instantiation_count[0]}'
        self.width = width
        self.kind = kind
        self.tags = None
        self.plotter = None
        self.args = []
        self.kwargs = {}

@dataclass
class GridManager:
    # noinspection PyUnresolvedReferences
    """
    Args:
        figsize: Optional if all sizes are relative or all are absolute.
            Must be given if absolute and relative sizes are mixed.
        grid: 2D arrangement of GridElements
    """
    grid: List[List[GridElement]]
    height_ratios: List[Tuple[float, str]]
    figsize: Tuple[float, float]
    fig_args: Dict[str, Any] = field(default_factory=dict)
    h_pad: float = 1/72
    w_pad: float = 1/72
    wspace: float = 0
    hspace: float = 0

    fig: Figure = field(init=False)
    axes_list: List[Axes] = field(init=False)
    axes_dict: Dict[str, Axes] = field(init=False)
    gs: gridspec.GridSpec = field(init=False)


    def _compute_height_ratios(self):
        heights = np.array([t[0] for t in self.height_ratios]).astype(float)
        anno = np.array([t[1] for t in self.height_ratios])
        total_abs_heights = heights[anno == 'abs'].sum()
        remaining_height = self.figsize[1] - total_abs_heights
        rel_heights = heights[anno == 'rel']
        heights[anno == 'rel'] = rel_heights / rel_heights.sum() * remaining_height
        self._height_ratios = heights


    def _compute_width_ratios(self):
        """Given figsize and row-wise width specs

        For each row, compute the absolute gridline positions that this row
        requires (values depend on figsize).
        Use the set of all gridline positions collected across the rows
        to calculate the width_ratios for the gridspec.

        Fills these attributes:
          - abs_grid_line_positions_per_row: List of sequences specifying
            the absolute grid line positions required for each row
          - all_unique_abs_grid_line_positions_sorted: array of all unique
            (within floating point precision) grid column boundary positions
          - width_ratios: for use in GridSpec. Will contain union of columns drawn
              from the specs for all rows.
        """

        all_unique_abs_grid_line_positions = np.array([], dtype='f8')
        self.abs_grid_line_positions_per_row: List = []
        for row in self.grid:
            # Each GridElement specifies its width with a float value width and
            # a kind: absolute (abs) or relative (rel)
            width_spec_value_arr = np.array([x.width for x in row]).astype('f8')
            width_spec_kind_arr = np.array([x.kind for x in row])

            # For each row, we first find all absolute widths, and subtract their total from
            # the width of the figure. This gives the width that is left to be distributed
            # among the remaining columns, according to their relative widths.
            space_remaining_for_cols_w_rel_widths = (
                    self.figsize[0] - width_spec_value_arr[width_spec_kind_arr == 'abs'].sum())

            # Now we can calculate an array of absolute widths for each column.
            abs_widths_value_arr = np.copy(width_spec_value_arr)
            rel_widths_values = width_spec_value_arr[width_spec_kind_arr == 'rel']
            abs_widths_value_arr[width_spec_kind_arr == 'rel'] = (
                    rel_widths_values / rel_widths_values.sum()
                    * space_remaining_for_cols_w_rel_widths)
            # The absolute positions of the grid lines are given by the cumsum
            # of the absolute column widths
            abs_width_cumsums = np.cumsum(abs_widths_value_arr)

            # Two rows may share a certain grid line position, but the calculation to
            # arrive at the grid line position may differ if the number or specs
            # of columns is different between the rows. In this case, grid line
            # positions must be matched to each other to avoid duplicates due to
            # floating point precision.
            precision_matched_widths = []
            for curr_width in abs_width_cumsums:
                for curr_column in all_unique_abs_grid_line_positions:
                    if np.isclose(curr_width, curr_column):
                        curr_width = curr_column
                        break
                else:
                    all_unique_abs_grid_line_positions = np.append(
                            all_unique_abs_grid_line_positions, curr_width)
                precision_matched_widths.append(curr_width)
            self.abs_grid_line_positions_per_row.append(precision_matched_widths)

        self.all_unique_abs_grid_line_positions_sorted = np.sort(
                all_unique_abs_grid_line_positions)

        # The float gridline position matching code is experimental
        # Therefore, we add an overly cautios warning
        width_warn_threshold = 0.4/2.54
        if (np.diff(self.all_unique_abs_grid_line_positions_sorted)
            < width_warn_threshold).any():
            warn(f'At least one column in the grid has a'
                 f'width smaller than {width_warn_threshold}')
            raise ValueError

        self.width_ratios = np.diff(
                np.insert(self.all_unique_abs_grid_line_positions_sorted, 0, 0))

    def _compute_element_gridspec_slices(self):
        self.elem_gs: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        for row_idx, (curr_grid_row, curr_row_boundaries) in enumerate(zip(self.grid, self.abs_grid_line_positions_per_row)):
            last_column = 0
            for elem, boundary in zip(curr_grid_row, curr_row_boundaries):
                curr_column = np.argmax(self.all_unique_abs_grid_line_positions_sorted == boundary)
                col_slice = (last_column, curr_column + 1)
                if elem.name in self.elem_gs:
                    assert self.elem_gs[elem.name]['col'] == col_slice
                    assert row_idx == self.elem_gs[elem.name]['row']['end']
                    self.elem_gs[elem.name]['row']['end'] += 1
                else:
                    self.elem_gs[elem.name] = {'row': {'start': row_idx,
                                                       'end': row_idx + 1},
                                               'col': col_slice,
                                               'grid_element': elem,
                                               }
                last_column = curr_column + 1

    def prepend_col(self, grid_element: GridElement, only_rows: Optional[List[int]] = None):
        times_used = 0
        for row_idx, row_list in enumerate(self.grid):
            if only_rows is not None and row_idx not in only_rows:
                row_list.insert(0, Spacer(width=grid_element.width,
                                          kind=grid_element.kind))
            else:
                times_used += 1
                curr_grid_element = deepcopy(grid_element)
                curr_grid_element.name += f'_{times_used}'
                row_list.insert(0, curr_grid_element)

    def insert_matched_row(self, i, grid_element: GridElement,
                           height: Tuple[float, str],
                           where='above',
                           only_cols: Optional[List[str]] = None):
        """Insert row above current row /i/

        For each GridElement in row /i/, insert either a spacer or the grid
        element and use the size properties of the matched object.
        """
        if where == 'above':
            pass
        elif where == 'below':
            i += 1
        else:
            raise ValueError(f'Unknown value {where} for argument where')
        new_row = []
        times_grid_element_was_inserted = 0
        grid_element_orig_name = grid_element.name
        for col_idx, grid_element_below_insertion_point in enumerate(self.grid[i]):
            if only_cols is not None and col_idx not in only_cols:
                new_row.append(self._spacer_like(grid_element_below_insertion_point))
            else:
                times_grid_element_was_inserted += 1
                grid_element_for_col = deepcopy(grid_element)
                grid_element_for_col.name = grid_element_orig_name + f'_{times_grid_element_was_inserted}'
                grid_element_for_col.width = grid_element_below_insertion_point.width
                grid_element_for_col.kind = grid_element_below_insertion_point.kind
                new_row.append(grid_element_for_col)
        self.grid.insert(i, new_row)
        self.height_ratios.insert(i, height)


    def _spacer_like(self, grid_element: GridElement):
        return Spacer(width=grid_element.width, kind=grid_element.kind)


    def create_or_update_figure(self):

        self._compute_width_ratios()
        self._compute_height_ratios()
        self._compute_element_gridspec_slices()
        if hasattr(self, 'fig'):
            self.fig.clear()
            self.fig.set_size_inches(self.figsize)
        else:
            self.fig = plt.figure(constrained_layout=True,
                                  figsize=self.figsize, **self.fig_args)
            # self.fig = plt.figure(tight_layout=True, figsize=self.figsize, **self.fig_args)
            # self.fig.set_constrained_layout_pads(
            #         h_pad=self.h_pad, w_pad=self.w_pad,
            #         hspace=self.hspace, wspace=self.wspace)
        self.gs = gridspec.GridSpec(nrows=len(self._height_ratios),
                                    ncols=len(self.all_unique_abs_grid_line_positions_sorted),
                                    width_ratios=self.width_ratios,
                                    height_ratios=self._height_ratios,
                                    figure=self.fig,
                                    hspace=0, wspace=0)
        self.axes_dict = {}
        # noinspection PyUnusedLocal
        self.axes_list = [[] for unused in self._height_ratios]
        for name, coord in self.elem_gs.items():
            if not name.startswith('spacer'):
                gs_tuple = self.gs[coord['row']['start']:coord['row']['end'], slice(*coord['col'])]
                ax = self.fig.add_subplot(gs_tuple)
                self.axes_dict[name] = ax
                self.axes_list[coord['row']['start']].append(ax)
                ge: GridElement = self.elem_gs[name]['grid_element']
                if ge.plotter is not None:
                    plotter_args = getfullargspec(ge.plotter).args
                    if 'ax' not in plotter_args and 'fig' not in plotter_args:
                        raise ValueError('plotter function must have kwarg ax, fig or both')
                    if 'ax' in plotter_args:
                        ge.kwargs['ax'] = self.axes_dict[name]
                    if 'fig' in plotter_args:
                        ge.kwargs['fig'] = self.fig
                    if 'gs_tuple' in plotter_args:
                        ge.kwargs['gs_tuple'] = gs_tuple

                    ge.plotter(*ge.args, **ge.kwargs)

                    # ax.plot([1, 2, 3])
                    # agg_line(**ge.kwargs)
                    # ge.kwargs['ax'].plot([1, 2, 3])

# def agg_line(df: pd.DataFrame, ax, cluster_ids: pd.Series, fn: Callable,
#              sharey=True, ylim=None):
    # agg_values = df.groupby(cluster_ids).agg(fn)
    # if sharey and ylim is None:
    #     ymax = np.max(agg_values.values)
    #     pad = abs(0.1 * ymax)
    #     padded_ymax = ymax + pad
    #     if ymax < 0 < padded_ymax:
    #         padded_ymax = 0
    #     ymin = np.min(agg_values.values)
    #     padded_ymin = ymin - pad
    #     if ymin > 0 > padded_ymin:
    #         padded_ymin = 0
    #     ylim = (padded_ymin, padded_ymax)
    # n_clusters = agg_values.shape[0]
    # gssub = ax.get_gridspec()[0].subgridspec(n_clusters, 1)
    # ax.remove()
    # for row_idx, (cluster_id, row_ser) in enumerate(agg_values.iterrows()):
    #     ax = fig.add_subplot(gssub[row_idx, 0])
    #     ax.set(xticks=[], xticklabels=[])
    #     if ylim is not None:
    #         ax.set_ylim(ylim)
    #     ax.plot(row_ser.values, marker='.', linestyle='-')
    # ax.set(xticks=[1, 2, 3], xticklabels=[1, 2, 3], xlabel='test')
    # ax.plot([1, 2, 3], label='inline label')
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
