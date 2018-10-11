from collections import defaultdict
from copy import deepcopy
from inspect import getfullargspec
from itertools import chain
from typing import Optional, List, Tuple, Any, Dict, DefaultDict, Union

import matplotlib.gridspec as gridspec
# mpl.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from codaplot.utils import warn
from dataclasses import dataclass
from dataclasses import field


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
                 plotter: Optional[staticmethod] = None,
                 tags: Optional[List[str]] = None,
                 **kwargs: Any,
                 ) -> None:
        self.name = name
        self.width = width
        self.kind = kind
        self.plotter = plotter
        self.kwargs = kwargs
        self.tags = [] if tags is None else tags
GE = GridElement


class Spacer(GridElement):
    """Empty Axes inserted for appropriate alignment of other elements"""

    # The name of a GridElement can be repeated across rows to tell GridManager
    # that a GridElement stretches across multiple rows. To avoid merging spacers
    # across multiple rows, we give each spacer a unique name. This is achieved
    # by counting the Spacer instantiations
    instantiation_count = [0]

    def __init__(self, width: float = 1, kind: str = 'rel') -> None:
        self.instantiation_count[0] += 1
        super().__init__(
                name = f'spacer_{self.instantiation_count[0]}',
                width = width,
                kind = kind,
                tags = [],
                plotter = None,
        )
        # IMPORTANT: We rely on Spacer names starting with 'spacer' in this
        # package. Don't change the prefix.

class FacetedGridElement(GridElement):
    """GridElement managing several axes according to faceting scheme

    The FacetedGridElement defines the placement and size of the element
    as a whole. Within this area of the figure, several axes are created
    according to the facetting scheme.
    """

    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 width: float = 1,
                 kind: str = 'rel',
                 plotter: Optional[staticmethod] = None,
                 row: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 **kwargs: Any,
                 ) -> None:
        kwargs['row'] = row
        kwargs['data'] = data
        super().__init__(
                name=name,
                width=width,
                kind=kind,
                plotter=plotter,
                tags=[] if tags is None else tags,
                **kwargs,
        )
        self.row = row
        self.data = data
        self._nrow = None

    @property
    def nrow(self):
        if self._nrow is None:
            if isinstance(self.row, str):
                if self.row in self.data:
                    self._nrow = self.data[self.row].nunique()
                else:
                    self._nrow = self.data.index.get_level_values(self.row).nunique()
            else:
                self._nrow = pd.Series(self.row).nunique()
        return self._nrow


@dataclass
class GridManager:
    # noinspection PyUnresolvedReferences
    """
    Args:
        figsize: Optional if all sizes are relative or all are absolute.
            Must be given if absolute and relative sizes are mixed.
        grid: 2D arrangement of GridElements
    """

    # Implementation notes
    # ====================
    #
    # FacetedGridElement
    # ------------------
    # - currently, *only row* facetting is implemented. The rest will follow if
    #   someone needs it.
    # - the axes within the element area are *not* inserted as subgridspec.
    #   This would break the alignment of this element and its contained axes
    #   with the rest of the plot. This is due to a current limitation in
    #   the constrained_layout solver. If the solver evolves to align subgridspecs,
    #   the implementation of this class could be significantly simplified...
    #   Instead of using a subgridspec, the gridspec for the figure is further
    #   subdivided. For each facet, a corresponding gridspec row or column is added.
    # - the SubplotSpec for the individual GridElements can then - as with the
    #   other GridElements - be determined by recording the absolute positions
    #   for each axes in the FacetedGridElement and using it to retrieve the
    #   corresponding line in the gridspec.
    # - this behavior is coded in GridManager._compute_height_ratios,
    #   GridManager._compute_width_ratios, GridManager._compute_element_gridspec_slices

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
    _height_ratios: np.ndarray = field(init=False)
    all_unique_heights: np.ndarray = field(init=False)
    abs_grid_line_positions_per_row: List[List[float]] = field(init=False)
    all_unique_abs_grid_line_positions_sorted: np.ndarray = field(init=False)
    width_ratios: np.ndarray = field(init=False)
    elem_gs: DefaultDict[str, Dict[str, Any]] = field(init=False)
    height_boundaries: List[List[Union[Tuple[float, float],
                                       List[Tuple[float, float]]
    ]]] = field(init=False)

    def _compute_height_ratios(self):
        """Compute height ratios for GridSpec and stats required for \
           assigning SubgridSpecs to the individual grid elements

           Similar to _compute_width_ratios.

           This function is a bit more complex because it can deal with
           FacetedGridElements.
        """

        heights = np.array([t[0] for t in self.height_ratios]).astype(float)
        anno = np.array([t[1] for t in self.height_ratios])
        total_abs_heights = heights[anno == 'abs'].sum()
        remaining_height = self.figsize[1] - total_abs_heights
        rel_heights = heights[anno == 'rel']
        heights[anno == 'rel'] = rel_heights / rel_heights.sum() * remaining_height
        height_cs = np.cumsum(heights)

        # noinspection PyUnusedLocal
        height_boundaries = [[[] for col in row] for row in self.grid]
        all_unique_heights = list(height_cs)
        for row_idx, row in enumerate(self.grid):
            for col_idx, grid_element in enumerate(row):
                if not isinstance(grid_element, FacetedGridElement):
                    height_boundaries[row_idx][col_idx] = (
                        height_cs[row_idx - 1] if row_idx != 0 else 0,
                        height_cs[row_idx])
                elif isinstance(grid_element, FacetedGridElement):
                    height_boundaries[row_idx][col_idx] = []
                    if row_idx == 0:
                        top_boundary = 0
                    else:
                        top_boundary = height_cs[row_idx - 1]
                    row_height = heights[row_idx]
                    new_height_candidates = (
                            top_boundary + row_height / grid_element.nrow
                            * np.arange(1, grid_element.nrow + 1))
                    last_height = top_boundary
                    for curr_height in new_height_candidates:
                        for previous_height in all_unique_heights:
                            if np.isclose(curr_height, previous_height):
                                curr_height = previous_height
                                break
                        else:
                            all_unique_heights.append(curr_height)
                        height_boundaries[row_idx][col_idx].append((last_height, curr_height))
                        last_height = curr_height
                else:
                    raise TypeError()
        self.all_unique_heights = np.insert(np.sort(all_unique_heights), 0, 0)
        self.height_boundaries = height_boundaries
        self._height_ratios = np.diff(self.all_unique_heights)


    def _compute_width_ratios(self):
        """ Compute width ratios for GridSpec and stats required for\
        assigning SubgridSpecs to the individual grid elements

        Given figsize and row-wise width specs for each element,
        compute the absolute gridline positions that the elements
        require (values depend on figsize).
        Use the set of all gridline positions collected across the rows
        to calculate the width_ratios for the gridspec.
        Record the absolute start and end boundaries of each element in each
        row to be able to find the correct SubgridSpec slices for each element

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
            last_width = 0
            for curr_width in abs_width_cumsums:
                for curr_column in all_unique_abs_grid_line_positions:
                    if np.isclose(curr_width, curr_column):
                        curr_width = curr_column
                        break
                else:
                    all_unique_abs_grid_line_positions = np.append(
                            all_unique_abs_grid_line_positions, curr_width)
                precision_matched_widths.append((last_width, curr_width))
                last_width = curr_width
            self.abs_grid_line_positions_per_row.append(precision_matched_widths)

        # for row_idx, row in enumerate(self.grid):
        #     for col_idx, grid_element in enumerate(row):
        #         if isinstance(grid_element, FacetedGridElement):
                    # if col_idx == 0:
                    #     left_gridline = 0
                    # else:
                    #     left_gridline = self.abs_grid_line_positions_per_row[row_idx][col_idx - 1]
                    # right_gridline = self.abs_grid_line_positions_per_row[row_idx][col_idx]

        self.all_unique_abs_grid_line_positions_sorted = np.concatenate(
                (np.array([0]), np.sort(all_unique_abs_grid_line_positions)), axis=0)

        # These width ratios are absolute widths in inch for each column in the grid
        self.width_ratios = np.diff(self.all_unique_abs_grid_line_positions_sorted)

        grid_col_size_warn_threshold = 0.3/2.54
        if (self.width_ratios
            < grid_col_size_warn_threshold).any():
            warn(f'At least one column in the grid has a'
                 f'width smaller than {grid_col_size_warn_threshold}')

        min_element_width_warn_threshold = 0.5/2.54
        for abs_element_boundary_left, abs_element_boundary_right in chain(
                *self.abs_grid_line_positions_per_row):
            if (abs_element_boundary_right - abs_element_boundary_left
                    < min_element_width_warn_threshold):
                warn(f'At least one GridElement has a width '
                     f'smaller than {min_element_width_warn_threshold}')


    def _compute_element_gridspec_slices(self):
        self.elem_gs = defaultdict(dict)
        for row_idx, row in enumerate(self.grid):
            for col_idx, grid_element in enumerate(row):
                seen = grid_element.name in self.elem_gs
                curr_row_bounds = self.abs_grid_line_positions_per_row[row_idx][col_idx]
                left_grid_col_line = np.argmax(
                        self.all_unique_abs_grid_line_positions_sorted
                        == curr_row_bounds[0])
                right_grid_col_line = np.argmax(
                        self.all_unique_abs_grid_line_positions_sorted
                        == curr_row_bounds[1])
                col_slice = (left_grid_col_line, right_grid_col_line)
                if seen:
                    assert self.elem_gs[grid_element.name]['col'] == col_slice
                else:
                    self.elem_gs[grid_element.name] = {
                        'col': col_slice,
                        'grid_element': grid_element,
                    }
                curr_col_bounds = self.height_boundaries[row_idx][col_idx]
                if isinstance(curr_col_bounds, tuple):
                    upper_grid_row_line = np.argmax(
                            self.all_unique_heights == curr_col_bounds[0]
                    )
                    lower_grid_row_line = np.argmax(
                            self.all_unique_heights == curr_col_bounds[1]
                    )
                    if seen:
                        self.elem_gs[grid_element.name]['row']['end'] = lower_grid_row_line
                    else:
                        self.elem_gs[grid_element.name]['row'] = {
                            'start': upper_grid_row_line,
                            'end': lower_grid_row_line,
                        }
                else:
                    upper_grid_row_lines = []
                    lower_grid_row_lines = []
                    for curr_col_bounds_tuple in curr_col_bounds:
                        upper_grid_row_lines.append(np.argmax(
                                self.all_unique_heights == curr_col_bounds_tuple[0]
                        ))
                        lower_grid_row_lines.append(np.argmax(
                                self.all_unique_heights == curr_col_bounds_tuple[1]
                        ))
                    if seen:
                        raise ValueError()
                    else:
                        self.elem_gs[grid_element.name]['row'] = {
                            'start': upper_grid_row_lines,
                            'end': lower_grid_row_lines,
                        }


    def prepend_col_from_element(
            self,grid_element: GridElement, only_rows: Optional[List[int]] = None):
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

    def prepend_col_from_sequence(self, grid_elements):
        if not len(grid_elements) == len(self.grid):
            raise ValueError('Argument grid elements must provide one '
                             'GridElement per row')
        for row, grid_element in zip(self.grid, grid_elements):
            row.insert(0, grid_element)

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


    @staticmethod
    def _spacer_like(grid_element: GridElement):
        """Return Spacer with same dimensions as the GridElement"""
        return Spacer(width=grid_element.width, kind=grid_element.kind)


    def create_or_update_figure(self):
        """Update GridSpec for self.fig and call all defined GridElement.plotter"""

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
                                    ncols=len(self.width_ratios),
                                    width_ratios=self.width_ratios,
                                    height_ratios=self._height_ratios,
                                    figure=self.fig,
                                    hspace=0, wspace=0)
        self.axes_dict = {}
        # noinspection PyUnusedLocal
        self.axes_list = [[] for unused in self._height_ratios]
        for name, coord in self.elem_gs.items():
            if not name.startswith('spacer'):
                if not isinstance(coord['row']['start'], list):
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

                        ge.plotter(**ge.kwargs)
                else:
                    axes = []
                    for row_start, row_end in zip(coord['row']['start'], coord['row']['end']):
                        gs_tuple = self.gs[row_start:row_end, slice(*coord['col'])]
                        axes.append(self.fig.add_subplot(gs_tuple))
                    self.axes_dict[name] = axes
                    self.axes_list[coord['row']['start'][0]].append(axes)

                    ge: GridElement = self.elem_gs[name]['grid_element']
                    if ge.plotter is not None:
                        plotter_args = getfullargspec(ge.plotter).args
                        if 'ax' not in plotter_args and 'fig' not in plotter_args:
                            raise ValueError('plotter function must have kwarg ax, fig or both')
                        if 'ax' in plotter_args:
                            ge.kwargs['ax'] = self.axes_dict[name]
                        if 'fig' in plotter_args:
                            ge.kwargs['fig'] = self.fig

                        ge.plotter(**ge.kwargs)

