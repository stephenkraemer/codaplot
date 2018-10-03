from collections import defaultdict
from inspect import getfullargspec
from typing import Optional, List, Tuple, Callable, Any, Dict

import matplotlib.gridspec as gridspec
# mpl.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


class GridElement:
    def __init__(self,
                 name: str,
                 width: float,
                 kind: str = 'rel',
                 plotter: Optional[Callable] = None,
                 *args: List[Any],
                 **kwargs: Dict[Any, Any],
                 ):
        self.name = name
        self.width = width
        self.kind = kind
        self.plotter = plotter
        self.args = args
        self.kwargs = kwargs
GE = GridElement

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
    figsize: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        self._create_or_update_figure()

    def _compute_height_ratios(self):
        heights = np.array([t[0] for t in self.height_ratios]).astype(float)
        anno = np.array([t[1] for t in self.height_ratios])
        total_abs_heights = heights[anno == 'abs'].sum()
        remaining_height = self.figsize[1] - total_abs_heights
        rel_heights = heights[anno == 'rel']
        heights[anno == 'rel'] = rel_heights / rel_heights.sum() * remaining_height
        self._height_ratios = heights


    def _compute_width_ratios(self):
        columns = []
        self.row_boundaries = []
        for row in self.grid:
            widths = np.array([x.width for x in row]).astype(float)
            kind = np.array([x.kind for x in row])
            absolute_widths = widths[kind == 'abs'].sum()
            remaining_width = self.figsize[0] - absolute_widths
            rel_widths = widths[kind == 'rel']
            widths[kind == 'rel'] = rel_widths / rel_widths.sum() * remaining_width
            widths = np.cumsum(widths)
            self.row_boundaries.append(widths)
            columns.extend(widths)

        self.columns_sortu = np.unique(np.sort(columns))
        self.width_ratios = np.diff(np.insert(self.columns_sortu, 0, 0))

    def _compute_element_gridspec_slices(self):
        self.elem_gs = defaultdict(dict)
        for row_idx, (curr_grid_row, curr_row_boundaries) in enumerate(zip(self.grid, self.row_boundaries)):
            last_column = 0
            for elem, boundary in zip(curr_grid_row, curr_row_boundaries):
                curr_column = np.argmax(self.columns_sortu == boundary)
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

    def _create_or_update_figure(self):
        self._compute_width_ratios()
        self._compute_height_ratios()
        self._compute_element_gridspec_slices()
        if hasattr(self, 'fig'):
            self.fig.clear()
            self.fig.set_size_inches(self.figsize)
        else:
            self.fig = plt.figure(constrained_layout=True, figsize = self.figsize)
        self.gs = gridspec.GridSpec(nrows=len(self._height_ratios),
                                    ncols=len(self.columns_sortu),
                                    width_ratios=self.width_ratios,
                                    height_ratios=self._height_ratios,
                                    figure=self.fig,
                                    hspace=0, wspace=0)
        self.axes_dict = {}
        self.axes_list = [[] for unused in self._height_ratios]
        for name, coord in self.elem_gs.items():
            if not name.startswith('spacer'):
                ax = self.fig.add_subplot(self.gs[coord['row']['start']:coord['row']['end'], slice(*coord['col'])])
                self.axes_dict[name] = ax
                self.axes_list[coord['row']['start']].append(ax)
                ge = self.elem_gs[name]['grid_element']
                if ge.plotter is not None:
                    plotter_args = getfullargspec(ge.plotter).args
                    if 'ax' not in plotter_args and 'fig' not in plotter_args:
                        raise ValueError('plotter function must have kwarg ax, fig or both')
                    if 'ax' in plotter_args:
                        ge.kwargs['ax'] = self.axes_dict[name]
                    if 'fig' in plotter_args:
                        ge.kwargs['fig'] = self.fig

                    ge.plotter(*ge.args, **ge.kwargs)


    def update_figure(self):
        self._create_or_update_figure()
