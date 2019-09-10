"""Use array of plotting functions to facilitate creation of complex gridspecs"""
import itertools
from inspect import getfullargspec
from typing import Callable, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
from dataclasses import dataclass, field

from IPython.display import display


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


@dataclass
class ArrayElement:
    name: str
    # without func, a placeholder Axes is created
    func: Optional[Callable] = None
    args: Tuple = tuple()
    kwargs: Dict = field(default_factory=dict)


# tests
def object_plotter(ax, a=3, b=10):
    """Simple test function which accepts ax, and fails if b is not overwritten"""
    ax.plot(range(a), range(b))


def test_plot():
    """Create simple array[ArrayElement] and create figure and axes containers

    displays figure if successful

    - empty legend ax
    - None placeholder without any axes
    - pyplot function
    - function taking Axes object
    """

    plot_array = np.array(
        [
            [
                ArrayElement(
                    "scatter1",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="green"),
                ),
                ArrayElement(
                    "scatter2",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="black"),
                ),
                None,
                (1, "abs"),
            ],
            [
                ArrayElement("plot", object_plotter, args=(), kwargs=dict(b=3)),
                ArrayElement(
                    "scatter4",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="blue"),
                ),
                ArrayElement("legend"),
                (1, "rel"),
            ],
            [
                ArrayElement("plot3", object_plotter, args=(), kwargs=dict(b=3)),
                ArrayElement(
                    "scatter34",
                    plt.scatter,
                    args=([1, 2, 3], [1, 2, 3]),
                    kwargs=dict(c="blue"),
                ),
                None,
                (2, "rel"),
            ],
            [(2, "rel"), (3, "rel"), (1, "abs"), None],
        ]
    )

    res = array_to_figure(plot_array, figsize=(8, 2))
    display(res["fig"])


def array_to_figure(plot_array: np.ndarray, figsize: Tuple[float]) -> Dict:
    """Convert array of plotting instructions into figure, provide access to Axes

    plot array format
    - Array of ArrayElements
    - all ArrayElements must have names
    - ArrayElement names must be unique, they will e.g. be used as identifiers
      in the returned Axes dict, but the unique names are also relied upon by
      internal code.


    [[ArrayElement, ArrayElement, (1, 'rel')],
     [ArrayElement, ArrayElement, (1, 'abs')],
     [  (1, 'abs'),   (1, 'rel'),       None][,


    Args:
        plot_array: array of ArrayElements detailing plotting instructions

    Returns:
        dict with keys
          - axes_d (dict name -> Axes)
          - axes_arr (array of Axes matching figure layout)
          - fig
    """

    # The margins of the array contain the height and width specs as (size, 'abs'|'rel')
    # retrieve the actual plot array (inner_array) and convert the size specs
    # to relative-only ratios, given the figsize
    inner_array = plot_array[:-1, :-1]
    widths = plot_array[-1, :-1]
    width_ratios = compute_gridspec_ratios(widths, figsize[0])
    heights = plot_array[:-1, -1]
    height_ratios = compute_gridspec_ratios(heights, figsize[1])

    assert (
        np.unique(
            [elem.name for elem in inner_array.ravel() if elem is not None],
            return_counts=True,
        )[-1]
        == 1
    ).all()

    fig = plt.figure(constrained_layout=True, dpi=180)
    # constrained layout pads need to be set directly,
    # setting hspace and wspace in GridSpec is not sufficient
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    gs = gridspec.GridSpec(
        inner_array.shape[0],
        inner_array.shape[1],
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        figure=fig,
    )

    axes_d = {}
    axes_arr = np.empty_like(inner_array)
    plot_returns = {}
    for t in itertools.product(
        range(inner_array.shape[0]), range(inner_array.shape[1])
    ):
        ge = inner_array[t]
        # Do not create Axes for Spacers (np.nan)
        if ge is None:
            continue
        # However, do create Axes for GridElements without plotting function
        ax = fig.add_subplot(gs[t])
        axes_d[ge.name] = ax
        axes_arr[t] = ax
        if ge.func is None:
            continue
        # Plotting func is given, create plot
        # Add Axes to func args if necessary
        # If ax is not in plotter args, we assume that this is a pyplot function relying on state
        plotter_args = getfullargspec(ge.func).args
        if "ax" in plotter_args:
            ge.kwargs["ax"] = ax
        plot_returns[ge.name] = ge.func(*ge.args, **ge.kwargs)

    return {"axes_d": axes_d, "axes_arr": axes_arr, "fig": fig}


def compute_gridspec_ratios(sizes: np.ndarray, total_size_in: float) -> np.ndarray:
    """Convert mixture of absolute and relative column/row sizes to relative-only

    The result depends on the figsize, the function will first allocate space for
    all absolute sizes, and then divide the rest proportionally between the relative
    sizes.

    Args:
        sizes: array of size specs, eg. [(1, 'rel'), (2, 'abs'), (3, 'rel')]
            all sizes in inch
        total_size_in: size of the corresponding dimension of the figure (inch)

    Returns:
        array of ratios which can be passed to width_ratios or height_ratios of
        the Gridspec constructor. Note that this result is only valid for the
        given figsize.

    """
    heights = np.array([t[0] for t in sizes]).astype(float)
    anno = np.array([t[1] for t in sizes])
    total_abs_heights = heights[anno == "abs"].sum()
    remaining_height = total_size_in - total_abs_heights
    rel_heights = heights[anno == "rel"]
    heights[anno == "rel"] = rel_heights / rel_heights.sum() * remaining_height
    return heights


def end():
    pass
