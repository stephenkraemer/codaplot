import matplotlib.legend as mlegend
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Literal
import codaplot.utils as coutils
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import _api
from matplotlib.offsetbox import (
    HPacker,
    VPacker,
    DrawingArea,
    TextArea,
)


class MyCenteredLabelFullHeightLegend(mlegend.Legend):
    """
    matplotlib.legend.Legend
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_legend_box(self, handles, labels, markerfirst=True):
        """
        Initialize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self._fontsize

        # legend_box is a HPacker, horizontally packed with columns.
        # Each column is a VPacker, vertically packed with legend items.
        # Each legend item is a HPacker packed with:
        # - handlebox: a DrawingArea which contains the legend handle.
        # - labelbox: a TextArea which contains the legend text.

        text_list = []  # the list of text instances
        handle_list = []  # the list of handle instances
        handles_and_labels = []

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.

        # EDIT: do not use descent
        # - descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.

        # EDIT: use full height
        # - height = fontsize * self.handleheight - descent
        # + height = fontsize * self.handleheight
        height = fontsize * self.handleheight

        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their coordinates should
        # be given in the display coordinates.

        # The transformation of each handle will be automatically set
        # to self.get_transform(). If the artist does not use its
        # default transform (e.g., Collections), you need to
        # manually set their transform to the self.get_transform().
        legend_handler_map = self.get_legend_handler_map()

        for orig_handle, label in zip(handles, labels):
            handler = self.get_legend_handler(legend_handler_map, orig_handle)
            if handler is None:
                _api.warn_external(
                    "Legend does not support {!r} instances.\nA proxy artist "
                    "may be used instead.\nSee: "
                    "https://matplotlib.org/users/legend_guide.html"
                    "#creating-artists-specifically-for-adding-to-the-legend-"
                    "aka-proxy-artists".format(orig_handle)
                )
                # No handle for this artist, so we just defer to None.
                handle_list.append(None)
            else:
                textbox = TextArea(
                    label,
                    multilinebaseline=True,
                    textprops=dict(
                        verticalalignment="baseline",
                        horizontalalignment="left",
                        fontproperties=self.prop,
                    ),
                )

                # EDIT: remove ydescent from handlebox DrawingArea
                # - handlebox = DrawingArea(width=self.handlelength * fontsize,
                # -                         height=height,
                # -                         xdescent=0., ydescent=descent)
                # + handlebox = DrawingArea(width=self.handlelength * fontsize,
                # +                         height=height,
                # +                         xdescent=0., ydescent=0.)
                handlebox = DrawingArea(
                    width=self.handlelength * fontsize,
                    height=height,
                    xdescent=0.0,
                    ydescent=0.0,
                )

                text_list.append(textbox._text)
                # Create the artist for the legend which represents the
                # original artist/handle.
                handle_list.append(
                    handler.legend_artist(self, orig_handle, fontsize, handlebox)
                )
                handles_and_labels.append((handlebox, textbox))

        columnbox = []
        # array_split splits n handles_and_labels into ncol columns, with the
        # first n%ncol columns having an extra entry.  filter(len, ...) handles
        # the case where n < ncol: the last ncol-n columns are empty and get
        # filtered out.
        for handles_and_labels_column in filter(
            len, np.array_split(handles_and_labels, self._ncol)
        ):
            # pack handlebox and labelbox into itembox
            # EDIT: change to center alignment in itembox
            # - itemboxes = [HPacker(pad=0,
            # -                      sep=self.handletextpad * fontsize,
            # -                      children=[h, t] if markerfirst else [t, h],
            # -                      align="baseline")
            # -              for h, t in handles_and_labels_column]
            # + itemboxes = [HPacker(pad=0,
            # +                      sep=self.handletextpad * fontsize,
            # +                      children=[h, t] if markerfirst else [t, h],
            # +                      align="center")
            # +              for h, t in handles_and_labels_column]
            itemboxes = [
                HPacker(
                    pad=0,
                    sep=self.handletextpad * fontsize,
                    children=[h, t] if markerfirst else [t, h],
                    align="center",
                )
                for h, t in handles_and_labels_column
            ]
            # pack columnbox
            alignment = "baseline" if markerfirst else "right"
            columnbox.append(
                VPacker(
                    pad=0,
                    sep=self.labelspacing * fontsize,
                    align=alignment,
                    children=itemboxes,
                )
            )

        mode = "expand" if self._mode == "expand" else "fixed"
        sep = self.columnspacing * fontsize
        self._legend_handle_box = HPacker(
            pad=0, sep=sep, align="baseline", mode=mode, children=columnbox
        )
        self._legend_title_box = TextArea("")
        self._legend_box = HPacker(
            pad=self.borderpad * fontsize,
            # sep=self.labelspacing * fontsize,
            sep=mpl.rcParams['axes.labelpad'],
            align="center",
            children=[self._legend_handle_box, self._legend_title_box],
        )
        self._legend_box.set_figure(self.figure)
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legendHandles = handle_list

    def set_title(self, title, prop=None):
        """
        Set the legend title. Fontproperties can be optionally set
        with *prop* parameter.
        """
        self._legend_title_box._text.set_text(title)
        if title:
            self._legend_title_box._text.set_visible(True)
            self._legend_title_box.set_visible(True)
        else:
            self._legend_title_box._text.set_visible(False)
            self._legend_title_box.set_visible(False)

        w, h, x, y = self._legend_handle_box.get_extent(self.parent.figure.canvas.renderer)
        print(w, h, x, y)

        if prop is not None:
            self._legend_title_box._text.set_fontproperties(prop)
        self._legend_title_box._text.set_rotation('vertical')
        # self._legend_title_box._text.set_va('bottom')
        # self._legend_title_box._text.set_ha('left')
        # self._legend_title_box._text.set_y((h - y)/2)
        # self._legend_title_box.set_clip_on(False)
        # self._legend_box.set_clip_on(False)

        # self.parent.figure.canvas.draw()

        self.stale = True

class VariableSizeRectanglePatchHandler:
    def __init__(self, max_height, max_width):
        self.max_height = max_height
        self.max_width = max_width

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        assert x0 == 0 and y0 == 0
        rel_width, rel_height = (
            orig_handle.get_width() / self.max_width,
            orig_handle.get_height() / self.max_height,
        )
        width_handlebox_coord = handlebox.width * rel_width
        height_handlebox_coord = handlebox.height * rel_height
        # move rectangle up a little bit to align with center of 'A', not of 'Ag'
        patch = mpatches.Rectangle(
            [
                handlebox.width - width_handlebox_coord,
                handlebox.height / 2 - height_handlebox_coord / 2,
            ],
            width_handlebox_coord,
            height_handlebox_coord,
            facecolor="black",
            edgecolor=None,
            lw=0,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


def create_legend_for_norm_size_patches(
    ax, handles, labels, title=None, **legend_kwargs
):
    """

    Parameters
    ----------
    pseudo_handles
        pseudo_handles are expected in increasing sort order, with monotonous increase of both height and width, so that the maximum size patch is the last in the list
    """

    # get max handle width and height in font size units (legend handle height widht is specified in font size units) - we need to make the legend handle height and length match these maximum dimensions

    # if figure is not drawn, the absolute patch sizes (in inch) are not correctly inferred
    ax.figure.canvas.draw()

    max_handle_width_in, max_handle_height_in = coutils.get_artist_size_inch(
        art=handles[-1], fig=ax.figure
    )

    max_handle_height_fontsize_units = max_handle_height_in / (
        mpl.rcParams["legend.fontsize"] / 72
    )
    max_handle_width_fontsize_units = max_handle_width_in / (
        mpl.rcParams["legend.fontsize"] / 72
    )
    # we draw legend handles by specifying the fraction of the maximum handle height and width
    # a given patch covers, to compute this, we need the maximum height and width in data coords

    max_width_data_coords, max_height_data_coords = (
        handles[-1].get_width(),
        handles[-1].get_height(),
    )

    leg = MyCenteredLabelFullHeightLegend(
        ax,
        handles=handles,
        labels=labels,
        handler_map={
            mpatches.Rectangle: VariableSizeRectanglePatchHandler(
                max_width=max_width_data_coords, max_height=max_height_data_coords
            )
        },
        handleheight=max_handle_height_fontsize_units,
        handlelength=max_handle_width_fontsize_units,
        title=title,
        **legend_kwargs,
    )

    leg.set_clip_on(False)

    return leg
