import matplotlib.legend as mlegend
from matplotlib import _api, docstring, colors, offsetbox
import matplotlib.patches as mpatches
import matplotlib.text as mtext
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, bbox_artist as mbbox_artist)
import matplotlib.transforms as mtransforms
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
from matplotlib.offsetbox import bbox_artist

class MyCenteredLabelFullHeightLegend(mlegend.Legend):
    """
    matplotlib.legend.Legend
    """

    # def _init_legend_box(self, handles, labels, markerfirst=True):
    #     """
    #     Initialize the legend_box. The legend_box is an instance of
    #     the OffsetBox, which is packed with legend handles and
    #     texts. Once packed, their location is calculated during the
    #     drawing time.
    #     """

    #     fontsize = self._fontsize

    #     # legend_box is a HPacker, horizontally packed with columns.
    #     # Each column is a VPacker, vertically packed with legend items.
    #     # Each legend item is a HPacker packed with:
    #     # - handlebox: a DrawingArea which contains the legend handle.
    #     # - labelbox: a TextArea which contains the legend text.

    #     text_list = []  # the list of text instances
    #     handle_list = []  # the list of handle instances
    #     handles_and_labels = []

    #     # The approximate height and descent of text. These values are
    #     # only used for plotting the legend handle.
    #     # descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.
    #     # height = fontsize * self.handleheight - descent

    #     height = fontsize * self.handleheight
    #     # each handle needs to be drawn inside a box of (x, y, w, h) =
    #     # (0, -descent, width, height).  And their coordinates should
    #     # be given in the display coordinates.

    #     # The transformation of each handle will be automatically set
    #     # to self.get_transform(). If the artist does not use its
    #     # default transform (e.g., Collections), you need to
    #     # manually set their transform to the self.get_transform().
    #     legend_handler_map = self.get_legend_handler_map()

    #     for orig_handle, label in zip(handles, labels):
    #         handler = self.get_legend_handler(legend_handler_map, orig_handle)
    #         if handler is None:
    #             _api.warn_external(
    #                 "Legend does not support {!r} instances.\nA proxy artist "
    #                 "may be used instead.\nSee: "
    #                 "https://matplotlib.org/users/legend_guide.html"
    #                 "#creating-artists-specifically-for-adding-to-the-legend-"
    #                 "aka-proxy-artists".format(orig_handle))
    #             # No handle for this artist, so we just defer to None.
    #             handle_list.append(None)
    #         else:
    #             textbox = TextArea(label, multilinebaseline=True,
    #                                textprops=dict(
    #                                    verticalalignment='baseline',
    #                                    horizontalalignment='left',
    #                                    fontproperties=self.prop))
    #             # handlebox = DrawingArea(width=self.handlelength * fontsize,
    #             #                         height=height,
    #             #                         xdescent=0., ydescent=descent)
    #             handlebox = DrawingArea(
    #                 width=self.handlelength * fontsize,
    #                 height=height,
    #                 xdescent=0.0,
    #                 ydescent=0.0,
    #             )

    #             text_list.append(textbox._text)
    #             # Create the artist for the legend which represents the
    #             # original artist/handle.
    #             handle_list.append(handler.legend_artist(self, orig_handle,
    #                                                      fontsize, handlebox))
    #             handles_and_labels.append((handlebox, textbox))

    #     columnbox = []
    #     # array_split splits n handles_and_labels into ncol columns, with the
    #     # first n%ncol columns having an extra entry.  filter(len, ...) handles
    #     # the case where n < ncol: the last ncol-n columns are empty and get
    #     # filtered out.
    #     for handles_and_labels_column \
    #             in filter(len, np.array_split(handles_and_labels, self._ncol)):
    #         # pack handlebox and labelbox into itembox
    #         itemboxes = [
    #             HPacker(
    #                 pad=0,
    #                 sep=self.handletextpad * fontsize,
    #                 children=[h, t] if markerfirst else [t, h],
    #                 align="center",
    #             )
    #             for h, t in handles_and_labels_column
    #         ]
    #         # pack columnbox
    #         alignment = "baseline" if markerfirst else "right"
    #         columnbox.append(VPacker(pad=0,
    #                                  sep=self.labelspacing * fontsize,
    #                                  align=alignment,
    #                                  children=itemboxes))

    #     mode = "expand" if self._mode == "expand" else "fixed"
    #     sep = self.columnspacing * fontsize
    #     self._legend_handle_box = HPacker(pad=0,
    #                                       sep=sep, align="baseline",
    #                                       mode=mode,
    #                                       children=columnbox)
    #     self._legend_title_box = TextArea("")
    #     self._legend_box = VPacker(pad=self.borderpad * fontsize,
    #                                sep=self.labelspacing * fontsize,
    #                                align="center",
    #                                children=[self._legend_title_box,
    #                                          self._legend_handle_box])
    #     self._legend_box.set_figure(self.figure)
    #     self._legend_box.axes = self.axes
    #     self.texts = text_list
    #     self.legendHandles = handle_list

    # def __init__(self, parent, side_title, **kwargs):
    #     self.side_title = side_title
    #     kwargs['title'] = 'a'
    #     super().__init__(parent, **kwargs)

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

        # self._legend_handle_box = HPacker(
        #     pad=0, sep=sep, align="baseline", mode=mode, children=columnbox
        # )

        self._legend_handle_box = HPacker(
            pad=0, sep=sep, align="top", mode=mode, children=columnbox
        )

        self._legend_title_box = TextArea("")

        # self._legend_box = VPacker(
        #     pad=self.borderpad * fontsize,
        #     sep=self.labelspacing * fontsize,
        #     align="center",
        #     children=[self._legend_title_box, self._legend_handle_box],
        # )

        self._legend_box = HPacker(
            pad=self.borderpad * fontsize,
            # sep=self.labelspacing * fontsize,
            sep=mpl.rcParams['axes.labelpad'],
            align="bottom",
            children=[self._legend_handle_box, self._legend_title_box],
            # children=[self._legend_handle_box],
        )

        self._legend_box.get_window_extent = get_window_extent

        self._legend_box.set_figure(self.figure)
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legendHandles = handle_list

    # def set_title(self, title, prop=None):
    #     """
    #     Set the legend title. Fontproperties can be optionally set
    #     with *prop* parameter.
    #     """
    #     self._legend_title_box._text.set_text(self.side_title)
    #     if title:
    #         self._legend_title_box._text.set_visible(True)
    #         self._legend_title_box.set_visible(True)
    #     else:
    #         self._legend_title_box._text.set_visible(False)
    #         self._legend_title_box.set_visible(False)

    #     if prop is not None:
    #         self._legend_title_box._text.set_fontproperties(
    #             prop
    #         )
    #     # import matplotlib.text
    #     # t = matplotlib.text.Text(0, 0, 'a')
    #     # dir(t)
    #     # self._legend_title_box._text.set_ha('left')
    #     self._legend_title_box._text.set_va('center')
    #     # import pdb; pdb.set_trace()
    #     w, h, x, y = self._legend_handle_box.get_extent(self.parent.figure.canvas.renderer)
    #     # self._legend_title_box._text.set_x(0)
    #     self._legend_title_box._text.set_y(h/2)
    #     self._legend_title_box._text.set_backgroundcolor('gray')
    #     self._legend_title_box._text.set_rotation(90)

    #     self.parent.figure.canvas.draw()

    #     self.stale = True

    # def get_title(self):
    #     return 'a'

    # def get_window_extent(self, renderer=None):
    #     return mtransforms.Bbox([[0, 0], [0, 0]])
    #     # # docstring inherited
    #     # if renderer is None:
    #     #     renderer = self.figure._cachedRenderer
    #     # return self._legend_box.get_window_extent(renderer=renderer)

    # def _auto_legend_data(self):
    #     """
    #     Return display coordinates for hit testing for "best" positioning.

    #     Returns
    #     -------
    #     bboxes
    #         List of bounding boxes of all patches.
    #     lines
    #         List of `.Path` corresponding to each line.
    #     offsets
    #         List of (x, y) offsets of all collection.
    #     """
    #     # import pdb; pdb.set_trace()
    #     assert self.isaxes  # always holds, as this is only called internally
    #     bboxes = []
    #     lines = []
    #     offsets = []
    #     for artist in self.parent._children:
    #         if isinstance(artist, Line2D):
    #             lines.append(
    #                 artist.get_transform().transform_path(artist.get_path()))
    #         elif isinstance(artist, Rectangle):
    #             bboxes.append(
    #                 artist.get_bbox().transformed(artist.get_data_transform()))
    #         elif isinstance(artist, Patch):
    #             bboxes.append(
    #                 artist.get_path().get_extents(artist.get_transform()))
    #         elif isinstance(artist, Collection):
    #             _, transOffset, hoffsets, _ = artist._prepare_points()
    #             for offset in transOffset.transform(hoffsets):
    #                 offsets.append(offset)
    #     return bboxes, lines, offsets

    # # def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
    # #     """
    # #     Place the *bbox* inside the *parentbbox* according to a given
    # #     location code. Return the (x, y) coordinate of the bbox.

    # #     Parameters
    # #     ----------
    # #     loc : int
    # #         A location code in range(1, 11). This corresponds to the possible
    # #         values for ``self._loc``, excluding "best".
    # #     bbox : `~matplotlib.transforms.Bbox`
    # #         bbox to be placed, in display coordinates.
    # #     parentbbox : `~matplotlib.transforms.Bbox`
    # #         A parent box which will contain the bbox, in display coordinates.
    # #     """
    # #     # import pdb; pdb.set_trace()
    # #     return offsetbox._get_anchored_bbox(
    # #         loc, bbox, parentbbox,
    # #         self.borderaxespad * renderer.points_to_pixels(self._fontsize))

def get_window_extent(self, renderer=None):
    return mtransforms.Bbox([[0, 0], [0, 0]])

class VariableSizeRectanglePatchHandler:
    def __init__(self, max_height, max_width):
        self.max_height = max_height
        self.max_width = max_width

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        # assert x0 == 0 and y0 == 0
        rel_width, rel_height = (
            orig_handle.get_width() / self.max_width,
            orig_handle.get_height() / self.max_height,
        )
        width_handlebox_coord = handlebox.width * rel_width
        height_handlebox_coord = handlebox.height * rel_height
        patch = mpatches.Rectangle(
            [0, 0],
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
        parent=ax,
        # side_title=title,
        handles=handles,
        labels=labels,
        handler_map={
            mpatches.Rectangle: VariableSizeRectanglePatchHandler(
                max_width=max_width_data_coords, max_height=max_height_data_coords
            )
        },
        handleheight=max_handle_height_fontsize_units,
        handlelength=max_handle_width_fontsize_units,
        **legend_kwargs,
    )

    # mbbox_artist(leg, ax.figure.canvas.renderer)

    return leg


import matplotlib.offsetbox as moffsetbox
'''
class TextArea(moffsetbox.OffsetBox):
    """
    The TextArea is a container artist for a single Text instance.

    The text is placed at (0, 0) with baseline+left alignment, by default. The
    width and height of the TextArea instance is the width and height of its
    child text.
    """

    @_api.delete_parameter("3.4", "minimumdescent")
    def __init__(self, s,
                 textprops=None,
                 multilinebaseline=False,
                 minimumdescent=True,
                 ):
        """
        Parameters
        ----------
        s : str
            The text to be displayed.
        textprops : dict, default: {}
            Dictionary of keyword parameters to be passed to the `.Text`
            instance in the TextArea.
        multilinebaseline : bool, default: False
            Whether the baseline for multiline text is adjusted so that it
            is (approximately) center-aligned with single-line text.
        minimumdescent : bool, default: True
            If `True`, the box has a minimum descent of "p".  This is now
            effectively always True.
        """
        if textprops is None:
            textprops = {}
        self._text = mtext.Text(0, 0, s, **textprops)
        super().__init__()
        self._children = [self._text]
        self.offset_transform = mtransforms.Affine2D()
        self._baseline_transform = mtransforms.Affine2D()
        self._text.set_transform(self.offset_transform +
                                 self._baseline_transform)
        self._multilinebaseline = multilinebaseline
        self._minimumdescent = minimumdescent

    def set_text(self, s):
        """Set the text of this area as a string."""
        self._text.set_text(s)
        self.stale = True

    def get_text(self):
        """Return the string representation of this area's text."""
        return self._text.get_text()

    def set_multilinebaseline(self, t):
        """
        Set multilinebaseline.

        If True, the baseline for multiline text is adjusted so that it is
        (approximately) center-aligned with single-line text.  This is used
        e.g. by the legend implementation so that single-line labels are
        baseline-aligned, but multiline labels are "center"-aligned with them.
        """
        self._multilinebaseline = t
        self.stale = True

    def get_multilinebaseline(self):
        """
        Get multilinebaseline.
        """
        return self._multilinebaseline

    @_api.deprecated("3.4")
    def set_minimumdescent(self, t):
        """
        Set minimumdescent.

        If True, extent of the single line text is adjusted so that
        its descent is at least the one of the glyph "p".
        """
        # The current implementation of Text._get_layout always behaves as if
        # this is True.
        self._minimumdescent = t
        self.stale = True

    @_api.deprecated("3.4")
    def get_minimumdescent(self):
        """
        Get minimumdescent.
        """
        return self._minimumdescent

    def set_transform(self, t):
        """
        set_transform is ignored.
        """

    def set_offset(self, xy):
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        """Return offset of the container."""
        return self._offset

    def get_window_extent(self, renderer):
        # docstring inherited
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset()
        return mtransforms.Bbox.from_bounds(ox - xd, oy - yd, w, h)

    def get_extent(self, renderer):
        _, h_, d_ = renderer.get_text_width_height_descent(
            "lp", self._text._fontproperties,
            ismath="TeX" if self._text.get_usetex() else False)

        bbox, info, yd = self._text._get_layout(renderer)
        w, h = bbox.size

        self._baseline_transform.clear()

        if len(info) > 1 and self._multilinebaseline:
            yd_new = 0.5 * h - 0.5 * (h_ - d_)
            self._baseline_transform.translate(0, yd - yd_new)
            yd = yd_new
        else:  # single line
            h_d = max(h_ - d_, h - yd)
            h = h_d + yd

        ha = self._text.get_horizontalalignment()
        if ha == 'left':
            xd = 0
        elif ha == 'center':
            xd = w / 2
        elif ha == 'right':
            xd = w

        return w, h, xd, yd
        # return w, h, xd, h / 2

    def draw(self, renderer):
        # docstring inherited
        self._text.draw(renderer)
        bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        self.stale = False

'''
